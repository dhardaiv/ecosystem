"""
Training loop with:
  - Teacher forcing (phase 1 & 2)
  - Scheduled sampling (phase 3) — with probability ss_rate, replace ground
    truth context with model predictions at each step in a multi-step window
  - Straight-through alive masking (hard mask forward, soft grad backward)
  - Alive mask noise injection (training robustness)
  - Three curriculum phases
  - Full wandb logging (every step for train, every epoch for val)
"""
import os
import math
import time
from typing import Optional, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb

from config import ModelConfig, TrainConfig, DataConfig, all_configs_as_dict
from model import EcosystemWorldModel
from model.heads import compute_loss
from data.dataset import compute_deltas, discretise_food
from training.utils import straight_through_alive, inject_alive_noise, apply_predictions_to_state


# ── Main Trainer ──────────────────────────────────────────────────────────────

class Trainer:
    def __init__(
        self,
        model: EcosystemWorldModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        train_cfg: TrainConfig,
        model_cfg: ModelConfig,
        val_episodes: Optional[List] = None,   # raw episodes for rollout eval
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = train_cfg
        self.mcfg = model_cfg
        self.val_episodes = val_episodes

        self.device = torch.device("cuda" if torch.cuda.is_available() else
                                   "mps"  if torch.backends.mps.is_available() else
                                   "cpu")
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=train_cfg.n_epochs, eta_min=train_cfg.lr * 0.05
        )

        self.global_step = 0
        self.best_val_loss = float("inf")
        self.ss_rate = train_cfg.ss_rate_init   # non-zero from epoch 1
        self.current_phase = 1
        self.rollout_k = train_cfg.rollout_k_start

        os.makedirs(os.path.dirname(train_cfg.checkpoint_path), exist_ok=True)

    # ── Curriculum Phase Management ───────────────────────────────────────────

    def _update_phase(self, epoch: int):
        cfg = self.cfg
        new_phase = self.current_phase

        if epoch >= cfg.phase3_epoch and self.current_phase < 3:
            new_phase = 3
        elif epoch >= cfg.phase2_epoch and self.current_phase < 2:
            new_phase = 2

        if new_phase != self.current_phase:
            self.current_phase = new_phase
            if new_phase == 2:
                self.cfg.lambda_aux = 0.1
                print(f"\n[Curriculum] → Phase 2: λ_aux = {self.cfg.lambda_aux}")
            elif new_phase == 3:
                self.cfg.lambda_aux = 0.1
                print(f"\n[Curriculum] → Phase 3: multi-step rollout k={self.rollout_k}")
            wandb.config.update({"train/phase": self.current_phase}, allow_val_change=True)
            wandb.log({"train/phase": self.current_phase, "step": self.global_step})

    # ── Single-step training pass (teacher forced or scheduled sampling) ──────

    def _forward_single(self, batch: dict) -> dict:
        """Teacher-forced single-step prediction."""
        agents_t  = batch["agents_t"].to(self.device)
        patches_t = batch["patches_t"].to(self.device)
        agents_t1 = batch["agents_t1"].to(self.device)
        patches_t1 = batch["patches_t1"].to(self.device)
        counts_t1 = batch["counts_t1"].to(self.device)

        # Inject alive noise for training robustness
        if self.cfg.alive_noise_rate > 0:
            agents_t = agents_t.clone()
            agents_t[:, :, 5] = inject_alive_noise(
                agents_t[:, :, 5], self.cfg.alive_noise_rate
            )

        preds = self.model(agents_t, patches_t)

        losses = compute_loss(
            preds, agents_t, agents_t1, patches_t1, counts_t1,
            self.cfg, self.mcfg,
        )
        return losses, preds, agents_t[:, :, 5]

    def _forward_multistep(self, batch: dict, k: int) -> dict:
        """
        Scheduled-sampling multi-step rollout loss.
        With probability ss_rate, use model predictions as context at each step;
        otherwise use ground truth (teacher forcing).

        Only supported for k=1 (single pairs in our dataset).
        We simulate k-step rollout by chaining k separate predictions from
        the episode state starting at batch['agents_t'].
        Since our dataset only has 1-step pairs, we reuse the model's own output
        as the next-step input for steps 2..k (scheduled sampling style).
        """
        agents  = batch["agents_t"].to(self.device)
        patches = batch["patches_t"].to(self.device)
        agents_t1  = batch["agents_t1"].to(self.device)
        patches_t1 = batch["patches_t1"].to(self.device)
        counts_t1  = batch["counts_t1"].to(self.device)

        total_loss = None
        for step_i in range(k):
            if self.cfg.alive_noise_rate > 0:
                agents = agents.clone()
                agents[:, :, 5] = inject_alive_noise(
                    agents[:, :, 5], self.cfg.alive_noise_rate
                )

            preds = self.model(agents, patches)
            if step_i == 0:
                # First step: always use ground truth targets
                losses = compute_loss(
                    preds, agents, agents_t1, patches_t1, counts_t1,
                    self.cfg, self.mcfg,
                )
                total_loss = losses
            else:
                # Subsequent steps: loss is dominated by consistency
                losses_i = compute_loss(
                    preds, agents, agents_t1, patches_t1, counts_t1,
                    self.cfg, self.mcfg,
                )
                # Accumulate (simple average over unroll)
                total_loss["loss_total"] = total_loss["loss_total"] + losses_i["loss_total"]

            # Determine next input: use GT or model prediction (scheduled sampling)
            use_model = torch.rand(1).item() < self.ss_rate
            if use_model and step_i < k - 1:
                agents, patches = apply_predictions_to_state(agents, patches, preds)
            else:
                agents  = agents_t1.clone()
                patches = patches_t1.clone()

        if k > 1:
            total_loss["loss_total"] = total_loss["loss_total"] / k

        return total_loss, preds, agents[:, :, 5]

    # ── Training epoch ────────────────────────────────────────────────────────

    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        accum = {k: 0.0 for k in ["loss_total", "loss_mse", "loss_bce", "loss_ce", "loss_aux", "loss_var_reg"]}
        n_batches = 0

        # Use multi-step whenever ss_rate > 0 (scheduled sampling active),
        # even in phases 1 and 2.  In phase 3 rollout_k grows up to k_max;
        # earlier phases use a fixed k=2 to expose the model to its own errors.
        use_rollout = self.ss_rate > 0 or self.current_phase >= 3
        if self.current_phase >= 3:
            k = min(self.rollout_k, self.cfg.rollout_k_max)
        else:
            k = 2  # minimal 2-step window for early scheduled sampling

        for batch in self.train_loader:
            self.optimizer.zero_grad()

            if use_rollout:
                losses, preds, alive_t = self._forward_multistep(batch, k)
            else:
                losses, preds, alive_t = self._forward_single(batch)

            loss = losses["loss_total"]
            loss.backward()

            grad_norm = nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.grad_clip
            )
            # Tighter clip on aux head to prevent early-training population
            # count gradients from overwhelming the shared transformer weights.
            nn.utils.clip_grad_norm_(
                self.model.heads.aux_head.parameters(),
                self.cfg.grad_clip_aux,
            )

            self.optimizer.step()

            # ── per-step wandb log ─────────────────────────────────────────
            alive_frac = alive_t.mean().item()
            aux_grad_norm = sum(
                p.grad.norm().item() ** 2
                for p in self.model.heads.aux_head.parameters()
                if p.grad is not None
            ) ** 0.5
            wandb.log({
                "train/loss_total":    losses["loss_total"].item(),
                "train/loss_mse":      losses["loss_mse"].item(),
                "train/loss_bce":      losses["loss_bce"].item(),
                "train/loss_ce":       losses["loss_ce"].item(),
                "train/loss_aux":      losses["loss_aux"].item(),
                "train/loss_var_reg":  losses["loss_var_reg"].item(),
                "train/alive_frac":    alive_frac,
                "train/grad_norm":     grad_norm.item(),
                "train/grad_norm_aux": aux_grad_norm,
                "train/ss_rate":       self.ss_rate,
                "train/phase":         self.current_phase,
                "step":                self.global_step,
            })

            for k_name in accum:
                val_to_add = losses[k_name]
                accum[k_name] += (val_to_add.item() if hasattr(val_to_add, 'item') else float(val_to_add))
            n_batches += 1
            self.global_step += 1

        return {k: v / max(n_batches, 1) for k, v in accum.items()}

    # ── Validation epoch ──────────────────────────────────────────────────────

    @torch.no_grad()
    def val_epoch(self, epoch: int) -> dict:
        self.model.eval()
        from sklearn.metrics import f1_score
        accum = {k: 0.0 for k in ["loss_total", "loss_mse", "loss_bce", "loss_ce", "loss_aux", "loss_var_reg"]}
        pos_mse_sum = 0.0
        energy_mse_sum = 0.0
        alive_acc_sum = 0.0
        all_alive_true, all_alive_pred = [], []
        n_batches = 0

        for batch in self.val_loader:
            agents_t   = batch["agents_t"].to(self.device)
            patches_t  = batch["patches_t"].to(self.device)
            agents_t1  = batch["agents_t1"].to(self.device)
            patches_t1 = batch["patches_t1"].to(self.device)
            counts_t1  = batch["counts_t1"].to(self.device)

            preds = self.model(agents_t, patches_t)
            losses = compute_loss(
                preds, agents_t, agents_t1, patches_t1, counts_t1,
                self.cfg, self.mcfg,
            )

            for k_name in accum:
                val_to_add = losses[k_name]
                accum[k_name] += (val_to_add.item() if hasattr(val_to_add, 'item') else float(val_to_add))

            # Position MSE (x, y only, alive at t)
            alive_t = agents_t[:, :, 5]
            pos_t1_pred = agents_t[:, :, 1:3] + preds["delta_pred"][:, :, :2]
            pos_t1_gt   = agents_t1[:, :, 1:3]
            pos_mse = ((pos_t1_pred - pos_t1_gt) ** 2).sum(-1)
            denom = alive_t.sum().clamp(min=1.0)
            pos_mse_sum += (alive_t * pos_mse).sum().item() / denom.item()

            # Energy MSE
            en_t1_pred = agents_t[:, :, 3] + preds["delta_pred"][:, :, 2]
            en_t1_gt   = agents_t1[:, :, 3]
            en_mse = (en_t1_pred - en_t1_gt) ** 2
            energy_mse_sum += (alive_t * en_mse).sum().item() / denom.item()

            # Alive accuracy / F1
            alive_pred_hard = (preds["alive_probs"] > 0.5).long().cpu().numpy().ravel()
            alive_gt_hard   = agents_t1[:, :, 5].long().cpu().numpy().ravel()
            alive_acc_sum += (alive_pred_hard == alive_gt_hard).mean()
            all_alive_true.extend(alive_gt_hard.tolist())
            all_alive_pred.extend(alive_pred_hard.tolist())

            n_batches += 1

        import numpy as np
        alive_f1 = f1_score(all_alive_true, all_alive_pred, pos_label=0,
                            zero_division=0)  # F1 on dead class

        metrics = {k: v / max(n_batches, 1) for k, v in accum.items()}
        metrics["val/position_mse"]  = pos_mse_sum / max(n_batches, 1)
        metrics["val/energy_mse"]    = energy_mse_sum / max(n_batches, 1)
        metrics["val/alive_accuracy"] = alive_acc_sum / max(n_batches, 1)
        metrics["val/alive_f1"]       = alive_f1

        wandb.log({
            "val/loss_total":     metrics["loss_total"],
            "val/loss_mse":       metrics["loss_mse"],
            "val/loss_bce":       metrics["loss_bce"],
            "val/loss_ce":        metrics["loss_ce"],
            "val/loss_aux":       metrics["loss_aux"],
            "val/loss_var_reg":   metrics["loss_var_reg"],
            "val/position_mse":   metrics["val/position_mse"],
            "val/energy_mse":     metrics["val/energy_mse"],
            "val/alive_accuracy": metrics["val/alive_accuracy"],
            "val/alive_f1":       metrics["val/alive_f1"],
            "epoch": epoch,
        })

        return metrics

    # ── Checkpoint helpers ────────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, val_loss: float):
        state = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "global_step": self.global_step,
            "ss_rate": self.ss_rate,
            "phase": self.current_phase,
        }
        torch.save(state, self.cfg.checkpoint_path)
        wandb.save(self.cfg.checkpoint_path)
        artifact = wandb.Artifact("world-model", type="model")
        artifact.add_file(self.cfg.checkpoint_path)
        wandb.log_artifact(artifact)
        print(f"  ✓ Checkpoint saved (val_loss={val_loss:.4f})")

    def load_checkpoint(self, path: Optional[str] = None):
        ckpt_path = path or self.cfg.checkpoint_path
        if not os.path.exists(ckpt_path):
            return
        state = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(state["model_state"])
        self.optimizer.load_state_dict(state["optimizer_state"])
        self.global_step = state.get("global_step", 0)
        self.ss_rate = state.get("ss_rate", 0.0)
        self.current_phase = state.get("phase", 1)
        print(f"Resumed from {ckpt_path} (epoch {state['epoch']}, "
              f"val_loss={state['val_loss']:.4f})")

    # ── Full training run ─────────────────────────────────────────────────────

    def fit(self):
        print(f"Training on {self.device}. "
              f"{sum(p.numel() for p in self.model.parameters()):,} parameters.")

        for epoch in range(1, self.cfg.n_epochs + 1):
            self._update_phase(epoch)

            # ss_rate grows every epoch from the start, capped at ss_rate_max
            self.ss_rate = min(
                self.cfg.ss_rate_max,
                self.ss_rate + self.cfg.ss_rate_increment,
            )
            if self.current_phase >= 3:
                self.rollout_k = min(
                    self.cfg.rollout_k_max,
                    self.cfg.rollout_k_start + (epoch - self.cfg.phase3_epoch) // 5,
                )

            train_metrics = self.train_epoch(epoch)

            if epoch % self.cfg.val_every_n_epochs == 0:
                val_metrics = self.val_epoch(epoch)
                val_loss = val_metrics["loss_total"]
                print(
                    f"Epoch {epoch:3d} | "
                    f"train_loss={train_metrics['loss_total']:.4f} | "
                    f"val_loss={val_loss:.4f} | "
                    f"val_f1(dead)={val_metrics['val/alive_f1']:.3f} | "
                    f"phase={self.current_phase}"
                )

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint(epoch, val_loss)

        if epoch % self.cfg.rollout_eval_every_n_epochs == 0 and self.val_episodes:
            from training.rollout import run_rollout_evaluation
            run_rollout_evaluation(
                self.model, self.val_episodes,
                horizon=self.cfg.rollout_horizon,
                device=self.device,
                model_cfg=self.mcfg,
                epoch=epoch,
            )

            self.scheduler.step()

        print("Training complete.")
