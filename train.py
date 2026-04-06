"""
train.py – Training script for WM-3: DreamerV3-based world model for multi-agent ecosystems.

Usage:
    python train.py --config config/training.yaml

The script implements the full Dreamer training loop:
    1. Collect experience in replay buffer (or use prefilled data)
    2. Train world model (φ) on real sequences
    3. Imagine trajectories using learned dynamics
    4. Train actor (θ) and critic (ψ) on imagined trajectories
"""

from __future__ import annotations

import argparse
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.cuda.amp import GradScaler, autocast
import yaml

from wm3 import WorldModel, symlog, symexp
from losses import lambda_return, symlog as symlog_loss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Training configuration loaded from YAML."""

    seed: int = 42

    # Batch settings
    batch_size: int = 16
    seq_len: int = 50
    prefill: int = 5000

    # World model optimizer
    wm_lr: float = 1e-4
    wm_eps: float = 1e-8
    wm_weight_decay: float = 0.0
    wm_max_grad_norm: float = 100.0

    # Actor optimizer
    actor_lr: float = 3e-5
    actor_eps: float = 1e-8
    actor_max_grad_norm: float = 100.0

    # Critic optimizer
    critic_lr: float = 3e-5
    critic_eps: float = 1e-8
    critic_max_grad_norm: float = 100.0

    # Loss weights
    decoder_weight: float = 1.0
    reward_weight: float = 1.0
    continue_weight: float = 1.0
    kl_weight: float = 0.5

    # KL settings
    free_bits: float = 1.0
    kl_balance: float = 0.8

    # Lambda return
    gamma: float = 0.997
    lam: float = 0.95

    # Schedule
    total_steps: int = 1_000_000
    warmup_steps: int = 1000
    log_every: int = 250
    eval_every: int = 10_000
    checkpoint_every: int = 50_000

    # Replay buffer
    buffer_capacity: int = 1_000_000
    min_episodes: int = 1

    # Mixed precision
    mixed_precision: bool = False
    mp_dtype: str = "float16"

    # Environment
    act_dim: int = 5
    scalar_dim: int = 16
    grid_shape: Tuple[int, int, int] = (3, 64, 64)
    imagination_horizon: int = 15

    # Actor entropy bonus
    entropy_coef: float = 3e-4

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        return cls(
            seed=data.get("seed", 42),
            batch_size=data.get("batch", {}).get("size", 16),
            seq_len=data.get("batch", {}).get("seq_len", 50),
            prefill=data.get("batch", {}).get("prefill", 5000),
            wm_lr=data.get("optimizer", {}).get("world_model", {}).get("lr", 1e-4),
            wm_eps=data.get("optimizer", {}).get("world_model", {}).get("eps", 1e-8),
            wm_weight_decay=data.get("optimizer", {}).get("world_model", {}).get("weight_decay", 0.0),
            wm_max_grad_norm=data.get("optimizer", {}).get("world_model", {}).get("max_grad_norm", 100.0),
            actor_lr=data.get("optimizer", {}).get("actor", {}).get("lr", 3e-5),
            actor_eps=data.get("optimizer", {}).get("actor", {}).get("eps", 1e-8),
            actor_max_grad_norm=data.get("optimizer", {}).get("actor", {}).get("max_grad_norm", 100.0),
            critic_lr=data.get("optimizer", {}).get("critic", {}).get("lr", 3e-5),
            critic_eps=data.get("optimizer", {}).get("critic", {}).get("eps", 1e-8),
            critic_max_grad_norm=data.get("optimizer", {}).get("critic", {}).get("max_grad_norm", 100.0),
            decoder_weight=data.get("loss_weights", {}).get("decoder", 1.0),
            reward_weight=data.get("loss_weights", {}).get("reward", 1.0),
            continue_weight=data.get("loss_weights", {}).get("continue", 1.0),
            kl_weight=data.get("loss_weights", {}).get("kl", 0.5),
            free_bits=data.get("kl", {}).get("free_bits", 1.0),
            kl_balance=data.get("kl", {}).get("balance", 0.8),
            gamma=data.get("lambda_return", {}).get("gamma", 0.997),
            lam=data.get("lambda_return", {}).get("lam", 0.95),
            total_steps=data.get("schedule", {}).get("total_steps", 1_000_000),
            warmup_steps=data.get("schedule", {}).get("warmup_steps", 1000),
            log_every=data.get("schedule", {}).get("log_every", 250),
            eval_every=data.get("schedule", {}).get("eval_every", 10_000),
            checkpoint_every=data.get("schedule", {}).get("checkpoint_every", 50_000),
            buffer_capacity=data.get("replay_buffer", {}).get("capacity", 1_000_000),
            min_episodes=data.get("replay_buffer", {}).get("min_episodes", 1),
            mixed_precision=data.get("mixed_precision", {}).get("enabled", False),
            mp_dtype=data.get("mixed_precision", {}).get("dtype", "float16"),
        )


@dataclass
class Episode:
    """Single episode of experience."""

    grids: List[Tensor] = field(default_factory=list)
    scalars: List[Tensor] = field(default_factory=list)
    actions: List[Tensor] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    continues: List[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.rewards)


class ReplayBuffer:
    """Episode-based replay buffer with sequence sampling."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.episodes: List[Episode] = []
        self.total_steps = 0

    def add_episode(self, episode: Episode) -> None:
        """Add an episode to the buffer."""
        self.episodes.append(episode)
        self.total_steps += len(episode)

        while self.total_steps > self.capacity and len(self.episodes) > 1:
            removed = self.episodes.pop(0)
            self.total_steps -= len(removed)

    def sample_sequences(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> Dict[str, Tensor]:
        """Sample random sequences from the buffer.

        Returns:
            dict with keys:
                - grids: (T, B, 3, 64, 64)
                - scalars: (T, B, scalar_dim)
                - actions: (T, B, act_dim)
                - rewards: (T, B)
                - continues: (T, B)
        """
        grids, scalars, actions, rewards, continues = [], [], [], [], []

        for _ in range(batch_size):
            valid_episodes = [ep for ep in self.episodes if len(ep) >= seq_len]
            if not valid_episodes:
                valid_episodes = self.episodes

            ep = random.choice(valid_episodes)
            max_start = max(0, len(ep) - seq_len)
            start = random.randint(0, max_start)
            end = min(start + seq_len, len(ep))

            grids.append(torch.stack(ep.grids[start:end]))
            scalars.append(torch.stack(ep.scalars[start:end]))
            actions.append(torch.stack(ep.actions[start:end]))
            rewards.append(torch.tensor(ep.rewards[start:end], dtype=torch.float32))
            continues.append(torch.tensor(ep.continues[start:end], dtype=torch.float32))

        actual_len = min(g.shape[0] for g in grids)
        grids = torch.stack([g[:actual_len] for g in grids], dim=1).to(device)
        scalars = torch.stack([s[:actual_len] for s in scalars], dim=1).to(device)
        actions = torch.stack([a[:actual_len] for a in actions], dim=1).to(device)
        rewards = torch.stack([r[:actual_len] for r in rewards], dim=1).to(device)
        continues = torch.stack([c[:actual_len] for c in continues], dim=1).to(device)

        return {
            "grids": grids,
            "scalars": scalars,
            "actions": actions,
            "rewards": rewards,
            "continues": continues,
        }

    def __len__(self) -> int:
        return self.total_steps


def categorical_kl_divergence(
    post_logits: Tensor,
    prior_logits: Tensor,
    free_bits: float = 1.0,
    balance: float = 0.8,
) -> Tuple[Tensor, Tensor]:
    """Compute KL divergence between categorical distributions with free bits and balancing.

    Args:
        post_logits: (B, z_cats, z_classes) posterior logits
        prior_logits: (B, z_cats, z_classes) prior logits
        free_bits: minimum KL per categorical (nats)
        balance: fraction of gradient to posterior (rest to prior)

    Returns:
        kl_loss: scalar KL loss for optimization
        kl_raw: scalar raw KL for logging
    """
    post_probs = F.softmax(post_logits, dim=-1)
    prior_probs = F.softmax(prior_logits, dim=-1)

    post_logprobs = F.log_softmax(post_logits, dim=-1)
    prior_logprobs = F.log_softmax(prior_logits, dim=-1)

    kl_per_cat = (post_probs * (post_logprobs - prior_logprobs)).sum(dim=-1)
    kl_raw = kl_per_cat.sum(dim=-1).mean()

    kl_clamped = torch.clamp(kl_per_cat, min=free_bits)

    kl_to_post = (post_probs.detach() * (post_logprobs - prior_logprobs.detach())).sum(dim=-1)
    kl_to_prior = (post_probs.detach() * (post_logprobs.detach() - prior_logprobs)).sum(dim=-1)

    kl_balanced = balance * kl_to_post + (1 - balance) * kl_to_prior
    kl_loss = torch.clamp(kl_balanced, min=free_bits).sum(dim=-1).mean()

    return kl_loss, kl_raw


def world_model_step(
    model: WorldModel,
    batch: Dict[str, Tensor],
    config: Config,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Compute world model loss on a batch of sequences.

    Args:
        model: WorldModel instance
        batch: dict with grids, scalars, actions, rewards, continues
        config: training configuration

    Returns:
        loss: scalar total loss
        metrics: dict of detached metrics for logging
    """
    T, B = batch["rewards"].shape
    device = batch["rewards"].device

    h, z = model.initial_state(B, device)
    a_prev = torch.zeros(B, model.act_dim, device=device)

    total_recon_loss = 0.0
    total_reward_loss = 0.0
    total_cont_loss = 0.0
    total_kl_loss = 0.0
    total_kl_raw = 0.0

    for t in range(T):
        obs = {
            "grid": batch["grids"][t],
            "scalars": batch["scalars"][t],
        }
        target_grid = batch["grids"][t]
        target_reward = batch["rewards"][t]
        target_cont = batch["continues"][t]

        h, z, s, preds, prior_logits, post_logits = model(
            obs, h, z, a_prev, agents=torch.empty(B, 0, model.x_dim, device=device)
        )

        recon_loss = F.mse_loss(preds["decoder"], target_grid)

        reward_pred = preds["reward"].squeeze(-1)
        reward_loss = F.mse_loss(reward_pred, symlog(target_reward))

        cont_logits = torch.logit(preds["cont"].squeeze(-1).clamp(1e-6, 1 - 1e-6))
        cont_loss = F.binary_cross_entropy_with_logits(cont_logits, target_cont)

        kl_loss, kl_raw = categorical_kl_divergence(
            post_logits, prior_logits,
            free_bits=config.free_bits,
            balance=config.kl_balance,
        )

        total_recon_loss = total_recon_loss + recon_loss
        total_reward_loss = total_reward_loss + reward_loss
        total_cont_loss = total_cont_loss + cont_loss
        total_kl_loss = total_kl_loss + kl_loss
        total_kl_raw = total_kl_raw + kl_raw

        a_prev = batch["actions"][t]

    total_recon_loss = total_recon_loss / T
    total_reward_loss = total_reward_loss / T
    total_cont_loss = total_cont_loss / T
    total_kl_loss = total_kl_loss / T
    total_kl_raw = total_kl_raw / T

    loss = (
        config.decoder_weight * total_recon_loss
        + config.reward_weight * total_reward_loss
        + config.continue_weight * total_cont_loss
        + config.kl_weight * total_kl_loss
    )

    metrics = {
        "loss/wm_total": loss.detach(),
        "loss/recon": total_recon_loss.detach(),
        "loss/reward": total_reward_loss.detach(),
        "loss/continue": total_cont_loss.detach(),
        "loss/kl": total_kl_loss.detach(),
        "loss/kl_raw": total_kl_raw.detach(),
    }

    return loss, metrics


def imagine_trajectories(
    model: WorldModel,
    batch: Dict[str, Tensor],
    horizon: int,
    device: torch.device,
) -> Dict[str, Tensor]:
    """Generate imagined trajectories for actor-critic training.

    Args:
        model: WorldModel instance
        batch: dict with initial observations
        horizon: imagination horizon
        device: torch device

    Returns:
        dict with stacked tensors:
            - states: (H, B, x_dim)
            - rewards: (H, B)
            - continues: (H, B)
            - values: (H, B)
            - action_logits: (H, B, act_dim)
    """
    B = batch["grids"].shape[1]

    with torch.no_grad():
        obs = {
            "grid": batch["grids"][0],
            "scalars": batch["scalars"][0],
        }
        h, z = model.initial_state(B, device)
        a_prev = torch.zeros(B, model.act_dim, device=device)
        h, z, s, _, _, _ = model(
            obs, h, z, a_prev,
            agents=torch.empty(B, 0, model.x_dim, device=device)
        )

    h = h.detach()
    z = z.detach()

    def actor_fn(s: Tensor) -> Tensor:
        return model.heads.actor_net(s.detach())

    traj = model.imagine(h, z, actor_fn, horizon)

    states = traj["s"]
    rewards = traj["reward"].squeeze(-1)
    continues = traj["cont"].squeeze(-1)

    with torch.no_grad():
        values = model.heads.critic_net(states).squeeze(-1)

    return {
        "states": states,
        "rewards": rewards,
        "continues": continues,
        "values": values,
        "h": traj["h"],
        "z": traj["z"],
        "actions": traj["a"],
    }


def actor_critic_step(
    model: WorldModel,
    traj: Dict[str, Tensor],
    config: Config,
) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
    """Compute actor and critic losses on imagined trajectories.

    Args:
        model: WorldModel instance
        traj: imagined trajectory dict
        config: training configuration

    Returns:
        actor_loss: scalar actor loss
        critic_loss: scalar critic loss
        metrics: dict of detached metrics
    """
    states = traj["states"]
    rewards = traj["rewards"]
    continues = traj["continues"]

    H, B = rewards.shape

    with torch.no_grad():
        values = model.heads.critic_net(states).squeeze(-1)
        bootstrap = model.heads.critic_net(states[-1:]).squeeze(-1)
        values_with_bootstrap = torch.cat([values, bootstrap], dim=0)

    lambda_returns = lambda_return(
        rewards=rewards,
        continues=continues,
        values=values_with_bootstrap,
        gamma=config.gamma,
        lam=config.lam,
    )

    returns_mean = lambda_returns.mean()
    returns_std = lambda_returns.std().clamp(min=1.0)
    returns_norm = (lambda_returns - returns_mean) / returns_std

    actor_logits = model.heads.actor_net(states.detach())
    action_dist = torch.distributions.Categorical(logits=actor_logits)
    entropy = action_dist.entropy()

    actor_loss = -returns_norm.mean() - config.entropy_coef * entropy.mean()

    critic_values = model.heads.critic_net(states.detach()).squeeze(-1)
    critic_loss = F.mse_loss(critic_values, lambda_returns.detach())

    metrics = {
        "loss/actor": actor_loss.detach(),
        "loss/critic": critic_loss.detach(),
        "actor/entropy": entropy.mean().detach(),
        "actor/returns_mean": returns_mean.detach(),
        "actor/returns_std": returns_std.detach(),
        "critic/value_mean": critic_values.mean().detach(),
    }

    return actor_loss, critic_loss, metrics


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create learning rate scheduler with linear warmup and cosine decay."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    path: Path,
    model: WorldModel,
    opt_wm: torch.optim.Optimizer,
    opt_actor: torch.optim.Optimizer,
    opt_critic: torch.optim.Optimizer,
    step: int,
    metrics: Dict[str, float],
) -> None:
    """Save training checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "opt_wm_state_dict": opt_wm.state_dict(),
            "opt_actor_state_dict": opt_actor.state_dict(),
            "opt_critic_state_dict": opt_critic.state_dict(),
            "metrics": metrics,
        },
        path,
    )
    logger.info(f"Saved checkpoint to {path}")


def load_checkpoint(
    path: Path,
    model: WorldModel,
    opt_wm: torch.optim.Optimizer,
    opt_actor: torch.optim.Optimizer,
    opt_critic: torch.optim.Optimizer,
) -> int:
    """Load training checkpoint. Returns the step number."""
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    opt_wm.load_state_dict(checkpoint["opt_wm_state_dict"])
    opt_actor.load_state_dict(checkpoint["opt_actor_state_dict"])
    opt_critic.load_state_dict(checkpoint["opt_critic_state_dict"])
    logger.info(f"Loaded checkpoint from {path} at step {checkpoint['step']}")
    return checkpoint["step"]


def generate_synthetic_episode(
    seq_len: int,
    act_dim: int,
    scalar_dim: int,
    grid_shape: Tuple[int, int, int],
) -> Episode:
    """Generate a synthetic episode for testing (replace with real env)."""
    episode = Episode()

    for t in range(seq_len):
        episode.grids.append(torch.rand(*grid_shape))
        episode.scalars.append(torch.randn(scalar_dim))
        episode.actions.append(F.one_hot(
            torch.randint(0, act_dim, (1,)).squeeze(),
            num_classes=act_dim,
        ).float())
        episode.rewards.append(random.gauss(0, 1))
        episode.continues.append(1.0 if t < seq_len - 1 else 0.0)

    return episode


class Trainer:
    """Main trainer class for WM-3."""

    def __init__(
        self,
        config: Config,
        checkpoint_dir: Path,
        device: torch.device,
    ):
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        self.device = device

        self._set_seed(config.seed)

        self.model = WorldModel(
            act_dim=config.act_dim,
            scalar_dim=config.scalar_dim,
        ).to(device)

        self.opt_wm = torch.optim.Adam(
            self.model.phi.parameters(),
            lr=config.wm_lr,
            eps=config.wm_eps,
            weight_decay=config.wm_weight_decay,
        )

        actor_params = list(self.model.heads.actor_net.parameters())
        self.opt_actor = torch.optim.Adam(
            actor_params,
            lr=config.actor_lr,
            eps=config.actor_eps,
        )

        critic_params = list(self.model.heads.critic_net.parameters())
        self.opt_critic = torch.optim.Adam(
            critic_params,
            lr=config.critic_lr,
            eps=config.critic_eps,
        )

        self.scheduler_wm = get_lr_scheduler(
            self.opt_wm, config.warmup_steps, config.total_steps
        )

        self.scaler = GradScaler(enabled=config.mixed_precision)

        self.buffer = ReplayBuffer(config.buffer_capacity)

        self.step = 0
        self.metrics_history: List[Dict[str, float]] = []

    def _set_seed(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def prefill_buffer(self) -> None:
        """Prefill replay buffer with random policy data."""
        logger.info(f"Prefilling buffer with {self.config.prefill} steps...")

        while len(self.buffer) < self.config.prefill:
            episode = generate_synthetic_episode(
                seq_len=self.config.seq_len,
                act_dim=self.config.act_dim,
                scalar_dim=self.config.scalar_dim,
                grid_shape=self.config.grid_shape,
            )
            self.buffer.add_episode(episode)

        logger.info(f"Buffer prefilled with {len(self.buffer)} steps")

    def train_step(self) -> Dict[str, float]:
        """Execute one training step."""
        self.model.train()

        batch = self.buffer.sample_sequences(
            self.config.batch_size,
            self.config.seq_len,
            self.device,
        )

        amp_dtype = torch.float16 if self.config.mp_dtype == "float16" else torch.bfloat16
        with autocast(enabled=self.config.mixed_precision, dtype=amp_dtype):
            wm_loss, wm_metrics = world_model_step(self.model, batch, self.config)

        self.opt_wm.zero_grad()
        self.scaler.scale(wm_loss).backward()
        self.scaler.unscale_(self.opt_wm)
        nn.utils.clip_grad_norm_(self.model.phi.parameters(), self.config.wm_max_grad_norm)
        self.scaler.step(self.opt_wm)
        self.scaler.update()
        self.scheduler_wm.step()

        with autocast(enabled=self.config.mixed_precision, dtype=amp_dtype):
            traj = imagine_trajectories(
                self.model, batch, self.config.imagination_horizon, self.device
            )
            actor_loss, critic_loss, ac_metrics = actor_critic_step(
                self.model, traj, self.config
            )

        self.opt_actor.zero_grad()
        self.scaler.scale(actor_loss).backward()
        self.scaler.unscale_(self.opt_actor)
        nn.utils.clip_grad_norm_(
            self.model.heads.actor_net.parameters(),
            self.config.actor_max_grad_norm,
        )
        self.scaler.step(self.opt_actor)

        self.opt_critic.zero_grad()
        self.scaler.scale(critic_loss).backward()
        self.scaler.unscale_(self.opt_critic)
        nn.utils.clip_grad_norm_(
            self.model.heads.critic_net.parameters(),
            self.config.critic_max_grad_norm,
        )
        self.scaler.step(self.opt_critic)

        self.scaler.update()

        metrics = {**wm_metrics, **ac_metrics}
        metrics["lr/wm"] = self.opt_wm.param_groups[0]["lr"]
        metrics["lr/actor"] = self.opt_actor.param_groups[0]["lr"]
        metrics["lr/critic"] = self.opt_critic.param_groups[0]["lr"]

        return {k: v.item() if isinstance(v, Tensor) else v for k, v in metrics.items()}

    def train(self) -> None:
        """Main training loop."""
        logger.info("Starting training...")
        logger.info(f"  Total steps: {self.config.total_steps}")
        logger.info(f"  Batch size: {self.config.batch_size}")
        logger.info(f"  Sequence length: {self.config.seq_len}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Mixed precision: {self.config.mixed_precision}")

        self.prefill_buffer()

        while self.step < self.config.total_steps:
            metrics = self.train_step()
            self.step += 1
            self.metrics_history.append(metrics)

            if self.step % self.config.log_every == 0:
                avg_metrics = {}
                for key in metrics.keys():
                    values = [m[key] for m in self.metrics_history[-self.config.log_every:]]
                    avg_metrics[key] = sum(values) / len(values)

                log_str = f"Step {self.step:>7d}"
                for key, value in sorted(avg_metrics.items()):
                    if key.startswith("loss/"):
                        log_str += f" | {key}: {value:.4f}"
                logger.info(log_str)

            if self.step % self.config.checkpoint_every == 0:
                save_checkpoint(
                    self.checkpoint_dir / f"checkpoint_{self.step}.pt",
                    self.model,
                    self.opt_wm,
                    self.opt_actor,
                    self.opt_critic,
                    self.step,
                    metrics,
                )

        logger.info("Training complete!")
        save_checkpoint(
            self.checkpoint_dir / "checkpoint_final.pt",
            self.model,
            self.opt_wm,
            self.opt_actor,
            self.opt_critic,
            self.step,
            metrics,
        )


def main():
    parser = argparse.ArgumentParser(description="Train WM-3 world model")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/training.yaml"),
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory for saving checkpoints",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on",
    )
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    device = torch.device(args.device)

    trainer = Trainer(config, args.checkpoint_dir, device)

    if args.resume:
        trainer.step = load_checkpoint(
            args.resume,
            trainer.model,
            trainer.opt_wm,
            trainer.opt_actor,
            trainer.opt_critic,
        )

    trainer.train()


if __name__ == "__main__":
    main()
