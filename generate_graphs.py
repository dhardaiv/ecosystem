"""
Standalone graph generation script.

Trains the world model from scratch on demo_episodes for a short curriculum,
then runs rollout evaluation and saves all graphs to graphs/training/ and
graphs/rollout/.  Does NOT require wandb.

Usage:
    python generate_graphs.py [--epochs 40] [--data_dir data/demo_episodes]
"""
import argparse
import os
import pickle
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from scipy.special import kl_div

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Project imports ──────────────────────────────────────────────────────────
import sys
sys.path.insert(0, os.path.dirname(__file__))

from config import ModelConfig, TrainConfig
from model import EcosystemWorldModel
from model.heads import compute_loss
from data.dataset import EcosystemDataset
from training.utils import apply_predictions_to_state, DIRECTION_VECS
from simulator.agents import SPECIES_PREY, SPECIES_PREDATOR


# ── Config tweaked for demo data ──────────────────────────────────────────────

def make_demo_configs(data_dir: str):
    """Build model + train configs sized to the demo dataset."""
    with open(os.path.join(data_dir, "train.pkl"), "rb") as f:
        eps = pickle.load(f)
    n_max = eps[0][0][0].shape[0]     # actual agent slots in demo
    grid_w = 40                        # same as SimConfig default

    mcfg = ModelConfig(
        d_model=64,
        n_layers=3,
        n_heads=4,
        d_ff=128,
        n_max_agents=n_max,
        grid_size=grid_w,
    )
    tcfg = TrainConfig(
        lambda_move=1.0,
        lambda_energy=1.0,
        lambda_bce=5.0,
        lambda_ce=0.5,
        lambda_aux=0.0,
        bce_dead_weight=10.0,
        energy_var_reg=0.1,
        energy_var_threshold=0.001,
        lr=3e-4,
        weight_decay=1e-4,
        batch_size=32,
        grad_clip=1.0,
        grad_clip_aux=0.1,
        ss_rate_init=0.0,
        ss_rate_max=0.0,
        alive_noise_rate=0.02,
        phase2_epoch=15,
        phase3_epoch=999,
    )
    return mcfg, tcfg, eps


# ── Training loop (no wandb) ──────────────────────────────────────────────────

def train(model, loader, val_loader, tcfg, mcfg, n_epochs, device):
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=tcfg.lr * 0.05)

    history = {k: [] for k in [
        "train_loss", "train_movement", "train_energy", "train_bce", "train_ce",
        "train_entropy", "train_frac_stay",
        "val_loss", "val_movement", "val_energy", "val_pos_mse", "val_energy_mse",
        "val_alive_acc",
    ]}

    dir_vecs  = DIRECTION_VECS.to(device)
    step_norm = 1.0 / mcfg.grid_size

    for epoch in range(1, n_epochs + 1):
        # Phase 2: enable aux loss after phase2_epoch
        if epoch == tcfg.phase2_epoch:
            tcfg.lambda_aux = 0.1
            print(f"  [Phase 2] lambda_aux = {tcfg.lambda_aux}")

        # ── Train ────────────────────────────────────────────────────────
        model.train()
        accum = {k: 0.0 for k in ["loss_total","loss_movement","loss_energy","loss_bce","loss_ce","movement_entropy","frac_stay_predicted"]}
        n = 0
        for batch in loader:
            agents_t   = batch["agents_t"].to(device)
            patches_t  = batch["patches_t"].to(device)
            agents_t1  = batch["agents_t1"].to(device)
            patches_t1 = batch["patches_t1"].to(device)
            counts_t1  = batch["counts_t1"].to(device)

            if tcfg.alive_noise_rate > 0:
                agents_t = agents_t.clone()
                flip = (torch.rand_like(agents_t[:,:,5]) < tcfg.alive_noise_rate) & (agents_t[:,:,5] > 0.5)
                agents_t[:,:,5] = agents_t[:,:,5] * (~flip).float()

            opt.zero_grad()
            preds  = model(agents_t, patches_t)
            losses = compute_loss(preds, agents_t, agents_t1, patches_t1, counts_t1, tcfg, mcfg)
            losses["loss_total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
            nn.utils.clip_grad_norm_(model.heads.aux_head.parameters(), tcfg.grad_clip_aux)
            opt.step()

            for k in accum:
                v = losses[k]
                accum[k] += (v.item() if hasattr(v, "item") else float(v))
            n += 1

        sched.step()
        avg = {k: v / max(n, 1) for k, v in accum.items()}
        history["train_loss"].append(avg["loss_total"])
        history["train_movement"].append(avg["loss_movement"])
        history["train_energy"].append(avg["loss_energy"])
        history["train_bce"].append(avg["loss_bce"])
        history["train_ce"].append(avg["loss_ce"])
        history["train_entropy"].append(avg["movement_entropy"])
        history["train_frac_stay"].append(avg["frac_stay_predicted"])

        # ── Val ──────────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            va = {k: 0.0 for k in ["loss_total","loss_movement","loss_energy"]}
            pos_mse_sum = en_mse_sum = alive_correct = alive_total = 0.0
            nv = 0
            for batch in val_loader:
                at  = batch["agents_t"].to(device)
                pt  = batch["patches_t"].to(device)
                at1 = batch["agents_t1"].to(device)
                pt1 = batch["patches_t1"].to(device)
                ct1 = batch["counts_t1"].to(device)

                p = model(at, pt)
                ls = compute_loss(p, at, at1, pt1, ct1, tcfg, mcfg)
                for k in va:
                    va[k] += (ls[k].item() if hasattr(ls[k],"item") else float(ls[k]))

                alive_t = at[:,:,5]
                probs   = F.softmax(p["movement_logits"], dim=-1)
                dxdy    = (probs.unsqueeze(-1) * dir_vecs).sum(-2)
                pos_pred = at[:,:,1:3] + dxdy * step_norm
                pm = ((pos_pred - at1[:,:,1:3])**2).sum(-1)
                d  = alive_t.sum().clamp(min=1)
                pos_mse_sum += (alive_t * pm).sum().item() / d.item()

                en_pred = at[:,:,3] + p["energy_delta"]
                em = (en_pred - at1[:,:,3])**2
                en_mse_sum += (alive_t * em).sum().item() / d.item()

                ap = (p["alive_probs"] > 0.5).long()
                ag = at1[:,:,5].long()
                alive_correct += (ap == ag).float().sum().item()
                alive_total   += ag.numel()
                nv += 1

            history["val_loss"].append(va["loss_total"] / max(nv,1))
            history["val_movement"].append(va["loss_movement"] / max(nv,1))
            history["val_energy"].append(va["loss_energy"] / max(nv,1))
            history["val_pos_mse"].append(pos_mse_sum / max(nv,1))
            history["val_energy_mse"].append(en_mse_sum / max(nv,1))
            history["val_alive_acc"].append(alive_correct / max(alive_total,1))

        print(f"Epoch {epoch:3d}/{n_epochs} | "
              f"train={avg['loss_total']:.4f} | val={va['loss_total']/max(nv,1):.4f} | "
              f"H={avg['movement_entropy']:.2f}bits | stay={avg['frac_stay_predicted']:.2f}")

    return history


# ── Rollout ────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_rollout(model, episodes, horizon, device, grid_size, n_ep=5):
    model.eval()
    selected = random.sample(episodes, min(n_ep, len(episodes)))
    results = []

    for episode in selected:
        T = min(horizon, len(episode) - 1)
        agents_t  = torch.from_numpy(episode[0][0]).unsqueeze(0).to(device)
        patches_t = torch.from_numpy(episode[0][1]).unsqueeze(0).to(device)

        pred_prey, pred_pred, true_prey, true_pred = [], [], [], []
        pred_xy_steps, true_xy_steps = [], []
        prey_spread_steps = []

        move_dir_counts = np.zeros(9, dtype=np.float64)  # cumulative direction histogram
        attn_last = None

        for step in range(T):
            preds = model(agents_t, patches_t)

            # Capture last-step attention weights
            if "attn_ws" in preds and preds["attn_ws"]:
                attn_last = preds["attn_ws"][-1][0].cpu().numpy()  # (H, N, N)

            alive_hard = (preds["alive_probs"][0] > 0.5).float()
            species    = agents_t[0, :, 0].long()
            pp  = (alive_hard * (species == SPECIES_PREY).float()).sum().item()
            pd  = (alive_hard * (species == SPECIES_PREDATOR).float()).sum().item()
            pred_prey.append(pp);  pred_pred.append(pd)

            _, _, gt = episode[step + 1]
            true_prey.append(int(gt[0]));  true_pred.append(int(gt[1]))

            alive_np   = alive_hard.cpu().numpy()
            species_np = agents_t[0, :, 0].cpu().numpy()
            xy_np      = agents_t[0, :, 1:3].cpu().numpy()
            pred_xy_steps.append(xy_np[alive_np > 0.5])

            prey_mask  = (alive_np > 0.5) & (species_np == SPECIES_PREY)
            prey_xy    = xy_np[prey_mask]
            spread = float(np.std(prey_xy[:,0]) + np.std(prey_xy[:,1])) if len(prey_xy)>1 else 0.0
            prey_spread_steps.append(spread)

            gt_alive_arr = episode[step + 1][0]
            gt_alive_mask = gt_alive_arr[:, 5] > 0.5
            true_xy_steps.append(gt_alive_arr[gt_alive_mask, 1:3])

            # Accumulate movement direction histogram over alive agents
            move_logits = preds["movement_logits"][0]   # (N_a, 9)
            pred_dirs   = move_logits.argmax(dim=-1).cpu().numpy()  # (N_a,)
            for d in pred_dirs[alive_np > 0.5]:
                move_dir_counts[int(d)] += 1

            agents_t, patches_t = apply_predictions_to_state(
                agents_t, patches_t, preds,
                grid_size=grid_size, soft=False,
            )

        results.append({
            "pred_prey":         np.array(pred_prey),
            "pred_pred":         np.array(pred_pred),
            "true_prey":         np.array(true_prey),
            "true_pred":         np.array(true_pred),
            "pred_xy_steps":     pred_xy_steps,
            "true_xy_steps":     true_xy_steps,
            "prey_spread_steps": np.array(prey_spread_steps),
            "move_dir_counts":   move_dir_counts,
            "attn_last":         attn_last,
        })

    return results


# ── Plot helpers ──────────────────────────────────────────────────────────────

COLORS = {
    "prey_pred": "#2196F3",
    "prey_true": "#90CAF9",
    "pred_pred": "#F44336",
    "pred_true": "#EF9A9A",
    "train":     "#1976D2",
    "val":       "#E64A19",
    "movement":  "#388E3C",
    "energy":    "#F57C00",
    "bce":       "#7B1FA2",
    "ce":        "#0097A7",
    "entropy":   "#00796B",
    "spread":    "#5D4037",
}


def save_training_graphs(history: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    epochs = list(range(1, len(history["train_loss"]) + 1))

    # ── 1. Loss curves ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Training Loss Curves", fontsize=14, fontweight="bold")

    specs = [
        ("Total loss",      "train_loss",     "val_loss",     "loss"),
        ("Movement loss",   "train_movement", "val_movement", "cross-entropy"),
        ("Energy loss",     "train_energy",   "val_energy",   "MSE"),
        ("BCE (alive)",     "train_bce",      None,           "binary CE"),
        ("Food CE",         "train_ce",       None,           "cross-entropy"),
        (None, None, None, None),  # placeholder
    ]
    for ax, (title, tk, vk, ylabel) in zip(axes.ravel(), specs):
        if title is None:
            ax.axis("off")
            continue
        ax.plot(epochs, history[tk], color=COLORS["train"], lw=1.8, label="train")
        if vk and vk in history:
            ax.plot(epochs, history[vk], color=COLORS["val"], lw=1.8, ls="--", label="val")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curves.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_dir}/loss_curves.png")

    # ── 2. Val metrics ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("Validation Metrics", fontsize=13, fontweight="bold")

    axes[0].plot(epochs, history["val_pos_mse"], color=COLORS["movement"], lw=2)
    axes[0].set_title("Val Position MSE"); axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE (normalised coords)"); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["val_energy_mse"], color=COLORS["energy"], lw=2)
    axes[1].set_title("Val Energy MSE"); axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE"); axes[1].grid(alpha=0.3)

    axes[2].plot(epochs, history["val_alive_acc"], color=COLORS["bce"], lw=2)
    axes[2].set_title("Val Alive Accuracy"); axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy"); axes[2].set_ylim(0, 1); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "val_metrics.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_dir}/val_metrics.png")

    # ── 3. Movement health ─────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Movement Head Health (Spatial Collapse Diagnostics)", fontsize=13, fontweight="bold")

    ax1.plot(epochs, history["train_entropy"], color=COLORS["entropy"], lw=2)
    ax1.axhline(1.5, color="red", ls="--", lw=1.5, label="collapse threshold (1.5 bits)")
    ax1.axhline(2.2, color="green", ls="--", lw=1.5, label="healthy minimum (2.2 bits)")
    ax1.set_title("Movement Entropy (bits)"); ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Entropy (bits)"); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    ax2.plot(epochs, history["train_frac_stay"], color=COLORS["spread"], lw=2)
    ax2.axhline(0.7, color="red", ls="--", lw=1.5, label="collapse threshold (70%)")
    ax2.axhline(0.4, color="orange", ls="--", lw=1.5, label="warning threshold (40%)")
    ax2.set_title("Fraction Predicting 'Stay'"); ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Fraction"); ax2.set_ylim(0, 1); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "alive_f1.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_dir}/alive_f1.png")


def save_rollout_graphs(rollout_results: list, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # ── 1. Population trajectories (up to 3 episodes) ─────────────────────
    n_plot = min(3, len(rollout_results))
    fig, axes = plt.subplots(1, n_plot, figsize=(6 * n_plot, 4))
    if n_plot == 1:
        axes = [axes]
    fig.suptitle("Population Trajectories: Predicted vs Ground Truth", fontsize=13, fontweight="bold")

    for i, (ax, res) in enumerate(zip(axes, rollout_results[:n_plot])):
        T = len(res["pred_prey"])
        steps = range(T)
        ax.plot(steps, res["pred_prey"], color=COLORS["prey_pred"], lw=2, label="prey (pred)")
        ax.plot(steps, res["true_prey"], color=COLORS["prey_true"], lw=2, ls="--", label="prey (GT)")
        ax.plot(steps, res["pred_pred"], color=COLORS["pred_pred"], lw=2, label="predator (pred)")
        ax.plot(steps, res["true_pred"], color=COLORS["pred_true"], lw=2, ls="--", label="predator (GT)")
        r_prey = np.corrcoef(res["pred_prey"], res["true_prey"])[0,1] if res["true_prey"].std()>1e-6 else 0.0
        r_pred = np.corrcoef(res["pred_pred"], res["true_pred"])[0,1] if res["true_pred"].std()>1e-6 else 0.0
        ax.set_title(f"Episode {i}  (r_prey={r_prey:.2f}, r_pred={r_pred:.2f})")
        ax.set_xlabel("Step"); ax.set_ylabel("Count")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "trajectory_ep0.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_dir}/trajectory_ep0.png")

    # ── 2. Spatial distributions at step 20 ───────────────────────────────
    n_ep = min(3, len(rollout_results))
    fig, axes = plt.subplots(2, n_ep, figsize=(5 * n_ep, 9))
    fig.suptitle("Spatial Distributions at Step 20: Predicted vs Ground Truth", fontsize=13, fontweight="bold")

    for col, res in enumerate(rollout_results[:n_ep]):
        t20 = min(19, len(res["pred_xy_steps"]) - 1)
        pred_xy = res["pred_xy_steps"][t20]
        true_xy = res["true_xy_steps"][t20]

        for row, (xy, label) in enumerate([(pred_xy, "Predicted"), (true_xy, "Ground Truth")]):
            ax = axes[row, col] if n_ep > 1 else axes[row]
            if len(xy) > 0:
                ax.scatter(xy[:, 0], xy[:, 1], s=15, alpha=0.7,
                           c=COLORS["prey_pred"] if row==0 else COLORS["prey_true"])
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_title(f"Ep {col} — {label} (step {t20})")
            ax.set_xlabel("x (norm)"); ax.set_ylabel("y (norm)")
            ax.set_aspect("equal"); ax.grid(alpha=0.2)
            spread = float(np.std(xy[:,0]) + np.std(xy[:,1])) if len(xy) > 1 else 0.0
            ax.text(0.02, 0.96, f"spread={spread:.3f}", transform=ax.transAxes,
                    fontsize=8, va="top",
                    color="green" if spread >= 0.15 else "red")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "spatial_distributions.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_dir}/spatial_distributions.png")

    # ── 3. Prey spatial spread over time ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.suptitle("Prey Spatial Spread over Rollout Steps", fontsize=13, fontweight="bold")

    for i, res in enumerate(rollout_results):
        ax.plot(res["prey_spread_steps"], alpha=0.6, lw=1.5, label=f"Ep {i}")

    spread_mean = np.mean([r["prey_spread_steps"] for r in rollout_results], axis=0)
    ax.plot(spread_mean, color="black", lw=2.5, label="Mean")
    ax.axhline(0.15, color="green", ls="--", lw=1.5, label="Healthy threshold (0.15)")
    ax.axhline(0.05, color="red",   ls="--", lw=1.5, label="Collapse threshold (0.05)")
    ax.set_xlabel("Rollout step"); ax.set_ylabel("Spread (std_x + std_y)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "trajectory_ep1.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_dir}/trajectory_ep1.png")

    # ── 4. Movement direction distribution (mean across all rollout episodes) ─
    fig, axes = plt.subplots(1, min(3, len(rollout_results)), figsize=(5 * min(3, len(rollout_results)), 4))
    if not hasattr(axes, "__len__"):
        axes = [axes]
    dir_names = ["stay", "N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    colors = plt.cm.Set3(np.linspace(0, 1, 9))

    for ep_i, (ax, res) in enumerate(zip(axes, rollout_results[:3])):
        counts = res["move_dir_counts"]
        total  = counts.sum()
        fracs  = counts / total if total > 0 else np.ones(9) / 9
        ax.bar(dir_names, fracs, color=colors)
        stay_pct = fracs[0] * 100
        ax.set_title(f"Episode {ep_i}\n(stay={stay_pct:.1f}%)", fontsize=10)
        ax.set_ylabel("Fraction"); ax.set_ylim(0, max(0.5, fracs.max() * 1.2))
        ax.axhline(1/9, color="grey", ls="--", lw=1, label="uniform (1/9)")
        ax.legend(fontsize=7); ax.grid(alpha=0.3, axis="y")
        # Colour the stay bar red if over-predicted
        if fracs[0] > 0.4:
            ax.patches[0].set_facecolor("#F44336")

    plt.suptitle("Predicted Movement Direction Distribution (Rollout)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "trajectory_ep2.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_dir}/trajectory_ep2.png")

    # ── 5. Attention heatmap ────────────────────────────────────────────────
    attn = rollout_results[0].get("attn_last")
    if attn is not None:
        attn_mean = attn.mean(axis=0)[:50, :50]
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(attn_mean, cmap="viridis", aspect="auto")
        ax.set_title("Attention Weights — Last Layer, Mean Across Heads\n(final rollout step)")
        ax.set_xlabel("Key token index"); ax.set_ylabel("Query token index")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "attention_heatmap.png"), dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  Saved {out_dir}/attention_heatmap.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",   type=int, default=40)
    parser.add_argument("--data_dir", type=str, default="data/demo_episodes")
    parser.add_argument("--horizon",  type=int, default=40)
    parser.add_argument("--n_rollout_ep", type=int, default=5)
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────
    data_dir = args.data_dir
    mcfg, tcfg, train_eps = make_demo_configs(data_dir)
    with open(os.path.join(data_dir, "val.pkl"),  "rb") as f:
        val_eps = pickle.load(f)

    train_ds  = EcosystemDataset(train_eps)
    val_ds    = EcosystemDataset(val_eps)
    train_ldr = DataLoader(train_ds, batch_size=tcfg.batch_size, shuffle=True)
    val_ldr   = DataLoader(val_ds,   batch_size=tcfg.batch_size, shuffle=False)

    n_agents = train_eps[0][0][0].shape[0]
    print(f"Data: {len(train_eps)} train eps, {len(val_eps)} val eps | "
          f"N_agents={n_agents} | d_model={mcfg.d_model}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = EcosystemWorldModel(mcfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")

    # ── Train ─────────────────────────────────────────────────────────────
    print(f"\nTraining for {args.epochs} epochs...")
    history = train(model, train_ldr, val_ldr, tcfg, mcfg, args.epochs, device)

    # ── Rollout ───────────────────────────────────────────────────────────
    print(f"\nRunning rollout on {args.n_rollout_ep} val episodes (horizon={args.horizon})...")
    rollout_results = run_rollout(
        model, val_eps, args.horizon, device, mcfg.grid_size, n_ep=args.n_rollout_ep
    )

    # ── Graphs ────────────────────────────────────────────────────────────
    print("\nSaving graphs...")
    save_training_graphs(history, "graphs/training")
    save_rollout_graphs(rollout_results, "graphs/rollout")

    print("\nDone. All graphs saved to graphs/training/ and graphs/rollout/")


if __name__ == "__main__":
    main()
