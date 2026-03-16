"""
Rollout evaluation suite.

Runs the world model autoregressively for `horizon` steps starting from the
ground-truth initial state of held-out episodes, then computes ecosystem-level
metrics and logs everything to wandb.

Metrics logged:
  rollout/pop_correlation_prey   — Pearson r of predicted vs true prey counts
  rollout/pop_correlation_pred   — Pearson r of predicted vs true predator counts
  rollout/extinction_detection   — fraction of extinctions detected within ±5 steps
  rollout/spatial_kl_t20         — KL divergence of spatial distributions at step 20
  rollout/energy_kl_t20          — KL divergence of energy distributions at step 20
  rollout/stability_rate         — fraction of rollouts with non-zero, non-exploding pop
  rollout/mean_ghost_count       — mean ghost agents per step
  rollout/trajectory             — population trajectory chart (for one random episode)
  rollout/attention_map          — attention weight heatmap from last transformer layer
"""
import math
from typing import Optional, List
import numpy as np
import torch
import wandb
from scipy.stats import pearsonr
from scipy.special import kl_div

from config import ModelConfig
from training.utils import apply_predictions_to_state, straight_through_alive
from simulator.agents import SPECIES_PREY, SPECIES_PREDATOR


# ── Rollout engine ────────────────────────────────────────────────────────────

@torch.no_grad()
def rollout_episode(
    model,
    episode: List[tuple],           # list of (agent_arr, patch_arr, counts)
    horizon: int,
    device: torch.device,
    threshold: float = 0.5,
) -> dict:
    """
    Roll out the world model for `horizon` steps from the episode start.

    Returns
    -------
    pred_prey   : (horizon,)  predicted prey counts
    pred_pred   : (horizon,)  predicted predator counts
    true_prey   : (horizon,)  ground truth prey counts
    true_pred   : (horizon,)  ground truth predator counts
    ghost_counts: (horizon,)  ghost agents per step (p>0.5 but GT dead)
    attn_last   : final attention weight matrix (H, N, N) or None
    pred_agents : list of (N_a,) arrays — alive masks per rollout step
    pred_xy     : list of (N_a, 2) arrays — predicted positions per step
    pred_energy : list of (N_a,) arrays — predicted energy per step
    """
    T = min(horizon, len(episode) - 1)

    agent_arr0, patch_arr0, _ = episode[0]
    agents_t  = torch.from_numpy(agent_arr0).unsqueeze(0).to(device)   # (1, N_a, 6)
    patches_t = torch.from_numpy(patch_arr0).unsqueeze(0).to(device)   # (1, N_p, 4)

    pred_prey, pred_pred = [], []
    true_prey, true_pred = [], []
    ghost_counts = []
    attn_last = None
    pred_agents_alive = []
    pred_xy_list = []
    pred_energy_list = []

    for step in range(T):
        preds = model(agents_t, patches_t)
        attn_last = preds["attn_ws"][-1][0].cpu().numpy()  # (H, N, N) for batch[0]

        alive_probs = preds["alive_probs"][0]               # (N_a,)
        alive_hard  = (alive_probs > threshold).float()

        # Count predictions
        species = agents_t[0, :, 0].long()                 # (N_a,)
        p_prey  = (alive_hard * (species == SPECIES_PREY).float()).sum().item()
        p_pred  = (alive_hard * (species == SPECIES_PREDATOR).float()).sum().item()
        pred_prey.append(p_prey)
        pred_pred.append(p_pred)

        # Ground truth at t+1
        _, _, gt_counts = episode[step + 1]
        true_prey.append(int(gt_counts[0]))
        true_pred.append(int(gt_counts[1]))

        # Ghost count: predicted alive but GT is dead
        gt_alive = torch.from_numpy(episode[step + 1][0][:, 5]).to(device)  # (N_a,)
        ghost_mask = (alive_hard > 0.5) & (gt_alive < 0.5)
        ghost_counts.append(ghost_mask.sum().item())

        # Save spatial/energy distributions
        alive_np = alive_hard.cpu().numpy()
        pred_agents_alive.append(alive_np)
        pred_xy_list.append(agents_t[0, :, 1:3].cpu().numpy())
        pred_energy_list.append(agents_t[0, :, 3].cpu().numpy())

        # Advance state
        agents_t, patches_t = apply_predictions_to_state(
            agents_t, patches_t, preds, threshold
        )

    return {
        "pred_prey":    np.array(pred_prey),
        "pred_pred":    np.array(pred_pred),
        "true_prey":    np.array(true_prey),
        "true_pred":    np.array(true_pred),
        "ghost_counts": np.array(ghost_counts),
        "attn_last":    attn_last,
        "pred_agents_alive": pred_agents_alive,
        "pred_xy":      pred_xy_list,
        "pred_energy":  pred_energy_list,
    }


# ── Metric helpers ────────────────────────────────────────────────────────────

def safe_pearsonr(a, b):
    """Pearson r, returns 0 if constant array."""
    if a.std() < 1e-6 or b.std() < 1e-6:
        return 0.0
    r, _ = pearsonr(a, b)
    return float(r)


def extinction_step(counts: np.ndarray) -> Optional[int]:
    """Return first step where count == 0, or None."""
    zeros = np.where(counts == 0)[0]
    return int(zeros[0]) if len(zeros) > 0 else None


def spatial_kl(pred_xy: np.ndarray, true_xy: np.ndarray, n_bins: int = 20) -> float:
    """
    KL divergence of 2D spatial histograms.
    pred_xy, true_xy: (N, 2) normalised coords, filtered to alive agents.
    """
    if len(pred_xy) == 0 or len(true_xy) == 0:
        return 0.0
    eps = 1e-8
    bins = np.linspace(0, 1, n_bins + 1)
    p_hist, _, _ = np.histogram2d(pred_xy[:, 0], pred_xy[:, 1], bins=bins, density=True)
    q_hist, _, _ = np.histogram2d(true_xy[:, 0], true_xy[:, 1], bins=bins, density=True)
    p_hist = p_hist.ravel() + eps
    q_hist = q_hist.ravel() + eps
    p_hist /= p_hist.sum()
    q_hist /= q_hist.sum()
    return float(np.sum(kl_div(p_hist, q_hist)))


def energy_kl(pred_energy: np.ndarray, true_energy: np.ndarray, n_bins: int = 20) -> float:
    """KL divergence of energy histograms."""
    if len(pred_energy) == 0 or len(true_energy) == 0:
        return 0.0
    eps = 1e-8
    bins = np.linspace(0, 1, n_bins + 1)
    p_hist, _ = np.histogram(pred_energy, bins=bins, density=True)
    q_hist, _ = np.histogram(true_energy, bins=bins, density=True)
    p_hist = p_hist + eps;  p_hist /= p_hist.sum()
    q_hist = q_hist + eps;  q_hist /= q_hist.sum()
    return float(np.sum(kl_div(p_hist, q_hist)))


# ── Main evaluation function ──────────────────────────────────────────────────

def run_rollout_evaluation(
    model,
    val_episodes: list,
    horizon: int,
    device: torch.device,
    model_cfg: ModelConfig,
    epoch: int,
    n_eval_episodes: int = 10,
):
    """
    Evaluate on up to n_eval_episodes held-out episodes and log metrics to wandb.
    """
    import random
    model.eval()

    selected = random.sample(val_episodes, min(n_eval_episodes, len(val_episodes)))

    all_prey_r, all_pred_r = [], []
    extinction_detected_prey, extinction_total_prey = 0, 0
    extinction_detected_pred, extinction_total_pred = 0, 0
    all_spatial_kl, all_energy_kl = [], []
    stability_count = 0
    all_ghost_counts = []

    traj_episode_result = None  # for trajectory plot

    for ep_idx, episode in enumerate(selected):
        res = rollout_episode(model, episode, horizon, device)

        # Population correlations
        all_prey_r.append(safe_pearsonr(res["pred_prey"], res["true_prey"]))
        all_pred_r.append(safe_pearsonr(res["pred_pred"], res["true_pred"]))

        # Extinction detection (within ±5 steps)
        for species_name, pred_counts, true_counts, det_ctr, tot_ctr in [
            ("prey",  res["pred_prey"], res["true_prey"],
             extinction_detected_prey, extinction_total_prey),
            ("pred",  res["pred_pred"], res["true_pred"],
             extinction_detected_pred, extinction_total_pred),
        ]:
            ext_gt = extinction_step(true_counts)
            if ext_gt is not None:
                tot_ctr += 1
                ext_pred = extinction_step(pred_counts)
                if ext_pred is not None and abs(ext_pred - ext_gt) <= 5:
                    det_ctr += 1
                if species_name == "prey":
                    extinction_detected_prey = det_ctr
                    extinction_total_prey = tot_ctr
                else:
                    extinction_detected_pred = det_ctr
                    extinction_total_pred = tot_ctr

        # Spatial and energy KL at step 20 (if available)
        t20 = min(19, len(res["pred_xy"]) - 1)
        alive_np = res["pred_agents_alive"][t20]
        pred_xy  = res["pred_xy"][t20][alive_np > 0.5]
        true_agents_t20 = episode[min(t20 + 1, len(episode) - 1)][0]
        true_alive = true_agents_t20[:, 5] > 0.5
        true_xy = true_agents_t20[true_alive, 1:3]
        all_spatial_kl.append(spatial_kl(pred_xy, true_xy))

        pred_en = res["pred_energy"][t20][alive_np > 0.5]
        true_en = true_agents_t20[true_alive, 3]
        all_energy_kl.append(energy_kl(pred_en, true_en))

        # Stability: non-zero and non-exploding population
        max_pop = max(res["pred_prey"].max(), res["pred_pred"].max())
        min_pop = min(res["pred_prey"].min(), res["pred_pred"].min())
        if min_pop >= 0 and max_pop < 1000:
            stability_count += 1

        # Ghost counts
        all_ghost_counts.extend(res["ghost_counts"].tolist())

        # Save first episode for trajectory plot
        if ep_idx == 0:
            traj_episode_result = res

    n_ep = len(selected)
    ext_rate = (extinction_detected_prey + extinction_detected_pred) / max(
        extinction_total_prey + extinction_total_pred, 1
    )

    metrics = {
        "rollout/pop_correlation_prey":   float(np.mean(all_prey_r)),
        "rollout/pop_correlation_pred":   float(np.mean(all_pred_r)),
        "rollout/extinction_detection":   ext_rate,
        "rollout/spatial_kl_t20":         float(np.mean(all_spatial_kl)),
        "rollout/energy_kl_t20":          float(np.mean(all_energy_kl)),
        "rollout/stability_rate":         stability_count / n_ep,
        "rollout/mean_ghost_count":       float(np.mean(all_ghost_counts)),
        "epoch": epoch,
    }

    # Population trajectory chart
    if traj_episode_result is not None:
        T = len(traj_episode_result["pred_prey"])
        metrics["rollout/trajectory"] = wandb.plot.line_series(
            xs=list(range(T)),
            ys=[
                traj_episode_result["pred_prey"].tolist(),
                traj_episode_result["true_prey"].tolist(),
                traj_episode_result["pred_pred"].tolist(),
                traj_episode_result["true_pred"].tolist(),
            ],
            keys=["pred_prey", "true_prey", "pred_predator", "true_predator"],
            title="Population trajectory (rollout vs ground truth)",
            xname="step",
        )

    # Attention heatmap (last layer, first head, first N agents)
    if traj_episode_result is not None and traj_episode_result["attn_last"] is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        attn = traj_episode_result["attn_last"]  # (H, N, N)
        # Show mean across heads, first 50 tokens (for readability)
        attn_mean = attn.mean(axis=0)[:50, :50]
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(attn_mean, cmap="viridis", aspect="auto")
        ax.set_title("Attention weights (last layer, mean across heads)")
        ax.set_xlabel("Key token index")
        ax.set_ylabel("Query token index")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        metrics["rollout/attention_map"] = wandb.Image(fig)
        plt.close(fig)

    wandb.log(metrics)
    print(f"  [Rollout eval] prey_r={metrics['rollout/pop_correlation_prey']:.3f} | "
          f"pred_r={metrics['rollout/pop_correlation_pred']:.3f} | "
          f"stability={metrics['rollout/stability_rate']:.2f} | "
          f"ghosts={metrics['rollout/mean_ghost_count']:.1f}")
