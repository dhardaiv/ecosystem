"""
Self-contained demo: generate data, train, evaluate, save graphs.

Graphs saved to  graphs/
  training/
    loss_curves.png          — total + per-head losses over training steps
    val_metrics.png          — val losses + position MSE + alive accuracy/F1 per epoch
  rollout/
    trajectory_ep<N>.png     — predicted vs true prey/predator counts (N held-out episodes)
    spatial_t0.png           — agent positions at step 0
    spatial_t20_pred.png     — predicted agent positions at step 20
    spatial_t20_true.png     — ground-truth agent positions at step 20
    energy_dist_t20.png      — predicted vs true energy distribution at step 20
    attention_heatmap.png    — mean attention weights (last transformer layer)
  inference/
    inference_trajectory.png — prey/predator counts over inference rollout
    ghost_and_food.png       — ghost count + food coverage over rollout

Uses a small model config so it finishes in a few minutes on CPU.
wandb is run in "disabled" mode — no login required.
"""
import os
import sys
import warnings
import pickle

warnings.filterwarnings("ignore")
os.environ["WANDB_MODE"] = "disabled"   # no wandb login needed

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

# ── Override configs for a fast demo ─────────────────────────────────────────
from config import SimConfig, ModelConfig, TrainConfig, DataConfig

SIM = SimConfig(
    grid_width=20,
    grid_height=20,
    n_initial_prey=15,
    n_initial_predators=4,
    initial_food_density=0.5,
    n_max_agents=80,
    n_food_patches=20 * 20,
)

MODEL_CFG = ModelConfig(
    d_model=64,
    n_layers=2,
    n_heads=4,
    d_ff=128,
    dropout=0.1,
    n_species=3,
    d_species_embed=16,
    d_continuous=4,
    n_mlp_hidden=32,
    spatial_encoding_dim=32,
    use_spatial_encoding=True,
    use_locality_mask=True,
    locality_radius=0.3,
    use_cross_type_mask=False,
    n_food_bins=5,
    n_max_agents=80,
    grid_size=20,
)

TRAIN_CFG = TrainConfig(
    lambda_mse=1.0,
    lambda_bce=2.0,
    lambda_ce=0.5,
    lambda_aux=0.0,
    bce_dead_weight=10.0,
    lr=3e-4,
    weight_decay=1e-4,
    batch_size=16,
    n_epochs=20,
    grad_clip=1.0,
    ss_rate_init=0.0,
    ss_rate_max=0.50,
    ss_rate_increment=0.01,
    alive_noise_rate=0.05,
    phase2_epoch=10,
    phase3_epoch=999,      # skip phase 3 in demo
    rollout_k_start=2,
    rollout_k_max=3,
    val_every_n_epochs=1,
    rollout_eval_every_n_epochs=5,
    rollout_horizon=30,
    checkpoint_path="checkpoints/demo_best.pt",
    wandb_project="ecosystem-world-model",
)

DATA_CFG = DataConfig(
    n_train_episodes=20,
    n_val_episodes=5,
    n_test_episodes=3,
    episode_length=60,
    data_dir="data/demo_episodes",
    prey_range=(8, 25),
    predator_range=(2, 8),
    food_density_range=(0.3, 0.8),
)

GRAPHS_DIR = "graphs"


# ── Utility: ensure directories exist ────────────────────────────────────────

def makedirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


makedirs(
    GRAPHS_DIR,
    os.path.join(GRAPHS_DIR, "training"),
    os.path.join(GRAPHS_DIR, "rollout"),
    os.path.join(GRAPHS_DIR, "inference"),
    os.path.dirname(TRAIN_CFG.checkpoint_path),
)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Generate data
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("STEP 1 / 4 — Generating simulator data")
print("="*60)

from simulator.generate_data import build_split, save_episodes

data_dir = DATA_CFG.data_dir
os.makedirs(data_dir, exist_ok=True)

train_eps = build_split(DATA_CFG.n_train_episodes, "train", SIM, DATA_CFG, base_seed=0)
val_eps   = build_split(DATA_CFG.n_val_episodes,   "val",   SIM, DATA_CFG, base_seed=10000)
test_eps  = build_split(DATA_CFG.n_test_episodes,  "test",  SIM, DATA_CFG, base_seed=20000)

save_episodes(train_eps, os.path.join(data_dir, "train.pkl"))
save_episodes(val_eps,   os.path.join(data_dir, "val.pkl"))
save_episodes(test_eps,  os.path.join(data_dir, "test.pkl"))

# Quick sanity: show population range in training data
prey_counts  = [ep[0][2][0] for ep in train_eps]
pred_counts  = [ep[0][2][1] for ep in train_eps]
print(f"  Training episodes: prey ∈ [{min(prey_counts)}, {max(prey_counts)}], "
      f"pred ∈ [{min(pred_counts)}, {max(pred_counts)}]")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Train
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("STEP 2 / 4 — Training world model")
print("="*60)

from data.dataset import make_dataloaders
from model import EcosystemWorldModel
from model.heads import compute_loss
from training.utils import straight_through_alive, inject_alive_noise, apply_predictions_to_state

wandb.init(project="ecosystem-world-model", mode="disabled")

device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps"  if torch.backends.mps.is_available() else "cpu")
print(f"  Device: {device}")

train_loader, val_loader, _ = make_dataloaders(
    data_dir=data_dir, batch_size=TRAIN_CFG.batch_size, num_workers=0
)

world_model = EcosystemWorldModel(MODEL_CFG).to(device)
n_params = sum(p.numel() for p in world_model.parameters())
print(f"  Model: {n_params:,} parameters")

optimizer = torch.optim.AdamW(
    world_model.parameters(), lr=TRAIN_CFG.lr, weight_decay=TRAIN_CFG.weight_decay
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=TRAIN_CFG.n_epochs, eta_min=TRAIN_CFG.lr * 0.05
)

# ── Logging storage (local, since wandb is disabled) ─────────────────────────
train_log   = []   # (step, loss_total, loss_mse, loss_bce, loss_ce, loss_aux)
val_log     = []   # (epoch, val_total, val_mse, val_bce, val_ce, pos_mse, alive_acc, alive_f1)
best_val    = float("inf")
global_step = 0

current_lambda_aux = 0.0   # bumped to 0.1 at phase 2

def train_epoch(epoch):
    global global_step, current_lambda_aux
    TRAIN_CFG.lambda_aux = current_lambda_aux
    world_model.train()
    for batch in train_loader:
        agents_t   = batch["agents_t"].to(device)
        patches_t  = batch["patches_t"].to(device)
        agents_t1  = batch["agents_t1"].to(device)
        patches_t1 = batch["patches_t1"].to(device)
        counts_t1  = batch["counts_t1"].to(device)

        if TRAIN_CFG.alive_noise_rate > 0:
            agents_t = agents_t.clone()
            agents_t[:, :, 5] = inject_alive_noise(agents_t[:, :, 5], TRAIN_CFG.alive_noise_rate)

        optimizer.zero_grad()
        preds  = world_model(agents_t, patches_t)
        losses = compute_loss(preds, agents_t, agents_t1, patches_t1, counts_t1, TRAIN_CFG, MODEL_CFG)
        losses["loss_total"].backward()
        nn.utils.clip_grad_norm_(world_model.parameters(), TRAIN_CFG.grad_clip)
        nn.utils.clip_grad_norm_(
            world_model.heads.aux_head.parameters(), TRAIN_CFG.grad_clip_aux
        )
        optimizer.step()

        train_log.append((
            global_step,
            losses["loss_total"].item(),
            losses["loss_mse"].item(),
            losses["loss_bce"].item(),
            losses["loss_ce"].item(),
            losses["loss_aux"].item(),
        ))
        global_step += 1

@torch.no_grad()
def val_epoch(epoch):
    from sklearn.metrics import f1_score
    world_model.eval()
    totals = {"total": 0, "mse": 0, "bce": 0, "ce": 0, "aux": 0,
              "pos_mse": 0, "alive_acc": 0}
    all_true, all_pred = [], []
    n = 0
    for batch in val_loader:
        agents_t   = batch["agents_t"].to(device)
        patches_t  = batch["patches_t"].to(device)
        agents_t1  = batch["agents_t1"].to(device)
        patches_t1 = batch["patches_t1"].to(device)
        counts_t1  = batch["counts_t1"].to(device)

        preds  = world_model(agents_t, patches_t)
        losses = compute_loss(preds, agents_t, agents_t1, patches_t1, counts_t1, TRAIN_CFG, MODEL_CFG)
        for k in ["total", "mse", "bce", "ce", "aux"]:
            v = losses[f"loss_{k}"]
            totals[k] += v.item() if hasattr(v, 'item') else float(v)

        alive_t = agents_t[:, :, 5]
        pos_pred = agents_t[:, :, 1:3] + preds["delta_pred"][:, :, :2]
        pos_gt   = agents_t1[:, :, 1:3]
        pos_mse  = ((pos_pred - pos_gt)**2).sum(-1)
        denom    = alive_t.sum().clamp(min=1.0)
        totals["pos_mse"]   += (alive_t * pos_mse).sum().item() / denom.item()
        totals["alive_acc"] += ((preds["alive_probs"] > 0.5).long() == agents_t1[:, :, 5].long()).float().mean().item()

        all_true.extend(agents_t1[:, :, 5].long().cpu().numpy().ravel().tolist())
        all_pred.extend((preds["alive_probs"] > 0.5).long().cpu().numpy().ravel().tolist())
        n += 1

    f1 = f1_score(all_true, all_pred, pos_label=0, zero_division=0)
    row = (epoch,
           totals["total"]/n, totals["mse"]/n, totals["bce"]/n, totals["ce"]/n,
           totals["pos_mse"]/n, totals["alive_acc"]/n, f1)
    val_log.append(row)
    return row[1]  # val_loss_total

print(f"  Training for {TRAIN_CFG.n_epochs} epochs...")
for epoch in range(1, TRAIN_CFG.n_epochs + 1):
    if epoch == TRAIN_CFG.phase2_epoch:
        current_lambda_aux = 0.1
        print(f"  [Phase 2] epoch {epoch}: λ_aux = {current_lambda_aux}")
    train_epoch(epoch)
    val_loss = val_epoch(epoch)
    if val_loss < best_val:
        best_val = val_loss
        os.makedirs(os.path.dirname(TRAIN_CFG.checkpoint_path), exist_ok=True)
        torch.save({"model_state": world_model.state_dict(), "epoch": epoch, "val_loss": val_loss},
                   TRAIN_CFG.checkpoint_path)
    if epoch % 5 == 0 or epoch == TRAIN_CFG.n_epochs:
        row = val_log[-1]
        print(f"  epoch {epoch:3d} | val_loss={row[1]:.4f} | pos_mse={row[5]:.4f} | "
              f"alive_acc={row[6]:.3f} | alive_f1={row[7]:.3f}")
    scheduler.step()

print(f"  Best val_loss = {best_val:.4f}  (checkpoint: {TRAIN_CFG.checkpoint_path})")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Plot training graphs
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("STEP 3 / 4 — Saving training graphs")
print("="*60)

train_arr = np.array(train_log)    # (N_steps, 6)
val_arr   = np.array(val_log)      # (N_epochs, 8)

# ── Loss curves ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("Training Loss Curves", fontsize=14, fontweight="bold")

steps = train_arr[:, 0]
labels = ["total", "mse", "bce", "ce", "aux"]
colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
for idx, (lbl, col) in enumerate(zip(labels, colours)):
    ax = axes.ravel()[idx]
    ax.plot(steps, train_arr[:, idx + 1], color=col, alpha=0.7, linewidth=0.8, label="train")
    ax.set_title(f"L_{lbl} (raw, unweighted)")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

# Phase boundary line on total loss
phase2_step = int(TRAIN_CFG.phase2_epoch / TRAIN_CFG.n_epochs * len(steps))
ax = axes.ravel()[0]
if phase2_step < len(steps):
    ax.axvline(steps[phase2_step], color="red", linestyle="--", alpha=0.6, label="phase 2")
    ax.legend()

axes.ravel()[-1].axis("off")
plt.tight_layout()
out = os.path.join(GRAPHS_DIR, "training", "loss_curves.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {out}")

# ── Validation metrics ────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("Validation Metrics per Epoch", fontsize=14, fontweight="bold")

epochs  = val_arr[:, 0]
v_labels = [
    ("Val loss total",  val_arr[:, 1], "#1f77b4"),
    ("Val loss MSE",    val_arr[:, 2], "#ff7f0e"),
    ("Val loss BCE",    val_arr[:, 3], "#2ca02c"),
    ("Val loss CE",     val_arr[:, 4], "#d62728"),
    ("Val position MSE", val_arr[:, 5], "#9467bd"),
    ("Alive accuracy",  val_arr[:, 6], "#8c564b"),
]
for ax, (title, vals, col) in zip(axes.ravel(), v_labels):
    ax.plot(epochs, vals, color=col, marker="o", markersize=3, linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
out = os.path.join(GRAPHS_DIR, "training", "val_metrics.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {out}")

# ── Alive F1 (dead class) ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(epochs, val_arr[:, 7], color="#e377c2", marker="o", markersize=4, linewidth=1.4)
ax.set_title("Validation: Alive F1 (dead class)")
ax.set_xlabel("epoch");  ax.set_ylabel("F1")
ax.grid(True, alpha=0.3)
plt.tight_layout()
out = os.path.join(GRAPHS_DIR, "training", "alive_f1.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Rollout evaluation graphs
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("STEP 4 / 4 — Rollout evaluation & inference graphs")
print("="*60)

# Load best checkpoint
ckpt = torch.load(TRAIN_CFG.checkpoint_path, map_location=device)
world_model.load_state_dict(ckpt["model_state"])
world_model.eval()

from training.rollout import rollout_episode
from simulator.agents import SPECIES_PREY, SPECIES_PREDATOR

HORIZON = TRAIN_CFG.rollout_horizon

# ── Population trajectories (one graph per val episode) ──────────────────────
all_prey_r, all_pred_r = [], []

for ep_idx, episode in enumerate(val_eps[:3]):
    res = rollout_episode(world_model, episode, HORIZON, device)

    T = len(res["pred_prey"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Rollout vs Ground Truth — val episode {ep_idx}", fontsize=13, fontweight="bold")

    axes[0].plot(res["true_prey"],  label="GT prey",       color="#2ca02c", linewidth=1.8)
    axes[0].plot(res["pred_prey"],  label="Pred prey",     color="#2ca02c", linewidth=1.8, linestyle="--")
    axes[0].plot(res["true_pred"],  label="GT predators",  color="#d62728", linewidth=1.8)
    axes[0].plot(res["pred_pred"],  label="Pred predators",color="#d62728", linewidth=1.8, linestyle="--")
    axes[0].set_title("Population counts")
    axes[0].set_xlabel("step"); axes[0].set_ylabel("count")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].bar(range(T), res["ghost_counts"], color="#ff7f0e", alpha=0.7, label="ghost agents")
    axes[1].set_title("Ghost agents per step (pred alive but GT dead)")
    axes[1].set_xlabel("step"); axes[1].set_ylabel("count")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(GRAPHS_DIR, "rollout", f"trajectory_ep{ep_idx}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")

    # Pearson r
    from scipy.stats import pearsonr
    if res["true_prey"].std() > 1e-6 and res["pred_prey"].std() > 1e-6:
        r_prey, _ = pearsonr(res["true_prey"], res["pred_prey"])
        all_prey_r.append(r_prey)
    if res["true_pred"].std() > 1e-6 and res["pred_pred"].std() > 1e-6:
        r_pred, _ = pearsonr(res["true_pred"], res["pred_pred"])
        all_pred_r.append(r_pred)

print(f"  Pop correlation: prey r={np.mean(all_prey_r):.3f}, pred r={np.mean(all_pred_r):.3f}")

# ── Spatial distribution: t=0 and t=20 ───────────────────────────────────────
episode = val_eps[0]
res     = rollout_episode(world_model, episode, HORIZON, device)
W, H    = SIM.grid_width, SIM.grid_height

# Step 0 — ground truth (show all alive agents)
ep0_agents = episode[0][0]      # (N_a, 6) agent_arr at t=0
alive0 = ep0_agents[:, 5] > 0.5
sp0    = ep0_agents[alive0, 0].astype(int)
xy0    = ep0_agents[alive0, 1:3]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Spatial distributions of agents", fontsize=13, fontweight="bold")

prey_mask0 = sp0 == SPECIES_PREY
pred_mask0 = sp0 == SPECIES_PREDATOR
axes[0].scatter(xy0[prey_mask0, 0] * W, xy0[prey_mask0, 1] * H,
                c="#2ca02c", s=20, alpha=0.7, label=f"prey ({prey_mask0.sum()})")
axes[0].scatter(xy0[pred_mask0, 0] * W, xy0[pred_mask0, 1] * H,
                c="#d62728", s=30, marker="^", alpha=0.9, label=f"pred ({pred_mask0.sum()})")
axes[0].set_title("Ground truth — step 0")
axes[0].set_xlim(0, W); axes[0].set_ylim(0, H)
axes[0].set_xlabel("x"); axes[0].set_ylabel("y"); axes[0].legend()

# Step 20 — ground truth
t20 = min(20, len(episode) - 1)
ep20_agents = episode[t20][0]
alive20_gt  = ep20_agents[:, 5] > 0.5
sp20_gt     = ep20_agents[alive20_gt, 0].astype(int)
xy20_gt     = ep20_agents[alive20_gt, 1:3]
prey_mask20_gt = sp20_gt == SPECIES_PREY
pred_mask20_gt = sp20_gt == SPECIES_PREDATOR
axes[1].scatter(xy20_gt[prey_mask20_gt, 0] * W, xy20_gt[prey_mask20_gt, 1] * H,
                c="#2ca02c", s=20, alpha=0.7, label=f"prey ({prey_mask20_gt.sum()})")
axes[1].scatter(xy20_gt[pred_mask20_gt, 0] * W, xy20_gt[pred_mask20_gt, 1] * H,
                c="#d62728", s=30, marker="^", alpha=0.9, label=f"pred ({pred_mask20_gt.sum()})")
axes[1].set_title("Ground truth — step 20")
axes[1].set_xlim(0, W); axes[1].set_ylim(0, H)
axes[1].set_xlabel("x"); axes[1].set_ylabel("y"); axes[1].legend()

# Step 20 — model prediction
alive20_pred = res["pred_agents_alive"][min(19, len(res["pred_agents_alive"])-1)]
xy20_pred    = res["pred_xy"][min(19, len(res["pred_xy"])-1)]
sp20_pred    = episode[0][0][:, 0].astype(int)   # species doesn't change
alive20_mask = alive20_pred > 0.5
xy_p   = xy20_pred[alive20_mask]
sp_p   = sp20_pred[alive20_mask]
prey_mp = sp_p == SPECIES_PREY
pred_mp = sp_p == SPECIES_PREDATOR
axes[2].scatter(xy_p[prey_mp, 0] * W, xy_p[prey_mp, 1] * H,
                c="#2ca02c", s=20, alpha=0.7, label=f"prey ({prey_mp.sum()})")
axes[2].scatter(xy_p[pred_mp, 0] * W, xy_p[pred_mp, 1] * H,
                c="#d62728", s=30, marker="^", alpha=0.9, label=f"pred ({pred_mp.sum()})")
axes[2].set_title("Model prediction — step 20")
axes[2].set_xlim(0, W); axes[2].set_ylim(0, H)
axes[2].set_xlabel("x"); axes[2].set_ylabel("y"); axes[2].legend()

plt.tight_layout()
out = os.path.join(GRAPHS_DIR, "rollout", "spatial_distributions.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {out}")

# ── Energy distribution at step 20 ───────────────────────────────────────────
t20_idx = min(19, len(res["pred_energy"]) - 1)
pred_energy_alive = res["pred_energy"][t20_idx][res["pred_agents_alive"][t20_idx] > 0.5]
gt_energy_alive   = ep20_agents[alive20_gt, 3]

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(gt_energy_alive,   bins=20, range=(0, 1), alpha=0.6, color="#2ca02c", label="GT energy (t=20)")
ax.hist(pred_energy_alive, bins=20, range=(0, 1), alpha=0.6, color="#1f77b4", label="Pred energy (t=20)")
ax.set_title("Energy distribution at step 20")
ax.set_xlabel("energy"); ax.set_ylabel("count")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
out = os.path.join(GRAPHS_DIR, "rollout", "energy_dist_t20.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {out}")

# ── Attention heatmap ─────────────────────────────────────────────────────────
if res["attn_last"] is not None:
    attn = res["attn_last"]     # (H, N, N)
    attn_mean = attn.mean(axis=0)

    # Show first min(60, N) tokens for readability
    n_show = min(60, attn_mean.shape[0])
    attn_crop = attn_mean[:n_show, :n_show]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Attention weights — last transformer layer (final rollout step)", fontsize=13, fontweight="bold")

    im0 = axes[0].imshow(attn_crop, cmap="viridis", aspect="auto", vmin=0)
    axes[0].set_title(f"Mean across heads (first {n_show} tokens)")
    axes[0].set_xlabel("Key token"); axes[0].set_ylabel("Query token")
    plt.colorbar(im0, ax=axes[0])

    # Per-head view (head 0)
    im1 = axes[1].imshow(attn[0, :n_show, :n_show], cmap="magma", aspect="auto", vmin=0)
    axes[1].set_title(f"Head 0 (first {n_show} tokens)")
    axes[1].set_xlabel("Key token"); axes[1].set_ylabel("Query token")
    plt.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    out = os.path.join(GRAPHS_DIR, "rollout", "attention_heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")

# ── Inference rollout graphs ──────────────────────────────────────────────────
from inference import state_from_episode, compute_step_metrics

inf_episode = test_eps[0]
agents_t, patches_t = state_from_episode(inf_episode, step=0)
agents_t  = agents_t.to(device)
patches_t = patches_t.to(device)

inf_history = {"step": [], "prey": [], "pred": [], "ghost": [], "energy": [], "food": []}

with torch.no_grad():
    for step in range(HORIZON):
        preds   = world_model(agents_t, patches_t)
        metrics = compute_step_metrics(agents_t, patches_t, preds)
        for k in ["prey", "pred", "ghost", "energy", "food"]:
            inf_history[k].append(metrics[f"{k}_count" if k not in ("energy","food") else
                                          ("mean_energy" if k == "energy" else "food_coverage")])
        inf_history["step"].append(step)
        if metrics["prey_count"] == 0 and metrics["pred_count"] == 0:
            break
        agents_t, patches_t = apply_predictions_to_state(agents_t, patches_t, preds)

steps_inf = inf_history["step"]

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle("Inference rollout (no ground truth)", fontsize=13, fontweight="bold")

axes[0].plot(steps_inf, inf_history["prey"], label="prey",      color="#2ca02c", linewidth=1.8)
axes[0].plot(steps_inf, inf_history["pred"], label="predators", color="#d62728", linewidth=1.8)
axes[0].set_title("Population counts"); axes[0].set_xlabel("step"); axes[0].set_ylabel("count")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

ax2 = axes[1].twinx()
axes[1].bar(steps_inf, inf_history["ghost"], alpha=0.5, color="#ff7f0e", label="ghosts (left)")
ax2.plot(steps_inf, inf_history["food"],  color="#1f77b4", linewidth=1.8, label="food coverage (right)")
axes[1].set_title("Ghost agents & food coverage")
axes[1].set_xlabel("step"); axes[1].set_ylabel("ghost count"); ax2.set_ylabel("food coverage")
lines1, labs1 = axes[1].get_legend_handles_labels()
lines2, labs2 = ax2.get_legend_handles_labels()
axes[1].legend(lines1 + lines2, labs1 + labs2, loc="upper right")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
out = os.path.join(GRAPHS_DIR, "inference", "inference_rollout.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {out}")

# ── Food patch heatmap (initial state) ───────────────────────────────────────
patch_arr0 = inf_episode[0][1]    # (N_p, 4)  x, y, food, alive
W2 = SIM.grid_width; H2 = SIM.grid_height
food_grid = np.zeros((W2, H2), dtype=np.float32)
for prow in patch_arr0:
    ix = int(round(prow[0] * (W2 - 1)))
    iy = int(round(prow[1] * (H2 - 1)))
    food_grid[ix, iy] = prow[2]

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(food_grid.T, origin="lower", cmap="YlGn", vmin=0, vmax=1, aspect="equal")
ax.set_title("Initial food patch levels (test episode)")
ax.set_xlabel("x"); ax.set_ylabel("y")
plt.colorbar(im, ax=ax, label="food level")
plt.tight_layout()
out = os.path.join(GRAPHS_DIR, "rollout", "food_heatmap_initial.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {out}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("COMPLETE — all graphs saved to:", os.path.abspath(GRAPHS_DIR))
print("="*60)
print(f"\n  graphs/training/  loss_curves.png  val_metrics.png  alive_f1.png")
print(f"  graphs/rollout/   trajectory_ep0-2.png  spatial_distributions.png")
print(f"                    energy_dist_t20.png  attention_heatmap.png  food_heatmap_initial.png")
print(f"  graphs/inference/ inference_rollout.png")
