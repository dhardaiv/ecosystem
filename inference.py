"""
Inference script — roll out the world model from a given initial state
without access to ground truth.

Per-step wandb logging tracks:
  - prey / predator count predictions
  - ghost agent count (p_alive in (0.3, 0.5) — uncertain zone)
  - mean energy of alive agents
  - food coverage (fraction of patches above 0.5)

Usage:
    python inference.py --checkpoint checkpoints/best.pt --steps 100
    python inference.py --checkpoint checkpoints/best.pt --steps 100 --episode data/episodes/test.pkl --ep_idx 0

If no --episode is given, runs from a fresh simulator initial state.
"""
import argparse
import os
import pickle
from typing import Optional
import torch
import numpy as np
import wandb

from config import ModelConfig, TrainConfig, SimConfig, MODEL, SIM, TRAIN
from model import EcosystemWorldModel
from training.utils import apply_predictions_to_state
from simulator.model import EcosystemModel
from simulator.generate_data import snapshot_to_arrays
from simulator.agents import SPECIES_PREY, SPECIES_PREDATOR


# ── State loading helpers ────────────────────────────────────────────────────

def state_from_episode(episode: list, step: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    agent_arr, patch_arr, _ = episode[step]
    agents_t  = torch.from_numpy(agent_arr).unsqueeze(0)   # (1, N_a, 6)
    patches_t = torch.from_numpy(patch_arr).unsqueeze(0)   # (1, N_p, 4)
    return agents_t, patches_t


def state_from_simulator(
    sim_cfg: SimConfig,
    n_prey: int = 30,
    n_pred: int = 8,
    food_density: float = 0.6,
    seed: int = 99,
) -> tuple[torch.Tensor, torch.Tensor]:
    model = EcosystemModel(sim_cfg=sim_cfg, n_prey=n_prey, n_pred=n_pred,
                           food_density=food_density, seed=seed)
    snap = model.datacollector.get_model_vars_dataframe()["snapshot"].iloc[0]
    agent_arr, patch_arr, _ = snapshot_to_arrays(
        snap, sim_cfg.n_max_agents, sim_cfg.grid_width, sim_cfg.grid_height
    )
    agents_t  = torch.from_numpy(agent_arr).unsqueeze(0)
    patches_t = torch.from_numpy(patch_arr).unsqueeze(0)
    return agents_t, patches_t


# ── Per-step metric computation ──────────────────────────────────────────────

def compute_step_metrics(
    agents_t: torch.Tensor,    # (1, N_a, 6)
    patches_t: torch.Tensor,   # (1, N_p, 4)
    preds: dict,
    threshold: float = 0.5,
    ghost_lo: float = 0.3,
    ghost_hi: float = 0.5,
) -> dict:
    alive_probs = preds["alive_probs"][0]           # (N_a,)
    alive_hard  = (alive_probs > threshold).float()
    species     = agents_t[0, :, 0].long()          # (N_a,)
    energy      = agents_t[0, :, 3]                 # (N_a,)
    food        = patches_t[0, :, 2]                # (N_p,)

    is_prey = (species == SPECIES_PREY).float()
    is_pred = (species == SPECIES_PREDATOR).float()

    prey_count  = (alive_hard * is_prey).sum().item()
    pred_count  = (alive_hard * is_pred).sum().item()

    # Ghost agents: uncertain — in (ghost_lo, ghost_hi)
    ghost_mask  = (alive_probs > ghost_lo) & (alive_probs < ghost_hi)
    ghost_count = ghost_mask.sum().item()

    # Mean energy of alive agents
    n_alive = alive_hard.sum().item()
    mean_energy = ((alive_hard * energy).sum() / max(n_alive, 1)).item()

    # Food coverage
    food_coverage = (food > 0.5).float().mean().item()

    return {
        "prey_count":    prey_count,
        "pred_count":    pred_count,
        "ghost_count":   ghost_count,
        "mean_energy":   mean_energy,
        "food_coverage": food_coverage,
    }


# ── Main inference loop ──────────────────────────────────────────────────────

def run_inference(
    checkpoint_path: str,
    n_steps: int = 100,
    episode_path: Optional[str] = None,
    ep_idx: int = 0,
    model_cfg: Optional[ModelConfig] = None,
    sim_cfg: Optional[SimConfig] = None,
    threshold: float = 0.5,
    offline: bool = False,
):
    model_cfg = model_cfg or MODEL
    sim_cfg   = sim_cfg   or SIM

    # ── Load model ────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else
                          "cpu")
    world_model = EcosystemWorldModel(model_cfg).to(device)

    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device)
        world_model.load_state_dict(state["model_state"])
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"Warning: checkpoint not found at {checkpoint_path}. Using random weights.")

    world_model.eval()

    # ── Prepare initial state ─────────────────────────────────────────────
    if episode_path and os.path.exists(episode_path):
        with open(episode_path, "rb") as f:
            episodes = pickle.load(f)
        episode = episodes[ep_idx % len(episodes)]
        agents_t, patches_t = state_from_episode(episode, step=0)
        print(f"Loaded episode {ep_idx} from {episode_path}")
    else:
        agents_t, patches_t = state_from_simulator(sim_cfg)
        print("Initialised from fresh simulator state.")

    agents_t  = agents_t.to(device)
    patches_t = patches_t.to(device)

    # ── wandb run ─────────────────────────────────────────────────────────
    wandb_mode = "offline" if offline else "online"
    run = wandb.init(
        project="ecosystem-world-model",
        tags=["inference"],
        job_type="rollout",
        mode=wandb_mode,
        config={
            "checkpoint": checkpoint_path,
            "n_steps": n_steps,
            "threshold": threshold,
        },
    )

    # ── Rollout loop ──────────────────────────────────────────────────────
    print(f"Rolling out for {n_steps} steps...")
    history = {"prey": [], "pred": [], "ghost": [], "energy": [], "food": []}

    with torch.no_grad():
        for step in range(n_steps):
            preds = world_model(agents_t, patches_t)
            metrics = compute_step_metrics(agents_t, patches_t, preds, threshold)

            wandb.log({
                "inference/step":          step,
                "inference/prey_count":    metrics["prey_count"],
                "inference/pred_count":    metrics["pred_count"],
                "inference/ghost_count":   metrics["ghost_count"],
                "inference/mean_energy":   metrics["mean_energy"],
                "inference/food_coverage": metrics["food_coverage"],
            })

            for key, arr in zip(
                ["prey", "pred", "ghost", "energy", "food"],
                [metrics["prey_count"], metrics["pred_count"], metrics["ghost_count"],
                 metrics["mean_energy"], metrics["food_coverage"]],
            ):
                history[key].append(arr)

            if (step + 1) % 10 == 0:
                print(f"  step={step+1:4d} | prey={metrics['prey_count']:5.1f} | "
                      f"pred={metrics['pred_count']:5.1f} | "
                      f"ghosts={metrics['ghost_count']:4.1f} | "
                      f"energy={metrics['mean_energy']:.3f} | "
                      f"food={metrics['food_coverage']:.3f}")

            # Early stop if population collapses
            if metrics["prey_count"] == 0 and metrics["pred_count"] == 0:
                print(f"  Ecosystem collapse detected at step {step}.")
                break

            # Advance state
            agents_t, patches_t = apply_predictions_to_state(
                agents_t, patches_t, preds, threshold,
                grid_size=sim_cfg.grid_size,
            )

    # ── Summary trajectory chart ──────────────────────────────────────────
    T = len(history["prey"])
    wandb.log({
        "inference/trajectory": wandb.plot.line_series(
            xs=list(range(T)),
            ys=[history["prey"], history["pred"]],
            keys=["prey_count", "pred_count"],
            title="Inference: population trajectory",
            xname="step",
        )
    })

    wandb.finish()
    print("Inference complete.")
    return history


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run ecosystem world model inference")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    parser.add_argument("--steps",      type=int, default=100)
    parser.add_argument("--episode",    type=str, default=None,
                        help="Path to a .pkl episode file")
    parser.add_argument("--ep_idx",     type=int, default=0)
    parser.add_argument("--threshold",  type=float, default=0.5)
    parser.add_argument("--offline",    action="store_true",
                        help="Log to wandb in offline mode (no internet required)")
    args = parser.parse_args()

    run_inference(
        checkpoint_path=args.checkpoint,
        n_steps=args.steps,
        episode_path=args.episode,
        ep_idx=args.ep_idx,
        threshold=args.threshold,
        offline=args.offline,
    )


if __name__ == "__main__":
    main()
