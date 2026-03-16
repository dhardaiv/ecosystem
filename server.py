"""
FastAPI backend server for the Ecosystem World Model visualiser.

Endpoints:
  POST /api/rollout/start   — initialise a new rollout
  POST /api/rollout/step    — advance one step
  POST /api/rollout/reset   — clear current rollout
  GET  /api/rollout/history — return all steps so far

Run with:
  python server.py
  # or
  uvicorn server:app --reload --port 8000
"""
import os
import sys
import uuid
import math
from typing import Optional

import torch
import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from config import ModelConfig, SimConfig, MODEL, SIM, TRAIN
from model import EcosystemWorldModel
from training.utils import apply_predictions_to_state
from simulator.model import EcosystemModel
from simulator.generate_data import snapshot_to_arrays
from simulator.agents import SPECIES_PREY, SPECIES_PREDATOR

# ── device ───────────────────────────────────────────────────────────────────
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)
print(f"Using device: {DEVICE}")

CHECKPOINT = os.path.join(os.path.dirname(__file__), "checkpoints", "demo_best.pt")


def infer_model_cfg_from_checkpoint(state_dict: dict) -> ModelConfig:
    """Read weight shapes from a saved state_dict and build the matching ModelConfig."""
    sd = state_dict
    d_model          = sd["transformer.blocks.0.attn.q_proj.weight"].shape[0]
    d_ff             = sd["transformer.blocks.0.ff.0.weight"].shape[0]
    d_species_embed  = sd["encoder.species_embed.weight"].shape[1]
    n_food_bins      = sd["heads.food_head.proj.weight"].shape[0]
    n_layers         = sum(1 for k in sd if "attn.q_proj.weight" in k)
    n_mlp_hidden     = sd["encoder.cont_mlp.0.weight"].shape[0]
    # spatial_enc.freqs shape = (n_freq,); d_enc = 4 * n_freq
    spatial_enc_dim  = int(sd["encoder.spatial_enc.freqs"].shape[0]) * 4
    # n_heads: d_model / head_dim; typical head_dim=16
    n_heads          = max(1, d_model // 16)

    return ModelConfig(
        d_model             = d_model,
        n_layers            = n_layers,
        n_heads             = n_heads,
        d_ff                = d_ff,
        d_species_embed     = d_species_embed,
        n_mlp_hidden        = n_mlp_hidden,
        spatial_encoding_dim= spatial_enc_dim,
        n_food_bins         = n_food_bins,
    )


# ── load model once at startup ────────────────────────────────────────────────
print("Loading world model...")
if os.path.exists(CHECKPOINT):
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
    model_cfg = infer_model_cfg_from_checkpoint(ckpt["model_state"])
    print(f"Detected config: d_model={model_cfg.d_model}, n_layers={model_cfg.n_layers}, "
          f"d_ff={model_cfg.d_ff}, spatial_enc_dim={model_cfg.spatial_encoding_dim}")
    world_model = EcosystemWorldModel(model_cfg).to(DEVICE)
    world_model.load_state_dict(ckpt["model_state"])
    print(f"Loaded checkpoint: {CHECKPOINT}")
else:
    model_cfg = MODEL
    world_model = EcosystemWorldModel(model_cfg).to(DEVICE)
    print(f"Warning: checkpoint not found at {CHECKPOINT}. Using random weights.")
world_model.eval()

# ── in-memory rollout store ───────────────────────────────────────────────────
_rollouts: dict = {}   # rollout_id -> rollout dict

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="Ecosystem World Model API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── helpers ───────────────────────────────────────────────────────────────────

def tensors_to_state(agents_t: torch.Tensor, patches_t: torch.Tensor,
                     preds: dict, sim_cfg: SimConfig, threshold: float = 0.5) -> dict:
    """
    Convert raw tensors + predictions to the JSON state format the frontend expects.

    agent cols:  0=species, 1=x, 2=y, 3=energy, 4=age, 5=alive
    patch cols:  0=x,       1=y, 2=food,         3=alive
    """
    a = agents_t[0]   # (N_agents, 6)
    p = patches_t[0]  # (N_patches, 4)

    alive_probs = preds["alive_probs"][0]  # (N_agents,)

    gw = sim_cfg.grid_width
    gh = sim_cfg.grid_height

    agents_out = []
    for i in range(a.shape[0]):
        species_id = int(a[i, 0].item())
        alive_val  = float(a[i, 5].item())
        # Only include agents that are alive (or recently alive slots with p_alive > 0.1)
        p_alive_val = float(alive_probs[i].item())
        if alive_val < 0.5 and p_alive_val < 0.1:
            continue
        agents_out.append({
            "id":      i,
            "type":    "prey" if species_id == SPECIES_PREY else "predator",
            "x":       round(float(a[i, 1].item()), 5),
            "y":       round(float(a[i, 2].item()), 5),
            "energy":  round(float(a[i, 3].item()), 5),
            "age":     round(float(a[i, 4].item()), 5),
            "alive":   int(round(alive_val)),
            "p_alive": round(p_alive_val, 5),
        })

    # Build a grid: accumulate max food per (px, py) cell to avoid duplicates
    grid_food: dict[tuple, float] = {}
    for j in range(p.shape[0]):
        # Invert normalisation: x = ix / W  =>  ix = round(x * W)
        px = int(round(float(p[j, 0].item()) * gw))
        py = int(round(float(p[j, 1].item()) * gh))
        px = max(0, min(gw - 1, px))
        py = max(0, min(gh - 1, py))
        food = float(p[j, 2].item())
        if food > 0.02:
            key = (px, py)
            if key not in grid_food or food > grid_food[key]:
                grid_food[key] = food

    patches_out = [
        {"x": px, "y": py, "food": round(food, 4)}
        for (px, py), food in grid_food.items()
    ]

    return {"agents": agents_out, "patches": patches_out}


def compute_metrics(agents_t: torch.Tensor, patches_t: torch.Tensor,
                    preds: dict, sim_cfg: SimConfig, threshold: float = 0.5) -> dict:
    alive_probs = preds["alive_probs"][0]      # (N_agents,)
    alive_hard  = (alive_probs > threshold).float()
    species     = agents_t[0, :, 0].long()
    energy      = agents_t[0, :, 3]

    is_prey = (species == SPECIES_PREY).float()
    is_pred = (species == SPECIES_PREDATOR).float()

    prey_count = (alive_hard * is_prey).sum().item()
    pred_count = (alive_hard * is_pred).sum().item()

    n_alive = alive_hard.sum().item()
    mean_energy_prey = ((alive_hard * is_prey * energy).sum() / max((alive_hard * is_prey).sum().item(), 1)).item()
    mean_energy_pred = ((alive_hard * is_pred * energy).sum() / max((alive_hard * is_pred).sum().item(), 1)).item()

    ghost_mask  = (alive_probs > 0.3) & (alive_probs < 0.5)
    ghost_count = int(ghost_mask.sum().item())

    # Dummy loss values — compute approximate versions from model output
    # (real training losses need ground truth; here we use prediction entropy as proxy)
    bce_logits = preds["alive_logit"][0]
    probs = torch.sigmoid(bce_logits)
    entropy = -(probs * (probs + 1e-8).log() + (1 - probs) * (1 - probs + 1e-8).log())
    loss_bce = float(entropy.mean().item())

    delta = preds["delta_pred"][0]
    loss_mse = float((delta ** 2).mean().item())

    food_logits = preds["food_logits"][0]   # (N_patches, n_bins)
    food_probs  = torch.softmax(food_logits, dim=-1)
    food_entropy = -(food_probs * (food_probs + 1e-8).log()).sum(-1)
    loss_ce = float(food_entropy.mean().item())

    loss_aux = 0.001  # placeholder

    return {
        "prey_count":       round(prey_count),
        "pred_count":       round(pred_count),
        "mean_energy_prey": round(mean_energy_prey, 4),
        "mean_energy_pred": round(mean_energy_pred, 4),
        "ghost_count":      ghost_count,
        "loss_mse":         round(loss_mse, 5),
        "loss_bce":         round(loss_bce, 5),
        "loss_ce":          round(loss_ce, 5),
        "loss_aux":         round(loss_aux, 5),
    }


# ── Request models ─────────────────────────────────────────────────────────────

class StartRequest(BaseModel):
    episode_seed:     int   = 42
    n_steps:          int   = 100
    noise_injection:  float = 0.0


class StepRequest(BaseModel):
    rollout_id: str


class ResetRequest(BaseModel):
    pass


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/api/rollout/start")
def start_rollout(req: StartRequest):
    # Build initial state from simulator
    sim_cfg = SIM
    sim = EcosystemModel(
        sim_cfg=sim_cfg,
        n_prey=sim_cfg.n_initial_prey,
        n_pred=sim_cfg.n_initial_predators,
        food_density=sim_cfg.initial_food_density,
        seed=req.episode_seed,
    )
    snap = sim.datacollector.get_model_vars_dataframe()["snapshot"].iloc[0]
    agent_arr, patch_arr, _ = snapshot_to_arrays(
        snap, sim_cfg.n_max_agents, sim_cfg.grid_width, sim_cfg.grid_height
    )

    agents_t  = torch.from_numpy(agent_arr).unsqueeze(0).to(DEVICE)
    patches_t = torch.from_numpy(patch_arr).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        preds = world_model(agents_t, patches_t)

    rollout_id = str(uuid.uuid4())[:8]
    initial_state = tensors_to_state(agents_t, patches_t, preds, sim_cfg)

    _rollouts[rollout_id] = {
        "agents_t":       agents_t,
        "patches_t":      patches_t,
        "step":           0,
        "n_steps":        req.n_steps,
        "noise":          req.noise_injection,
        "sim_cfg":        sim_cfg,
        "history":        [],
    }

    return {
        "rollout_id":   rollout_id,
        "grid_width":   sim_cfg.grid_width,
        "grid_height":  sim_cfg.grid_height,
        "n_max_agents": sim_cfg.n_max_agents,
        "initial_state": initial_state,
    }


@app.post("/api/rollout/step")
def step_rollout(req: StepRequest):
    r = _rollouts.get(req.rollout_id)
    if r is None:
        raise HTTPException(status_code=404, detail=f"Rollout {req.rollout_id} not found")

    agents_t  = r["agents_t"]
    patches_t = r["patches_t"]
    sim_cfg   = r["sim_cfg"]
    threshold = 0.5

    with torch.no_grad():
        preds = world_model(agents_t, patches_t)

    # Inject noise to alive mask if requested
    noise = r["noise"]
    if noise > 0:
        from training.utils import inject_alive_noise
        noisy_alive = inject_alive_noise(agents_t[0, :, 5], noise)
        agents_t = agents_t.clone()
        agents_t[0, :, 5] = noisy_alive

    agents_next, patches_next = apply_predictions_to_state(
        agents_t, patches_t, preds, threshold
    )

    # Advance state
    r["agents_t"]  = agents_next
    r["patches_t"] = patches_next
    r["step"] += 1

    state   = tensors_to_state(agents_next, patches_next, preds, sim_cfg, threshold)
    metrics = compute_metrics(agents_t, patches_t, preds, sim_cfg, threshold)

    step_record = {"step": r["step"], "state": state, "metrics": metrics}
    r["history"].append(step_record)

    return step_record


@app.post("/api/rollout/reset")
def reset_rollout():
    _rollouts.clear()
    return {"status": "ok"}


@app.get("/api/rollout/history")
def get_history(rollout_id: str):
    r = _rollouts.get(rollout_id)
    if r is None:
        raise HTTPException(status_code=404, detail=f"Rollout {rollout_id} not found")
    return {"steps": r["history"]}


@app.get("/api/health")
def health():
    return {"status": "ok", "device": str(DEVICE)}


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
