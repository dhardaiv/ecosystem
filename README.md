# Ecosystem World Model

A transformer-based world model that learns the transition dynamics **p(s_{t+1} | s_t)** of a Mesa-simulated predator–prey–food ecosystem. The model predicts how the ecosystem evolves one (or multiple) steps forward given the current full state.

---

## Architecture Overview

```
Simulator (Mesa)  →  (s_t, s_{t+1}) pairs
                           │
                    EntityEncoder
              (species_embed + cont_MLP + spatial_enc)
                           │
               EcosystemTransformer (L=4 blocks)
              ┌─────────────────────────────┐
              │  MaskedMHSA (3 masks)       │
              │  ┣ Alive mask               │
              │  ┣ Locality mask (radius r) │
              │  ┗ Cross-type mask          │
              │  + Feed-forward + LayerNorm │
              └─────────────────────────────┘
                           │
              ┌────────────┼────────────────┐
           MSEHead      BCEHead       FoodCEHead  AuxHead
         (Δx,Δy,Δe)  p(alive_t+1)  food_bin     (n_prey, n_pred)
```

**Token layout per step:** `[agent_tokens (N_max padded) | patch_tokens (W×H full grid)]`

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate training data

```bash
python -m simulator.generate_data --train 200 --val 30 --test 20 --steps 200
```

Generated data is stored in `data/episodes/` as compressed pickle files.

### 3. Train

```bash
python train.py
```

Optional flags:
```bash
python train.py --epochs 150 --batch_size 64 --lr 1e-4
python train.py --resume checkpoints/best.pt     # resume from checkpoint
```

### 4. Inference

```bash
# From a fresh simulator state
python inference.py --checkpoint checkpoints/best.pt --steps 100

# From a held-out test episode
python inference.py --checkpoint checkpoints/best.pt \
    --episode data/episodes/test.pkl --ep_idx 0 --steps 100

# Offline (no internet needed for wandb)
python inference.py --checkpoint checkpoints/best.pt --steps 100 --offline
```

---

## Project Structure

```
world/
├── config.py                   # All hyperparameters (SimConfig, ModelConfig, TrainConfig, DataConfig)
├── train.py                    # Main training entrypoint
├── inference.py                # Rollout inference with wandb logging
├── requirements.txt
│
├── simulator/
│   ├── agents.py               # PreyAgent, PredatorAgent, FoodPatch
│   ├── model.py                # EcosystemModel (Mesa Model)
│   └── generate_data.py        # Script to run simulator and save dataset
│
├── data/
│   └── dataset.py              # EcosystemDataset, DataLoader, delta helpers
│
├── model/
│   ├── __init__.py             # EcosystemWorldModel (full assembled model)
│   ├── encoder.py              # EntityEncoder + SpatialEncoding
│   ├── transformer.py          # EcosystemTransformer, MaskedMHSA, attention mask builder
│   └── heads.py                # MSEHead, BCEHead, FoodCEHead, AuxHead, compute_loss()
│
└── training/
    ├── trainer.py              # Trainer: curriculum, scheduled sampling, wandb logging
    └── rollout.py              # Rollout evaluation suite (ecosystem-level metrics)
```

---

## Key Design Decisions

### Per-entity tokenisation
One token per agent/patch per timestep. Agent ordering is arbitrary — no sequential positional encoding. Spatial sinusoidal encoding ensures physically close agents share similar representations.

### Three simultaneous attention masks
Applied as additive `−∞` bias to attention scores:
1. **Alive mask** — dead and padding tokens cannot be attended to as keys
2. **Locality mask** — agents only attend within radius `r` (normalised grid units)
3. **Cross-type mask** (optional) — separates agent↔agent and patch↔patch streams

### Death cascade mitigation
Three complementary strategies prevent ghost agents from corrupting multi-step rollouts:
1. **Alive noise injection** — randomly flip alive=1→0 during training
2. **Straight-through estimator** — hard binary mask forward, soft gradients backward
3. **Hard threshold** — `p < 0.5 → dead` before advancing state

### Curriculum learning (3 phases)
- **Phase 1** (epochs 1–19): single-step teacher-forced prediction, `λ_aux = 0`
- **Phase 2** (epochs 20–39): introduce auxiliary population count head (`λ_aux = 0.1`)
- **Phase 3** (epochs 40+): multi-step rollout loss with scheduled sampling

### Delta prediction
The MSE head predicts `(Δx, Δy, Δenergy)` rather than absolute next-state values — the same movement pattern is valid regardless of absolute position, making the learned function smoother and more generalisable.

---

## Loss Function

```
L_total = λ1·L_mse + λ2·L_bce + λ3·L_ce + λ4·L_aux
```

| Head | Loss | Default λ | Notes |
|------|------|-----------|-------|
| MSE  | Masked MSE on Δx, Δy, Δenergy | 1.0 | Alive agents only |
| BCE  | Binary cross-entropy p(alive) | 2.0 | Dead class upweighted ×10 |
| CE   | Cross-entropy over 5 food bins | 0.5 | All patches |
| Aux  | MSE on global count predictions | 0.1 | Phase 2+ only |

---

## wandb Logging

All metrics are logged to the `ecosystem-world-model` project.

**Training** (every step): `train/loss_*`, `train/alive_frac`, `train/grad_norm`, `train/ss_rate`, `train/phase`

**Validation** (every epoch): `val/loss_*`, `val/position_mse`, `val/energy_mse`, `val/alive_accuracy`, `val/alive_f1`

**Rollout eval** (every 5 epochs): `rollout/pop_correlation_*`, `rollout/extinction_detection`, `rollout/spatial_kl_t20`, `rollout/energy_kl_t20`, `rollout/stability_rate`, `rollout/mean_ghost_count`, `rollout/trajectory` (chart), `rollout/attention_map` (image)

**Inference**: `inference/prey_count`, `inference/pred_count`, `inference/ghost_count`, `inference/mean_energy`, `inference/food_coverage`

---

## Configuration

All hyperparameters live in `config.py` as frozen dataclasses:

```python
from config import SIM, MODEL, TRAIN, DATA
```

Override any value before passing to the relevant class, or add CLI flags in `train.py`.
