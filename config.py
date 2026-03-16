"""Central configuration for the ecosystem world model."""
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class SimConfig:
    """Simulator hyperparameters."""
    grid_width: int = 40
    grid_height: int = 40
    n_initial_prey: int = 30
    n_initial_predators: int = 8
    initial_food_density: float = 0.6  # fraction of cells with food

    # Prey dynamics
    prey_move_energy_cost: float = 0.02
    prey_metabolism: float = 0.01       # energy lost per step (idle)
    prey_eat_gain: float = 0.25         # energy gained per food unit eaten
    prey_reproduce_threshold: float = 0.8
    prey_reproduce_cost: float = 0.4
    prey_initial_energy: float = 0.5

    # Predator dynamics
    predator_move_energy_cost: float = 0.03
    predator_metabolism: float = 0.02
    predator_eat_gain: float = 0.5
    predator_reproduce_threshold: float = 0.8
    predator_reproduce_cost: float = 0.4
    predator_initial_energy: float = 0.5
    predator_hunt_radius: int = 2

    # Food patch regrowth
    food_regrow_rate: float = 0.03      # food units regrown per step

    # Max agents padded in a snapshot (agents + patches)
    n_max_agents: int = 200             # agents only
    n_food_patches: int = 40 * 40       # all grid cells are patches


@dataclass
class ModelConfig:
    """Transformer world model hyperparameters."""
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 256                     # feed-forward hidden dim
    dropout: float = 0.1

    # Entity encoder
    n_species: int = 3                  # prey=0, predator=1, patch=2
    d_species_embed: int = 32
    d_continuous: int = 4               # x, y, energy, age (patches: x, y, food, 0)
    n_mlp_hidden: int = 64

    # Spatial sinusoidal encoding
    use_spatial_encoding: bool = True
    spatial_encoding_dim: int = 64      # must be even; added to token

    # Attention masking
    use_locality_mask: bool = True
    locality_radius: float = 0.25       # fraction of grid width
    use_cross_type_mask: bool = False   # separate agent/patch streams

    # Food discretisation
    n_food_bins: int = 5                # [0,0.2), [0.2,0.4), ..., [0.8,1]

    # N_max for padding (agents only)
    n_max_agents: int = 200
    grid_size: int = 40                 # for normalisation


@dataclass
class TrainConfig:
    """Training loop hyperparameters."""
    # Loss weights
    lambda_move: float = 1.0            # movement cross-entropy
    lambda_energy: float = 1.0          # energy delta MSE
    lambda_bce: float = 5.0             # raised from 2.0 — harder signal on dead class
    lambda_ce: float = 0.5
    lambda_aux: float = 0.0             # 0 during phase 1

    # Dead-class upweighting in BCE
    bce_dead_weight: float = 10.0

    # Energy variance penalty (penalises near-zero Δenergy variance across batch)
    energy_var_reg: float = 0.1         # weight on variance penalty term
    energy_var_threshold: float = 0.001 # var floor; penalty = relu(threshold - var)

    # Movement sampling temperature (used at rollout time; 1.0 = argmax-equivalent default)
    movement_temperature: float = 1.0

    # Optimiser
    lr: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 32
    n_epochs: int = 100
    grad_clip: float = 1.0
    grad_clip_aux: float = 0.1      # tighter clip applied to aux head only

    # Scheduled sampling (teacher-forcing → self-rollout)
    ss_rate_init: float = 0.10          # non-zero from epoch 1
    ss_rate_max: float = 0.50           # cap — never fully discard ground truth
    ss_rate_increment: float = 0.01     # added to ss_rate each epoch

    # Alive mask noise injection
    alive_noise_rate: float = 0.05

    # Curriculum phases (epoch thresholds)
    phase2_epoch: int = 20              # introduce lambda_aux
    phase3_epoch: int = 40              # introduce multi-step rollout

    # Multi-step rollout window (phase 3)
    rollout_k_start: int = 2
    rollout_k_max: int = 5

    # Validation / evaluation
    val_every_n_epochs: int = 1
    rollout_eval_every_n_epochs: int = 5
    rollout_horizon: int = 50

    # Checkpointing
    checkpoint_path: str = "checkpoints/best.pt"

    # wandb
    wandb_project: str = "ecosystem-world-model"
    wandb_entity: Optional[str] = None  # set to your wandb username if needed


@dataclass
class DataConfig:
    """Data generation and dataset settings."""
    n_train_episodes: int = 200
    n_val_episodes: int = 30
    n_test_episodes: int = 20
    episode_length: int = 200           # steps per episode
    data_dir: str = "data/episodes"
    dataset_cache: str = "data/dataset.pkl"

    # Varied IC sampling ranges (for diverse episodes)
    prey_range: tuple = (10, 60)
    predator_range: tuple = (3, 15)
    food_density_range: tuple = (0.3, 0.9)


# ── convenience accessors ────────────────────────────────────────────────────

SIM = SimConfig()
MODEL = ModelConfig()
TRAIN = TrainConfig()
DATA = DataConfig()


def all_configs_as_dict() -> dict:
    """Flatten all configs into a single dict for wandb.config."""
    d = {}
    for cfg, prefix in [(SIM, "sim"), (MODEL, "model"), (TRAIN, "train"), (DATA, "data")]:
        for k, v in asdict(cfg).items():
            d[f"{prefix}/{k}"] = v
    return d
