"""
Generate training, validation, and test datasets from the Mesa simulator.

Each episode produces a list of snapshot dicts [s_0, ..., s_T].
We save each episode as a list of numpy arrays and store all episodes in a
compressed .npz archive, one per split.

Run:
    python -m simulator.generate_data
"""
import os
import argparse
import pickle
from typing import List, Tuple
import numpy as np
from tqdm import tqdm

from config import SimConfig, DataConfig, SIM, DATA
from simulator.model import EcosystemModel, MAX_AGE_NORM
from simulator.agents import SPECIES_PREY, SPECIES_PREDATOR, SPECIES_PATCH


# ── Snapshot → numpy array ───────────────────────────────────────────────────

AGENT_FIELDS = ["entity_id", "species", "x", "y", "energy", "age", "alive"]


def snapshot_to_arrays(snapshot: dict, n_max_agents: int, grid_w: int, grid_h: int):
    """
    Convert a raw snapshot dict to three padded numpy arrays.

    Returns
    -------
    agent_arr  : float32 (n_max_agents, 6)  columns: species, x, y, energy, age, alive
    patch_arr  : float32 (grid_w*grid_h, 4) columns: x, y, food, alive (always 1)
    counts     : int32   (2,)               [n_prey, n_pred]
    """
    agents = snapshot["agents"]
    patches = snapshot["patches"]

    # Agent array — padded to n_max_agents
    agent_arr = np.zeros((n_max_agents, 6), dtype=np.float32)
    actual_n = min(len(agents), n_max_agents)
    for i, a in enumerate(agents[:actual_n]):
        agent_arr[i] = [a["species"], a["x"], a["y"], a["energy"], a["age"], a["alive"]]

    # Patch array (fixed size, no padding needed)
    n_patches = grid_w * grid_h
    patch_arr = np.zeros((n_patches, 4), dtype=np.float32)
    for j, p in enumerate(patches[:n_patches]):
        patch_arr[j] = [p["x"], p["y"], p["energy"], p["alive"]]

    # Global counts
    n_prey = sum(1 for a in agents if a["species"] == SPECIES_PREY and a["alive"] > 0.5)
    n_pred = sum(1 for a in agents if a["species"] == SPECIES_PREDATOR and a["alive"] > 0.5)
    counts = np.array([n_prey, n_pred], dtype=np.int32)

    return agent_arr, patch_arr, counts


def run_episode(
    sim_cfg: SimConfig,
    data_cfg: DataConfig,
    n_prey: int,
    n_pred: int,
    food_density: float,
    seed: int,
) -> List[Tuple]:
    """
    Run one episode and return a list of (agent_arr, patch_arr, counts) tuples,
    one per timestep (including step 0).
    """
    model = EcosystemModel(
        sim_cfg=sim_cfg,
        n_prey=n_prey,
        n_pred=n_pred,
        food_density=food_density,
        seed=seed,
    )
    snapshots = model.run(data_cfg.episode_length)

    W = sim_cfg.grid_width
    H = sim_cfg.grid_height
    n_max = sim_cfg.n_max_agents

    return [snapshot_to_arrays(s, n_max, W, H) for s in snapshots]


# ── Dataset builder ──────────────────────────────────────────────────────────

def build_split(
    n_episodes: int,
    split_name: str,
    sim_cfg: SimConfig,
    data_cfg: DataConfig,
    base_seed: int,
) -> List[List[Tuple]]:
    """Run n_episodes and return a list of episodes."""
    rng = np.random.default_rng(base_seed)
    episodes = []
    prey_lo, prey_hi = data_cfg.prey_range
    pred_lo, pred_hi = data_cfg.predator_range
    fd_lo, fd_hi = data_cfg.food_density_range

    for i in tqdm(range(n_episodes), desc=f"Generating {split_name}"):
        n_prey = int(rng.integers(prey_lo, prey_hi + 1))
        n_pred = int(rng.integers(pred_lo, pred_hi + 1))
        food_density = float(rng.uniform(fd_lo, fd_hi))
        seed = int(rng.integers(0, 2**31))
        ep = run_episode(sim_cfg, data_cfg, n_prey, n_pred, food_density, seed)
        episodes.append(ep)

    return episodes


def save_episodes(episodes: list, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(episodes, f)
    print(f"Saved {len(episodes)} episodes → {path}")


def load_episodes(path: str) -> list:
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate ecosystem dataset")
    parser.add_argument("--train", type=int, default=DATA.n_train_episodes)
    parser.add_argument("--val",   type=int, default=DATA.n_val_episodes)
    parser.add_argument("--test",  type=int, default=DATA.n_test_episodes)
    parser.add_argument("--steps", type=int, default=DATA.episode_length)
    parser.add_argument("--out",   type=str, default=DATA.data_dir)
    args = parser.parse_args()

    sim_cfg = SIM
    data_cfg = DataConfig(
        n_train_episodes=args.train,
        n_val_episodes=args.val,
        n_test_episodes=args.test,
        episode_length=args.steps,
        data_dir=args.out,
    )

    print(f"Generating data: {args.train} train / {args.val} val / {args.test} test "
          f"episodes of {args.steps} steps each.")
    print(f"Grid: {sim_cfg.grid_width}×{sim_cfg.grid_height}, "
          f"N_max_agents={sim_cfg.n_max_agents}")

    train_eps = build_split(args.train, "train", sim_cfg, data_cfg, base_seed=0)
    val_eps   = build_split(args.val,   "val",   sim_cfg, data_cfg, base_seed=10000)
    test_eps  = build_split(args.test,  "test",  sim_cfg, data_cfg, base_seed=20000)

    save_episodes(train_eps, os.path.join(args.out, "train.pkl"))
    save_episodes(val_eps,   os.path.join(args.out, "val.pkl"))
    save_episodes(test_eps,  os.path.join(args.out, "test.pkl"))

    print("Done.")


if __name__ == "__main__":
    main()
