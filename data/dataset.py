"""
PyTorch Dataset and DataLoader for the ecosystem world model.

Each sample is a consecutive pair (s_t, s_{t+1}) from a single episode.

Batch tensors:
  agents_t       (B, N_a, 6)  — species(int), x, y, energy, age, alive
  patches_t      (B, N_p, 4)  — x, y, food, alive
  agents_t1      (B, N_a, 6)  — same layout, next step
  patches_t1     (B, N_p, 4)
  counts_t       (B, 2)       — [n_prey, n_pred] at t
  counts_t1      (B, 2)       — [n_prey, n_pred] at t+1

N_a = sim_cfg.n_max_agents (padded)
N_p = grid_w * grid_h (full grid, never padded)
"""
import os
import pickle
from typing import Optional, List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config import SimConfig, DataConfig, SIM, DATA


# ── Dataset ──────────────────────────────────────────────────────────────────

class EcosystemDataset(Dataset):
    """
    Iterates over consecutive (s_t, s_{t+1}) pairs from a list of episodes.

    Episodes are lists of tuples (agent_arr, patch_arr, counts).
    """

    def __init__(self, episodes: List[List[Tuple]]):
        self.pairs: List[Tuple[int, int]] = []   # (episode_idx, step_idx)
        self.episodes = episodes

        for ep_idx, episode in enumerate(episodes):
            T = len(episode)
            for t in range(T - 1):
                self.pairs.append((ep_idx, t))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        ep_idx, t = self.pairs[idx]
        episode = self.episodes[ep_idx]

        agents_t,  patches_t,  counts_t  = episode[t]
        agents_t1, patches_t1, counts_t1 = episode[t + 1]

        return {
            "agents_t":   torch.from_numpy(agents_t),    # (N_a, 6)
            "patches_t":  torch.from_numpy(patches_t),   # (N_p, 4)
            "counts_t":   torch.from_numpy(counts_t),    # (2,)
            "agents_t1":  torch.from_numpy(agents_t1),   # (N_a, 6)
            "patches_t1": torch.from_numpy(patches_t1),  # (N_p, 4)
            "counts_t1":  torch.from_numpy(counts_t1),   # (2,)
        }


# ── Helpers for computing delta targets ──────────────────────────────────────

def compute_deltas(agents_t: torch.Tensor, agents_t1: torch.Tensor) -> torch.Tensor:
    """
    Compute (Δx, Δy, Δenergy) for alive agents.

    agents_t  : (B, N_a, 6)  cols: species, x, y, energy, age, alive
    agents_t1 : (B, N_a, 6)

    Returns delta_targets : (B, N_a, 3) — (Δx, Δy, Δenergy)
    """
    # Columns: 0=species, 1=x, 2=y, 3=energy, 4=age, 5=alive
    pos_energy_t  = agents_t[..., 1:4]   # (B, N_a, 3)  x, y, energy
    pos_energy_t1 = agents_t1[..., 1:4]
    return pos_energy_t1 - pos_energy_t  # (B, N_a, 3)


def discretise_food(food_level: torch.Tensor, n_bins: int = 5) -> torch.Tensor:
    """
    Discretise food level in [0,1] into n_bins integer class labels.
    patches : (B, N_p) float32 → (B, N_p) int64
    """
    bins = torch.linspace(0.0, 1.0, n_bins + 1)
    labels = torch.bucketize(food_level, bins[1:-1])  # (B, N_p) int64
    return labels.long()


# ── Loaders ──────────────────────────────────────────────────────────────────

def load_episodes(path: str) -> list:
    with open(path, "rb") as f:
        return pickle.load(f)


def compute_death_rate(episodes: list) -> dict:
    """
    Audit the fraction of (s_t, s_{t+1}) pairs that contain at least one death
    event, and the per-slot death probability across the full dataset.

    A death event is: alive at t (col 5 == 1) AND dead at t+1 (col 5 == 0).

    Returns
    -------
    dict with keys:
      n_pairs                    — total transition pairs examined
      n_pairs_with_death         — pairs where ≥1 agent dies
      fraction_pairs_with_death  — above as a fraction
      total_death_events         — sum of individual death events
      total_alive_at_t           — sum of alive agent-slots at t
      per_slot_death_rate        — death events / alive slots (death probability per step)
    """
    n_pairs = 0
    n_pairs_with_death = 0
    total_death_events = 0
    total_alive_at_t = 0

    for episode in episodes:
        for t in range(len(episode) - 1):
            agents_t,  _, _ = episode[t]
            agents_t1, _, _ = episode[t + 1]
            alive_t  = agents_t[:, 5]
            alive_t1 = agents_t1[:, 5]
            deaths = ((alive_t == 1) & (alive_t1 == 0)).sum()
            n_pairs += 1
            if deaths > 0:
                n_pairs_with_death += 1
            total_death_events += int(deaths)
            total_alive_at_t   += int((alive_t == 1).sum())

    return {
        "n_pairs":                    n_pairs,
        "n_pairs_with_death":         n_pairs_with_death,
        "fraction_pairs_with_death":  n_pairs_with_death / max(n_pairs, 1),
        "total_death_events":         total_death_events,
        "total_alive_at_t":           total_alive_at_t,
        "per_slot_death_rate":        total_death_events / max(total_alive_at_t, 1),
    }


def make_dataloaders(
    data_dir: str = DATA.data_dir,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train_loader, val_loader, test_loader)."""
    train_eps = load_episodes(os.path.join(data_dir, "train.pkl"))
    val_eps   = load_episodes(os.path.join(data_dir, "val.pkl"))
    test_eps  = load_episodes(os.path.join(data_dir, "test.pkl"))

    train_ds = EcosystemDataset(train_eps)
    val_ds   = EcosystemDataset(val_eps)
    test_ds  = EcosystemDataset(test_eps)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader
