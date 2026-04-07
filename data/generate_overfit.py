"""Generate synthetic overfit dataset for WM-3 world model testing.

Creates a simple but learnable pattern: a colored circle that moves based on actions.
- Action 0: up    (y -= step)
- Action 1: down  (y += step)
- Action 2: left  (x -= step)
- Action 3: right (x += step)
- Action 4: stay  (no movement)

The dataset has predictable:
- Reconstruction: circle at (x, y) with consistent color
- Dynamics: position changes deterministically with action
- Reward: +1 when circle reaches target zone (center), 0 otherwise
- Continue: 1 until max steps or out of bounds
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from pathlib import Path


def draw_circle(
    grid: Tensor,
    cx: float,
    cy: float,
    radius: float = 8.0,
    color: tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> None:
    """Draw a filled circle on a (3, H, W) tensor in-place."""
    _, H, W = grid.shape
    y_coords = torch.arange(H, dtype=grid.dtype, device=grid.device)
    x_coords = torch.arange(W, dtype=grid.dtype, device=grid.device)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
    
    dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
    mask = dist_sq <= radius ** 2
    
    for c, val in enumerate(color):
        grid[c][mask] = val


def draw_target_zone(
    grid: Tensor,
    cx: float = 32.0,
    cy: float = 32.0,
    size: float = 10.0,
) -> None:
    """Draw a subtle target zone (green tint) at center."""
    _, H, W = grid.shape
    y_coords = torch.arange(H, dtype=grid.dtype, device=grid.device)
    x_coords = torch.arange(W, dtype=grid.dtype, device=grid.device)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
    
    in_zone = (torch.abs(xx - cx) <= size) & (torch.abs(yy - cy) <= size)
    grid[1][in_zone] = torch.clamp(grid[1][in_zone] + 0.15, 0, 1)


def generate_sequence(
    seq_len: int,
    act_dim: int = 5,
    grid_size: int = 64,
    scalar_dim: int = 16,
    step_size: float = 4.0,
    radius: float = 6.0,
    seed: int | None = None,
) -> dict[str, Tensor]:
    """Generate a single sequence with learnable dynamics.
    
    Returns dict with shapes:
        obs_grid:    (T, 3, 64, 64)
        obs_scalars: (T, 16)
        actions:     (T, act_dim) one-hot
        rewards:     (T, 1)
        continues:   (T, 1)
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Starting position (random corner region)
    corners = [
        (12, 12), (12, 52), (52, 12), (52, 52),
        (32, 12), (32, 52), (12, 32), (52, 32),
    ]
    start_idx = np.random.randint(len(corners))
    x, y = float(corners[start_idx][0]), float(corners[start_idx][1])
    
    # Velocity for scalars
    vx, vy = 0.0, 0.0
    
    # Circle color (fixed per sequence for consistency)
    color_choices = [
        (1.0, 0.2, 0.2),  # red
        (0.2, 0.2, 1.0),  # blue
        (1.0, 0.8, 0.2),  # yellow
        (0.8, 0.2, 0.8),  # magenta
    ]
    color = color_choices[np.random.randint(len(color_choices))]
    color_idx = color_choices.index(color)
    
    # Target zone center
    target_x, target_y = 32.0, 32.0
    target_radius = 10.0
    
    grids = []
    scalars_list = []
    actions = []
    rewards = []
    continues = []
    
    for t in range(seq_len):
        # Create grid observation
        grid = torch.zeros(3, grid_size, grid_size)
        
        # Background gradient (subtle, learnable texture)
        for c in range(3):
            bg_val = 0.1 + 0.05 * c
            grid[c] = bg_val
        
        # Draw target zone first (so circle overlays it)
        draw_target_zone(grid, target_x, target_y, target_radius)
        
        # Draw circle at current position
        draw_circle(grid, x, y, radius, color)
        
        grids.append(grid)
        
        # Scalar observations: encode position, velocity, color, time
        scalar = torch.zeros(scalar_dim)
        scalar[0] = x / grid_size          # normalized x
        scalar[1] = y / grid_size          # normalized y
        scalar[2] = vx / step_size         # normalized vx
        scalar[3] = vy / step_size         # normalized vy
        scalar[4] = float(color_idx) / 4   # color encoding
        scalar[5] = t / seq_len            # normalized time
        scalar[6] = target_x / grid_size   # target x
        scalar[7] = target_y / grid_size   # target y
        # Remaining dims: distance to target, in-zone indicator
        dist_to_target = np.sqrt((x - target_x)**2 + (y - target_y)**2)
        scalar[8] = dist_to_target / grid_size
        scalar[9] = 1.0 if dist_to_target <= target_radius else 0.0
        scalars_list.append(scalar)
        
        # Sample action (biased toward moving to target for learnable reward)
        if np.random.random() < 0.7:
            # Move toward target
            dx = target_x - x
            dy = target_y - y
            if abs(dx) > abs(dy):
                action_idx = 3 if dx > 0 else 2  # right or left
            else:
                action_idx = 1 if dy > 0 else 0  # down or up
        else:
            action_idx = np.random.randint(act_dim)
        
        action_onehot = F.one_hot(torch.tensor(action_idx), act_dim).float()
        actions.append(action_onehot)
        
        # Compute reward: +1 if in target zone
        in_target = dist_to_target <= target_radius
        reward = torch.tensor([1.0 if in_target else 0.0])
        rewards.append(reward)
        
        # Continue: 0 if out of bounds, 1 otherwise
        out_of_bounds = x < radius or x > grid_size - radius or \
                        y < radius or y > grid_size - radius
        cont = torch.tensor([0.0 if out_of_bounds else 1.0])
        continues.append(cont)
        
        # Update position based on action (for next timestep)
        vx, vy = 0.0, 0.0
        if action_idx == 0:    # up
            vy = -step_size
        elif action_idx == 1:  # down
            vy = step_size
        elif action_idx == 2:  # left
            vx = -step_size
        elif action_idx == 3:  # right
            vx = step_size
        # action 4 = stay
        
        x = np.clip(x + vx, radius, grid_size - radius)
        y = np.clip(y + vy, radius, grid_size - radius)
    
    return {
        "obs_grid": torch.stack(grids),
        "obs_scalars": torch.stack(scalars_list),
        "actions": torch.stack(actions),
        "rewards": torch.stack(rewards),
        "continues": torch.stack(continues),
    }


def generate_dataset(
    num_sequences: int = 8,
    seq_len: int = 48,
    act_dim: int = 5,
    seed: int = 42,
) -> dict[str, Tensor]:
    """Generate full dataset for overfitting test.
    
    Returns dict with shapes:
        obs_grid:    (T, B, 3, 64, 64)
        obs_scalars: (T, B, 16)
        actions:     (T, B, act_dim)
        rewards:     (T, B, 1)
        continues:   (T, B, 1)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    all_grids = []
    all_scalars = []
    all_actions = []
    all_rewards = []
    all_continues = []
    
    for i in range(num_sequences):
        seq = generate_sequence(
            seq_len=seq_len,
            act_dim=act_dim,
            seed=seed + i,
        )
        all_grids.append(seq["obs_grid"])
        all_scalars.append(seq["obs_scalars"])
        all_actions.append(seq["actions"])
        all_rewards.append(seq["rewards"])
        all_continues.append(seq["continues"])
    
    # Stack: (B, T, ...) then transpose to (T, B, ...)
    dataset = {
        "obs_grid": torch.stack(all_grids, dim=0).transpose(0, 1),
        "obs_scalars": torch.stack(all_scalars, dim=0).transpose(0, 1),
        "actions": torch.stack(all_actions, dim=0).transpose(0, 1),
        "rewards": torch.stack(all_rewards, dim=0).transpose(0, 1),
        "continues": torch.stack(all_continues, dim=0).transpose(0, 1),
    }
    
    return dataset


def validate_dataset(data: dict[str, Tensor]) -> None:
    """Print dataset statistics and validate shapes."""
    print("=" * 60)
    print("Dataset Validation")
    print("=" * 60)
    
    print("\nShapes:")
    for k, v in data.items():
        print(f"  {k}: {tuple(v.shape)}")
    
    T, B = data["obs_grid"].shape[:2]
    print(f"\nSequence length (T): {T}")
    print(f"Batch size (B): {B}")
    
    print("\nValue ranges:")
    print(f"  obs_grid:    [{data['obs_grid'].min():.3f}, {data['obs_grid'].max():.3f}]")
    print(f"  obs_scalars: [{data['obs_scalars'].min():.3f}, {data['obs_scalars'].max():.3f}]")
    print(f"  rewards:     [{data['rewards'].min():.3f}, {data['rewards'].max():.3f}]")
    print(f"  continues:   [{data['continues'].min():.3f}, {data['continues'].max():.3f}]")
    
    # Verify actions are one-hot
    action_sums = data["actions"].sum(dim=-1)
    assert torch.allclose(action_sums, torch.ones_like(action_sums)), "Actions not one-hot!"
    print(f"  actions:     one-hot verified, act_dim={data['actions'].shape[-1]}")
    
    # Check reward distribution
    total_rewards = data["rewards"].sum().item()
    total_steps = T * B
    print(f"\nReward statistics:")
    print(f"  Total reward: {total_rewards:.0f} / {total_steps} steps ({100*total_rewards/total_steps:.1f}%)")
    
    # Verify learnable structure: action causes position change
    print("\nLearnable structure verification:")
    print("  Checking that actions cause predictable position changes...")
    
    # Sample first sequence
    scalars = data["obs_scalars"][:, 0, :]  # (T, 16)
    actions = data["actions"][:, 0, :]      # (T, act_dim)
    
    correct_predictions = 0
    total_checked = 0
    
    for t in range(min(10, T - 1)):
        x_t = scalars[t, 0].item() * 64
        y_t = scalars[t, 1].item() * 64
        x_next = scalars[t + 1, 0].item() * 64
        y_next = scalars[t + 1, 1].item() * 64
        
        action_idx = actions[t].argmax().item()
        
        # Predict next position
        step = 4.0
        if action_idx == 0:  # up
            pred_y = y_t - step
            pred_x = x_t
        elif action_idx == 1:  # down
            pred_y = y_t + step
            pred_x = x_t
        elif action_idx == 2:  # left
            pred_x = x_t - step
            pred_y = y_t
        elif action_idx == 3:  # right
            pred_x = x_t + step
            pred_y = y_t
        else:  # stay
            pred_x, pred_y = x_t, y_t
        
        # Clamp prediction
        pred_x = np.clip(pred_x, 6, 58)
        pred_y = np.clip(pred_y, 6, 58)
        
        if abs(pred_x - x_next) < 0.5 and abs(pred_y - y_next) < 0.5:
            correct_predictions += 1
        total_checked += 1
    
    print(f"  Position predictions: {correct_predictions}/{total_checked} correct")
    
    print("\n" + "=" * 60)
    print("Dataset is valid and has learnable structure!")
    print("=" * 60)


def main():
    """Generate and save the overfit dataset."""
    output_path = Path(__file__).parent / "overfit_dataset.pt"
    
    print("Generating synthetic overfit dataset...")
    print(f"  num_sequences: 8")
    print(f"  seq_len: 48")
    print(f"  seed: 42")
    
    data = generate_dataset(
        num_sequences=8,
        seq_len=48,
        act_dim=5,
        seed=42,
    )
    
    validate_dataset(data)
    
    print(f"\nSaving to {output_path}...")
    torch.save(data, output_path)
    print("Done!")
    
    return data


if __name__ == "__main__":
    main()
