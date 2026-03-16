"""
Shared rollout utilities used by both trainer.py and rollout.py.
Kept separate to avoid circular imports.
"""
import torch
import torch.nn.functional as F


# Movement direction vectors: class index → (dx, dy) in grid-cell units.
# Must stay in sync with _DIRECTION_VECS in model/heads.py.
DIRECTION_VECS = torch.tensor([
    [ 0,  0],   # 0: stay
    [ 0,  1],   # 1: N
    [ 1,  1],   # 2: NE
    [ 1,  0],   # 3: E
    [ 1, -1],   # 4: SE
    [ 0, -1],   # 5: S
    [-1, -1],   # 6: SW
    [-1,  0],   # 7: W
    [-1,  1],   # 8: NW
], dtype=torch.float32)  # (9, 2)


def straight_through_alive(alive_probs: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Hard binary mask for forward pass; gradients flow through soft probs (STE)."""
    hard = (alive_probs > threshold).float()
    return hard - alive_probs.detach() + alive_probs


def inject_alive_noise(alive: torch.Tensor, noise_rate: float) -> torch.Tensor:
    """
    Randomly flip a fraction of alive=1 bits to dead to make the model
    robust to its own survival-prediction errors during rollout.
    """
    if noise_rate <= 0.0:
        return alive
    flip = torch.rand_like(alive) < noise_rate
    flip = flip & (alive > 0.5)
    return alive * (~flip).float()


def apply_predictions_to_state(
    agents_t: torch.Tensor,    # (B, N_agents, 6)
    patches_t: torch.Tensor,   # (B, N_patches, 4)
    preds: dict,
    threshold: float = 0.5,
    grid_size: int = 40,
    soft: bool = False,
) -> tuple:
    """
    Apply model predictions to produce next-step state tensors.

    Position update uses the MovementHead's 9-class logits:
      soft=False  — argmax (deterministic; used during evaluation / rollout)
      soft=True   — expected direction via softmax (differentiable; used during
                    scheduled-sampling training so gradients flow through)

    Returns (agents_next, patches_next).
    """
    agents_next  = agents_t.clone()
    patches_next = patches_t.clone()

    # ── Position: movement direction → normalised (dx, dy) ───────────────
    movement_logits = preds["movement_logits"]          # (B, N_a, 9)
    device = movement_logits.device
    dir_vecs = DIRECTION_VECS.to(device)                # (9, 2)
    step_norm = 1.0 / grid_size                         # one cell in [0, 1] space

    if soft:
        # Differentiable expected direction (for scheduled-sampling training)
        probs  = F.softmax(movement_logits, dim=-1)     # (B, N_a, 9)
        dx_dy  = (probs.unsqueeze(-1) * dir_vecs).sum(-2)   # (B, N_a, 2)
    else:
        # Hard argmax for rollout / evaluation
        pred_class = movement_logits.argmax(dim=-1)     # (B, N_a)
        dx_dy = dir_vecs[pred_class]                    # (B, N_a, 2)

    agents_next[:, :, 1:3] = (agents_t[:, :, 1:3] + dx_dy * step_norm).clamp(0.0, 1.0)

    # ── Energy: continuous delta ──────────────────────────────────────────
    energy_delta = preds["energy_delta"]                # (B, N_a)
    agents_next[:, :, 3] = (agents_t[:, :, 3] + energy_delta).clamp(0.0, 1.0)

    # ── Alive mask — straight-through estimator ───────────────────────────
    alive_probs = preds["alive_probs"]                  # (B, N_a)
    new_alive   = straight_through_alive(alive_probs, threshold)
    agents_next[:, :, 5] = new_alive

    # ── Food patches — soft differentiable bin-centre weighted sum ────────
    food_logits = preds["food_logits"]                  # (B, N_p, n_bins)
    n_bins = food_logits.shape[-1]
    bin_centres = torch.linspace(0.0, 1.0, n_bins + 1, device=device)
    bin_centres = (bin_centres[:-1] + bin_centres[1:]) / 2.0
    food_probs  = torch.softmax(food_logits, dim=-1)    # (B, N_p, n_bins)
    food_soft   = (food_probs * bin_centres).sum(-1)    # (B, N_p)
    patches_next[:, :, 2] = food_soft.clamp(0.0, 1.0)

    return agents_next, patches_next
