"""
Shared rollout utilities used by both trainer.py and rollout.py.
Kept separate to avoid circular imports.
"""
import torch


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
) -> tuple:
    """
    Apply model predictions to produce next-step state tensors.

    Returns (agents_next, patches_next).
    """
    agents_next  = agents_t.clone()
    patches_next = patches_t.clone()

    # Apply deltas (cols 1=x, 2=y, 3=energy)
    delta = preds["delta_pred"]                         # (B, N_a, 3)
    agents_next[:, :, 1:4] = (agents_t[:, :, 1:4] + delta).clamp(0.0, 1.0)

    # Alive mask — straight-through estimator
    alive_probs = preds["alive_probs"]                  # (B, N_a)
    new_alive   = straight_through_alive(alive_probs, threshold)
    agents_next[:, :, 5] = new_alive

    # Food patches — soft differentiable bin-centre weighted sum
    food_logits = preds["food_logits"]                  # (B, N_p, n_bins)
    n_bins = food_logits.shape[-1]
    bin_centres = torch.linspace(0.0, 1.0, n_bins + 1, device=food_logits.device)
    bin_centres = (bin_centres[:-1] + bin_centres[1:]) / 2.0   # (n_bins,)
    food_probs   = torch.softmax(food_logits, dim=-1)           # (B, N_p, n_bins)
    food_soft    = (food_probs * bin_centres).sum(-1)           # (B, N_p)
    patches_next[:, :, 2] = food_soft.clamp(0.0, 1.0)

    return agents_next, patches_next
