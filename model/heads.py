"""
Output prediction heads and combined loss function.

Four heads:
  MSEHead     — predicts Δx, Δy, Δenergy for alive agents
  BCEHead     — predicts p(alive at t+1) per agent slot
  FoodCEHead  — predicts discretised food bin per patch (cross-entropy)
  AuxHead     — predicts global [n_prey, n_pred] counts

Combined loss:
  L = λ1·L_mse + λ2·L_bce + λ3·L_ce + λ4·L_aux
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ModelConfig, TrainConfig


# ── Individual Heads ──────────────────────────────────────────────────────────

class MSEHead(nn.Module):
    """
    Predicts (Δx, Δy, Δenergy) for each agent slot.
    Output is applied to alive agents only (masking happens in the loss).
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, 3)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden: (B, N_agents, d_model) → (B, N_agents, 3)"""
        return self.proj(hidden)


class BCEHead(nn.Module):
    """Predicts log-odds of being alive at t+1 per agent slot."""
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, 1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden: (B, N_agents, d_model) → (B, N_agents) logits"""
        return self.proj(hidden).squeeze(-1)  # (B, N_agents)


class FoodCEHead(nn.Module):
    """Predicts food bin class logits for each patch."""
    def __init__(self, d_model: int, n_bins: int = 5):
        super().__init__()
        self.n_bins = n_bins
        self.proj = nn.Linear(d_model, n_bins)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden: (B, N_patches, d_model) → (B, N_patches, n_bins)"""
        return self.proj(hidden)  # (B, N_patches, n_bins)


class AuxHead(nn.Module):
    """
    Predicts global [n_prey, n_pred] counts via differentiable sum over
    alive-probability-weighted token representations.

    The aux head never needs gradients to flow through hard alive masks
    (it uses soft probabilities from the BCE head).
    """
    def __init__(self, d_model: int):
        super().__init__()
        # Small MLP: pool alive-weighted tokens → scalar count predictions
        self.prey_head  = nn.Linear(d_model, 1)
        self.pred_head  = nn.Linear(d_model, 1)

    def forward(
        self,
        hidden: torch.Tensor,       # (B, N_agents, d_model)
        alive_probs: torch.Tensor,  # (B, N_agents) — sigmoid of BCE logits
        species: torch.Tensor,      # (B, N_agents) int64 — 0=prey, 1=pred
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        n_prey_hat : (B,) float — predicted prey count
        n_pred_hat : (B,) float — predicted predator count
        """
        is_prey = (species == 0).float()     # (B, N_agents)
        is_pred = (species == 1).float()

        # Weighted hidden representations
        prey_hidden = hidden * (alive_probs * is_prey).unsqueeze(-1)   # (B, N, D)
        pred_hidden = hidden * (alive_probs * is_pred).unsqueeze(-1)

        # Differentiable sum (like counting alive-weighted tokens)
        # Use a learned linear head on the pooled representation,
        # plus a direct count from the alive probabilities as a residual.
        prey_pool = prey_hidden.sum(dim=1)           # (B, D)
        pred_pool = pred_hidden.sum(dim=1)

        prey_count_direct = (alive_probs * is_prey).sum(dim=1)   # (B,)
        pred_count_direct = (alive_probs * is_pred).sum(dim=1)

        n_prey = self.prey_head(prey_pool).squeeze(-1) + prey_count_direct
        n_pred = self.pred_head(pred_pool).squeeze(-1) + pred_count_direct

        return n_prey, n_pred


# ── Full Output Module ────────────────────────────────────────────────────────

class WorldModelHeads(nn.Module):
    """
    Thin wrapper that owns all four heads and routes hidden states to them.

    The hidden state input is split:
      hidden[:, :n_agents, :]   → MSE + BCE + Aux heads
      hidden[:, n_agents:, :]   → FoodCE head (patches)
    """

    def __init__(self, model_cfg: ModelConfig):
        super().__init__()
        D = model_cfg.d_model
        self.mse_head   = MSEHead(D)
        self.bce_head   = BCEHead(D)
        self.food_head  = FoodCEHead(D, n_bins=model_cfg.n_food_bins)
        self.aux_head   = AuxHead(D)

    def forward(
        self,
        hidden: torch.Tensor,    # (B, N_total, d_model)
        species: torch.Tensor,   # (B, N_agents) int64 — agent species (not patches)
        n_agents: int,
    ) -> dict:
        agent_h = hidden[:, :n_agents, :]         # (B, N_agents, d_model)
        patch_h = hidden[:, n_agents:, :]         # (B, N_patches, d_model)

        delta_pred  = self.mse_head(agent_h)      # (B, N_agents, 3)
        alive_logit = self.bce_head(agent_h)      # (B, N_agents)
        food_logits = self.food_head(patch_h)     # (B, N_patches, n_bins)

        alive_probs = torch.sigmoid(alive_logit)  # (B, N_agents)
        n_prey, n_pred = self.aux_head(agent_h, alive_probs, species)

        return {
            "delta_pred":   delta_pred,    # (B, N_agents, 3)
            "alive_logit":  alive_logit,   # (B, N_agents)
            "alive_probs":  alive_probs,   # (B, N_agents)
            "food_logits":  food_logits,   # (B, N_patches, n_bins)
            "n_prey_hat":   n_prey,        # (B,)
            "n_pred_hat":   n_pred,        # (B,)
        }


# ── Combined Loss ─────────────────────────────────────────────────────────────

def compute_loss(
    preds: dict,
    agents_t: torch.Tensor,     # (B, N_agents, 6) — current step
    agents_t1: torch.Tensor,    # (B, N_agents, 6) — next step ground truth
    patches_t1: torch.Tensor,   # (B, N_patches, 4) — next step patches
    counts_t1: torch.Tensor,    # (B, 2) — [n_prey, n_pred] ground truth
    train_cfg: TrainConfig,
    model_cfg: ModelConfig,
) -> dict:
    """
    Compute all four loss terms and their weighted sum.

    Returns a dict with individual (raw/unweighted) losses and L_total.
    """
    device = agents_t.device

    # ── Unpack predictions ────────────────────────────────────────────────
    delta_pred  = preds["delta_pred"]    # (B, N_agents, 3)
    alive_logit = preds["alive_logit"]   # (B, N_agents)
    food_logits = preds["food_logits"]   # (B, N_patches, n_bins)
    n_prey_hat  = preds["n_prey_hat"]    # (B,)
    n_pred_hat  = preds["n_pred_hat"]    # (B,)

    # ── Unpack ground truth ───────────────────────────────────────────────
    # agents columns: 0=species, 1=x, 2=y, 3=energy, 4=age, 5=alive
    alive_mask_t  = agents_t[..., 5]    # (B, N_agents) alive at t
    alive_mask_t1 = agents_t1[..., 5]  # (B, N_agents) alive at t+1 (BCE target)

    pos_e_t  = agents_t[..., 1:4]      # (B, N_agents, 3)  x, y, energy at t
    pos_e_t1 = agents_t1[..., 1:4]     # ground truth at t+1
    delta_gt = pos_e_t1 - pos_e_t      # (B, N_agents, 3)  ground truth deltas

    food_t1 = patches_t1[..., 2]       # (B, N_patches) ground truth food level

    n_prey_gt = counts_t1[:, 0].float()   # (B,)
    n_pred_gt = counts_t1[:, 1].float()   # (B,)

    # ── L_mse (alive agents only) ─────────────────────────────────────────
    # Apply alive_mask_t: only agents alive at t contribute to delta loss
    m = alive_mask_t                             # (B, N_agents)
    denom_mse = m.sum().clamp(min=1.0)
    sq_err = ((delta_pred - delta_gt) ** 2).sum(-1)   # (B, N_agents)
    L_mse = (m * sq_err).sum() / denom_mse

    # ── L_bce (all agent slots, with dead-class upweighting) ─────────────
    # Upweight positive class (alive) = 1, dead class = dead_weight
    pos_weight = torch.ones(1, device=device) * train_cfg.bce_dead_weight
    bce_per_agent = F.binary_cross_entropy_with_logits(
        alive_logit, alive_mask_t1, pos_weight=pos_weight, reduction="none"
    )
    # Invert: we want to upweight DEAD (y=0 → positive class for dead)
    # More precisely: dead_weight applies when target=0. 
    # Standard BCE pos_weight upweights y=1. To upweight y=0 (dead),
    # we flip: compute weighted manually.
    dead_weight = train_cfg.bce_dead_weight
    bce_per_agent_manual = -(
        alive_mask_t1       * torch.log(torch.sigmoid(alive_logit)   + 1e-8)
      + (1 - alive_mask_t1) * torch.log(1 - torch.sigmoid(alive_logit) + 1e-8)
      * dead_weight
    )
    N = bce_per_agent_manual.shape[-1]
    L_bce = bce_per_agent_manual.mean()

    # ── L_ce (food patch bins) ────────────────────────────────────────────
    # Discretise ground truth food level
    bins = torch.linspace(0.0, 1.0, model_cfg.n_food_bins + 1, device=device)
    food_labels = torch.bucketize(food_t1.contiguous(), bins[1:-1].contiguous())  # (B, N_patches) int64
    B, N_p, n_bins = food_logits.shape
    L_ce = F.cross_entropy(
        food_logits.view(B * N_p, n_bins),
        food_labels.view(B * N_p),
        reduction="mean",
    )

    # ── L_aux (global population counts) ─────────────────────────────────
    L_aux = ((n_prey_hat - n_prey_gt) ** 2 + (n_pred_hat - n_pred_gt) ** 2).mean()

    # ── Weighted total ────────────────────────────────────────────────────
    L_total = (
        train_cfg.lambda_mse * L_mse
      + train_cfg.lambda_bce * L_bce
      + train_cfg.lambda_ce  * L_ce
      + train_cfg.lambda_aux * L_aux
    )

    return {
        "loss_total":   L_total,
        "loss_mse":     L_mse.detach(),
        "loss_bce":     L_bce.detach(),
        "loss_ce":      L_ce.detach(),
        "loss_aux":     L_aux.detach(),
        "loss_total_for_backward": L_total,  # aliased for clarity
    }
