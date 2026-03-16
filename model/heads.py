"""
Output prediction heads and combined loss function.

Five heads:
  MovementHead — predicts movement direction as 9-class classification (replaces Δx, Δy regression)
  EnergyHead   — predicts Δenergy for alive agents (continuous MSE + variance penalty)
  BCEHead      — predicts p(alive at t+1) per agent slot
  FoodCEHead   — predicts discretised food bin per patch (cross-entropy)
  AuxHead      — predicts global [n_prey, n_pred] counts

Movement directions:
  0: stay (0, 0)   1: N  (0,+1)   2: NE (+1,+1)   3: E  (+1, 0)   4: SE (+1,-1)
  5: S   (0,-1)   6: SW (-1,-1)   7: W  (-1, 0)   8: NW (-1,+1)

Combined loss:
  L = λ_move·L_movement + λ_energy·L_energy + λ_bce·L_bce + λ_ce·L_ce + λ_aux·L_aux
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ModelConfig, TrainConfig


# ── Movement encoding constants ───────────────────────────────────────────────

# Maps (dx+1, dy+1) → movement class index, where dx,dy ∈ {-1, 0, +1}
_DIRECTION_TABLE = torch.tensor([
    [6, 7, 8],   # dx = -1: SW, W, NW
    [5, 0, 1],   # dx =  0:  S, stay, N
    [4, 3, 2],   # dx = +1: SE, E, NE
], dtype=torch.long)  # (3, 3)

# Maps class index → (dx, dy) offset in grid cell units
_DIRECTION_VECS = torch.tensor([
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


def encode_movement(
    delta_x_norm: torch.Tensor,   # (B, N_agents) normalised Δx
    delta_y_norm: torch.Tensor,   # (B, N_agents) normalised Δy
    grid_size: int,
) -> torch.Tensor:
    """
    Convert normalised position deltas to movement class indices (0–8).

    Handles toroidal wrap-around: an agent stepping off one edge may produce a
    large apparent delta (e.g. 0.975 on a 40-cell grid for a 1-cell westward
    wrap). We correct to the shorter-path delta before rounding.
    """
    dx = delta_x_norm * grid_size   # in cell units
    dy = delta_y_norm * grid_size

    # Toroidal correction — pick the path with |delta| ≤ grid_size/2
    half = grid_size / 2.0
    dx = torch.where(dx.abs() > half, dx - dx.sign() * grid_size, dx)
    dy = torch.where(dy.abs() > half, dy - dy.sign() * grid_size, dy)

    dx = dx.round().long().clamp(-1, 1)   # (B, N_agents) ∈ {-1, 0, +1}
    dy = dy.round().long().clamp(-1, 1)

    table = _DIRECTION_TABLE.to(delta_x_norm.device)
    return table[dx + 1, dy + 1]   # (B, N_agents) int64 ∈ {0..8}


# ── Individual Heads ──────────────────────────────────────────────────────────

class MovementHead(nn.Module):
    """
    Predicts movement direction as a 9-class classification over cardinal + diagonal
    directions and stay. Cross-entropy loss avoids the regression-to-mean collapse
    that continuous (Δx, Δy) MSE produces when movement is stochastic.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.classifier = nn.Linear(d_model, 9)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden: (B, N_agents, d_model) → (B, N_agents, 9) logits"""
        return self.classifier(hidden)


class EnergyHead(nn.Module):
    """
    Predicts Δenergy (continuous scalar) for each agent slot.
    MSE loss is appropriate here because energy changes are bounded and
    the regression target is less multimodal than position.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, 1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden: (B, N_agents, d_model) → (B, N_agents) scalar Δenergy"""
        return self.proj(hidden).squeeze(-1)


class BCEHead(nn.Module):
    """Predicts log-odds of being alive at t+1 per agent slot."""
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, 1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden: (B, N_agents, d_model) → (B, N_agents) logits"""
        return self.proj(hidden).squeeze(-1)


class FoodCEHead(nn.Module):
    """Predicts food bin class logits for each patch."""
    def __init__(self, d_model: int, n_bins: int = 5):
        super().__init__()
        self.n_bins = n_bins
        self.proj = nn.Linear(d_model, n_bins)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden: (B, N_patches, d_model) → (B, N_patches, n_bins)"""
        return self.proj(hidden)


class AuxHead(nn.Module):
    """
    Predicts global [n_prey, n_pred] counts as normalised fractions in [0, 1].

    Output is sigmoid-constrained so predictions are always in a valid range
    and the supervision signal is commensurate with other loss terms (~[0,1]).
    Mean pooling (not sum) over alive-weighted tokens decouples the
    representation from raw population size.
    """
    def __init__(self, d_model: int, n_max: int):
        super().__init__()
        self.n_max = n_max
        self.prey_head = nn.Linear(d_model, 1)
        self.pred_head = nn.Linear(d_model, 1)

    def forward(
        self,
        hidden: torch.Tensor,       # (B, N_agents, d_model)
        alive_probs: torch.Tensor,  # (B, N_agents)
        species: torch.Tensor,      # (B, N_agents) int64 — 0=prey, 1=pred
    ) -> tuple[torch.Tensor, torch.Tensor]:
        is_prey = (species == 0).float()
        is_pred = (species == 1).float()

        prey_w = alive_probs * is_prey
        pred_w = alive_probs * is_pred

        prey_denom = prey_w.sum(dim=1, keepdim=True).clamp(min=1.0)
        pred_denom = pred_w.sum(dim=1, keepdim=True).clamp(min=1.0)

        prey_repr = (hidden * prey_w.unsqueeze(-1)).sum(dim=1) / prey_denom
        pred_repr = (hidden * pred_w.unsqueeze(-1)).sum(dim=1) / pred_denom

        n_hat_prey_norm = torch.sigmoid(self.prey_head(prey_repr)).squeeze(-1)
        n_hat_pred_norm = torch.sigmoid(self.pred_head(pred_repr)).squeeze(-1)

        return n_hat_prey_norm, n_hat_pred_norm

    def to_counts(self, n_hat_norm: torch.Tensor) -> torch.Tensor:
        return (n_hat_norm * self.n_max).round()


# ── Full Output Module ────────────────────────────────────────────────────────

class WorldModelHeads(nn.Module):
    """
    Thin wrapper that owns all five heads and routes hidden states to them.

    The hidden state input is split:
      hidden[:, :n_agents, :]   → Movement + Energy + BCE + Aux heads
      hidden[:, n_agents:, :]   → FoodCE head (patches)
    """

    def __init__(self, model_cfg: ModelConfig):
        super().__init__()
        D = model_cfg.d_model
        self.movement_head = MovementHead(D)
        self.energy_head   = EnergyHead(D)
        self.bce_head      = BCEHead(D)
        self.food_head     = FoodCEHead(D, n_bins=model_cfg.n_food_bins)
        self.aux_head      = AuxHead(D, n_max=model_cfg.n_max_agents)

    def forward(
        self,
        hidden: torch.Tensor,    # (B, N_total, d_model)
        species: torch.Tensor,   # (B, N_agents) int64 — agent species (not patches)
        n_agents: int,
    ) -> dict:
        agent_h = hidden[:, :n_agents, :]    # (B, N_agents, d_model)
        patch_h = hidden[:, n_agents:, :]    # (B, N_patches, d_model)

        movement_logits = self.movement_head(agent_h)   # (B, N_agents, 9)
        energy_delta    = self.energy_head(agent_h)     # (B, N_agents)
        alive_logit     = self.bce_head(agent_h)        # (B, N_agents)
        food_logits     = self.food_head(patch_h)       # (B, N_patches, n_bins)

        alive_probs = torch.sigmoid(alive_logit)
        n_prey, n_pred = self.aux_head(agent_h, alive_probs, species)

        return {
            "movement_logits": movement_logits,   # (B, N_agents, 9)
            "energy_delta":    energy_delta,      # (B, N_agents)
            "alive_logit":     alive_logit,       # (B, N_agents)
            "alive_probs":     alive_probs,       # (B, N_agents)
            "food_logits":     food_logits,       # (B, N_patches, n_bins)
            "n_prey_hat":      n_prey,            # (B,)
            "n_pred_hat":      n_pred,            # (B,)
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
    Compute all loss terms and their weighted sum.

    Returns a dict with individual (raw/unweighted) losses, movement diagnostics,
    and L_total.
    """
    device = agents_t.device

    # ── Unpack predictions ────────────────────────────────────────────────
    movement_logits = preds["movement_logits"]   # (B, N_agents, 9)
    energy_delta    = preds["energy_delta"]      # (B, N_agents)
    alive_logit     = preds["alive_logit"]       # (B, N_agents)
    food_logits     = preds["food_logits"]       # (B, N_patches, n_bins)
    n_prey_hat      = preds["n_prey_hat"]        # (B,)
    n_pred_hat      = preds["n_pred_hat"]        # (B,)

    # ── Unpack ground truth ───────────────────────────────────────────────
    # agents columns: 0=species, 1=x, 2=y, 3=energy, 4=age, 5=alive
    alive_mask_t  = agents_t[..., 5]    # (B, N_agents)
    alive_mask_t1 = agents_t1[..., 5]  # (B, N_agents)

    pos_e_t  = agents_t[..., 1:4]      # (B, N_agents, 3) — x, y, energy
    pos_e_t1 = agents_t1[..., 1:4]
    delta_gt = pos_e_t1 - pos_e_t      # (B, N_agents, 3)

    food_t1 = patches_t1[..., 2]       # (B, N_patches)

    N_max = float(model_cfg.n_max_agents)
    n_prey_gt = counts_t1[:, 0].float() / N_max
    n_pred_gt = counts_t1[:, 1].float() / N_max

    # Agents alive at BOTH t and t+1: dying agents have garbage delta targets.
    m = alive_mask_t * alive_mask_t1             # (B, N_agents) float
    alive_bool = m.bool()                        # (B, N_agents) bool

    # ── L_movement (cross-entropy over 9 direction classes) ───────────────
    # Ground truth movement class from normalised position deltas
    move_class_gt = encode_movement(
        delta_gt[..., 0], delta_gt[..., 1], model_cfg.grid_size
    )   # (B, N_agents) int64

    move_logits_alive = movement_logits[alive_bool]   # (N_alive, 9)
    move_class_alive  = move_class_gt[alive_bool]     # (N_alive,)

    if move_logits_alive.shape[0] > 0:
        L_movement = F.cross_entropy(move_logits_alive, move_class_alive, reduction="mean")
    else:
        L_movement = movement_logits.new_zeros(())

    # ── L_energy (MSE + variance penalty) ────────────────────────────────
    delta_energy_gt = delta_gt[..., 2]            # (B, N_agents)
    energy_alive    = energy_delta[alive_bool]    # (N_alive,)
    energy_gt_alive = delta_energy_gt[alive_bool]

    if energy_alive.shape[0] > 1:
        L_energy_mse = F.mse_loss(energy_alive, energy_gt_alive)
        # Variance penalty: kick in only when predicted Δenergy has near-zero
        # spread across the batch — catches the zero-delta attractor.
        energy_var = energy_alive.var()
        L_energy_var = F.relu(
            torch.tensor(train_cfg.energy_var_threshold, device=device) - energy_var
        )
        L_energy = L_energy_mse + train_cfg.energy_var_reg * L_energy_var
    else:
        L_energy = energy_delta.new_zeros(())

    # ── L_bce (all agent slots, with dead-class upweighting) ─────────────
    dead_weight = train_cfg.bce_dead_weight
    bce_per_agent = -(
        alive_mask_t1       * torch.log(torch.sigmoid(alive_logit)     + 1e-8)
      + (1 - alive_mask_t1) * torch.log(1 - torch.sigmoid(alive_logit) + 1e-8)
      * dead_weight
    )
    L_bce = bce_per_agent.mean()

    # ── L_ce (food patch bins) ────────────────────────────────────────────
    bins = torch.linspace(0.0, 1.0, model_cfg.n_food_bins + 1, device=device)
    food_labels = torch.bucketize(food_t1.contiguous(), bins[1:-1].contiguous())
    B, N_p, n_bins = food_logits.shape
    L_ce = F.cross_entropy(
        food_logits.view(B * N_p, n_bins),
        food_labels.view(B * N_p),
        reduction="mean",
    )

    # ── L_aux (global population counts) ─────────────────────────────────
    if train_cfg.lambda_aux > 0:
        L_aux = (
            F.mse_loss(n_prey_hat, n_prey_gt)
            + F.mse_loss(n_pred_hat, n_pred_gt)
        )
    else:
        L_aux = torch.zeros((), device=device)

    # ── Weighted total ────────────────────────────────────────────────────
    L_total = (
        train_cfg.lambda_move   * L_movement
      + train_cfg.lambda_energy * L_energy
      + train_cfg.lambda_bce    * L_bce
      + train_cfg.lambda_ce     * L_ce
      + train_cfg.lambda_aux    * L_aux
    )

    # ── Movement diagnostics (no grad) ────────────────────────────────────
    with torch.no_grad():
        if move_logits_alive.shape[0] > 0:
            move_probs_mean = F.softmax(move_logits_alive, dim=-1).mean(dim=0).clamp(min=1e-8)
            movement_entropy = -(move_probs_mean * move_probs_mean.log()).sum().item()
            frac_stay = (move_logits_alive.argmax(dim=-1) == 0).float().mean().item()
        else:
            movement_entropy = 0.0
            frac_stay = 0.0

    return {
        "loss_total":            L_total,
        "loss_movement":         L_movement.detach(),
        "loss_energy":           L_energy.detach(),
        "loss_bce":              L_bce.detach(),
        "loss_ce":               L_ce.detach(),
        "loss_aux":              L_aux.detach(),
        "movement_entropy":      movement_entropy,    # float — bits
        "frac_stay_predicted":   frac_stay,           # float — fraction
        "loss_total_for_backward": L_total,
    }
