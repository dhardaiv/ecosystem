"""
Transformer world model with three simultaneous attention masks.

Masks applied at every layer:
  1. Alive mask     — keys/values of dead tokens are zeroed out.
  2. Locality mask  — each token can only attend to entities within
                       a spatial radius r (optional).
  3. Cross-type mask — separate agent-to-agent and agent-to-patch
                       attention streams (optional).

The model processes a combined sequence of shape (B, N_total, d_model)
where N_total = n_max_agents (agents, padded) + n_patches (patches, full grid).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ModelConfig


# ── Masked Multi-Head Self-Attention ─────────────────────────────────────────

class MaskedMHSA(nn.Module):
    """
    Multi-head self-attention with additive attention bias.

    Additive bias (–∞ for forbidden pairs) is built outside this module and
    passed in as `attn_bias` of shape (B, n_heads, N, N) or (1, 1, N, N).
    This lets us combine alive, locality, and cross-type masks cleanly.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,              # (B, N, d_model)
        attn_bias: torch.Tensor,      # (B, n_heads, N, N) additive bias
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        out       : (B, N, d_model)
        attn_w    : (B, n_heads, N, N)  — saved for logging
        """
        B, N, D = x.shape
        H, Dh = self.n_heads, self.d_head

        Q = self.q_proj(x).view(B, N, H, Dh).transpose(1, 2)   # (B, H, N, Dh)
        K = self.k_proj(x).view(B, N, H, Dh).transpose(1, 2)
        V = self.v_proj(x).view(B, N, H, Dh).transpose(1, 2)

        scale = math.sqrt(Dh)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale    # (B, H, N, N)
        scores = scores + attn_bias                               # additive mask bias

        attn_w = F.softmax(scores, dim=-1)                        # (B, H, N, N)
        attn_w = self.dropout(attn_w)

        out = torch.matmul(attn_w, V)                            # (B, H, N, Dh)
        out = out.transpose(1, 2).contiguous().view(B, N, D)      # (B, N, D)
        out = self.out_proj(out)
        return out, attn_w


# ── Transformer Block ─────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = MaskedMHSA(d_model, n_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,         # (B, N, D)
        attn_bias: torch.Tensor, # (B, H, N, N)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Pre-norm convention (more stable at depth > 4)
        attn_out, attn_w = self.attn(self.norm1(x), attn_bias)
        x = x + self.drop(attn_out)
        x = x + self.drop(self.ff(self.norm2(x)))
        return x, attn_w


# ── Attention Mask Builder ────────────────────────────────────────────────────

NEG_INF = -1e9


def build_attention_bias(
    alive: torch.Tensor,           # (B, N_total) float32
    xy: torch.Tensor,              # (B, N_total, 2) float32 — normalised coords
    species: torch.Tensor,         # (B, N_total) int64
    n_agents: int,                 # number of agent slots (not patch slots)
    n_heads: int,
    cfg: ModelConfig,
) -> torch.Tensor:
    """
    Builds a combined additive attention bias (B, n_heads, N, N).
    Dead/padding tokens contribute –∞ as column bias so they can never
    be attended to (keys), but we still let their query attend to others
    (they'll be zeroed in the output via alive mask in loss computation).
    """
    B, N = alive.shape
    device = alive.device

    # Start with zeros; we will add –∞ where attention is forbidden
    bias = torch.zeros(B, n_heads, N, N, device=device)

    # ── Mask 1: Alive mask on keys ────────────────────────────────────────
    # dead_col[b, :, j] = NEG_INF when alive[b, j] == 0
    dead_col = (1.0 - alive).unsqueeze(1).unsqueeze(2) * NEG_INF  # (B, 1, 1, N)
    bias = bias + dead_col                                          # (B, H, N, N)

    # ── Mask 2: Locality mask ─────────────────────────────────────────────
    if cfg.use_locality_mask:
        # Pairwise squared Euclidean distance in normalised coords
        xy_i = xy.unsqueeze(2)                        # (B, N, 1, 2)
        xy_j = xy.unsqueeze(1)                        # (B, 1, N, 2)
        dist2 = ((xy_i - xy_j) ** 2).sum(-1)          # (B, N, N)
        r2 = cfg.locality_radius ** 2
        far_mask = (dist2 > r2).float() * NEG_INF     # (B, N, N)
        bias = bias + far_mask.unsqueeze(1)            # (B, 1, N, N) broadcast H

    # ── Mask 3: Cross-type mask ───────────────────────────────────────────
    # Agents only attend to agents; patches only attend to patches.
    if cfg.use_cross_type_mask:
        # species < 2 → agent, species == 2 → patch
        is_agent = (species < 2).float()               # (B, N)
        is_patch = (species == 2).float()              # (B, N)
        # Agent querying patch key → forbidden
        ct_mask_val = (
            is_agent.unsqueeze(2) * is_patch.unsqueeze(1) +   # agent→patch
            is_patch.unsqueeze(2) * is_agent.unsqueeze(1)     # patch→agent
        ) * NEG_INF                                    # (B, N, N)
        bias = bias + ct_mask_val.unsqueeze(1)         # (B, 1, N, N)

    return bias


# ── Full Transformer World Model ──────────────────────────────────────────────

class EcosystemTransformer(nn.Module):
    """
    Full transformer world model.

    Accepts pre-encoded tokens (output of EntityEncoder) and passes them
    through L transformer blocks with combined attention masking.

    Returns
    -------
    hidden   : (B, N_total, d_model)   — final token representations
    attn_ws  : list of (B, H, N, N)    — per-layer attention weights
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.final_norm = nn.LayerNorm(cfg.d_model)

    def forward(
        self,
        tokens: torch.Tensor,      # (B, N_total, d_model) — from EntityEncoder
        alive: torch.Tensor,       # (B, N_total) float32
        xy: torch.Tensor,          # (B, N_total, 2) — normalised coords
        species: torch.Tensor,     # (B, N_total) int64
        n_agents: int,             # agent slot count (for cross-type mask)
    ) -> tuple[torch.Tensor, list]:
        attn_bias = build_attention_bias(
            alive, xy, species, n_agents, self.cfg.n_heads, self.cfg
        )
        x = tokens
        attn_ws = []
        for block in self.blocks:
            x, attn_w = block(x, attn_bias)
            attn_ws.append(attn_w)
        x = self.final_norm(x)
        return x, attn_ws
