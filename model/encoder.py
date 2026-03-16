"""
Entity encoder for the ecosystem world model.

Produces one token per entity (agent or food patch) of dimension d_model.

Pipeline for a single entity:
  1. species_embed(species_id)           → (d_species_embed,)
  2. MLP([x, y, energy, age])            → (d_model - d_species_embed,)
  3. token = LayerNorm(species_emb + continuous_proj)  → (d_model,)
  4. token += sinusoidal_spatial_enc(x, y)             → (d_model,)

For patches: x, y, food_level, 0.0 (age placeholder).
"""
import math
import torch
import torch.nn as nn
from config import ModelConfig


# ── Sinusoidal Spatial Encoding ──────────────────────────────────────────────

class SpatialEncoding(nn.Module):
    """
    2D sinusoidal encoding over normalised (x, y) coordinates.

    Produces a vector of length `d_enc` (must be divisible by 4):
      [sin(x/f_0), cos(x/f_0), sin(y/f_0), cos(y/f_0),
       sin(x/f_1), cos(x/f_1), sin(y/f_1), cos(y/f_1), ...]
    """

    def __init__(self, d_enc: int = 64):
        super().__init__()
        assert d_enc % 4 == 0, "d_enc must be divisible by 4"
        self.d_enc = d_enc
        n_freq = d_enc // 4
        # Log-spaced frequencies exactly as in the original transformer paper
        freqs = torch.pow(10000.0, -torch.arange(n_freq, dtype=torch.float32) / n_freq)
        self.register_buffer("freqs", freqs)  # (n_freq,)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        """
        xy : (..., 2)  normalised x and y in [0, 1]
        Returns : (..., d_enc)
        """
        x = xy[..., 0:1]  # (..., 1)
        y = xy[..., 1:2]
        freqs = self.freqs  # (n_freq,)

        x_enc = x * freqs                         # (..., n_freq)
        y_enc = y * freqs
        enc = torch.cat([
            torch.sin(x_enc), torch.cos(x_enc),
            torch.sin(y_enc), torch.cos(y_enc),
        ], dim=-1)                                # (..., d_enc)
        return enc


# ── Entity Encoder ────────────────────────────────────────────────────────────

class EntityEncoder(nn.Module):
    """
    Encodes a padded sequence of entity tokens (agents + patches) into
    d_model-dimensional representations.

    Inputs
    ------
    species   : (B, N) int64   — species ids (0=prey, 1=pred, 2=patch)
    cont      : (B, N, 4) float32 — [x, y, energy/food, age]
    alive     : (B, N) float32 — alive mask (for patches, always 1)

    Output
    ------
    tokens : (B, N, d_model) float32 — entity tokens with spatial encoding
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.species_embed = nn.Embedding(cfg.n_species, cfg.d_species_embed)

        # MLP: continuous attrs → (d_model - d_species_embed)
        d_cont_out = cfg.d_model - cfg.d_species_embed
        self.cont_mlp = nn.Sequential(
            nn.Linear(cfg.d_continuous, cfg.n_mlp_hidden),
            nn.GELU(),
            nn.Linear(cfg.n_mlp_hidden, d_cont_out),
        )

        self.layer_norm = nn.LayerNorm(cfg.d_model)

        if cfg.use_spatial_encoding:
            d_enc = cfg.spatial_encoding_dim
            self.spatial_enc = SpatialEncoding(d_enc=d_enc)
            # Project spatial encoding up to d_model
            self.spatial_proj = nn.Linear(d_enc, cfg.d_model, bias=False)
        else:
            self.spatial_enc = None
            self.spatial_proj = None

    def forward(
        self,
        species: torch.Tensor,    # (B, N)  int64
        cont: torch.Tensor,       # (B, N, 4) float32
        alive: torch.Tensor,      # (B, N)  float32
    ) -> torch.Tensor:
        """Returns tokens : (B, N, d_model)."""
        # Species embedding
        sp_emb = self.species_embed(species)           # (B, N, d_species_embed)

        # Continuous MLP
        cont_out = self.cont_mlp(cont)                 # (B, N, d_model - d_species_embed)

        # Concatenate and normalise
        tokens = torch.cat([sp_emb, cont_out], dim=-1) # (B, N, d_model)
        tokens = self.layer_norm(tokens)

        # Zero out dead/padding slots so they don't pollute layer-norm stats
        tokens = tokens * alive.unsqueeze(-1)          # (B, N, d_model)

        # Add spatial encoding
        if self.spatial_enc is not None:
            xy = cont[..., :2]                         # (B, N, 2)
            sp = self.spatial_enc(xy)                  # (B, N, d_enc)
            sp = self.spatial_proj(sp)                 # (B, N, d_model)
            tokens = tokens + sp * alive.unsqueeze(-1)

        return tokens                                   # (B, N, d_model)
