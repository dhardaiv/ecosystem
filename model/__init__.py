"""
Full assembled world model: EntityEncoder → EcosystemTransformer → WorldModelHeads.
"""
import torch
import torch.nn as nn
from config import ModelConfig, TrainConfig, SIM

from model.encoder import EntityEncoder
from model.transformer import EcosystemTransformer
from model.heads import WorldModelHeads, compute_loss


class EcosystemWorldModel(nn.Module):
    """
    End-to-end world model.

    forward() accepts the current-step state tensors and returns predictions
    for the next step.

    Input layout
    ------------
    agents_t  : (B, N_agents, 6) — species(int cast to float), x, y, energy, age, alive
    patches_t : (B, N_patches, 4) — x, y, food, alive

    The combined token sequence is  [agent_tokens | patch_tokens]
    of total length N_total = N_agents + N_patches.
    """

    def __init__(self, model_cfg: ModelConfig):
        super().__init__()
        self.cfg = model_cfg
        self.encoder = EntityEncoder(model_cfg)
        self.transformer = EcosystemTransformer(model_cfg)
        self.heads = WorldModelHeads(model_cfg)

    def forward(
        self,
        agents_t: torch.Tensor,   # (B, N_agents, 6)
        patches_t: torch.Tensor,  # (B, N_patches, 4)
    ) -> dict:
        B = agents_t.shape[0]
        N_a = agents_t.shape[1]
        N_p = patches_t.shape[1]

        # ── Parse agent fields ─────────────────────────────────────────────
        # columns: 0=species, 1=x, 2=y, 3=energy, 4=age, 5=alive
        agent_species = agents_t[:, :, 0].long()         # (B, N_a)
        agent_cont    = agents_t[:, :, 1:5]              # (B, N_a, 4) x,y,energy,age
        agent_alive   = agents_t[:, :, 5]                # (B, N_a)
        agent_xy      = agents_t[:, :, 1:3]              # (B, N_a, 2)

        # ── Parse patch fields ────────────────────────────────────────────
        patch_xy      = patches_t[:, :, :2]              # (B, N_p, 2)
        patch_cont    = torch.cat([
            patches_t[:, :, :2],                         # x, y
            patches_t[:, :, 2:3],                        # food level
            torch.zeros(B, N_p, 1, device=patches_t.device),  # age placeholder
        ], dim=-1)                                        # (B, N_p, 4)
        patch_species = torch.full((B, N_p), 2, dtype=torch.long, device=patches_t.device)
        patch_alive   = patches_t[:, :, 3]               # (B, N_p)  always 1

        # ── Encode entities ───────────────────────────────────────────────
        agent_tokens = self.encoder(agent_species, agent_cont, agent_alive)
        patch_tokens = self.encoder(patch_species, patch_cont, patch_alive)

        # ── Concatenate into full sequence ────────────────────────────────
        tokens  = torch.cat([agent_tokens, patch_tokens], dim=1)  # (B, N_total, D)
        alive   = torch.cat([agent_alive, patch_alive], dim=1)    # (B, N_total)
        xy      = torch.cat([agent_xy, patch_xy], dim=1)          # (B, N_total, 2)
        species = torch.cat([agent_species, patch_species], dim=1) # (B, N_total)

        # ── Transformer ───────────────────────────────────────────────────
        hidden, attn_ws = self.transformer(tokens, alive, xy, species, n_agents=N_a)

        # ── Prediction heads ──────────────────────────────────────────────
        preds = self.heads(hidden, agent_species, n_agents=N_a)
        preds["attn_ws"] = attn_ws  # carry attention weights for logging

        return preds
