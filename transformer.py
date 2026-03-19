from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_chebyshev_mask(
    positions: torch.Tensor, alive_mask: torch.Tensor, R: int, N_max: int
) -> torch.Tensor:
    """
    Returns additive mask [B,N,N+1]:
      0.0 for allowed attention, -1e9 for blocked.
    Last column corresponds to global patch token and is always 0.0.
    """
    B, N, _ = positions.shape
    if N != N_max:
        raise ValueError(f"Expected N={N_max}, got {N}")

    pos_i = positions.unsqueeze(2)  # [B,N,1,2]
    pos_j = positions.unsqueeze(1)  # [B,1,N,2]
    delta = (pos_i - pos_j).abs()
    cheb = torch.maximum(delta[..., 0], delta[..., 1])  # [B,N,N]
    within = cheb <= R

    alive_j = alive_mask.squeeze(-1).bool().unsqueeze(1).expand(B, N, N)
    not_self = ~torch.eye(N, device=positions.device, dtype=torch.bool).unsqueeze(0)
    allowed = within & alive_j & not_self
    attn_mask_agents = torch.where(
        allowed,
        torch.zeros((B, N, N), device=positions.device, dtype=torch.float32),
        torch.full((B, N, N), -1e9, device=positions.device, dtype=torch.float32),
    )
    patch_col = torch.zeros((B, N, 1), device=positions.device, dtype=torch.float32)
    return torch.cat([attn_mask_agents, patch_col], dim=-1)


class LocalAttention(nn.Module):
    def __init__(self, d_m: int, H_heads: int, d_k: int, R: int, H_grid: int, W_grid: int):
        super().__init__()
        self.d_m = d_m
        self.H_heads = H_heads
        self.d_k = d_k
        self.R = R
        self.H_grid = H_grid
        self.W_grid = W_grid
        self.q_proj = nn.Linear(d_m, H_heads * d_k)
        self.k_proj = nn.Linear(d_m, H_heads * d_k)
        self.v_proj = nn.Linear(d_m, H_heads * d_k)
        self.out_proj = nn.Linear(H_heads * d_k, d_m)
        self.rel_bias = nn.Parameter(torch.zeros(H_heads, 2 * R + 1, 2 * R + 1))
        nn.init.normal_(self.rel_bias, mean=0.0, std=0.02)

    def _relative_bias(self, positions: torch.Tensor) -> torch.Tensor:
        B, N, _ = positions.shape
        pi = positions.unsqueeze(2)  # [B,N,1,2]
        pj = positions.unsqueeze(1)  # [B,1,N,2]
        delta = pj - pi  # j - i
        dx = delta[..., 0] + self.R
        dy = delta[..., 1] + self.R
        valid = (
            (dx >= 0)
            & (dx <= 2 * self.R)
            & (dy >= 0)
            & (dy <= 2 * self.R)
        )
        dx = dx.clamp(0, 2 * self.R).long()
        dy = dy.clamp(0, 2 * self.R).long()
        flat = dx * (2 * self.R + 1) + dy  # [B,N,N]
        table = self.rel_bias.reshape(self.H_heads, -1)  # [Hh,(2R+1)^2]
        idx = flat.unsqueeze(1).expand(B, self.H_heads, N, N)
        gathered = torch.gather(
            table.unsqueeze(0).expand(B, self.H_heads, -1),
            2,
            idx.reshape(B, self.H_heads, -1),
        ).reshape(B, self.H_heads, N, N)
        return gathered * valid.unsqueeze(1).float()

    def forward(
        self,
        z: torch.Tensor,
        patch_token: torch.Tensor,
        positions: torch.Tensor,
        alive_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, N, _ = z.shape
        alive = alive_mask.float()

        z_agents = z * alive
        tokens = torch.cat([z_agents, patch_token], dim=1)  # [B,N+1,d_m]

        q = self.q_proj(z_agents).reshape(B, N, self.H_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(tokens).reshape(B, N + 1, self.H_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(tokens).reshape(B, N + 1, self.H_heads, self.d_k).transpose(1, 2)

        q = q * alive.unsqueeze(1)  # dead queries are zeroed

        logits = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k**0.5)  # [B,H,N,N+1]
        add_mask = get_chebyshev_mask(positions, alive_mask, self.R, N).unsqueeze(1)
        logits = logits + add_mask

        rel = self._relative_bias(positions)  # [B,H,N,N]
        logits[:, :, :, :N] = logits[:, :, :, :N] + rel

        dead_q = ~alive.squeeze(-1).bool()
        logits = logits.masked_fill(dead_q.unsqueeze(1).unsqueeze(-1), -1e9)

        attn = torch.softmax(logits, dim=-1)
        out = torch.matmul(attn, v)  # [B,H,N,d_k]
        out = out.transpose(1, 2).reshape(B, N, self.H_heads * self.d_k)
        out = self.out_proj(out)
        out = out * alive
        return out


class TemporalCrossAttention(nn.Module):
    def __init__(self, d_m: int, H_heads: int, d_k: int, T: int):
        super().__init__()
        self.d_m = d_m
        self.H_heads = H_heads
        self.d_k = d_k
        self.T = T
        self.q_proj = nn.Linear(d_m, H_heads * d_k)
        self.k_proj = nn.Linear(d_m, H_heads * d_k)
        self.v_proj = nn.Linear(d_m, H_heads * d_k)
        self.out_proj = nn.Linear(H_heads * d_k, d_m)
        self.temporal_bias = nn.Parameter(torch.zeros(T))
        nn.init.normal_(self.temporal_bias, mean=0.0, std=0.02)

    def forward(self, z: torch.Tensor, context: torch.Tensor, alive_mask: torch.Tensor) -> torch.Tensor:
        """
        z: [B,N,d_m]
        context: [B,T,N,d_m]
        """
        B, N, _ = z.shape
        alive = alive_mask.float()
        q = self.q_proj(z * alive).reshape(B * N, self.H_heads, 1, self.d_k)

        hist = context.transpose(1, 2).reshape(B * N, self.T, self.d_m)  # [B*N,T,d_m]
        k = self.k_proj(hist).reshape(B * N, self.T, self.H_heads, self.d_k).permute(0, 2, 1, 3)
        v = self.v_proj(hist).reshape(B * N, self.T, self.H_heads, self.d_k).permute(0, 2, 1, 3)

        logits = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k**0.5)  # [B*N,H,1,T]
        logits = logits + self.temporal_bias.view(1, 1, 1, self.T)
        attn = torch.softmax(logits, dim=-1)
        out = torch.matmul(attn, v).squeeze(-2)  # [B*N,H,d_k]
        out = out.transpose(1, 2).reshape(B, N, self.H_heads * self.d_k)
        out = self.out_proj(out)
        out = out * alive
        return out


class TransformerLayer(nn.Module):
    def __init__(
        self,
        d_m: int,
        H_heads: int,
        d_k: int,
        R: int,
        H_grid: int,
        W_grid: int,
        T: int,
    ):
        super().__init__()
        self.ln_temp = nn.LayerNorm(d_m)
        self.ln_spatial = nn.LayerNorm(d_m)
        self.ln_ffn = nn.LayerNorm(d_m)
        self.temp_attn = TemporalCrossAttention(d_m=d_m, H_heads=H_heads, d_k=d_k, T=T)
        self.local_attn = LocalAttention(
            d_m=d_m, H_heads=H_heads, d_k=d_k, R=R, H_grid=H_grid, W_grid=W_grid
        )
        self.fuse = nn.Linear(2 * d_m, d_m)
        self.ffn = nn.Sequential(
            nn.Linear(d_m, 4 * d_m),
            nn.GELU(),
            nn.Linear(4 * d_m, d_m),
        )

    def forward(
        self,
        z: torch.Tensor,
        context: torch.Tensor,
        patch_token: torch.Tensor,
        positions: torch.Tensor,
        alive_mask: torch.Tensor,
    ) -> torch.Tensor:
        c_temp = self.temp_attn(self.ln_temp(z), context, alive_mask)
        c_spatial = self.local_attn(self.ln_spatial(z), patch_token, positions, alive_mask)
        z_fused = z + self.fuse(torch.cat([c_temp, c_spatial], dim=-1))
        z_out = z_fused + self.ffn(self.ln_ffn(z_fused))
        return z_out * alive_mask.float()


class WorldModelTransformer(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.T = cfg["T"]
        self.N_max = cfg["N_max"]
        self.d_m = cfg["d_m"]
        self.d_z = cfg["d_z"]
        self.d_a = cfg["d_a"]
        self.H = cfg["H"]
        self.W = cfg["W"]
        self.n_actions = cfg["n_actions"]

        self.W_a = nn.Linear(self.n_actions, self.d_a)
        self.W_in = nn.Linear(self.d_z + self.d_a, self.d_m)
        self.W_g = nn.Linear(self.d_z, self.d_m)
        self.in_ln = nn.LayerNorm(self.d_m)
        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    d_m=cfg["d_m"],
                    H_heads=cfg["H_heads"],
                    d_k=cfg["d_k"],
                    R=cfg["R"],
                    H_grid=cfg["H"],
                    W_grid=cfg["W"],
                    T=cfg["T"],
                )
                for _ in range(cfg["L"])
            ]
        )

        self.register_buffer("context", torch.zeros(1, self.T, self.N_max, self.d_m), persistent=False)
        self.register_buffer("context_valid", torch.zeros(1, self.T, dtype=torch.bool), persistent=False)
        self.register_buffer("ctx_ptr", torch.zeros(1, dtype=torch.long), persistent=False)

    def _ensure_context(self, B: int, device: torch.device, dtype: torch.dtype):
        if self.context.shape[0] != B or self.context.device != device or self.context.dtype != dtype:
            self.context = torch.zeros(B, self.T, self.N_max, self.d_m, device=device, dtype=dtype)
            self.context_valid = torch.zeros(B, self.T, device=device, dtype=torch.bool)
            self.ctx_ptr = torch.zeros(1, device=device, dtype=torch.long)

    def reset_context(self, B: int, device: torch.device, dtype: torch.dtype):
        self.context = torch.zeros(B, self.T, self.N_max, self.d_m, device=device, dtype=dtype)
        self.context_valid = torch.zeros(B, self.T, device=device, dtype=torch.bool)
        self.ctx_ptr = torch.zeros(1, device=device, dtype=torch.long)

    def push(self, z_new: torch.Tensor):
        B = z_new.shape[0]
        self._ensure_context(B, z_new.device, z_new.dtype)
        ptr = int(self.ctx_ptr.item())
        # Context is a history buffer; detach to avoid backprop across training steps.
        self.context[:, ptr] = z_new.detach()
        self.context_valid[:, ptr] = True
        self.ctx_ptr[0] = (ptr + 1) % self.T

    def get_ordered_context(self) -> torch.Tensor:
        ptr = int(self.ctx_ptr.item())
        idx = torch.arange(self.T, device=self.context.device)
        ordered = (idx + ptr) % self.T  # oldest -> newest
        ctx = self.context.index_select(1, ordered)
        valid = self.context_valid.index_select(1, ordered).unsqueeze(-1).unsqueeze(-1)
        return ctx * valid.float()

    def forward(
        self,
        z_t: torch.Tensor,
        a_t: torch.Tensor,
        z_g_t: torch.Tensor,
        positions: torch.Tensor,
        alive_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, N, _ = z_t.shape
        if N != self.N_max:
            raise ValueError(f"Expected N={self.N_max}, got {N}")

        a_onehot = F.one_hot(a_t.long(), num_classes=self.n_actions).float()
        a_emb = self.W_a(a_onehot)
        z0 = self.in_ln(self.W_in(torch.cat([z_t, a_emb], dim=-1)))
        z0 = z0 * alive_mask.float()

        self.push(z0)
        context = self.get_ordered_context()  # [B,T,N,d_m]

        patch_token = self.W_g(z_g_t).unsqueeze(1)  # [B,1,d_m]
        z = z0
        for layer in self.layers:
            z = layer(z, context, patch_token, positions, alive_mask)
        return z * alive_mask.float()


if __name__ == "__main__":
    cfg = dict(
        H=16,
        W=16,
        N_max=64,
        d_z=64,
        d_a=16,
        d_m=128,
        d_k=32,
        H_heads=4,
        L=4,
        T=8,
        R=2,
        n_actions=8,
    )
    B = 2
    z_t = torch.randn(B, cfg["N_max"], cfg["d_z"])
    actions = torch.randint(0, cfg["n_actions"], (B, cfg["N_max"]))
    z_g = torch.randn(B, cfg["d_z"])
    positions = torch.randint(0, cfg["H"], (B, cfg["N_max"], 2))
    alive = (torch.rand(B, cfg["N_max"], 1) > 0.3).float()
    model = WorldModelTransformer(cfg)
    zL = model(z_t, actions, z_g, positions, alive)
    assert zL.shape == (B, cfg["N_max"], cfg["d_m"])
    print("transformer.py self-test passed.")
