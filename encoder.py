import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEnc(nn.Module):
    def __init__(self, d_pos: int, H: int, W: int):
        super().__init__()
        if d_pos % 2 != 0:
            raise ValueError("d_pos must be even for sin/cos pair encoding.")
        self.d_pos = d_pos
        self.H = H
        self.W = W
        pe = self._build_buffer(d_pos=d_pos, H=H, W=W)
        self.register_buffer("pos_enc", pe, persistent=True)

    @staticmethod
    def _axis_pe(length: int, d_pos: int) -> torch.Tensor:
        pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)  # [L,1]
        k = torch.arange(d_pos // 2, dtype=torch.float32).unsqueeze(0)  # [1,d_pos/2]
        denom = torch.pow(10000.0, (2.0 * k) / float(d_pos))
        angles = pos / denom
        pe = torch.zeros(length, d_pos, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(angles)
        pe[:, 1::2] = torch.cos(angles)
        return pe

    @classmethod
    def _build_buffer(cls, d_pos: int, H: int, W: int) -> torch.Tensor:
        x_pe = cls._axis_pe(H, d_pos).unsqueeze(1).expand(H, W, d_pos)
        y_pe = cls._axis_pe(W, d_pos).unsqueeze(0).expand(H, W, d_pos)
        return torch.cat([x_pe, y_pe], dim=-1)  # [H,W,2*d_pos]

    def forward(self) -> torch.Tensor:
        return self.pos_enc


class PatchEncoder(nn.Module):
    """Per-agent local patch encoder. Input [B,N,C,5,5] -> [B,N,d_patch]."""

    def __init__(self, C: int, d_patch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(C, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * 5 * 5, d_patch)
        self.ln = nn.LayerNorm(d_patch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C, P1, P2 = x.shape
        if P1 != 5 or P2 != 5:
            raise ValueError("PatchEncoder requires patch_size=5.")
        y = x.reshape(B * N, C, P1, P2)
        y = F.relu(self.conv1(y))
        y = F.relu(self.conv2(y))
        y = y.reshape(B * N, -1)
        y = self.fc(y)
        y = self.ln(y)
        return y.reshape(B, N, -1)


class AgentEncoder(nn.Module):
    def __init__(self, d_x: int, d_h: int, d_z: int):
        super().__init__()
        self.in_ln = nn.LayerNorm(d_x)
        self.fc1 = nn.Linear(d_x, d_h)
        self.ln1 = nn.LayerNorm(d_h)
        self.fc2 = nn.Linear(d_h, d_h)
        self.ln2 = nn.LayerNorm(d_h)
        self.mu_head = nn.Linear(d_h, d_z)
        self.logvar_head = nn.Linear(d_h, d_z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.in_ln(x)
        h = F.gelu(self.fc1(h))
        h = self.ln1(h)
        h = F.gelu(self.fc2(h))
        h = self.ln2(h)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        if self.training and torch.is_grad_enabled():
            eps = torch.randn_like(mu)
            z = mu + torch.exp(0.5 * logvar) * eps
        else:
            z = mu
        return z, mu, logvar


def assemble_input(
    e: torch.Tensor,
    pos: torch.Tensor,
    species: torch.Tensor,
    patch_cnn_out: torch.Tensor,
    pos_enc_buffer: torch.Tensor,
) -> torch.Tensor:
    """
    e: [B,N,1], pos: [B,N,2] int(x,y), species: [B,N,3], patch_cnn_out: [B,N,d_patch]
    pos_enc_buffer: [H,W,2*d_pos]
    """
    B, N, _ = pos.shape
    H, W, Dp = pos_enc_buffer.shape
    pos_x = pos[..., 0].clamp(0, H - 1)
    pos_y = pos[..., 1].clamp(0, W - 1)
    flat_idx = (pos_x * W + pos_y).reshape(-1)
    pe_flat = pos_enc_buffer.reshape(H * W, Dp)
    pos_emb = pe_flat.index_select(0, flat_idx).reshape(B, N, Dp)
    return torch.cat([e, pos_emb, species, patch_cnn_out], dim=-1)


class FeedPatchEncoder(nn.Module):
    """Global feed density map encoder: [B,H,W] -> mu/logvar/z [B,d_z]."""

    def __init__(self, H: int, W: int, d_z: int):
        super().__init__()
        self.H = H
        self.W = W
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * H * W, 256)
        self.fc2 = nn.Linear(256, 2 * d_z)
        self.d_z = d_z

    def forward(self, f_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, H, W = f_map.shape
        if H != self.H or W != self.W:
            raise ValueError(f"Expected feed map [{self.H},{self.W}], got [{H},{W}]")
        x = f_map.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.reshape(B, -1)
        x = F.gelu(self.fc1(x))
        stats = self.fc2(x)
        mu, logvar = torch.chunk(stats, 2, dim=-1)
        if self.training and torch.is_grad_enabled():
            eps = torch.randn_like(mu)
            z = mu + torch.exp(0.5 * logvar) * eps
        else:
            z = mu
        return z, mu, logvar


if __name__ == "__main__":
    cfg = dict(H=16, W=16, N_max=64, d_pos=16, d_patch=128, d_z=64, C=5)
    B, N = 2, cfg["N_max"]
    pos_enc_mod = SinusoidalPosEnc(cfg["d_pos"], cfg["H"], cfg["W"])
    pe = pos_enc_mod()
    assert pe.shape == (cfg["H"], cfg["W"], 2 * cfg["d_pos"])

    patch_encoder = PatchEncoder(cfg["C"], cfg["d_patch"])
    patches = torch.randn(B, N, cfg["C"], 5, 5)
    patch_out = patch_encoder(patches)
    assert patch_out.shape == (B, N, cfg["d_patch"])

    e = torch.rand(B, N, 1)
    pos = torch.randint(0, cfg["H"], size=(B, N, 2))
    species = F.one_hot(torch.randint(0, 3, size=(B, N)), num_classes=3).float()
    x = assemble_input(e, pos, species, patch_out, pe)
    assert x.shape == (B, N, 164)

    agent_enc = AgentEncoder(d_x=164, d_h=256, d_z=cfg["d_z"])
    z, mu, logvar = agent_enc(x)
    assert z.shape == mu.shape == logvar.shape == (B, N, cfg["d_z"])

    feed_enc = FeedPatchEncoder(cfg["H"], cfg["W"], cfg["d_z"])
    fmap = torch.rand(B, cfg["H"], cfg["W"])
    zg, mug, lvg = feed_enc(fmap)
    assert zg.shape == mug.shape == lvg.shape == (B, cfg["d_z"])
    print("encoder.py self-test passed.")
