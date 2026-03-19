from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class PredictionHeads(nn.Module):
    def __init__(self, d_m: int, d_z: int, n_actions: int):
        super().__init__()
        self.d_z = d_z
        self.state_head = nn.Linear(d_m, 2 * d_z)
        self.action_head = nn.Linear(d_m, n_actions)
        self.reward_head = nn.Sequential(
            nn.Linear(d_m, d_m // 2),
            nn.GELU(),
            nn.Linear(d_m // 2, 1),
            nn.Tanh(),
        )
        self.survival_head = nn.Linear(d_m, 1)

    def forward(self, zL: torch.Tensor) -> Dict[str, torch.Tensor]:
        state_stats = self.state_head(zL)
        mu_hat, logvar_hat = torch.chunk(state_stats, 2, dim=-1)
        if self.training and torch.is_grad_enabled():
            eps = torch.randn_like(mu_hat)
            z_next = mu_hat + torch.exp(0.5 * logvar_hat) * eps
        else:
            z_next = mu_hat

        return {
            "mu_hat": mu_hat,
            "logvar_hat": logvar_hat,
            "z_next": z_next,
            "action_logits": self.action_head(zL),
            "reward": self.reward_head(zL),
            "alive_logit": self.survival_head(zL),
        }


class PatchDecoder(nn.Module):
    def __init__(self, d_z: int, H: int, W: int):
        super().__init__()
        self.H = H
        self.W = W
        self.fc1 = nn.Linear(d_z, 256)
        self.fc2 = nn.Linear(256, H * W)

    def forward(self, z_g: torch.Tensor, F_total: torch.Tensor) -> torch.Tensor:
        """
        z_g: [B,d_z]
        F_total: scalar or [B,1]
        returns [B,H,W]
        """
        B = z_g.shape[0]
        logits = self.fc2(F.gelu(self.fc1(z_g))).reshape(B, self.H * self.W)
        probs = torch.softmax(logits, dim=-1).reshape(B, self.H, self.W)
        if not torch.is_tensor(F_total):
            F_total = torch.tensor(F_total, device=z_g.device, dtype=z_g.dtype)
        if F_total.ndim == 0:
            F_total = F_total.view(1, 1).expand(B, 1)
        elif F_total.ndim == 1:
            F_total = F_total.view(B, 1)
        return probs * F_total.view(B, 1, 1)


if __name__ == "__main__":
    B, N, d_m, d_z, n_actions = 2, 64, 128, 64, 8
    zL = torch.randn(B, N, d_m)
    heads = PredictionHeads(d_m=d_m, d_z=d_z, n_actions=n_actions)
    out = heads(zL)
    assert out["mu_hat"].shape == (B, N, d_z)
    assert out["logvar_hat"].shape == (B, N, d_z)
    assert out["z_next"].shape == (B, N, d_z)
    assert out["action_logits"].shape == (B, N, n_actions)
    assert out["reward"].shape == (B, N, 1)
    assert out["alive_logit"].shape == (B, N, 1)

    dec = PatchDecoder(d_z=d_z, H=16, W=16)
    z_g = torch.randn(B, d_z)
    feed_pred = dec(z_g, F_total=torch.ones(B))
    assert feed_pred.shape == (B, 16, 16)
    print("heads.py self-test passed.")
