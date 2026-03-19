from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def kl_two_gaussians(
    mu1: torch.Tensor, logvar1: torch.Tensor, mu2: torch.Tensor, logvar2: torch.Tensor
) -> torch.Tensor:
    """
    KL[N(mu1,sig1) || N(mu2,sig2)] summed over last dim.
    """
    var1 = torch.exp(logvar1)
    var2 = torch.exp(logvar2)
    term = 0.5 * (
        (logvar2 - logvar1)
        + (var1 + (mu1 - mu2) ** 2) / var2
        - 1.0
    )
    return term.sum(dim=-1)


def kl_prior(mu: torch.Tensor, logvar: torch.Tensor, free_bits: float = 0.5) -> torch.Tensor:
    kl_dims = 0.5 * (mu**2 + logvar.exp() - logvar - 1.0)
    kl_dims = kl_dims.clamp(min=free_bits)
    return kl_dims.sum(dim=-1)


class WorldModelLoss(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg

    def compute_all_losses(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        alive_mask: torch.Tensor,
        beta_kl: float,
    ) -> Dict[str, torch.Tensor]:
        """
        alive_mask: [B,N,1] for masking all terms.
        """
        mask = alive_mask.float().squeeze(-1)  # [B,N]
        mask_sum = mask.sum().clamp(min=1.0)

        l_state_raw = kl_two_gaussians(
            targets["mu_t1"], targets["logvar_t1"], preds["mu_hat"], preds["logvar_hat"]
        )  # [B,N]
        l_state = (l_state_raw * mask).sum() / mask_sum

        l_kl_raw = kl_prior(targets["mu_t"], targets["logvar_t"], free_bits=self.cfg["free_bits"])
        l_kl = (l_kl_raw * mask).sum() / mask_sum

        B, N, A = preds["action_logits"].shape
        ce = F.cross_entropy(
            preds["action_logits"].reshape(B * N, A),
            targets["action_tgt"].reshape(B * N),
            reduction="none",
        ).reshape(B, N)
        l_action = (ce * mask).sum() / mask_sum

        mse = F.mse_loss(preds["reward"], targets["delta_e"], reduction="none").squeeze(-1)
        l_reward = (mse * mask).sum() / mask_sum

        bce = F.binary_cross_entropy_with_logits(
            preds["alive_logit"], targets["alive_next"].float(), reduction="none"
        ).squeeze(-1)
        l_alive = (bce * mask).sum() / mask_sum

        total = (
            l_state
            + targets.get("l_multi", torch.tensor(0.0, device=l_state.device))
            + beta_kl * l_kl
            + self.cfg["lambda_a"] * l_action
            + self.cfg["lambda_r"] * l_reward
            + l_alive
        )
        return {
            "L_total": total,
            "L_state": l_state,
            "L_multi": targets.get("l_multi", torch.tensor(0.0, device=l_state.device)),
            "L_KL": l_kl,
            "L_action": l_action,
            "L_reward": l_reward,
            "L_alive": l_alive,
        }

    def multistep_loss(
        self,
        model,
        z0: torch.Tensor,
        actions_seq: List[torch.Tensor],
        targets_seq: List[Dict[str, torch.Tensor]],
        positions_seq: List[torch.Tensor],
        alive_seq: List[torch.Tensor],
        z_g_seq: List[torch.Tensor],
        K: int = 5,
        gamma: float = 0.95,
    ) -> torch.Tensor:
        total = torch.tensor(0.0, device=z0.device)
        z = z0
        K_eff = min(K, len(actions_seq), len(targets_seq), len(positions_seq), len(alive_seq), len(z_g_seq))
        for k in range(K_eff):
            preds = model.step(z, actions_seq[k], positions_seq[k], alive_seq[k], z_g_seq[k])
            loss_k = self.compute_all_losses(
                preds=preds,
                targets=targets_seq[k],
                alive_mask=targets_seq[k]["alive_next"],
                beta_kl=0.0,
            )["L_total"]
            total = total + (gamma**k) * loss_k
            z = preds["z_next"].detach()
        return total


if __name__ == "__main__":
    cfg = dict(free_bits=0.5, lambda_a=1.0, lambda_r=0.5)
    B, N, d_z, A = 2, 64, 64, 8
    preds = {
        "mu_hat": torch.randn(B, N, d_z),
        "logvar_hat": torch.randn(B, N, d_z),
        "action_logits": torch.randn(B, N, A),
        "reward": torch.randn(B, N, 1).tanh(),
        "alive_logit": torch.randn(B, N, 1),
    }
    targets = {
        "mu_t1": torch.randn(B, N, d_z),
        "logvar_t1": torch.randn(B, N, d_z),
        "mu_t": torch.randn(B, N, d_z),
        "logvar_t": torch.randn(B, N, d_z),
        "action_tgt": torch.randint(0, A, (B, N)),
        "delta_e": torch.randn(B, N, 1).clamp(-1, 1),
        "alive_next": (torch.rand(B, N, 1) > 0.2).float(),
        "l_multi": torch.tensor(0.0),
    }
    wm_loss = WorldModelLoss(cfg)
    out = wm_loss.compute_all_losses(preds, targets, alive_mask=targets["alive_next"], beta_kl=0.1)
    assert "L_total" in out and torch.isfinite(out["L_total"])
    print("losses.py self-test passed.")
