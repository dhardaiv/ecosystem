"""
losses.py – Loss functions for WM-3, a multi-agent DreamerV3 world model
for a predator–prey ecosystem.

Exports
-------
    symlog, symlogꜝ (symlog_inv)
    lambda_return
    actor_loss
    critic_loss

Note: World model loss is computed directly in train.py using the WorldModel's
actual API (forward method with obs dict, h, z, a_prev, agents).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Any, Dict, Tuple, Union

__all__ = [
    "symlog",
    "symlogꜝ",
    "symlog_inv",
    "lambda_return",
    "actor_loss",
    "critic_loss",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Symmetric logarithmic transforms
# ═══════════════════════════════════════════════════════════════════════════════


def symlog(x: Tensor) -> Tensor:
    """sign(x) · ln(|x| + 1).  Numerically stable via log1p (no NaN at x=0)."""
    return torch.sign(x) * torch.log1p(x.abs())


def symlogꜝ(x: Tensor) -> Tensor:
    """Inverse symlog: sign(x) · (exp(|x|) − 1).  Stable near zero via expm1."""
    return torch.sign(x) * torch.expm1(x.abs())


symlog_inv = symlogꜝ


# ═══════════════════════════════════════════════════════════════════════════════
# λ-return with continue masking
# ═══════════════════════════════════════════════════════════════════════════════


def lambda_return(
    rewards: Tensor,
    continues: Tensor,
    values: Tensor,
    gamma: float,
    lam: Union[float, Tensor],
) -> Tensor:
    """
    GAE-style λ-returns with per-step continue masking.

        V_λᵗ = rₜ·cₜ + γ·cₜ·[(1−λ)·V̂_{t+1} + λ·V_λ^{t+1}]

    Death (cₜ = 0) at any step zeros that step's reward and all future
    contributions, correctly handling variable-length episodes.

    Parameters
    ----------
    rewards   : [T, B]     per-step rewards
    continues : [T, B]     binary continue mask (0 on death)
    values    : [T+1, B]   critic estimates; values[-1] bootstraps.
                            Also accepts [T, B], in which case bootstrap = 0.
    gamma     : float       discount factor
    lam       : float | [T, B]  GAE λ; pass a per-step tensor for per-species λ

    Returns
    -------
    Tensor [T, B] of λ-returns.
    """
    T = rewards.shape[0]
    device, dtype = rewards.device, rewards.dtype

    if not isinstance(lam, Tensor):
        lam = torch.tensor(lam, device=device, dtype=dtype)

    if values.shape[0] == T + 1:
        bootstrap = values[T]
        vals = values
    else:
        bootstrap = torch.zeros_like(rewards[0])
        vals = torch.cat([values, bootstrap.unsqueeze(0)], dim=0)

    returns = torch.empty_like(rewards)
    next_ret = bootstrap

    for t in reversed(range(T)):
        l = lam[t] if lam.dim() >= 1 and lam.shape[0] == T else lam
        returns[t] = (
            rewards[t] * continues[t]
            + gamma * continues[t] * ((1.0 - l) * vals[t + 1] + l * next_ret)
        )
        next_ret = returns[t]

    return returns


# ═══════════════════════════════════════════════════════════════════════════════
# Actor loss (trains θ — policy network; must NOT propagate into φ)
# ═══════════════════════════════════════════════════════════════════════════════


def actor_loss(
    imagination_traj: Dict[str, Any],
    pi_theta: Any,
    eta: float = 3e-4,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """
    ℒ_actor = −𝔼[V_λ_norm] − η · H[π]

    Gradients back-propagate through the imagined dynamics into the actor
    (straight-through / reparameterised), but must not reach φ.

    Parameters
    ----------
    imagination_traj : dict
        'states'         : [H, B, D]  imagined latent states
        'lambda_returns' : [H, B]     pre-computed λ-returns (gradient-retaining)
        'phi_params'     : optional iterable of φ parameters for the guard
    pi_theta : callable
        Actor network.  ``pi_theta(states)`` → ``torch.distributions.Distribution``
    eta : float
        Entropy bonus coefficient.

    Returns
    -------
    (scalar loss, dict of detached sub-terms)
    """
    states = imagination_traj["states"]
    returns = imagination_traj["lambda_returns"]

    # Normalise returns: scale-invariant across predator / prey reward regimes
    mu, sigma = returns.mean(), returns.std()
    returns_norm = (returns - mu) / torch.clamp(sigma, min=1.0)

    dist = pi_theta(states)
    entropy = dist.entropy()

    # The −η H[π] term in the minimised loss MAXIMISES entropy, acting as an
    # exploration bonus.  In the predator-prey setting this prevents prey from
    # collapsing to a single hiding strategy and predators from fixating on one
    # hunting pattern.
    loss = -returns_norm.mean() - eta * entropy.mean()

    # Guard: verify no gradient leaks to world-model (φ) parameters.
    phi_params = imagination_traj.get("phi_params")
    if phi_params is not None:
        phi_params = list(phi_params)
        grads = torch.autograd.grad(
            loss,
            phi_params,
            allow_unused=True,
            retain_graph=True,
        )
        if any(g is not None for g in grads):
            raise RuntimeError(
                "Actor loss has gradient path to φ.  "
                "Detach the imagination initial state from the encoder/RSSM graph."
            )

    metrics = {
        "loss_actor": loss.detach(),
        "entropy": entropy.mean().detach(),
        "returns_mean": returns.mean().detach(),
        "returns_std": sigma.detach(),
    }
    return loss, metrics


# ═══════════════════════════════════════════════════════════════════════════════
# Critic loss (trains ψ — value network; must NOT propagate into φ)
# ═══════════════════════════════════════════════════════════════════════════════


def critic_loss(
    imagination_traj: Dict[str, Any],
    V_psi: Any,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """
    ℒ_critic = MSE(V_ψ(sg(sₜ)), sg(V_λ))

    Both the state input and the regression target are stop-gradiented so
    the critic purely regresses toward the current λ-return estimate.

    Parameters
    ----------
    imagination_traj : dict
        'states'         : [H, B, D]  imagined latent states
        'lambda_returns' : [H, B]     pre-computed λ-returns
    V_psi : callable
        Critic network.  ``V_psi(states)`` → ``Tensor [H, B]``

    Returns
    -------
    (scalar loss, dict of detached sub-terms)
    """
    states = imagination_traj["states"].detach()          # sg(sₜ)
    targets = imagination_traj["lambda_returns"].detach()  # sg(V_λ)

    values = V_psi(states)
    loss = F.mse_loss(values, targets)

    metrics = {
        "loss_critic": loss.detach(),
        "value_mean": values.mean().detach(),
        "target_mean": targets.mean().detach(),
    }
    return loss, metrics


# ═══════════════════════════════════════════════════════════════════════════════
# Usage example
# ═══════════════════════════════════════════════════════════════════════════════
#
# See train.py for the full training loop implementation. Key points:
#
# 1. World model loss is computed via world_model_step() in train.py, which:
#    - Iterates through sequences calling model.forward(obs, h, z, a_prev, agents)
#    - Computes reconstruction, reward, continue, and KL losses
#    - Uses categorical_kl_divergence() with free bits and KL balancing
#
# 2. Actor-critic training uses actor_critic_step() which:
#    - Imagines trajectories via model.imagine(h, z, actor_fn, horizon)
#    - Computes λ-returns with EMA target critic for stable value estimation
#    - Actor loss: -(log_prob * returns_norm).mean() - eta * entropy.mean()
#    - Critic loss: MSE(critic(states), lambda_returns.detach())
#
# Example lambda_return usage:
#
# ```python
# from losses import lambda_return
#
# returns = lambda_return(
#     rewards=rewards,      # [T, B]
#     continues=continues,  # [T, B]
#     values=values,        # [T+1, B] or [T, B]
#     gamma=0.997,
#     lam=0.95,
# )
# ```
