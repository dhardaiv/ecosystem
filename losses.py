"""
losses.py – Loss functions for WM-3, a multi-agent DreamerV3 world model
for a predator–prey ecosystem.

Exports
-------
    symlog, symlogꜝ (symlog_inv)
    lambda_return
    world_model_loss
    actor_loss
    critic_loss
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Any, Dict, Tuple, Union

from wm3 import WorldModel

__all__ = [
    "symlog",
    "symlogꜝ",
    "symlog_inv",
    "lambda_return",
    "world_model_loss",
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
# World-model loss (trains φ — encoder + RSSM)
# ═══════════════════════════════════════════════════════════════════════════════


def world_model_loss(
    obs: Tensor,
    actions: Tensor,
    rewards: Tensor,
    continues: Tensor,
    phi: WorldModel,
    *,
    beta: float = 1.0,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """
    ℒ_WM = ℒ_recon + ℒ_rew + ℒ_cont + β · ℒ_KL

    Parameters
    ----------
    obs       : [T, B, ...]  observations (e.g. images)
    actions   : [T, B, A]    actions
    rewards   : [T, B]       rewards
    continues : [T, B]       continue flags
    phi       : WorldModel   encoder + RSSM + decoder + heads
    beta      : float        KL coefficient (default 1.0)

    Returns
    -------
    (scalar total loss, dict of detached sub-losses for logging)
    """
    embed = phi.encode(obs)
    posteriors, priors, states = phi.rssm.observe(embed, actions)

    # ℒ_recon: pixel cross-entropy −log p(oₜ | sₜ)
    obs_dist = phi.decode(states)
    loss_recon = -obs_dist.log_prob(obs).mean()

    # ℒ_rew: MSE in symlog space
    rew_pred = phi.reward_head(states)
    loss_rew = F.mse_loss(rew_pred, symlog(rewards))

    # ℒ_cont: binary cross-entropy on continue logits
    cont_logits = phi.continue_head(states)
    loss_cont = F.binary_cross_entropy_with_logits(cont_logits, continues)

    # ℒ_KL with free-bits clamp (1.0 nat):
    # Per-element KL is clamped to ≥ 1.0 before averaging.  This prevents
    # posterior collapse: the optimiser cannot trivially zero-out KL, so the
    # stochastic state is forced to carry meaningful information.
    kl = torch.distributions.kl_divergence(posteriors, priors)
    loss_kl = torch.clamp(kl, min=1.0).mean()  # free bits = 1.0 nat

    loss = loss_recon + loss_rew + loss_cont + beta * loss_kl

    metrics = {
        "loss_recon": loss_recon.detach(),
        "loss_rew": loss_rew.detach(),
        "loss_cont": loss_cont.detach(),
        "loss_kl": loss_kl.detach(),
        "loss_wm": loss.detach(),
        "kl_raw": kl.mean().detach(),
    }
    return loss, metrics


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
# """
# import torch
# from wm3 import WorldModel, Actor, Critic
# from losses import (
#     world_model_loss, actor_loss, critic_loss, lambda_return, symlog,
# )
#
# # ── Initialise ──────────────────────────────────────────────────────────
# phi       = WorldModel(obs_shape=(3, 64, 64), act_dim=5, stoch_dim=32)
# pi_theta  = Actor(state_dim=phi.state_dim, act_dim=5)
# V_psi     = Critic(state_dim=phi.state_dim)
#
# opt_wm     = torch.optim.Adam(phi.parameters(),       lr=1e-4)
# opt_actor  = torch.optim.Adam(pi_theta.parameters(),   lr=3e-5)
# opt_critic = torch.optim.Adam(V_psi.parameters(),      lr=3e-5)
#
# gamma, lam = 0.997, 0.95
#
# # ── Training step ──────────────────────────────────────────────────────
#
# # 1) World-model update on real experience
# loss_wm, wm_metrics = world_model_loss(obs, actions, rewards, continues, phi)
# opt_wm.zero_grad()
# loss_wm.backward()
# opt_wm.step()
#
# # 2) Imagine trajectories (φ weights frozen, gradients flow only to actor)
# with torch.no_grad():
#     init_state = phi.rssm.posterior(phi.encode(obs[:1]))
# imagination_traj = phi.imagine(init_state.detach(), pi_theta, horizon=15)
#
# # 3) Compute λ-returns (gradient graph retained for actor backprop)
# with torch.no_grad():
#     values = V_psi(imagination_traj["states"])
# returns = lambda_return(
#     imagination_traj["rewards"],
#     imagination_traj["continues"],
#     values,
#     gamma,
#     lam,    # scalar λ; pass a [H, B] tensor for per-species λ
# )
# imagination_traj["lambda_returns"] = returns
# imagination_traj["phi_params"] = list(phi.parameters())  # for the guard
#
# # 4) Actor update (straight-through gradients through dynamics)
# loss_a, actor_metrics = actor_loss(imagination_traj, pi_theta, eta=3e-4)
# opt_actor.zero_grad()
# loss_a.backward()
# opt_actor.step()
#
# # 5) Critic update (pure regression, both inputs stop-gradiented)
# loss_c, critic_metrics = critic_loss(imagination_traj, V_psi)
# opt_critic.zero_grad()
# loss_c.backward()
# opt_critic.step()
#
# # ── Logging ────────────────────────────────────────────────────────────
# all_metrics = {**wm_metrics, **actor_metrics, **critic_metrics}
# for k, v in all_metrics.items():
#     print(f"  {k:20s}: {v.item():.4f}")
# """
