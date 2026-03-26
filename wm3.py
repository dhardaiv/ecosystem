"""WM-3: DreamerV3-based world model for multi-agent predator-prey ecosystems."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Callable, Dict, List, Tuple, Union

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def symlog(x: Tensor) -> Tensor:
    """Symmetric logarithmic compression: sign(x) · ln(1 + |x|)."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: Tensor) -> Tensor:
    """Inverse of symlog: sign(x) · (exp(|x|) − 1)."""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


# ---------------------------------------------------------------------------
# RSSM
# ---------------------------------------------------------------------------

class RSSM(nn.Module):
    """Recurrent state-space model with GRU + categorical latents.

    Deterministic path:  h ∈ ℝ^{h_dim}  (GRU hidden state)
    Stochastic path:     z as z_cats × z_classes one-hot categories
                         sampled via Gumbel-Softmax (τ, straight-through).
    """

    def __init__(
        self,
        act_dim: int,
        h_dim: int = 512,
        e_dim: int = 512,
        z_cats: int = 32,
        z_classes: int = 32,
        tau: float = 0.5,
    ):
        super().__init__()
        self.h_dim = h_dim
        self.z_cats = z_cats
        self.z_classes = z_classes
        self.z_dim = z_cats * z_classes
        self.tau = tau

        self.pre_gru = nn.Sequential(
            nn.Linear(self.z_dim + act_dim, h_dim),
            nn.SiLU(),
        )
        self.gru = nn.GRUCell(h_dim, h_dim)

        # q(zₜ | hₜ, eₜ)
        self.post_net = nn.Sequential(
            nn.Linear(h_dim + e_dim, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, self.z_dim),
        )

        # p(zₜ | hₜ)
        self.prior_net = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, self.z_dim),
        )

    def _sample_z(self, logits: Tensor) -> Tensor:
        """Sample z via Gumbel-Softmax (train) or argmax (eval)."""
        B = logits.shape[0]
        logits_3d = logits.view(B, self.z_cats, self.z_classes)
        if self.training:
            return F.gumbel_softmax(logits_3d, tau=self.tau, hard=True, dim=-1)
        return F.one_hot(logits_3d.argmax(-1), self.z_classes).float()

    def step(
        self,
        h_prev: Tensor,
        z_prev: Tensor,
        a_prev: Tensor,
        embed: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Full RSSM step with posterior (training).

        Returns:
            h:            (B, h_dim)
            z:            (B, z_cats, z_classes)
            prior_logits: (B, z_cats, z_classes)
            post_logits:  (B, z_cats, z_classes)
        """
        z_flat = z_prev.flatten(1)
        x = self.pre_gru(torch.cat([z_flat, a_prev], dim=-1))
        h = self.gru(x, h_prev)

        prior_logits = self.prior_net(h).view(-1, self.z_cats, self.z_classes)
        post_logits = self.post_net(
            torch.cat([h, embed], dim=-1)
        ).view(-1, self.z_cats, self.z_classes)

        z = self._sample_z(post_logits.flatten(1))
        return h, z, prior_logits, post_logits

    def prior_step(
        self,
        h_prev: Tensor,
        z_prev: Tensor,
        a_prev: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Prior-only step (imagination — no encoder).

        Returns:
            h:            (B, h_dim)
            z:            (B, z_cats, z_classes)
            prior_logits: (B, z_cats, z_classes)
        """
        z_flat = z_prev.flatten(1)
        x = self.pre_gru(torch.cat([z_flat, a_prev], dim=-1))
        h = self.gru(x, h_prev)

        prior_logits = self.prior_net(h).view(-1, self.z_cats, self.z_classes)
        z = self._sample_z(prior_logits.flatten(1))
        return h, z, prior_logits


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """CNN (64×64×3 grid patch) + MLP (scalar features) → eₜ ∈ ℝ^{e_dim}."""

    def __init__(self, scalar_dim: int = 16, e_dim: int = 512):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),    # 64 → 32
            nn.SiLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),   # 32 → 16
            nn.SiLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 16 → 8
            nn.SiLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), #  8 → 4
            nn.SiLU(),
            nn.Flatten(),                                  # → 4096
        )
        cnn_out = 256 * 4 * 4

        self.scalar_net = nn.Sequential(
            nn.Linear(scalar_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
        )

        self.fuse = nn.Sequential(
            nn.Linear(cnn_out + 128, e_dim),
            nn.SiLU(),
            nn.Linear(e_dim, e_dim),
        )

    def forward(self, grid: Tensor, scalars: Tensor) -> Tensor:
        """
        Args:
            grid:    (B, 3, 64, 64) RGB grid patch.
            scalars: (B, scalar_dim) auxiliary scalar features.
        Returns:
            (B, e_dim) embedding vector.
        """
        return self.fuse(
            torch.cat([self.cnn(grid), self.scalar_net(scalars)], dim=-1)
        )


# ---------------------------------------------------------------------------
# AttentionAggregator
# ---------------------------------------------------------------------------

class AttentionAggregator(nn.Module):
    """Scaled dot-product attention producing a permutation-invariant joint state.

    Q derives from the querying (self) agent; K from every agent including self.
    Because softmax + weighted-sum is a set operation, sₜ is invariant to the
    ordering of the *other* agents.
    """

    def __init__(self, x_dim: int = 1536, qk_dim: int = 64):
        super().__init__()
        self.W_Q = nn.Linear(x_dim, qk_dim, bias=False)
        self.W_K = nn.Linear(x_dim, qk_dim, bias=False)
        self.scale = qk_dim ** 0.5

    def forward(
        self,
        x_self: Tensor,
        others: Union[Tensor, List[Tensor]],
    ) -> Tensor:
        """
        Args:
            x_self: (B, x_dim)  querying agent's [h ‖ z_flat].
            others: (B, N, x_dim) tensor **or** list of (B, x_dim) — other agents.
                    An empty list / zero-length dim-1 is valid (solo agent).
        Returns:
            s: (B, x_dim) aggregated joint state.
        """
        if isinstance(others, list):
            if len(others) == 0:
                return x_self
            others = torch.stack(others, dim=1)
        if others.shape[1] == 0:
            return x_self

        all_agents = torch.cat(
            [x_self.unsqueeze(1), others], dim=1,
        )  # (B, N+1, x_dim)

        Q = self.W_Q(x_self)                                 # (B, qk)
        K = self.W_K(all_agents)                              # (B, N+1, qk)
        alpha = F.softmax(
            torch.einsum("bd,bnd->bn", Q, K) / self.scale,
            dim=-1,
        )                                                     # (B, N+1)
        return torch.einsum("bn,bnd->bd", alpha, all_agents)  # (B, x_dim)


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------

class Heads(nn.Module):
    """Prediction heads operating on joint state sₜ.

    - decoder:  sₜ → 64×64×3 logits  (transposed CNN)
    - reward:   sₜ → scalar           (symlog space)
    - continue: sₜ → scalar           (sigmoid — survival probability)
    - critic:   sg(sₜ) → scalar V     (stop-gradient applied internally)
    - actor:    sg(sₜ) → logits       (stop-gradient applied internally)
    """

    def __init__(self, s_dim: int = 1536, act_dim: int = 5):
        super().__init__()
        self.s_dim = s_dim

        self.decoder_fc = nn.Sequential(
            nn.Linear(s_dim, 256 * 4 * 4),
            nn.SiLU(),
        )
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  #  4 → 8
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   #  8 → 16
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),    # 16 → 32
            nn.SiLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),     # 32 → 64
        )

        self.reward_net = nn.Sequential(
            nn.Linear(s_dim, 512), nn.SiLU(),
            nn.Linear(512, 256),   nn.SiLU(),
            nn.Linear(256, 1),
        )

        self.cont_net = nn.Sequential(
            nn.Linear(s_dim, 512), nn.SiLU(),
            nn.Linear(512, 256),   nn.SiLU(),
            nn.Linear(256, 1),
        )

        self.critic_net = nn.Sequential(
            nn.Linear(s_dim, 512), nn.SiLU(),
            nn.Linear(512, 256),   nn.SiLU(),
            nn.Linear(256, 1),
        )

        self.actor_net = nn.Sequential(
            nn.Linear(s_dim, 512), nn.SiLU(),
            nn.Linear(512, 256),   nn.SiLU(),
            nn.Linear(256, act_dim),
        )

    def forward(self, s: Tensor) -> Dict[str, Tensor]:
        """Run all heads.  Critic and actor receive ``s.detach()``."""
        s_sg = s.detach()
        B = s.shape[0]

        dec = self.decoder_cnn(self.decoder_fc(s).view(B, 256, 4, 4))

        return {
            "decoder":      dec,                                # (B, 3, 64, 64)
            "reward":       self.reward_net(s),                 # (B, 1)
            "cont":         torch.sigmoid(self.cont_net(s)),    # (B, 1)
            "critic":       self.critic_net(s_sg),              # (B, 1)
            "actor_logits": self.actor_net(s_sg),               # (B, act_dim)
        }


# ---------------------------------------------------------------------------
# WorldModel
# ---------------------------------------------------------------------------

class WorldModel(nn.Module):
    """WM-3 world model: RSSM + Encoder + AttentionAggregator + Heads.

    Encoder and RSSM parameters live under ``self.phi`` (an ``nn.ModuleDict``)
    so that the caller can construct a dedicated optimizer via
    ``torch.optim.Adam(model.phi.parameters(), ...)``.
    """

    def __init__(
        self,
        act_dim: int,
        scalar_dim: int = 16,
        *,
        h_dim: int = 512,
        z_cats: int = 32,
        z_classes: int = 32,
        tau: float = 0.5,
        qk_dim: int = 64,
    ):
        super().__init__()
        self.act_dim = act_dim
        self.h_dim = h_dim
        self.z_cats = z_cats
        self.z_classes = z_classes
        self.x_dim = h_dim + z_cats * z_classes  # 1536

        self.phi = nn.ModuleDict({
            "encoder": Encoder(scalar_dim=scalar_dim, e_dim=h_dim),
            "rssm": RSSM(
                act_dim=act_dim, h_dim=h_dim, e_dim=h_dim,
                z_cats=z_cats, z_classes=z_classes, tau=tau,
            ),
        })
        self.aggregator = AttentionAggregator(x_dim=self.x_dim, qk_dim=qk_dim)
        self.heads = Heads(s_dim=self.x_dim, act_dim=act_dim)

    # -- convenience accessors (no extra parameter registration) --

    @property
    def encoder(self) -> Encoder:
        return self.phi["encoder"]

    @property
    def rssm(self) -> RSSM:
        return self.phi["rssm"]

    # -- state initialisation --

    def initial_state(
        self, batch_size: int, device: torch.device | None = None,
    ) -> Tuple[Tensor, Tensor]:
        """Initialise (h₀, z₀) with h₀=0 and a valid one-hot z₀."""
        device = device or next(self.parameters()).device
        h = torch.zeros(batch_size, self.h_dim, device=device)
        # Use class-0 one-hot for each categorical latent at t=0.
        z = torch.zeros(batch_size, self.z_cats, self.z_classes, device=device)
        z[..., 0] = 1.0
        return h, z

    # -- main forward --

    def forward(
        self,
        obs: Dict[str, Tensor],
        h_prev: Tensor,
        z_prev: Tensor,
        a_prev: Tensor,
        agents: Union[Tensor, List[Tensor]],
    ) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Tensor], Tensor, Tensor]:
        """Single-step forward through the full world model.

        Args:
            obs:     dict with ``grid`` (B, 3, 64, 64) and ``scalars`` (B, D).
            h_prev:  (B, 512)    previous deterministic state.
            z_prev:  (B, 32, 32) previous stochastic state (one-hot per cat).
            a_prev:  (B, act_dim) previous action.
            agents:  other agents' [h‖z_flat] — (B, N, 1536) or list of (B, 1536).

        Returns:
            h:            (B, 512)
            z:            (B, 32, 32)
            s:            (B, 1536)   aggregated joint state
            preds:        dict from Heads (decoder, reward, cont, critic, actor_logits)
            prior_logits: (B, 32, 32)
            post_logits:  (B, 32, 32)
        """
        embed = self.encoder(obs["grid"], obs["scalars"])
        h, z, prior_logits, post_logits = self.rssm.step(
            h_prev, z_prev, a_prev, embed,
        )

        x_self = torch.cat([h, z.flatten(1)], dim=-1)
        s = self.aggregator(x_self, agents)
        preds = self.heads(s)

        return h, z, s, preds, prior_logits, post_logits

    # -- imagination rollout --

    def imagine(
        self,
        h: Tensor,
        z: Tensor,
        actor: Callable[[Tensor], Tensor],
        horizon: int = 15,
    ) -> Dict[str, Tensor]:
        """Open-loop rollout through the learned prior (no encoder).

        Gradients flow through the dynamics and actor for policy optimisation;
        the caller's optimizer config determines which parameters are updated.

        Args:
            h:       (B, 512)    initial deterministic state.
            z:       (B, 32, 32) initial stochastic state.
            actor:   callable  s (B, 1536) → action logits (B, act_dim).
            horizon: number of imagination steps.

        Returns:
            dict with stacked tensors (B, T, …):
                h      (B, T, 512)
                z      (B, T, 32, 32)
                s      (B, T, 1536)
                a      (B, T, act_dim)
                reward (B, T, 1)
                cont   (B, T, 1)
        """
        buf: Dict[str, List[Tensor]] = {
            k: [] for k in ("h", "z", "s", "a", "reward", "cont")
        }

        for _ in range(horizon):
            s = torch.cat([h, z.flatten(1)], dim=-1)  # (B, 1536)

            logits = actor(s)
            a = F.gumbel_softmax(logits, tau=1.0, hard=True, dim=-1)

            buf["h"].append(h)
            buf["z"].append(z)
            buf["s"].append(s)
            buf["a"].append(a)
            buf["reward"].append(self.heads.reward_net(s))
            buf["cont"].append(torch.sigmoid(self.heads.cont_net(s)))

            h, z, _ = self.rssm.prior_step(h, z, a)

        return {k: torch.stack(v, dim=1) for k, v in buf.items()}
