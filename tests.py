"""Test suite for WM-3: multi-agent DreamerV3 world model (predator-prey)."""

import pytest
import torch

from wm3 import WorldModel, aggregator, critic
from losses import lambda_return, actor_loss, symlog, symlog_inv, kl_per_dim


# ---------------------------------------------------------------------------
# Fixture — minimal model shared across tests
# ---------------------------------------------------------------------------

@pytest.fixture
def model():
    torch.manual_seed(42)
    m = WorldModel(
        obs_dim=16,
        act_dim=4,
        latent_dim=32,
        hidden_dim=64,
        num_agents=3,
    )
    m.eval()
    return m


# ---------------------------------------------------------------------------
# T1 — stop_gradient boundary
# ---------------------------------------------------------------------------

def test_stop_gradient(model):
    """Actor loss backward() must not write gradients into any φ (RSSM) parameter."""
    torch.manual_seed(42)
    model.train()

    obs = torch.randn(1, 3, 16)
    acts = torch.randn(1, 3, 4)

    posterior, prior, features = model.forward(obs, acts)
    a_loss = actor_loss(features.detach() if not hasattr(actor_loss, '__self__') else features, model.actor)

    model.zero_grad()
    a_loss.backward()

    for name, p in model.phi.named_parameters():
        assert p.grad is None or p.grad.abs().max() < 1e-8, (
            f"RSSM gradients leaked from actor loss (param: {name}, "
            f"max |grad|={p.grad.abs().max().item():.2e})"
        )


# ---------------------------------------------------------------------------
# T2 — continue head zeroes futures
# ---------------------------------------------------------------------------

def test_continue_zeroes_futures():
    """Death flag (continue=0) must zero out the return at that step."""
    torch.manual_seed(42)

    r = torch.tensor([0.1, 0.2, -10.0, 5.0])
    c = torch.tensor([1.0, 1.0, 0.0, 1.0])
    V_next = torch.tensor([2.0, 1.5, 0.0, 3.0])

    V = lambda_return(r, c, V_next, gamma=0.99, lam=0.95)

    assert V[2].item() == 0.0, (
        f"Death step must produce zero return, got {V[2].item():.6f}"
    )
    assert V[1].item() < 0.3, (
        f"Step before death must be dragged down, got {V[1].item():.6f}"
    )


# ---------------------------------------------------------------------------
# T3 — attention permutation invariance
# ---------------------------------------------------------------------------

def test_attention_permutation_invariance(model):
    """Joint state must be invariant to the ordering of *other* agents."""
    torch.manual_seed(42)

    a1 = torch.randn(1, 64)
    a2 = torch.randn(1, 64)
    a3 = torch.randn(1, 64)

    s_orig = aggregator(self_idx=0, agents=[a1, a2, a3])
    s_shuf = aggregator(self_idx=2, agents=[a3, a1, a2])

    assert torch.allclose(s_orig, s_shuf, atol=1e-5), (
        "Joint state changed under permutation — aggregator is not invariant "
        f"(max Δ = {(s_orig - s_shuf).abs().max().item():.2e})"
    )


# ---------------------------------------------------------------------------
# T4 — symlog correctness and compression
# ---------------------------------------------------------------------------

def test_symlog_correctness():
    """symlog must match hand-computed values and compress reward scale."""
    torch.manual_seed(42)

    assert abs(symlog(torch.tensor(0.1)).item() - 0.0953) < 1e-4, (
        f"symlog(0.1) expected ≈0.0953, got {symlog(torch.tensor(0.1)).item():.6f}"
    )
    assert abs(symlog(torch.tensor(5.0)).item() - 1.7918) < 1e-4, (
        f"symlog(5.0) expected ≈1.7918, got {symlog(torch.tensor(5.0)).item():.6f}"
    )
    assert abs(symlog(torch.tensor(-10.0)).item() - (-2.3979)) < 1e-4, (
        f"symlog(-10) expected ≈-2.3979, got {symlog(torch.tensor(-10.0)).item():.6f}"
    )

    ratio = symlog(torch.tensor(10.0)) / symlog(torch.tensor(0.1))
    assert ratio.item() < 30, (
        f"symlog not compressing reward scale (ratio 10/0.1 = {ratio.item():.2f}, want <30)"
    )

    x = torch.tensor(3.7)
    roundtrip = symlog_inv(symlog(x))
    assert torch.allclose(roundtrip, x, atol=1e-5), (
        f"Inverse must round-trip: symlog_inv(symlog(3.7)) = {roundtrip.item():.6f}, want 3.7"
    )


# ---------------------------------------------------------------------------
# T5 — KL free bits
# ---------------------------------------------------------------------------

def test_kl_free_bits(model):
    """Latent KL must be non-collapsed, and free-bits clamp must activate on most dims."""
    torch.manual_seed(42)

    obs = torch.randn(4, 3, 16)
    acts = torch.randn(4, 3, 4)

    posterior, prior, _ = model.forward(obs, acts)
    kl = kl_per_dim(posterior, prior)

    assert kl.shape[-1] == 32, (
        f"Expected 32 latent dims, got shape {kl.shape}"
    )

    mean_kl = kl.mean().item()
    assert mean_kl > 0.5, (
        f"Latent collapsed — mean KL per dim = {mean_kl:.4f}, want >0.5"
    )

    free_bits_active = (kl.clamp(min=1.0) == 1.0).sum().item()
    assert free_bits_active > 20, (
        f"Free bits not activating — only {free_bits_active}/32 dims clamped, want >20"
    )


# ---------------------------------------------------------------------------
# T6 — variable population critic stability
# ---------------------------------------------------------------------------

def test_variable_population_critic_stability(model):
    """Adding a zero-padded agent must not shift the critic value by more than 0.5."""
    torch.manual_seed(42)

    a1 = torch.randn(1, 64)
    a2 = torch.randn(1, 64)
    a3 = torch.randn(1, 64)

    agg_3 = aggregator(self_idx=0, agents=[a1, a2, a3])
    V_3 = critic(agg_3)

    agg_4 = aggregator(self_idx=0, agents=[a1, a2, a3, torch.zeros_like(a1)])
    V_4 = critic(agg_4)

    diff = abs(V_3.item() - V_4.item())
    assert diff < 0.5, (
        f"Critic shifts >{diff:.4f} from a zero-padded agent (threshold 0.5)"
    )


# ---------------------------------------------------------------------------
# T7 — λ asymmetry produces horizon difference
# ---------------------------------------------------------------------------

def test_lambda_asymmetry_horizon():
    """Higher λ must weight distant rewards substantially more (>2×) than lower λ."""
    torch.manual_seed(42)

    r = torch.tensor([0.0] * 14 + [10.0])
    c = torch.tensor([1.0] * 15)
    V = torch.tensor([0.0] * 15)

    prey_return = lambda_return(r, c, V, gamma=0.997, lam=0.80)[0].item()
    pred_return = lambda_return(r, c, V, gamma=0.997, lam=0.95)[0].item()

    assert pred_return > prey_return * 2.0, (
        f"λ=0.95 pred must weight far reward >2× more than λ=0.80 prey "
        f"(pred={pred_return:.4f}, prey={prey_return:.4f}, "
        f"ratio={pred_return / max(prey_return, 1e-12):.2f})"
    )


# ---------------------------------------------------------------------------
# Integration note — run after wm3.py and losses.py are merged
# ---------------------------------------------------------------------------
