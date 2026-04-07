"""
Generate sanity-check plots for the WM-3 codebase.

This repo currently doesn't include a complete training loop / logger, so these
plots focus on the implemented, unit-testable math utilities (symlog and
lambda-returns).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
# Headless-safe backend. Some environments default to an interactive backend
# (e.g., "macosx") which can crash in non-GUI contexts.
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import torch

from losses import lambda_return, symlog, symlog_inv
from train import Config, world_model_step
from wm3 import WorldModel


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_symlog(out_dir: Path) -> None:
    xs = torch.linspace(-50, 50, 4001)
    ys = symlog(xs)
    inv = symlog_inv(ys)

    plt.figure(figsize=(7, 4))
    plt.plot(xs.numpy(), ys.numpy(), label="symlog(x)")
    plt.plot(xs.numpy(), xs.numpy(), "--", label="identity", alpha=0.6)
    plt.title("symlog compression")
    plt.xlabel("x")
    plt.ylabel("symlog(x)")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "symlog_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(xs.numpy(), (inv - xs).abs().numpy())
    plt.title("symlog round-trip error: |symlog_inv(symlog(x)) - x|")
    plt.xlabel("x")
    plt.ylabel("absolute error")
    plt.yscale("log")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "symlog_roundtrip_error.png", dpi=200)
    plt.close()


def plot_lambda_return(out_dir: Path) -> None:
    # A toy scenario: sparse terminal reward and a "death" (continue=0) event.
    T = 40
    rewards = torch.zeros(T)
    rewards[10] = 1.0
    rewards[25] = 5.0

    continues_alive = torch.ones(T)
    continues_death = torch.ones(T)
    continues_death[20:] = 0.0  # death at t=20 (inclusive) zeros future contributions

    # critic bootstrap values (all zeros for clarity)
    values = torch.zeros(T + 1)

    gamma = 0.997
    lams = [0.0, 0.8, 0.95, 1.0]

    ts = np.arange(T)

    plt.figure(figsize=(8, 4))
    for lam in lams:
        V = lambda_return(
            rewards.view(T, 1),
            continues_alive.view(T, 1),
            values.view(T + 1, 1),
            gamma=gamma,
            lam=lam,
        ).squeeze(1)
        plt.plot(ts, V.numpy(), label=f"λ={lam}")
    plt.title("λ-returns (no death)")
    plt.xlabel("t")
    plt.ylabel("V_λ(t)")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "lambda_returns_alive.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4))
    for lam in lams:
        V = lambda_return(
            rewards.view(T, 1),
            continues_death.view(T, 1),
            values.view(T + 1, 1),
            gamma=gamma,
            lam=lam,
        ).squeeze(1)
        plt.plot(ts, V.numpy(), label=f"λ={lam}")
    plt.axvline(20, color="k", linestyle="--", alpha=0.4, label="death @ t=20")
    plt.title("λ-returns with continue-mask death")
    plt.xlabel("t")
    plt.ylabel("V_λ(t)")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "lambda_returns_death.png", dpi=200)
    plt.close()


def plot_loss_per_iteration(out_dir: Path) -> None:
    """
    Produce an intuitive loss-vs-iteration curve.

    The repo doesn't yet include an environment + replay buffer + full training
    script, so we run a small "synthetic training" loop that exercises the
    current WorldModel forward pass and optimizes a standard Dreamer-style
    objective on random data.
    """
    torch.manual_seed(0)
    device = torch.device("cpu")

    act_dim = 5
    B = 32
    iters = 400

    model = WorldModel(act_dim=act_dim).to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)

    losses = []
    recon_losses = []
    rew_losses = []
    cont_losses = []
    kl_losses = []

    beta_kl = 0.5

    for _ in range(iters):
        # Synthetic "observation"
        grid = torch.rand(B, 3, 64, 64, device=device)  # target in [0,1]
        scalars = torch.randn(B, 16, device=device)
        obs = {"grid": grid, "scalars": scalars}

        # Previous model state and action
        h_prev, z_prev = model.initial_state(B, device=device)
        a_prev = torch.nn.functional.one_hot(
            torch.randint(0, act_dim, (B,), device=device),
            num_classes=act_dim,
        ).float()

        # No other agents for this synthetic loop
        agents = torch.zeros(B, 0, model.x_dim, device=device)

        h, z, s, preds, prior_logits, post_logits = model(obs, h_prev, z_prev, a_prev, agents)

        # Reconstruction: treat decoder output as logits for Bernoulli pixels
        recon = torch.nn.functional.binary_cross_entropy_with_logits(
            preds["decoder"], grid, reduction="mean"
        )

        # Reward prediction in symlog space (targets are random, just to drive gradients)
        reward_target = symlog(torch.randn(B, 1, device=device))
        rew = torch.nn.functional.mse_loss(preds["reward"], reward_target, reduction="mean")

        # Continue prediction: BCE on a random survival label
        cont_target = torch.rand(B, 1, device=device).bernoulli()
        # preds["cont"] is sigmoid output; convert to logits for stable BCE
        cont_logits = torch.logit(preds["cont"].clamp(1e-5, 1 - 1e-5))
        cont = torch.nn.functional.binary_cross_entropy_with_logits(
            cont_logits, cont_target, reduction="mean"
        )

        # KL between categorical post and prior across z_cats; average over batch and cats
        # Shapes: (B, z_cats, z_classes)
        log_p = torch.log_softmax(post_logits, dim=-1)
        log_q = torch.log_softmax(prior_logits, dim=-1)
        p = torch.softmax(post_logits, dim=-1)
        kl = (p * (log_p - log_q)).sum(dim=-1).mean()

        loss = recon + rew + cont + beta_kl * kl

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        losses.append(loss.item())
        recon_losses.append(recon.item())
        rew_losses.append(rew.item())
        cont_losses.append(cont.item())
        kl_losses.append(kl.item())

    xs = np.arange(iters)
    plt.figure(figsize=(8, 4))
    plt.plot(xs, losses, label="total")
    plt.plot(xs, recon_losses, label="recon", alpha=0.8)
    plt.plot(xs, rew_losses, label="reward", alpha=0.8)
    plt.plot(xs, cont_losses, label="continue", alpha=0.8)
    plt.plot(xs, np.array(kl_losses) * beta_kl, label=f"β·KL (β={beta_kl})", alpha=0.8)
    plt.title("Synthetic training loss per iteration (WM-3 forward pass)")
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_per_iteration.png", dpi=200)
    plt.close()


def plot_loss_overfit_dataset(out_dir: Path) -> None:
    """Train WM-3 on the fixed overfit dataset (learnable structure, not noise)."""
    data_path = Path(__file__).resolve().parent / "data" / "overfit_dataset.pt"
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Missing {data_path}; run: python3 data/generate_overfit.py"
        )

    raw = torch.load(data_path, map_location="cpu")
    rewards = raw["rewards"]
    continues = raw["continues"]
    if rewards.dim() == 3:
        rewards = rewards.squeeze(-1)
    if continues.dim() == 3:
        continues = continues.squeeze(-1)

    batch = {
        "grids": raw["obs_grid"].float(),
        "scalars": raw["obs_scalars"].float(),
        "actions": raw["actions"].float(),
        "rewards": rewards.float(),
        "continues": continues.float(),
    }

    torch.manual_seed(0)
    device = torch.device("cpu")
    act_dim = batch["actions"].shape[-1]
    model = WorldModel(act_dim=act_dim).to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)

    config = Config()
    iters = 200

    totals, recons, rews, conts, kls = [], [], [], [], []

    for _ in range(iters):
        loss, metrics = world_model_step(model, batch, config)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        totals.append(metrics["loss/wm_total"].item())
        recons.append(metrics["loss/recon"].item())
        rews.append(metrics["loss/reward"].item())
        conts.append(metrics["loss/continue"].item())
        kls.append(metrics["loss/kl"].item())

    xs = np.arange(iters)
    plt.figure(figsize=(8, 4))
    plt.plot(xs, totals, label="total")
    plt.plot(xs, recons, label="recon", alpha=0.85)
    plt.plot(xs, rews, label="reward", alpha=0.85)
    plt.plot(xs, conts, label="continue", alpha=0.85)
    plt.plot(xs, np.array(kls) * config.kl_weight, label=f"β·KL (β={config.kl_weight})", alpha=0.85)
    plt.title("WM-3 training on overfit_dataset.pt (structured synthetic data)")
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_overfit_dataset.png", dpi=200)
    plt.close()


def main() -> None:
    out_dir = _ensure_dir(Path(__file__).resolve().parent / "plots")
    plot_symlog(out_dir)
    plot_lambda_return(out_dir)
    plot_loss_per_iteration(out_dir)
    plot_loss_overfit_dataset(out_dir)
    print(f"Wrote plots to: {out_dir}")


if __name__ == "__main__":
    main()

