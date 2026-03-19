from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder import AgentEncoder, FeedPatchEncoder, PatchEncoder, SinusoidalPosEnc, assemble_input
from env import PredatorPreyEnv
from heads import PatchDecoder, PredictionHeads
from losses import WorldModelLoss
from transformer import WorldModelTransformer


CFG = dict(
    H=16,
    W=16,
    N_max=64,
    d_pos=16,  # per-axis sinusoidal dim -> p_tilde in R^32
    d_patch=128,  # CNN patch output dim
    d_z=64,  # latent dim
    d_a=16,  # action embedding dim
    d_m=128,  # transformer model dim
    d_k=32,  # attention key dim per head (d_m / H_heads)
    H_heads=4,  # attention heads
    L=4,  # transformer layers
    T=8,  # temporal context window
    R=2,  # Chebyshev attention radius
    C=5,  # patch channels: [prey,predator,energy,feed,alive]
    patch_size=5,  # 2*R+1
    n_actions=8,
    beta_kl=0.1,  # KL weight (anneal from 0)
    free_bits=0.5,  # per-dim KL floor
    gamma=0.95,  # multi-step rollout discount
    K_rollout=5,  # unroll steps
    lambda_a=1.0,  # action loss weight
    lambda_r=0.5,  # reward loss weight
)


def anneal_beta(step: int, cfg: dict, warmup: int = 10000) -> float:
    return min(cfg["beta_kl"], cfg["beta_kl"] * step / warmup)


def pad_state(state: Dict[str, np.ndarray], cfg: dict, device: torch.device) -> Dict[str, torch.Tensor]:
    N_max, H, W = cfg["N_max"], cfg["H"], cfg["W"]
    n = min(len(state["e"]), N_max)

    e = torch.zeros(1, N_max, 1, device=device)
    pos = torch.zeros(1, N_max, 2, dtype=torch.long, device=device)
    species = torch.zeros(1, N_max, 3, device=device)
    alive = torch.zeros(1, N_max, 1, device=device)
    patch = torch.from_numpy(state["patch"]).float().to(device).unsqueeze(0)

    if n > 0:
        e[0, :n, 0] = torch.from_numpy(state["e"][:n]).float().to(device)
        pos[0, :n] = torch.from_numpy(state["pos"][:n]).long().to(device)
        species[0, :n] = torch.from_numpy(state["species"][:n]).float().to(device)
        alive[0, :n, 0] = torch.from_numpy(state["alive"][:n]).float().to(device)

    pos[..., 0] = pos[..., 0].clamp(0, H - 1)
    pos[..., 1] = pos[..., 1].clamp(0, W - 1)
    return {"e": e, "pos": pos, "species": species, "patch": patch, "alive": alive}


def pad_actions(actions: np.ndarray, cfg: dict, device: torch.device) -> torch.Tensor:
    N_max = cfg["N_max"]
    n = min(len(actions), N_max)
    out = torch.zeros(1, N_max, dtype=torch.long, device=device)
    if n > 0:
        out[0, :n] = torch.from_numpy(actions[:n]).long().to(device)
    return out


def build_spatial_grids(
    e: torch.Tensor,
    pos: torch.Tensor,
    species: torch.Tensor,
    alive: torch.Tensor,
    H: int,
    W: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns prey_grid,pred_grid,energy_grid,alive_grid each [B,H,W].
    """
    B, N, _ = pos.shape
    prey = torch.zeros(B, H, W, device=e.device)
    pred = torch.zeros(B, H, W, device=e.device)
    eng = torch.zeros(B, H, W, device=e.device)
    alive_grid = torch.zeros(B, H, W, device=e.device)

    for b in range(B):
        for i in range(N):
            if alive[b, i, 0] <= 0:
                continue
            x = int(pos[b, i, 0].item())
            y = int(pos[b, i, 1].item())
            prey[b, x, y] = species[b, i, 0]
            pred[b, x, y] = species[b, i, 1]
            eng[b, x, y] = e[b, i, 0]
            alive_grid[b, x, y] = 1.0
    return prey, pred, eng, alive_grid


def extract_agent_patches(
    e: torch.Tensor,
    pos: torch.Tensor,
    species: torch.Tensor,
    feed_map: torch.Tensor,
    alive: torch.Tensor,
    cfg: dict,
) -> torch.Tensor:
    """
    Returns [B,N,C,5,5] with channel order [prey,predator,energy,feed,alive].
    """
    B, N, _ = pos.shape
    H, W = cfg["H"], cfg["W"]
    R = cfg["R"]
    P = cfg["patch_size"]
    prey, pred, eng, alive_grid = build_spatial_grids(e, pos, species, alive, H, W)

    pad = (R, R, R, R)
    prey_p = F.pad(prey, pad)
    pred_p = F.pad(pred, pad)
    eng_p = F.pad(eng, pad)
    feed_p = F.pad(feed_map, pad)
    alive_p = F.pad(alive_grid, pad)

    out = torch.zeros(B, N, cfg["C"], P, P, device=e.device)
    for b in range(B):
        for i in range(N):
            x = int(pos[b, i, 0].item()) + R
            y = int(pos[b, i, 1].item()) + R
            xs, xe = x - R, x + R + 1
            ys, ye = y - R, y + R + 1
            out[b, i, 0] = prey_p[b, xs:xe, ys:ye]
            out[b, i, 1] = pred_p[b, xs:xe, ys:ye]
            out[b, i, 2] = eng_p[b, xs:xe, ys:ye]
            out[b, i, 3] = feed_p[b, xs:xe, ys:ye]
            out[b, i, 4] = alive_p[b, xs:xe, ys:ye]
    return out * alive.unsqueeze(-1).unsqueeze(-1)


class MultiAgentWorldModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.pos_enc = SinusoidalPosEnc(cfg["d_pos"], cfg["H"], cfg["W"])
        self.patch_encoder = PatchEncoder(cfg["C"], cfg["d_patch"])
        self.agent_encoder = AgentEncoder(d_x=164, d_h=256, d_z=cfg["d_z"])
        self.feed_encoder = FeedPatchEncoder(cfg["H"], cfg["W"], cfg["d_z"])
        self.transformer = WorldModelTransformer(cfg)
        self.heads = PredictionHeads(cfg["d_m"], cfg["d_z"], cfg["n_actions"])
        self.patch_decoder = PatchDecoder(cfg["d_z"], cfg["H"], cfg["W"])

    def encode_agents(self, state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        patches = extract_agent_patches(
            e=state["e"],
            pos=state["pos"],
            species=state["species"],
            feed_map=state["patch"],
            alive=state["alive"],
            cfg=self.cfg,
        )
        patch_out = self.patch_encoder(patches)
        x = assemble_input(
            e=state["e"],
            pos=state["pos"],
            species=state["species"],
            patch_cnn_out=patch_out,
            pos_enc_buffer=self.pos_enc(),
        )
        z, mu, logvar = self.agent_encoder(x)
        z = z * state["alive"]
        mu = mu * state["alive"]
        logvar = logvar * state["alive"]
        return z, mu, logvar

    def step(
        self,
        z: torch.Tensor,
        actions: torch.Tensor,
        positions: torch.Tensor,
        alive_mask: torch.Tensor,
        z_g: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        zL = self.transformer(z, actions, z_g, positions, alive_mask)
        preds = self.heads(zL)
        for k in ["mu_hat", "logvar_hat", "z_next", "action_logits", "reward", "alive_logit"]:
            preds[k] = preds[k] * alive_mask
        return preds


def build_targets_for_step(
    model: MultiAgentWorldModel,
    s_t: Dict[str, torch.Tensor],
    s_t1: Dict[str, torch.Tensor],
    actions_t: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    with torch.set_grad_enabled(model.training):
        _, mu_t, logvar_t = model.encode_agents(s_t)
        _, mu_t1, logvar_t1 = model.encode_agents(s_t1)
    delta_e = (s_t1["e"] - s_t["e"]).clamp(-1.0, 1.0)
    return {
        "mu_t": mu_t,
        "logvar_t": logvar_t,
        "mu_t1": mu_t1,
        "logvar_t1": logvar_t1,
        "action_tgt": actions_t.long(),
        "delta_e": delta_e,
        "alive_next": s_t1["alive"],
    }


def train_step(model, batch, optimizer, cfg, global_step: int):
    """
    batch: (states_t, states_t1, actions, patches_t, patches_t1, alive_t, alive_t1)
    """
    states_t, states_t1, actions, patches_t, patches_t1, alive_t, alive_t1 = batch
    del patches_t, patches_t1  # state dict already carries patches
    model.train()
    optimizer.zero_grad(set_to_none=True)

    loss_fn = WorldModelLoss(cfg)
    beta = anneal_beta(global_step, cfg)

    z_t, mu_t, logvar_t = model.encode_agents(states_t)
    z_t1, mu_t1, logvar_t1 = model.encode_agents(states_t1)
    z_g_t, _, _ = model.feed_encoder(states_t["patch"])
    zL = model.transformer(z_t, actions, z_g_t, states_t["pos"], alive_t)
    preds = model.heads(zL)
    feed_pred = model.patch_decoder(z_g_t, F_total=states_t["patch"].sum(dim=(1, 2)))

    targets = {
        "mu_t": mu_t,
        "logvar_t": logvar_t,
        "mu_t1": mu_t1,
        "logvar_t1": logvar_t1,
        "action_tgt": actions.long(),
        "delta_e": (states_t1["e"] - states_t["e"]).clamp(-1.0, 1.0),
        "alive_next": alive_t1.float(),
    }

    l_multi = torch.tensor(0.0, device=z_t.device)
    if "rollout" in states_t:
        roll = states_t["rollout"]
        actions_seq = roll["actions_seq"]
        positions_seq = [s["pos"] for s in roll["states_seq"]]
        alive_seq = [s["alive"] for s in roll["states_seq"]]
        z_g_seq = []
        targets_seq = []
        for k in range(len(actions_seq)):
            z_gk, _, _ = model.feed_encoder(roll["states_seq"][k]["patch"])
            z_g_seq.append(z_gk)
            targets_seq.append(
                build_targets_for_step(
                    model=model,
                    s_t=roll["states_seq"][k],
                    s_t1=roll["next_states_seq"][k],
                    actions_t=actions_seq[k],
                )
            )
        l_multi = loss_fn.multistep_loss(
            model=model,
            z0=z_t,
            actions_seq=actions_seq,
            targets_seq=targets_seq,
            positions_seq=positions_seq,
            alive_seq=alive_seq,
            z_g_seq=z_g_seq,
            K=cfg["K_rollout"],
            gamma=cfg["gamma"],
        )
    targets["l_multi"] = l_multi

    losses = loss_fn.compute_all_losses(preds, targets, alive_mask=alive_t1, beta_kl=beta)
    losses["L_total"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    with torch.no_grad():
        for t in [z_t, z_t1, zL, preds["mu_hat"], preds["action_logits"], preds["reward"], preds["alive_logit"], feed_pred]:
            if not torch.isfinite(t).all():
                raise RuntimeError("NaN/Inf detected in tensors.")

    loss_dict = {k: float(v.detach().cpu().item()) for k, v in losses.items()}
    debug = {
        "z_t": z_t.detach(),
        "z_L": zL.detach(),
        "mu_hat": preds["mu_hat"].detach(),
        "action_logits": preds["action_logits"].detach(),
        "reward": preds["reward"].detach(),
        "alive_logit": preds["alive_logit"].detach(),
        "feed_pred": feed_pred.detach(),
    }
    return loss_dict, debug


def test_full_rollout():
    cfg = CFG
    device = torch.device("cpu")

    env = PredatorPreyEnv(H=16, W=16, N_prey=8, N_pred=4, feed_regen_rate=0.03)
    model = MultiAgentWorldModel(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Collect trajectory.
    states_np = []
    actions_torch = []
    s = env.reset()
    states_np.append(s)
    for _ in range(20):
        n_alive = len(s["e"])
        a_np = np.random.randint(0, cfg["n_actions"], size=(n_alive,))
        s_next, _, _ = env.step(a_np)
        actions_torch.append(pad_actions(a_np, cfg, device))
        states_np.append(s_next)
        s = s_next

    states = [pad_state(s_np, cfg, device) for s_np in states_np]
    B = 1
    for step in range(5):
        t = step
        states_t = dict(states[t])
        states_t1 = states[t + 1]
        actions = actions_torch[t]
        rollout_states_seq = [states[t + k] for k in range(cfg["K_rollout"])]
        rollout_next_states_seq = [states[t + k + 1] for k in range(cfg["K_rollout"])]
        rollout_actions_seq = [actions_torch[t + k] for k in range(cfg["K_rollout"])]
        states_t["rollout"] = {
            "states_seq": rollout_states_seq,
            "next_states_seq": rollout_next_states_seq,
            "actions_seq": rollout_actions_seq,
        }

        batch = (
            states_t,
            states_t1,
            actions,
            states_t["patch"],
            states_t1["patch"],
            states_t["alive"],
            states_t1["alive"],
        )
        loss_dict, dbg = train_step(model, batch, optimizer, cfg, global_step=step + 1)

        d_z, d_m = cfg["d_z"], cfg["d_m"]
        N_max = cfg["N_max"]
        H, W = cfg["H"], cfg["W"]
        assert dbg["z_t"].shape == (B, N_max, d_z)
        assert dbg["z_L"].shape == (B, N_max, d_m)
        assert dbg["mu_hat"].shape == (B, N_max, d_z)
        assert dbg["action_logits"].shape == (B, N_max, 8)
        assert dbg["reward"].shape == (B, N_max, 1)
        assert dbg["alive_logit"].shape == (B, N_max, 1)
        assert dbg["feed_pred"].shape == (B, H, W)
        for tensor_name, tensor_val in dbg.items():
            assert torch.isfinite(tensor_val).all(), f"{tensor_name} has NaN/Inf"
        print(f"train_step={step} losses={loss_dict}")

    print("test_full_rollout passed.")


if __name__ == "__main__":
    test_full_rollout()
