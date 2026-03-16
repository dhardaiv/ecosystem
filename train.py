"""
Main training entry point.

Usage:
    # 1. Generate data first
    python -m simulator.generate_data --train 200 --val 30 --test 20 --steps 200

    # 2. Train
    python train.py

    # 3. Inference
    python inference.py --checkpoint checkpoints/best.pt --steps 100
"""
import os
import pickle
import argparse
import wandb

from config import MODEL, TRAIN, DATA, all_configs_as_dict
from model import EcosystemWorldModel
from data.dataset import make_dataloaders, load_episodes
from training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train ecosystem world model")
    parser.add_argument("--data_dir",   type=str, default=DATA.data_dir)
    parser.add_argument("--epochs",     type=int, default=TRAIN.n_epochs)
    parser.add_argument("--batch_size", type=int, default=TRAIN.batch_size)
    parser.add_argument("--lr",         type=float, default=TRAIN.lr)
    parser.add_argument("--resume",     type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--wandb_entity", type=str, default=None)
    args = parser.parse_args()

    # Apply CLI overrides
    TRAIN.n_epochs   = args.epochs
    TRAIN.batch_size = args.batch_size
    TRAIN.lr         = args.lr

    # ── wandb init ────────────────────────────────────────────────────────
    cfg_dict = all_configs_as_dict()
    cfg_dict.update({
        "d_model":   MODEL.d_model,
        "n_layers":  MODEL.n_layers,
        "n_heads":   MODEL.n_heads,
        "n_max_agents": MODEL.n_max_agents,
        "grid_size": MODEL.grid_size,
        "lambda_mse": TRAIN.lambda_mse,
        "lambda_bce": TRAIN.lambda_bce,
        "lambda_ce":  TRAIN.lambda_ce,
        "lambda_aux": TRAIN.lambda_aux,
        "lr":         TRAIN.lr,
        "batch_size": TRAIN.batch_size,
        "scheduled_sampling_rate": TRAIN.scheduled_sampling_rate,
        "alive_noise_rate":        TRAIN.alive_noise_rate,
        "phase": 1,
    })

    wandb.init(
        project=TRAIN.wandb_project,
        entity=args.wandb_entity or TRAIN.wandb_entity,
        config=cfg_dict,
    )

    # ── Data ─────────────────────────────────────────────────────────────
    print(f"Loading data from {args.data_dir}...")
    train_loader, val_loader, _ = make_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
    )

    val_episodes_path = os.path.join(args.data_dir, "val.pkl")
    val_episodes = load_episodes(val_episodes_path) if os.path.exists(val_episodes_path) else None

    # ── Model ─────────────────────────────────────────────────────────────
    model = EcosystemWorldModel(MODEL)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")
    wandb.config.update({"n_parameters": n_params}, allow_val_change=True)

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_cfg=TRAIN,
        model_cfg=MODEL,
        val_episodes=val_episodes,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    # ── Train ─────────────────────────────────────────────────────────────
    trainer.fit()

    wandb.finish()


if __name__ == "__main__":
    main()
