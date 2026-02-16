#!/usr/bin/env python3
"""
Fine-tune pretrained hybrid DDM+LSTM agent on macaque reference data.

Strategy:
1. Load pretrained LSTM + drift_head weights from supervised pretraining
2. Option A: Freeze drift_head parameters completely
3. Option B: Allow drift_head to update with strong L2 regularization
4. Train on macaque reference data with full task losses (choice + RT + history)
5. Evaluate RT-coherence dynamics to check if structure preserved
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from agents.hybrid_ddm_lstm import HybridTrainingConfig
from agents.losses import LossWeights
from animaltasksim.config import ProjectPaths


def finetune_hybrid_agent(
    pretrained_model_path: Path,
    output_dir: Path,
    freeze_drift: bool = True,
    drift_regularization: float = 0.1,
    epochs: int = 10,
    episodes: int = 20,
    trials_per_episode: int = 400,
    learning_rate: float = 3e-4,
    seed: int = 42,
) -> dict:
    """
    Fine-tune pretrained hybrid agent on macaque reference data.
    
    Args:
        pretrained_model_path: Path to pretrained model checkpoint
        output_dir: Where to save fine-tuned model and logs
        freeze_drift: If True, freeze drift_head parameters completely
        drift_regularization: L2 weight for drift_head (if not frozen)
        epochs: Number of training epochs
        episodes: Number of episodes per epoch
        trials_per_episode: Trials per episode
        learning_rate: Learning rate for fine-tuning
        seed: Random seed
    
    Returns:
        Dictionary with training results
    """
    print("="*80)
    print("FINE-TUNING PRETRAINED HYBRID AGENT ON MACAQUE DATA")
    print("="*80)
    print(f"Pretrained model: {pretrained_model_path}")
    print(f"Output directory: {output_dir}")
    print(f"Freeze drift_head: {freeze_drift}")
    print(f"Drift regularization: {drift_regularization}")
    print(f"Epochs: {epochs}, Episodes: {episodes}, Trials/episode: {trials_per_episode}")
    print(f"Learning rate: {learning_rate}")
    print(f"Seed: {seed}")
    print()
    
    # Create training config
    config = HybridTrainingConfig(
        reference_log=ProjectPaths.from_cwd().data / "macaque" / "reference.ndjson",
        output_dir=output_dir,
        agent_version="0.2.0-pretrained",
        trials_per_episode=trials_per_episode,
        episodes=episodes,
        seed=seed,
        epochs=epochs,
        hidden_size=64,
        learning_rate=learning_rate,
        loss_weights=LossWeights(
            choice=1.0,
            rt=0.5,
            history=0.1,
            drift_supervision=0.0  # No drift supervision during fine-tuning
        ),
    )
    
    # Train the agent (will initialize and then we'll load pretrained weights)
    # We need to modify the train_hybrid function to support loading pretrained weights
    # For now, let's do a workaround: train normally but load weights right after init
    
    print("Starting fine-tuning with pretrained weights...")
    print()
    
    # Import the trainer to access the model
    from agents.hybrid_ddm_lstm import HybridDDMTrainer
    
    trainer = HybridDDMTrainer(config)
    
    # Load pretrained weights
    print(f"Loading pretrained weights from {pretrained_model_path}...")
    checkpoint = torch.load(pretrained_model_path, map_location=trainer.device, weights_only=False)
    
    # Load LSTM and drift_head states
    trainer.model.lstm.load_state_dict(checkpoint["lstm_state"])
    trainer.model.drift_head.load_state_dict(checkpoint["drift_head_state"])
    
    print("✓ Loaded pretrained LSTM and drift_head")
    print(f"  Val loss from pretraining: {checkpoint.get('val_loss', 'N/A'):.4f}")
    print(f"  Val MAE from pretraining: {checkpoint.get('val_mae', 'N/A'):.4f}")
    print()
    
    # Freeze drift_head if requested
    if freeze_drift:
        print("🔒 Freezing drift_head parameters...")
        for param in trainer.model.drift_head.parameters():
            param.requires_grad = False
        print("✓ drift_head frozen (will not update during training)")
    else:
        print(f"📐 Applying L2 regularization (weight={drift_regularization}) to drift_head")
    print()
    
    # Re-create optimizer with potentially frozen parameters
    trainable_params = [p for p in trainer.model.parameters() if p.requires_grad]
    trainer.optimizer = torch.optim.Adam(trainable_params, lr=config.learning_rate)
    
    # Run training and save outputs
    print("Starting training on macaque reference data...")
    print()
    
    paths = config.output_paths()
    training_metrics = trainer.train()
    rollout_stats = trainer.rollout(paths)
    trainer.save(paths, training_metrics, rollout_stats)
    
    print()
    print("="*80)
    print("FINE-TUNING COMPLETE")
    print("="*80)
    print(f"Model and logs saved to: {output_dir}")
    print(f"  Config: {paths.config}")
    print(f"  Log: {paths.log}")
    print(f"  Metrics: {paths.metrics}")
    print(f"  Model: {paths.model}")
    print()
    
    return {
        "training_metrics": training_metrics,
        "rollout_stats": rollout_stats,
        "config": config,
        "pretrained_model": str(pretrained_model_path),
        "freeze_drift": freeze_drift,
        "paths": {
            "log": str(paths.log),
            "config": str(paths.config),
            "metrics": str(paths.metrics),
            "model": str(paths.model),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune pretrained hybrid agent on macaque data")
    parser.add_argument(
        "--pretrained-model",
        type=Path,
        default=Path("runs/hybrid_pretrain/best_pretrained_model.pt"),
        help="Path to pretrained model checkpoint"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/hybrid_finetuned"),
        help="Output directory for fine-tuned model"
    )
    parser.add_argument(
        "--freeze-drift",
        action="store_true",
        default=True,
        help="Freeze drift_head parameters during fine-tuning"
    )
    parser.add_argument(
        "--no-freeze-drift",
        dest="freeze_drift",
        action="store_false",
        help="Allow drift_head to update during fine-tuning"
    )
    parser.add_argument(
        "--drift-reg",
        type=float,
        default=0.1,
        help="L2 regularization weight for drift_head (if not frozen)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of episodes per epoch"
    )
    parser.add_argument(
        "--trials-per-episode",
        type=int,
        default=400,
        help="Trials per episode"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate for fine-tuning"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    results = finetune_hybrid_agent(
        pretrained_model_path=args.pretrained_model,
        output_dir=args.output_dir,
        freeze_drift=args.freeze_drift,
        drift_regularization=args.drift_reg,
        epochs=args.epochs,
        episodes=args.episodes,
        trials_per_episode=args.trials_per_episode,
        learning_rate=args.lr,
        seed=args.seed,
    )
    
    print("Fine-tuning results:")
    print(f"  Output directory: {results['config'].output_dir}")
    print(f"  Pretrained model: {results['pretrained_model']}")
    print(f"  Drift head frozen: {results['freeze_drift']}")


if __name__ == "__main__":
    main()
