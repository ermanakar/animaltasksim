#!/usr/bin/env python3
"""
Supervised pretraining for hybrid DDM+LSTM agent on synthetic DDM data.

Strategy:
1. Load synthetic dataset with known (coherence, true_drift_gain, RT) labels
2. Train LSTM to predict drift_gain from trial features (coherence, history)
3. Loss = MSE(predicted_drift_gain, true_drift_gain)
4. Save pretrained weights for fine-tuning on real macaque data
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from agents.hybrid_ddm_lstm import HybridDDMModel


@dataclass
class PretrainConfig:
    """Configuration for supervised pretraining."""
    
    # Data
    data_path: Path = Path("data/synthetic/ddm_pretraining.ndjson")
    train_split: float = 0.8
    
    # Training
    batch_size: int = 128
    epochs: int = 20
    learning_rate: float = 1e-3
    
    # Model (must match HybridDDMLSTMAgent architecture)
    input_size: int = 7  # coherence, abs_coh, sign, prev_action, prev_reward, prev_correct, trial_norm
    hidden_size: int = 64
    
    # Validation
    val_every: int = 1  # Validate every N epochs
    early_stopping_patience: int = 5
    
    # Output
    output_dir: Path = Path("runs/hybrid_pretrain")
    seed: int = 42


class SyntheticDDMDataset(Dataset):
    """PyTorch dataset for synthetic DDM pretraining data."""
    
    def __init__(self, data_path: Path, config: PretrainConfig):
        self.trials = []
        
        # Load NDJSON
        with open(data_path) as f:
            for line in f:
                trial = json.loads(line)
                self.trials.append(trial)
        
        print(f"Loaded {len(self.trials)} trials from {data_path}")
        
        # Build sequence of trials (simulating online learning)
        # For each trial, we have:
        # - Current trial features: coherence, abs_coh, sign
        # - History features: prev_action, prev_reward, prev_correct (start at 0)
        # - Target: true_drift_gain
        
        self.features = []
        self.targets = []
        
        prev_action = 0
        prev_reward = 0
        prev_correct = 0
        
        for idx, trial in enumerate(self.trials):
            coherence = trial["coherence"]
            abs_coh = abs(coherence)
            sign = 1.0 if coherence > 0 else -1.0
            trial_norm = idx / len(self.trials)
            
            # Compute correct/reward from action and coherence
            # In synthetic data: action is 0 (left) or 1 (right)
            # correct if: (action == 1 and coherence > 0) or (action == 0 and coherence < 0)
            # For zero coherence, randomly assign correct=1 with 50% probability
            action = trial["action"]
            if coherence > 0:
                correct = 1 if action == 1 else 0
            elif coherence < 0:
                correct = 1 if action == 0 else 0
            else:  # coherence == 0
                correct = np.random.randint(0, 2)  # Random 50/50
            reward = correct  # Reward = correct in this task
            
            # Features: [coherence, abs_coh, sign, prev_action, prev_reward, prev_correct, trial_norm]
            features = np.array([
                coherence,
                abs_coh,
                sign,
                float(prev_action),
                float(prev_reward),
                float(prev_correct),
                trial_norm
            ], dtype=np.float32)
            
            # Target: true_drift_gain
            target = np.array([trial["true_drift_gain"]], dtype=np.float32)
            
            self.features.append(features)
            self.targets.append(target)
            
            # Update history for next trial
            prev_action = action
            prev_reward = reward
            prev_correct = correct
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.features[idx]), torch.from_numpy(self.targets[idx])


def pretrain_hybrid_agent(config: PretrainConfig) -> dict[str, Any]:
    """
    Pretrain hybrid agent on synthetic DDM data.
    
    Returns:
        Dictionary with training metrics and final model state
    """
    # Set seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_path = config.output_dir / "pretrain_config.json"
    with open(config_path, "w") as f:
        json.dump({
            "data_path": str(config.data_path),
            "train_split": config.train_split,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "seed": config.seed,
        }, f, indent=2)
    
    print("="*80)
    print("SUPERVISED PRETRAINING ON SYNTHETIC DDM DATA")
    print("="*80)
    print(f"Config: {config}")
    print()
    
    # Load dataset
    full_dataset = SyntheticDDMDataset(config.data_path, config)
    
    # Train/validation split
    n_train = int(len(full_dataset) * config.train_split)
    n_val = len(full_dataset) - n_train
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(config.seed)
    )
    
    print(f"Train: {len(train_dataset)} trials, Val: {len(val_dataset)} trials")
    print()
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False
    )
    
    # Create model (only need LSTM + drift_head for pretraining)
    device = torch.device("cpu")
    model = HybridDDMModel(
        feature_dim=config.input_size,
        hidden_size=config.hidden_size,
        device=device,
        drift_scale=1.0  # Start with no scaling for supervised learning
    )
    model.to(device)
    
    # Optimizer (only for LSTM + drift_head parameters)
    optimizer = torch.optim.Adam(
        list(model.lstm.parameters()) + list(model.drift_head.parameters()),
        lr=config.learning_rate
    )
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_mae": [],
    }
    
    print("Starting supervised pretraining...")
    print()
    
    for epoch in range(config.epochs):
        # Training phase
        model.train()
        train_losses = []
        
        for features, targets in train_loader:
            optimizer.zero_grad()
            
            # Forward pass through LSTM
            # Need to add batch dimension and handle hidden state
            # features: [batch, 7]
            # We'll process each batch as a mini-sequence
            
            batch_size = features.shape[0]
            hidden = model.init_state(batch_size)
            
            # Reshape for LSTMCell: just use features directly [batch, input_size]
            # LSTMCell processes one timestep at a time
            h, c = model.lstm(features, hidden)
            
            # Predict drift_gain
            predicted_drift = model.drift_head(h)  # [batch, 1]
            
            # Compute loss
            loss = criterion(predicted_drift, targets)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.lstm.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model.drift_head.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        history["train_loss"].append(avg_train_loss)
        
        # Validation phase
        if (epoch + 1) % config.val_every == 0:
            model.eval()
            val_losses = []
            val_maes = []
            
            with torch.no_grad():
                for features, targets in val_loader:
                    batch_size = features.shape[0]
                    hidden = model.init_state(batch_size)
                    
                    h, c = model.lstm(features, hidden)
                    
                    predicted_drift = model.drift_head(h)
                    
                    loss = criterion(predicted_drift, targets)
                    mae = torch.abs(predicted_drift - targets).mean()
                    
                    val_losses.append(loss.item())
                    val_maes.append(mae.item())
            
            avg_val_loss = np.mean(val_losses)
            avg_val_mae = np.mean(val_maes)
            history["val_loss"].append(avg_val_loss)
            history["val_mae"].append(avg_val_mae)
            
            print(f"Epoch {epoch+1:3d}/{config.epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | "
                  f"Val MAE: {avg_val_mae:.4f}")
            
            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                
                # Save best model
                torch.save({
                    "lstm_state": model.lstm.state_dict(),
                    "drift_head_state": model.drift_head.state_dict(),
                    "epoch": epoch,
                    "val_loss": avg_val_loss,
                    "val_mae": avg_val_mae,
                }, config.output_dir / "best_pretrained_model.pt")
                
                print(f"  → New best model saved (val_loss={avg_val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= config.early_stopping_patience:
                    print(f"\nEarly stopping triggered at epoch {epoch+1}")
                    break
    
    print()
    print("="*80)
    print("PRETRAINING COMPLETE")
    print("="*80)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {config.output_dir / 'best_pretrained_model.pt'}")
    print()
    
    # Save training history
    history_path = config.output_dir / "pretrain_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    
    return {
        "best_val_loss": best_val_loss,
        "history": history,
        "model_path": str(config.output_dir / "best_pretrained_model.pt"),
    }


def main():
    """Run supervised pretraining."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pretrain hybrid agent on synthetic DDM data")
    parser.add_argument("--data-path", type=Path, default=Path("data/synthetic/ddm_pretraining.ndjson"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/hybrid_pretrain"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    config = PretrainConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
    )
    
    results = pretrain_hybrid_agent(config)
    
    print("Results:")
    print(f"  Best validation loss: {results['best_val_loss']:.4f}")
    print(f"  Model saved to: {results['model_path']}")


if __name__ == "__main__":
    main()
