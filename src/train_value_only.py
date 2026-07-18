"""
Train ONLY the value head on self-play game outcomes.
Backbone + policy head are frozen.
Value head is reinitialized before training.
"""

import os, sys, glob, random, copy
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import Config
from src.network import load_model, save_model, ChessNet
from src.board import board_to_tensor

torch.set_float32_matmul_precision('high')


def reinit_value_head(model):
    """Reinitialize value head conv + fc layers with fresh weights."""
    for name, mod in model.named_modules():
        # value head specific layers
        if 'value_head' in name:
            if isinstance(mod, (torch.nn.Conv2d, torch.nn.Linear)):
                torch.nn.init.orthogonal_(mod.weight, gain=1.0)
                if mod.bias is not None:
                    torch.nn.init.zeros_(mod.bias)
            elif isinstance(mod, torch.nn.BatchNorm2d):
                mod.reset_running_stats()
                torch.nn.init.constant_(mod.weight, 1.0)
                torch.nn.init.zeros_(mod.bias)
    print("Value head reinitialized.")


def freeze_backbone(model):
    """Freeze all params except value head."""
    for name, p in model.named_parameters():
        if 'value_head' in name:
            p.requires_grad = True
        else:
            p.requires_grad = False
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Frozen: {n_frozen/1e6:.1f}M  Trainable (value): {n_train/1e3:.0f}K")


def train_value_head(model, game_dir, config, epochs=20, batch_size=512, lr=0.001, max_samples=100000):
    """Train only value head on self-play game results."""
    files = sorted(glob.glob(os.path.join(game_dir, '*.pt')))
    if not files:
        print(f"No .pt files found in {game_dir}")
        return

    samples = []
    for fp in files:
        if max_samples > 0 and len(samples) >= max_samples:
            break
        data = torch.load(fp, map_location='cpu', weights_only=False)
        for s in data['samples']:
            if max_samples > 0 and len(samples) >= max_samples:
                break
            samples.append((s['tensor'], s['value']))

    if len(samples) < batch_size:
        print(f"Only {len(samples)} samples")
        return

    print(f"Loaded {len(samples)} samples from {len(files)} games")

    model.train()
    model = model.to(config.device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=1e-4
    )

    indices = list(range(len(samples)))
    for epoch in range(epochs):
        random.shuffle(indices)
        total_loss = 0.0
        n_batches = 0
        for start in range(0, len(indices), batch_size):
            sel = [samples[i] for i in indices[start:start+batch_size]]
            inputs = torch.stack([s[0] for s in sel]).to(config.device)
            v_label = torch.tensor([s[1] for s in sel], dtype=torch.float32).to(config.device)

            optimizer.zero_grad()
            _, v_pred = model(inputs)
            v_pred = v_pred.squeeze(-1)
            loss = ((v_pred - v_label) ** 2).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg = total_loss / max(n_batches, 1)
        print(f"  epoch {epoch+1}/{epochs}: loss={avg:.6f}", flush=True)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", default="data/self_play_games")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--max-samples", type=int, default=100000)
    args = p.parse_args()

    config = Config()
    config.device = 'cuda'

    model = load_model(args.model, config).cuda()
    reinit_value_head(model)
    freeze_backbone(model)
    train_value_head(model, args.data, config, epochs=args.epochs, lr=args.lr,
                     max_samples=args.max_samples)

    # Save as a separate model for testing
    out_path = args.model.replace(".pt", "_value_trained.pt")
    save_model(model, out_path)
    print(f"Saved to {out_path}")
