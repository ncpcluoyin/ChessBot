"""
Train on self-play games (saved by self_play.py).
"""

import os, sys, glob, random
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import Config
from src.network import load_model, save_model
from src.board import board_to_tensor

torch.set_float32_matmul_precision('high')


def train_selfplay(model, game_dir, config, epochs=5, batch_size=512, lr=0.001, cleanup=False, max_samples=50000):
    """Load self-play games from game_dir and train. If cleanup, delete .pt/.pgn after training."""
    files = sorted(glob.glob(os.path.join(game_dir, '*.pt')))
    if not files:
        print(f"No .pt files found in {game_dir}")
        return 0.0

    # Load samples, limit to max_samples
    samples = []
    for fp in files:
        if max_samples > 0 and len(samples) >= max_samples:
            break
        data = torch.load(fp, map_location='cpu', weights_only=False)
        for s in data['samples']:
            if max_samples > 0 and len(samples) >= max_samples:
                break
            samples.append((s['tensor'], s['policy'], s['value']))

    if len(samples) < batch_size:
        print(f"Only {len(samples)} samples, need at least {batch_size}")
        return 0.0

    print(f"Loaded {len(samples)} samples from {len(files)} games")

    model.train()
    model = model.to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    indices = list(range(len(samples)))
    total_loss = 0.0
    n_batches = 0

    for epoch in range(epochs):
        random.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            sel = [samples[i] for i in indices[start:start+batch_size]]
            inputs = torch.stack([s[0] for s in sel]).to(config.device)
            td = torch.stack([s[1] for s in sel]).to(config.device)
            v_label = torch.tensor([s[2] for s in sel], dtype=torch.float32).to(config.device)

            optimizer.zero_grad()
            pol, v_pred = model(inputs)
            v_pred = v_pred.squeeze(-1)
            pol_loss = -(td * pol).sum(dim=-1).mean()
            val_loss = ((v_pred - v_label) ** 2).mean()
            loss = pol_loss + 12.0 * val_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg = total_loss / max(n_batches, 1)
        pol_avg = float(-(td * pol).sum(dim=-1).mean())
        val_avg = float(((v_pred - v_label) ** 2).mean())
        print(f"  epoch {epoch+1}/{epochs}: loss={avg:.4f}  policy={pol_avg:.4f}  value={val_avg:.4f}",
              flush=True)

    # Cleanup: delete .pt and .pgn files after training
    if cleanup and files:
        n_pt = len(glob.glob(os.path.join(game_dir, '*.pt')))
        n_pgn = len(glob.glob(os.path.join(game_dir, '*.pgn')))
        for f in glob.glob(os.path.join(game_dir, '*.pt')):
            os.remove(f)
        for f in glob.glob(os.path.join(game_dir, '*.pgn')):
            os.remove(f)
        print(f"Cleaned {n_pt} .pt + {n_pgn} .pgn files")

    return total_loss / max(n_batches, 1)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", default="data/self_play_games")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--max-samples", type=int, default=50000, help="Max samples to load (0=unlimited)")
    p.add_argument("--cleanup", action="store_true", help="Delete old .pt/.pgn after training")
    args = p.parse_args()

    config = Config()
    config.device = 'cuda'
    model = load_model(args.model, config).cuda()
    train_selfplay(model, args.data, config, epochs=args.epochs, lr=args.lr,
                   cleanup=args.cleanup, max_samples=args.max_samples)
    save_model(model, args.model)
    print(f"Model saved to {args.model}")
