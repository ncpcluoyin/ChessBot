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


def _save_ema_model(model, trainable_params, ema_params, ema_path):
    """将 EMA 权重写入模型副本并保存。"""
    import copy
    model_cpu = copy.deepcopy(model).cpu()
    state = model_cpu.state_dict()
    ema_idx = 0
    for name, param in model_cpu.named_parameters():
        if ema_idx < len(ema_params):
            state[name].copy_(ema_params[ema_idx])
            ema_idx += 1
    model_cpu.load_state_dict(state)
    torch.save(model_cpu.state_dict(), ema_path)
    del model_cpu


def train_selfplay(model, game_dir, config, epochs=5, batch_size=512, lr=0.001,
                   cleanup=False, max_samples=50000, ema_decay=0.999):
    """Load self-play games from game_dir and train. If cleanup, delete .pt/.pgn after training."""
    files = sorted(glob.glob(os.path.join(game_dir, '*.pt')))
    if not files:
        print(f"No .pt files found in {game_dir}")
        return 0.0, None, None

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
        return 0.0, None, None

    print(f"Loaded {len(samples)} samples from {len(files)} games")

    model.train()
    model = model.to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # ── EMA 初始化 ──
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    ema_params = [p.detach().clone() for p in trainable_params]

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

            # ── EMA 更新 ──
            for p, ema_p in zip(trainable_params, ema_params):
                ema_p.mul_(ema_decay).add_(p.detach(), alpha=1 - ema_decay)

            total_loss += loss.item()
            n_batches += 1

        avg = total_loss / max(n_batches, 1)
        pol_avg = float(-(td * pol).sum(dim=-1).detach().mean())
        val_avg = float(((v_pred - v_label) ** 2).detach().mean())
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

    return total_loss / max(n_batches, 1), ema_params, trainable_params


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
    loss, ema_params, trainable_params = train_selfplay(
        model, args.data, config, epochs=args.epochs, lr=args.lr,
        cleanup=args.cleanup, max_samples=args.max_samples)
    if ema_params is None:
        print("No data, skipping model save")
        sys.exit(0)
    save_model(model, args.model)
    print(f"Model saved to {args.model}")

    # 保存 EMA 模型
    ema_path = args.model.replace(".pt", "_ema.pt")
    _save_ema_model(model, trainable_params, ema_params, ema_path)
    print(f"EMA model saved to {ema_path}")
