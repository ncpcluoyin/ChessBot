# ChessBot

A chess engine built with pure CNN + MCTS, trained via Stockfish distillation.

## Architecture

| Component | Detail |
|-----------|--------|
| Backbone | 10 InceptionResBlocks, 512 filters |
| Conv types | 3×2 (horizontal) + 2×3 (vertical), alternating per layer |
| Params | 22.63M |
| Board encoding | 19 planes, rank-flip (STM perspective) |
| Policy head | Conv1×1 512→64 → FC 4096→512 → 4672, log_softmax |
| Value head | Conv1×1 512→64 → FC 4096→512 → 256 → 1, tanh |
| MCTS | GPU batched, 2000 sims default, 12 workers |
| Intuition mode | NN-only (no MCTS), legal-mask filtered |

## Files

- `src/network.py` — ChessNet model definition
- `src/board.py` — 19-plane rank-flip encoding, 4672 move encoding
- `src/train.py` — Distillation training loop (EMA, cosine LR, balanced sampling)
- `src/sf_dataset.py` — Online 63-sq → rank-flip conversion + balanced batch sampling
- `src/mcts/` — GPU batched MCTS engine, persistent workers
- `src/uci.py` — UCI protocol handler with IntuitionMode option
- `scripts/download_hf_dataset.py` — Download HuggingFace supervised dataset

## Training

```bash
distill_daemon.bat
```

Trains on HuggingFace Stockfish data (800K games), default 1800 epochs, batch_size=512.

## Usage

```bash
# UCI mode (MCTS)
chessbot_fp.bat

# Intuition mode (NN-only, no MCTS)
chessbot_intuition.bat
```

## UCI Options

- `IntuitionMode` — true/false, bypass MCTS for direct NN policy
- `BatchSize` — GPU batch size (default 32)
- `Simulations` — MCTS simulations (default 2000)
- `NumWorkers` — MCTS worker threads (default 12)

## Checkpoints

- `model_sf.pt` — Latest weights
- `model_sf_ema.pt` — Exponential Moving Average weights (0.999 decay)
- `model_sf_checkpoint.pt` — Full training state (optimizer, epoch, total_epochs)
