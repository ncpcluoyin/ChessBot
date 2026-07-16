"""
Download Lichess chess-position-evaluations from HuggingFace and
convert to ChessBot training format.

Dataset columns: fen, line (PV UCI), cp (centipawn), mate (mate in N)

Usage:
  .venv311\Scripts\python.exe scripts\download_hf_dataset.py ^
      --num-positions 2000000 --output data\hf_supervised_samples

The output files use the same format as sf_supervised_samples:
  hf_batch_0000.pt, hf_batch_0001.pt, ...
"""

import argparse, math, os, sys, time
import torch
import chess
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── 63-sq 编码常数 (硬编码, 不依赖 board.py) ──
_QUEEN_DIRS = [(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1)]
_KNIGHT_OFFS = [(-1,-2),(-1,2),(-2,-1),(-2,1),(1,-2),(1,2),(2,-1),(2,1)]
_UNDERPROS = [(-1,chess.KNIGHT),(-1,chess.BISHOP),(-1,chess.ROOK),
              (0,chess.KNIGHT),(0,chess.BISHOP),(0,chess.ROOK),
              (1,chess.KNIGHT),(1,chess.BISHOP),(1,chess.ROOK)]

def _63sq_move_to_index(move, board):
    """63-sq 编码: 黑走棋时 180°旋转 (63 - sq)。"""
    from_sq = move.from_square; to_sq = move.to_square
    if board.turn == chess.BLACK:
        from_sq, to_sq = 63 - from_sq, 63 - to_sq
    dx = chess.square_file(to_sq) - chess.square_file(from_sq)
    dy = chess.square_rank(to_sq) - chess.square_rank(from_sq)
    def _qi():
        for d,(ddx,ddy) in enumerate(_QUEEN_DIRS):
            dist = max(abs(dx),abs(dy))
            if dist==0 or dist>7: continue
            if ddx*dist==dx and ddy*dist==dy: return d*7+dist-1
        return None
    def _ki():
        for k,(kdx,kdy) in enumerate(_KNIGHT_OFFS):
            if kdx==dx and kdy==dy: return 56+k
        return None
    def _ui():
        for u,(udx,upromo) in enumerate(_UNDERPROS):
            if udx==dx and upromo==promo: return 64+u
        return None
    for promo in [move.promotion]:
        if promo == chess.QUEEN:
            qi = _qi()
            if qi is not None: return from_sq*73+qi
        if promo and promo in (chess.KNIGHT,chess.BISHOP,chess.ROOK):
            ui = _ui()
            if ui is not None: return from_sq*73+ui
    qi = _qi()
    if qi is not None: return from_sq*73+qi
    ki = _ki()
    if ki is not None: return from_sq*73+ki
    raise ValueError(f"无法编码 {move.uci()}")


# ── cp → value conversion (same formula as training sigmoid inverse) ──

K = 271.6  # 1/0.003682

def cp_to_value(cp):
    """Convert SF cp to value in [-1, 1]."""
    cp = max(min(cp, 10000), -10000)
    win_prob = 1.0 / (1.0 + math.exp(-cp / K))
    return 2.0 * win_prob - 1.0


def convert_row(row):
    """Convert one dataset row to (fen, [(move_idx, prob)], value)."""
    fen = row["fen"]
    cp = row.get("cp")
    mate = row.get("mate")
    uci_line = row.get("line", "")

    # Skip if no evaluation at all
    if not fen or not uci_line:
        return None

    # Need either cp or mate
    if cp is None and mate is None:
        return None

    # Extract first UCI move from PV line
    parts = uci_line.strip().split()
    if not parts:
        return None
    best_uci = parts[0]

    board = chess.Board(fen)
    try:
        move = chess.Move.from_uci(best_uci)
    except Exception:
        return None
    if move not in board.legal_moves:
        return None

    # Policy: one-hot on best move
    try:
        idx = _63sq_move_to_index(move, board)
    except ValueError:
        return None
    moves_probs = [(idx, 1.0)]

    # Value: mate -> 0.99（走棋方可将杀）
    if cp is not None:
        value_stm = cp_to_value(cp)
    else:
        value_stm = 0.99

    # Store as white's perspective (dataset negates for black-turn positions)
    value = value_stm if fen.split()[1] == 'w' else -value_stm

    return (fen, moves_probs, value)


def download_and_convert(num_positions, output_dir, batch_size=5000, resume=False, skip=0):
    """Download dataset and convert to ChessBot .pt files."""
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    from datasets import load_dataset

    os.makedirs(output_dir, exist_ok=True)

    # ── Checkpoint ──
    ckpt_path = os.path.join(output_dir, "_download_ckpt.pt")
    downloaded_count = 0
    total_samples = 0
    file_idx = 0
    if resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        downloaded_count = ckpt.get("count", 0)
        total_samples = ckpt.get("count", 0)
        file_idx = ckpt.get("file_idx", 0)
        print(f"Resuming from position {downloaded_count}, file_idx={file_idx}")

    # 额外跳过 (用于验证集: 跳过训练数据部分)
    effective_skip = max(skip, downloaded_count)

    print(f"Downloading Lichess/chess-position-evaluations (streaming)...")
    ds = load_dataset(
        "Lichess/chess-position-evaluations",
        split="train",
        streaming=True,
    )

    # Skip already-downloaded positions (or skip offset for val set)
    if effective_skip > 0:
        print(f"Skipping first {effective_skip} positions...")
        ds = ds.skip(effective_skip)

    samples = []
    t0 = time.time()
    skipped = 0

    for row in ds:
        result = convert_row(row)
        if result is None:
            skipped += 1
            continue

        samples.append(result)
        total_samples += 1

        if len(samples) >= batch_size:
            _save_batch(samples, output_dir, file_idx)
            elapsed = time.time() - t0
            rate = (total_samples - effective_skip) / elapsed if elapsed > 0 else 0
            print(f"  [{total_samples - effective_skip}/{num_positions}] {file_idx+1} files  "
                  f"skipped={skipped}  {rate:.0f} pos/s  "
                  f"({elapsed:.0f}s)", flush=True)
            torch.save({"count": total_samples, "file_idx": file_idx + 1}, ckpt_path)
            samples.clear()
            file_idx += 1

        if total_samples >= num_positions + effective_skip:
            break

    if samples:
        _save_batch(samples, output_dir, file_idx)
        file_idx += 1

    elapsed = time.time() - t0
    new = total_samples - effective_skip
    torch.save({"count": total_samples, "file_idx": file_idx}, ckpt_path)
    print(f"\nDone: {new} new positions (total {total_samples}) in {file_idx} files  "
          f"{new/elapsed:.0f} pos/s  ({elapsed:.0f}s)")
    print(f"Saved to {output_dir}/hf_batch_*.pt")


def _save_batch(samples, output_dir, file_idx):
    """Save a batch as hf_batch_XXXX.pt in ChessBot format."""
    data = []
    game_lens = []

    chunk = 100
    for i in range(0, len(samples), chunk):
        chunk_data = samples[i:i + chunk]
        data.extend(chunk_data)
        game_lens.append(len(chunk_data))

    out = {"data": data, "game_lens": game_lens}
    path = os.path.join(output_dir, f"hf_batch_{file_idx:04d}.pt")
    torch.save(out, path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--num-positions", type=int, default=2000000,
                   help="Total positions to download")
    p.add_argument("--output", default="data/hf_supervised_samples",
                   help="Output directory")
    p.add_argument("--batch-size", type=int, default=5000,
                   help="Positions per .pt file")
    p.add_argument("--resume", action="store_true",
                   help="Resume from last download checkpoint")
    p.add_argument("--skip", type=int, default=0,
                   help="Skip N positions (for val set: skip training data)")
    args = p.parse_args()

    download_and_convert(args.num_positions, args.output, args.batch_size,
                         resume=args.resume, skip=args.skip)
