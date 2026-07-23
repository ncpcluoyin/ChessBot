"""
从 Lichess/chess-position-evaluations 流式下载易位走法样本。
只保留 Stockfish top-1 为易位的局面, 20x 扩增。
"""
import os, sys, gc, time
import numpy as np
import torch
import chess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "castling_samples")
BATCH_SIZE = 5000
OVER_SAMPLE = 20
TARGET_CASTLING = 80000  # 目标易位样本数 (扩增前, 约扫4000万局面)
MAX_SCAN = 40_000_000

CASTLING_UCIS = {'e1g1', 'e1c1', 'e8g8', 'e8c8'}


def cp_to_value(cp):
    """Centipawn to [-1, 1]."""
    return max(-1.0, min(1.0, cp / 1000.0))


def convert_row(row):
    """Convert one row. Returns (fen, [(idx, 1.0)], value) or None."""
    fen = row["fen"]
    uci_line = row.get("line", "")
    if not fen or not uci_line:
        return None

    parts = uci_line.strip().split()
    if not parts:
        return None
    best_uci = parts[0]

    # 快速过滤: 只有易位走法才继续
    if best_uci not in CASTLING_UCIS:
        return None

    cp = row.get("cp")
    mate = row.get("mate")
    if cp is None and mate is None:
        return None

    try:
        board = chess.Board(fen)
        move = chess.Move.from_uci(best_uci)
        if move not in board.legal_moves:
            return None
    except:
        return None

    # 用现有 board.py 的转换
    try:
        from src.board import move_to_index
        idx = move_to_index(move, board)
    except:
        return None

    moves_probs = [(int(idx), 1.0)]

    if cp is not None:
        value_stm = cp_to_value(cp)
    else:
        value_stm = 0.99

    # 保持白方视角
    value = value_stm if fen.split()[1] == 'w' else -value_stm
    return (fen, moves_probs, value)


def download():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from datasets import load_dataset

    ds = load_dataset(
        "Lichess/chess-position-evaluations",
        split="train",
        streaming=True,
    )

    batch_out = []
    batch_n = 0
    found = 0
    scanned = 0
    t0 = time.time()

    for row in ds:
        scanned += 1
        if scanned > MAX_SCAN:
            break
        if scanned % 100000 == 0:
            el = time.time() - t0
            print(f"  scanned {scanned}, found {found} castling, {scanned/el:.0f} pos/s")

        r = convert_row(row)
        if r is None:
            continue
        found += 1

        for _ in range(OVER_SAMPLE):
            batch_out.append(r)

        if len(batch_out) >= BATCH_SIZE:
            batch_n += 1
            path = os.path.join(OUTPUT_DIR, f"castling_batch_{batch_n:04d}.pt")
            torch.save({'data': batch_out, 'game_lens': [1]*len(batch_out)}, path)
            print(f"  batch {batch_n}: {len(batch_out)} samples, {found} castling total")
            batch_out = []
            gc.collect()

        if found >= TARGET_CASTLING:
            break

    if batch_out:
        batch_n += 1
        path = os.path.join(OUTPUT_DIR, f"castling_batch_{batch_n:04d}.pt")
        torch.save({'data': batch_out, 'game_lens': [1]*len(batch_out)}, path)
        print(f"  batch {batch_n}: {len(batch_out)} samples")

    el = time.time() - t0
    print(f"\nDone: scanned {scanned}, found {found} castling")
    print(f"After {OVER_SAMPLE}x oversample: {found * OVER_SAMPLE} samples → {OUTPUT_DIR}")
    print(f"Time: {el:.0f}s")


if __name__ == '__main__':
    download()
