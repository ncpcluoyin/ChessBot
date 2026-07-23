"""
从 Lichess/chess-position-evaluations 流式下载, 只保存易位走法样本。
配合原有 8000 万数据集通过 castling_ratio 混合使用。
"""
import os, sys, gc, time
import numpy as np
import torch
import chess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "castling_samples")
BATCH_SIZE = 5000
TARGET_FILES = 9999  # 跑完整个数据集
CASTLING_UCIS = {'e1g1', 'e1c1', 'e8g8', 'e8c8'}


def cp_to_value(cp):
    return max(-1.0, min(1.0, cp / 1000.0))


def convert_row(row):
    """返回 (fen, [(idx, 1.0)], value) 或 None"""
    fen = row["fen"]
    uci_line = row.get("line", "")
    if not fen or not uci_line:
        return None
    parts = uci_line.strip().split()
    if not parts or parts[0] not in CASTLING_UCIS:
        return None
    cp = row.get("cp")
    mate = row.get("mate")
    if cp is None and mate is None:
        return None
    try:
        board = chess.Board(fen)
        move = chess.Move.from_uci(parts[0])
        if move not in board.legal_moves:
            return None
        from src.board import move_to_index
        idx = int(move_to_index(move, board))
    except:
        return None
    value_stm = cp_to_value(cp) if cp is not None else 0.99
    value = value_stm if fen.split()[1] == 'w' else -value_stm
    return (fen, [(idx, 1.0)], value)


def download(skip=80000000):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from datasets import load_dataset

    ds = load_dataset("Lichess/chess-position-evaluations", split="train", streaming=True)
    if skip > 0:
        print(f"Skipping first {skip} positions...")
        ds = ds.skip(skip)
    
    batch, batch_n, found, scanned = [], 0, 0, 0
    no_line, not_castling, no_eval, illegals = 0, 0, 0, 0
    
    for row in ds:
        scanned += 1
        if scanned % 500000 == 0:
            el = time.time() - t0
            print(f"  scanned {scanned}  found {found}  | no_line={no_line} not_castling={not_castling} no_eval={no_eval} illegal={illegals}  | {scanned/el:.0f} pos/s")

        fen = row["fen"]
        uci_line = row.get("line", "")
        if not fen or not uci_line:
            no_line += 1
            continue
        parts = uci_line.strip().split()
        if not parts or parts[0] not in CASTLING_UCIS:
            not_castling += 1
            continue
        cp = row.get("cp")
        mate = row.get("mate")
        if cp is None and mate is None:
            no_eval += 1
            continue
        try:
            board = chess.Board(fen)
            move = chess.Move.from_uci(parts[0])
            if move not in board.legal_moves:
                illegals += 1
                continue
            from src.board import move_to_index
            idx = int(move_to_index(move, board))
        except:
            illegals += 1
            continue
        value_stm = cp_to_value(cp) if cp is not None else 0.99
        value = value_stm if fen.split()[1] == 'w' else -value_stm
        
        found += 1
        batch.append((fen, [(idx, 1.0)], value))

        if len(batch) >= BATCH_SIZE:
            batch_n += 1
            path = os.path.join(OUTPUT_DIR, f"castling_batch_{batch_n:04d}.pt")
            torch.save({'data': batch, 'game_lens': [1]*len(batch)}, path)
            print(f"  batch {batch_n}: {len(batch)} samples, {found} total")
            batch = []
            gc.collect()

        if batch_n >= TARGET_FILES:
            break

    if batch:
        batch_n += 1
        path = os.path.join(OUTPUT_DIR, f"castling_batch_{batch_n:04d}.pt")
        torch.save({'data': batch, 'game_lens': [1]*len(batch)}, path)
        print(f"  batch {batch_n}: {len(batch)} samples")

    el = time.time() - t0
    print(f"\nDone: scanned {scanned}, saved {found} castling in {batch_n} files")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Time: {el:.0f}s")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--skip', type=int, default=80000000,
                   help='Skip N positions (default 80M = existing data)')
    args = p.parse_args()
    download(skip=args.skip)
