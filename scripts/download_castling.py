"""
从 Lichess/chess-position-evaluations 流式下载, 只保存易位走法样本。
索引用 63-sq 编码, 与现有 8000 万数据集兼容。
"""
import os, sys, gc, time, math
import numpy as np
import torch
import chess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "castling_samples")
BATCH_SIZE = 5000
TARGET_FILES = 9999

# ── 63-sq 编码 (兼容旧数据格式, SFDistillDataset._convert_moves 会转成新格式) ──
_QUEEN_DIRS = [(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1)]
_KNIGHT_OFFS = [(-1,-2),(-1,2),(-2,-1),(-2,1),(1,-2),(1,2),(2,-1),(2,1)]
_UNDERPROS = [(-1,chess.KNIGHT),(-1,chess.BISHOP),(-1,chess.ROOK),
              (0,chess.KNIGHT),(0,chess.BISHOP),(0,chess.ROOK),
              (1,chess.KNIGHT),(1,chess.BISHOP),(1,chess.ROOK)]

def _63sq_move_to_index(move, board):
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
    raise ValueError(f"cannot encode {move.uci()}")

K = 271.6
def cp_to_value(cp):
    cp = max(min(cp, 10000), -10000)
    win_prob = 1.0 / (1.0 + math.exp(-cp / K))
    return 2.0 * win_prob - 1.0


def download(skip=0):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from datasets import load_dataset

    ds = load_dataset("Lichess/chess-position-evaluations", split="train", streaming=True)
    if skip > 0:
        print(f"Skipping first {skip} positions...")
        ds = ds.skip(skip)
    else:
        print("Starting from beginning...")

    batch, batch_n, found, scanned = [], 0, 0, 0
    no_line, not_castling, no_eval, illegals = 0, 0, 0, 0
    t0 = time.time()

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
        if not parts:
            not_castling += 1
            continue
        try:
            mv = chess.Move.from_uci(parts[0])
            board = chess.Board(fen)
            if not board.is_castling(mv) or mv not in board.legal_moves:
                not_castling += 1
                continue
        except:
            illegals += 1
            continue
        cp = row.get("cp")
        mate = row.get("mate")
        if cp is None and mate is None:
            no_eval += 1
            continue
        try:
            idx = int(_63sq_move_to_index(mv, board))
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
    p.add_argument('--skip', type=int, default=0, help='Skip N positions')
    args = p.parse_args()
    download(skip=args.skip)
