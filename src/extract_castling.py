"""
从 HF 数据中提取易位走法样本 (e1g1/e1c1/e8g8/e8c8), 重复扩增后保存。
"""
import os, sys, glob, gc
import numpy as np
import torch
import chess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.board import move_to_index

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
HF_DIR = os.path.join(DATA_DIR, "hf_supervised_samples")
OUT_DIR = os.path.join(DATA_DIR, "castling_samples")
OVER_SAMPLE = 20  # 每个易位样本重复 20 倍

CASTLING_MOVES = {chess.Move.from_uci(u) for u in ('e1g1', 'e1c1', 'e8g8', 'e8c8')}


def extract():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(HF_DIR, "hf_batch_*.pt")))
    total = len(files)
    castling_count = 0
    batch_out = []
    batch_n = 0

    for fi, f in enumerate(files):
        data = torch.load(f, map_location='cpu', weights_only=True)
        items = data['data']

        for fen, policy_list, value in items:
            try:
                board = chess.Board(fen)
            except:
                continue

            # 找出当前局面中的易位走法索引
            legal_castling = set()
            for mv in board.legal_moves:
                if mv in CASTLING_MOVES:
                    legal_castling.add(move_to_index(mv, board))

            if not legal_castling:
                continue
            if not policy_list:
                continue

            # Stockfish top-1 是否是易位?
            top_idx = policy_list[0][0]
            if top_idx not in legal_castling:
                continue

            # 重复扩增
            for _ in range(OVER_SAMPLE):
                batch_out.append((fen, policy_list, value))
                castling_count += 1

            if len(batch_out) >= 5000:
                batch_n += 1
                path = os.path.join(OUT_DIR, f"castling_batch_{batch_n:04d}.pt")
                torch.save({'data': batch_out, 'game_lens': [1] * len(batch_out)}, path)
                print(f"  Saved {path} ({len(batch_out)} samples)")
                batch_out = []
                gc.collect()

        if (fi + 1) % 200 == 0:
            print(f"  [{fi+1}/{total}] -> {castling_count} castling samples")

    if batch_out:
        batch_n += 1
        path = os.path.join(OUT_DIR, f"castling_batch_{batch_n:04d}.pt")
        torch.save({'data': batch_out, 'game_lens': [1] * len(batch_out)}, path)
        print(f"  Saved {path} ({len(batch_out)} samples)")

    print(f"\nDone: {castling_count} castling samples in {batch_n} files → {OUT_DIR}")
    return castling_count


if __name__ == '__main__':
    extract()
