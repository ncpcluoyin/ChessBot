"""
Compare move selection quality across simulation budgets.
40000 sims = ground truth. Stockfish depth=25 eval for each candidate move.
"""

import os, sys, time, gc, subprocess
import numpy as np
import chess
import chess.engine
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import Config
from src.network import load_model
from src.board import move_to_index
from src.mcts import get_mcts_engine

torch.set_float32_matmul_precision('high')

SIM_BUDGETS = [100, 200, 400, 1000, 4000, 10000, 40000]
GROUND = 40000
SF_PATH = r"stockfish-windows-x86-64-avxvnni.exe"


def score_with_sf(board, move, sf_engine, depth=25):
    """Evaluate position after `move` using Stockfish at `depth`, return centipawn score (STM)."""
    b = board.copy()
    b.push(move)
    info = sf_engine.analyse(b, chess.engine.Limit(depth=depth))
    score = info["score"].pov(chess.WHITE)
    return score.score(mate_score=10000)


def pick_move(result, board):
    legals = list(board.legal_moves)
    if not legals or result is None or result.policy is None:
        return None, 0.0
    indices = [move_to_index(mv, board) for mv in legals]
    best = max(indices, key=lambda i: result.policy[i])
    return legals[indices.index(best)], float(result.policy[best])


def evaluate():
    config = Config()
    config.num_mcts_workers = 8
    model_path = "data/models/model_sf.pt"

    print(f"Loading model: {model_path}")
    model = load_model(model_path, config).cuda().eval()
    engine = get_mcts_engine(model, config)
    engine._ensure_pool()

    print(f"Starting Stockfish: {SF_PATH}")
    sf = chess.engine.SimpleEngine.popen_uci(SF_PATH)
    sf.configure({"Threads": 4, "Hash": 512})

    for game_idx in range(3):
        print(f"\n{'='*70}")
        print(f"Game {game_idx+1}/3")
        print(f"{'='*70}")

        board = chess.Board()
        for pos_idx in range(10):
            if board.is_game_over():
                break

            fen = board.fen()
            stm = "w" if board.turn == chess.WHITE else "b"
            print(f"\n  [{pos_idx+1}] move {board.fullmove_number}{stm}  {fen[:60]}")

            # Run all sim budgets at this position
            results = {}
            for sims in SIM_BUDGETS:
                gc.collect()
                result = engine.search(board.copy(), num_simulations=sims)
                mv, prob = pick_move(result, board)
                results[sims] = (mv, prob)

            ground_move, _ = results[GROUND]
            if ground_move is None:
                print("    (no legal move, skipping)")
                break

            # Score ground-truth move with Stockfish
            sf_ground = score_with_sf(board, ground_move, sf)
            print(f"    {GROUND:5d} sims: {ground_move.uci():6s}  SF={sf_ground:+5d}  (ground truth)")

            for sims in SIM_BUDGETS[:-1]:
                mv, prob = results[sims]
                if mv is None:
                    continue
                sf_score = score_with_sf(board, mv, sf)
                match = "OK" if mv == ground_move else "XX"
                delta = sf_score - sf_ground
                worse = " <<<" if delta < -30 else (" >>>" if delta > 30 else "")
                print(f"    {sims:5d} sims: {mv.uci():6s}  p={prob*100:5.1f}%  "
                      f"SF={sf_score:+5d} (d={delta:+4d})  {match}{worse}", flush=True)

            # Advance with ground-truth move
            board.push(ground_move)

    sf.quit()
    engine.shutdown()


if __name__ == '__main__':
    evaluate()
