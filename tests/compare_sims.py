"""
Compare move selection across simulation budgets against both Stockfish depth=25 and 40000-sim MCTS.
"""

import os, sys, time, gc
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
SF_PATH = r"stockfish-windows-x86-64-avxvnni.exe"


def sf_best_move(board, sf, depth=25):
    """Return Stockfish's best move and its score (cp, STM)."""
    info = sf.play(board, chess.engine.Limit(depth=depth))
    best = info.move
    # eval after playing best move
    b = board.copy()
    b.push(best)
    info2 = sf.analyse(b, chess.engine.Limit(depth=depth))
    score = info2["score"].pov(chess.WHITE)
    return best, score.score(mate_score=10000)


def eval_move(board, move, sf, depth=25):
    """Evaluate position after `move`, return centipawn score (STM)."""
    b = board.copy()
    b.push(move)
    info = sf.analyse(b, chess.engine.Limit(depth=depth))
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
    model_path = "data/models/model_sf_ema.pt"

    print(f"Loading model: {model_path}")
    model = load_model(model_path, config).cuda().eval()
    engine = get_mcts_engine(model, config)
    engine._ensure_pool()

    print(f"Starting Stockfish: {SF_PATH}")
    sf = chess.engine.SimpleEngine.popen_uci(SF_PATH)
    sf.configure({"Threads": 4, "Hash": 512})

    for game_idx in range(3):
        print(f"\n{'='*80}")
        print(f"Game {game_idx+1}/3")
        print(f"{'='*80}")

        board = chess.Board()
        for pos_idx in range(10):
            if board.is_game_over():
                break

            fen = board.fen()
            stm = "w" if board.turn == chess.WHITE else "b"
            print(f"\n  [{pos_idx+1}] move {board.fullmove_number}{stm}  {fen[:60]}")

            # --- Reference: Stockfish best move + score ---
            sf_ref_move, sf_ref_score = sf_best_move(board, sf)
            print(f"    Stockfish d25: {sf_ref_move.uci():6s}  SF={sf_ref_score:+5d}")

            # --- Run all MCTS budgets ---
            results = {}
            for sims in SIM_BUDGETS:
                gc.collect()
                result = engine.search(board.copy(), num_simulations=sims)
                mv, prob = pick_move(result, board)
                results[sims] = (mv, prob)

            ground_move, _ = results[40000]
            if ground_move is None:
                break
            sf_ground = eval_move(board, ground_move, sf)
            print(f"    MCTS 40000:   {ground_move.uci():6s}  SF={sf_ground:+5d}  (dSF={sf_ground-sf_ref_score:+4d})")

            # --- Compare each budget ---
            for sims in SIM_BUDGETS[:-1]:
                mv, prob = results[sims]
                if mv is None:
                    continue
                sf_score = eval_move(board, mv, sf)
                match_40000 = "OK" if mv == ground_move else "XX"
                match_sf = "OK" if mv == sf_ref_move else "XX"
                d_sf = sf_score - sf_ref_score
                d_40000 = sf_score - sf_ground
                worse = " <<<" if d_sf < -30 else ""
                better = " >>>" if d_sf > 30 else ""
                print(f"    {sims:5d} sims: {mv.uci():6s}  p={prob*100:5.1f}%  "
                      f"SF={sf_score:+5d} (dSF={d_sf:+4d}, d40k={d_40000:+4d})  "
                      f"vsSF={match_sf} vs40k={match_40000}{worse}{better}", flush=True)

            # Advance with 40000-sim move
            board.push(ground_move)

    sf.quit()
    engine.shutdown()


if __name__ == '__main__':
    evaluate()
