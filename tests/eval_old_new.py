"""
新旧模型对战验证。
加载 model_sf.pt (新) vs model_sf_old.pt (旧), 各执先手 N 盘, 报告战绩。
"""

import os, sys, gc, time
import numpy as np
import chess
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import Config
from src.network import load_model
from src.board import move_to_index
from src.mcts import get_mcts_engine

torch.set_float32_matmul_precision('high')


def _pick_move(policy, board):
    """贪心选合法走法中概率最高的."""
    legals = list(board.legal_moves)
    if not legals:
        return None
    indices = [move_to_index(mv, board) for mv in legals]
    best = max(indices, key=lambda i: policy[i])
    return legals[indices.index(best)]


def play_game(engine_new, engine_old, model_new_turn=True, sims=800, max_moves=200):
    """model_new_turn=True: 新模型执先手."""
    board = chess.Board()
    move_count = 0

    while not board.is_game_over() and move_count < max_moves:
        engine = engine_new if (board.turn == chess.WHITE) == model_new_turn else engine_old
        result = engine.search(board, num_simulations=sims)
        if result is None or result.policy is None:
            break
        move = _pick_move(result.policy, board)
        if move is None:
            break
        board.push(move)
        move_count += 1

    outcome = board.outcome()
    if outcome is None:
        return None  # 超步数平局
    if outcome.winner == chess.WHITE:
        return 'new' if model_new_turn else 'old'
    elif outcome.winner == chess.BLACK:
        return 'old' if model_new_turn else 'new'
    else:
        return 'draw'


def evaluate(model_new_path, model_old_path, games=20, sims=400):
    config = Config()
    config.num_mcts_workers = 8

    print(f"Loading: {model_new_path}")
    model_new = load_model(model_new_path, config).cuda().eval()
    print(f"Loading: {model_old_path}")
    model_old = load_model(model_old_path, config).cuda().eval()

    engine_new = get_mcts_engine(model_new, config)
    engine_new._ensure_pool()
    engine_old = get_mcts_engine(model_old, config)
    engine_old._ensure_pool()

    half = games // 2
    total = {'new': 0, 'old': 0, 'draw': 0}
    t0 = time.time()

    print(f"\n{'='*50}")
    print(f"新 vs 旧: {games} 局, {sims} sims")
    print(f"{'='*50}")

    for g in range(games):
        new_turn = (g < half)
        label = "新执先" if new_turn else "新执黑"
        result = play_game(engine_new, engine_old, new_turn, sims)
        if result is None:
            print(f"  局 {g+1}/{games} [{label}] = 平局 (超步数)", flush=True)
            total['draw'] += 1
        else:
            total[result] += 1
            winner = "新" if result == 'new' else "旧"
            print(f"  局 {g+1}/{games} [{label}] = {winner} 胜", flush=True)

    elapsed = time.time() - t0
    new_score = total['new'] + 0.5 * total['draw']
    old_score = total['old'] + 0.5 * total['draw']
    new_pct = new_score / games * 100

    print(f"\n{'='*50}")
    print(f"新 {total['new']} 胜 / 旧 {total['old']} 胜 / 平 {total['draw']}")
    print(f"新评分: {new_score:.1f}/{games} = {new_pct:.1f}%")
    print(f"耗时: {elapsed:.0f}s = {elapsed/games:.1f}s/局")

    engine_new.shutdown()
    engine_old.shutdown()

    return new_pct


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--new", default="data/models/model_sf.pt")
    p.add_argument("--old", default="data/models/model_sf_old.pt")
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--sims", type=int, default=400)
    args = p.parse_args()

    for f in [args.new, args.old]:
        if not os.path.exists(f):
            print(f"File not found: {f}")
            sys.exit(1)

    evaluate(args.new, args.old, args.games, args.sims)
