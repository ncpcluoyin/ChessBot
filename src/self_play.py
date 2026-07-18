"""
Self-play game generation. Saves games to disk, one .pt per game.
"""

import gc, os, signal, sys, time, hashlib, json, glob
import numpy as np
import torch
import chess

from src.board import board_to_tensor, move_to_index
from src.config import Config
from src.network import load_model
from src.mcts import get_mcts_engine

TEMP_THRESHOLD = 30
TEMPERATURE = 1.0
TOP_K_SAMPLE = 3
DIR_ALPHA = 0.3
DIR_EPSILON = 0.25
MIN_MOVES = 20
MAX_MOVES = 200


def play_one_game(mcts, config, stop_event=None):
    board = chess.Board()
    samples = []
    move_count = 0

    while not board.is_game_over() and move_count < MAX_MOVES:
        if stop_event and stop_event.is_set():
            break
        use_dd = move_count < TEMP_THRESHOLD
        result = mcts.search(board, num_simulations=config.self_play_simulations,
                              use_dirichlet=use_dd, stop_event=stop_event)
        if result is None or result.policy is None:
            break
        tensor = board_to_tensor(board)
        samples.append({
            'tensor': tensor,
            'policy': torch.from_numpy(result.policy.copy()).float(),
        })
        if move_count < TEMP_THRESHOLD:
            move = _sample_move(result.policy, board)
        else:
            move = _greedy_move(result.policy, board)
        if move is None:
            break
        board.push(move)
        move_count += 1
        if abs(result.root_value) > 5.0 and move_count > MIN_MOVES:
            break

    outcome = board.outcome()
    if outcome is None:
        val = result.root_value if result else 0.0
        white_result = val if board.turn == chess.WHITE else -val
        white_result = max(-1.0, min(1.0, white_result))
    elif outcome.winner == chess.WHITE:
        white_result = 1.0
    elif outcome.winner == chess.BLACK:
        white_result = -1.0
    else:
        white_result = 0.0

    # STM-perspective value labels
    for j, s in enumerate(samples):
        s['value'] = white_result if (j % 2 == 0) else -white_result
        s['result'] = white_result
        s['fen'] = board.fen()  # approximate, not exact per position

    return samples, white_result, move_count


def _sample_move(policy, board):
    legals = list(board.legal_moves)
    if not legals:
        return None
    indices = [move_to_index(mv, board) for mv in legals]
    probs = np.array([max(policy[idx], 0.0) for idx in indices], dtype=np.float64)
    if TOP_K_SAMPLE > 0 and TOP_K_SAMPLE < len(legals):
        top_idx = np.argpartition(probs, -TOP_K_SAMPLE)[-TOP_K_SAMPLE:]
        mask = np.zeros_like(probs, dtype=bool)
        mask[top_idx] = True
        probs[~mask] = 0.0
    s = probs.sum()
    if s <= 0:
        return legals[np.random.randint(len(legals))]
    probs /= probs.sum()
    return legals[np.random.choice(len(legals), p=probs)]


def _greedy_move(policy, board):
    legals = list(board.legal_moves)
    if not legals:
        return None
    indices = [move_to_index(mv, board) for mv in legals]
    best = max(indices, key=lambda i: policy[i])
    return legals[indices.index(best)]


def _game_hash(samples):
    h = hashlib.md5()
    for s in samples:
        h.update(s['tensor'].numpy().tobytes())
    return h.hexdigest()[:16]


def generate_games(mcts, config, num_games, output_dir, stop_event=None, verbose=True):
    os.makedirs(output_dir, exist_ok=True)
    existing = [f for f in os.listdir(output_dir) if f.startswith('game_') and f.endswith('.pt')]
    start_idx = 0
    if existing:
        nums = [int(f.replace('game_','').replace('.pt','')) for f in existing]
        start_idx = max(nums) + 1

    seen_hashes = set()
    stats = {'wins': 0, 'losses': 0, 'draws': 0, 'total_positions': 0}
    t0 = time.time()

    for i in range(num_games):
        if stop_event and stop_event.is_set():
            break
        gc.collect()
        samples, result, length = play_one_game(mcts, config, stop_event)
        if len(samples) < 5:
            continue
        gh = _game_hash(samples)
        if gh in seen_hashes:
            continue
        seen_hashes.add(gh)

        path = os.path.join(output_dir, f'game_{start_idx + i:04d}.pt')
        torch.save({
            'samples': samples,
            'result': result,
            'length': length,
            'hash': gh,
        }, path)

        stats['total_positions'] += len(samples)
        if result > 0.5: stats['wins'] += 1
        elif result < -0.5: stats['losses'] += 1
        else: stats['draws'] += 1

        if verbose:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{num_games}] {length:3d} pos  "
                  f"W/B/D {stats['wins']}/{stats['losses']}/{stats['draws']}  "
                  f"({elapsed:.0f}s)", flush=True)

    elapsed = time.time() - t0
    total = stats['wins'] + stats['losses'] + stats['draws']
    print(f"  Generated {total} games, {stats['total_positions']} positions, {elapsed:.0f}s",
          flush=True)
    return stats


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--sims", type=int, default=800)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--output", default="data/self_play_games")
    args = p.parse_args()

    config = Config()
    config.self_play_simulations = args.sims
    config.num_mcts_workers = args.workers
    config.max_game_length = MAX_MOVES

    model = load_model(args.model, config).cuda().eval()
    print(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    engine = get_mcts_engine(model, config)
    engine._ensure_pool()

    def _on_int(sig, frame):
        print("\n[Interrupted]")
        if stop_evt: stop_evt.set()
    import signal as _sig
    stop_evt = None
    _sig.signal(_sig.SIGINT, _on_int)

    try:
        generate_games(engine, config, args.games, args.output)
    finally:
        engine.shutdown()
