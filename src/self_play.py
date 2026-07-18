"""
Self-play game generation. Saves games to disk as .pt + .pgn.
Press Ctrl+C to stop gracefully.
"""

import gc, os, signal, sys, time, hashlib, json, glob, threading
from datetime import datetime
import numpy as np
import torch
import chess
import chess.pgn

from src.board import board_to_tensor, move_to_index
from src.config import Config
from src.network import load_model
from src.mcts import get_mcts_engine

TEMP_THRESHOLD = 30
TOP_K_SAMPLE = 3
MIN_MOVES = 20
MAX_MOVES = 200


def play_one_game(mcts, config, stop_event=None):
    board = chess.Board()
    samples = []
    moves_history = []
    move_count = 0

    while not board.is_game_over() and move_count < MAX_MOVES:
        if stop_event and stop_event.is_set():
            break
        use_dd = move_count < TEMP_THRESHOLD
        result = mcts.search(board, num_simulations=config.self_play_simulations,
                              use_dirichlet=use_dd, stop_event=stop_event)
        if result is None or result.policy is None:
            break
        samples.append({
            'tensor': board_to_tensor(board),
            'policy': torch.from_numpy(result.policy.copy()).float(),
        })
        if move_count < TEMP_THRESHOLD:
            move = _sample_move(result.policy, board)
        else:
            move = _greedy_move(result.policy, board)
        if move is None:
            break
        moves_history.append(move)
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

    for j, s in enumerate(samples):
        s['value'] = white_result if (j % 2 == 0) else -white_result
        s['result'] = white_result

    pgn_text = _build_pgn(moves_history, outcome, white_result)
    return samples, white_result, move_count, pgn_text


def _build_pgn(moves, outcome, white_result):
    game = chess.pgn.Game()
    node = game
    for mv in moves:
        node = node.add_variation(mv)
    if outcome:
        if outcome.winner == chess.WHITE:
            game.headers["Result"] = "1-0"
        elif outcome.winner == chess.BLACK:
            game.headers["Result"] = "0-1"
        else:
            game.headers["Result"] = "1/2-1/2"
    elif white_result > 0.5:
        game.headers["Result"] = "1-0"
    elif white_result < -0.5:
        game.headers["Result"] = "0-1"
    else:
        game.headers["Result"] = "1/2-1/2"
    exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
    return game.accept(exporter)


def _sample_move(policy, board):
    legals = list(board.legal_moves)
    if not legals:
        return None
    indices = [move_to_index(mv, board) for mv in legals]
    probs = np.array([max(policy[idx], 0.0) for idx in indices], dtype=np.float64)

    # 王车易位偏置: 短易位 4x, 长易位 2x
    for i, mv in enumerate(legals):
        uci = mv.uci()
        if uci in ('e1g1', 'e8g8'):
            probs[i] *= 4.0
        elif uci in ('e1c1', 'e8c8'):
            probs[i] *= 2.0
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


def generate_games(mcts, config, output_dir, stop_event=None, verbose=True, max_games=0):
    """Generate games until stopped or max_games reached."""
    os.makedirs(output_dir, exist_ok=True)
    prefix = datetime.now().strftime('%Y%m%d_%H%M%S')
    count = 0
    stats = {'wins': 0, 'losses': 0, 'draws': 0}
    t0 = time.time()

    while (stop_event is None or not stop_event.is_set()) and (max_games == 0 or count < max_games):
        gc.collect()
        samples, result, length, pgn_text = play_one_game(mcts, config, stop_event)
        if len(samples) < 5:
            continue

        path = os.path.join(output_dir, f'{prefix}_{count:04d}.pt')
        torch.save({
            'samples': samples,
            'result': result,
            'length': length,
            'pgn': pgn_text,
        }, path)
        pgn_path = os.path.join(output_dir, f'{prefix}_{count:04d}.pgn')
        with open(pgn_path, 'w') as f:
            f.write(pgn_text)

        if result > 0.5: stats['wins'] += 1
        elif result < -0.5: stats['losses'] += 1
        else: stats['draws'] += 1
        count += 1

        if verbose:
            elapsed = time.time() - t0
            rate = count / elapsed * 3600 if elapsed > 0 else 0
            print(f"  [{prefix}_{count-1:04d}] {length:3d} pos  "
                  f"W/B/D {stats['wins']}/{stats['losses']}/{stats['draws']}  "
                  f"{rate:.0f} games/hr", flush=True)

    elapsed = time.time() - t0
    total = stats['wins'] + stats['losses'] + stats['draws']
    print(f"\\nGenerated {total} games, {elapsed:.0f}s total", flush=True)
    return stats


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--games", type=int, default=0, help="0=continuous, N=stop after N games")
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

    stop_evt = threading.Event()
    def _on_int(sig, frame):
        print("\n[Stopping...]")
        stop_evt.set()
    import signal as _sig
    _sig.signal(_sig.SIGINT, _on_int)

    if args.games > 0:
        # Fixed count mode (for daemon loop)
        _games_remaining = [args.games]
        def _stop_after_n():
            _games_remaining[0] -= 1
            return _games_remaining[0] <= 0

    print(f"Generating games to {args.output} (Ctrl+C to stop)...")
    try:
        generate_games(engine, config, args.output, stop_evt, max_games=args.games)
    finally:
        engine.shutdown()
