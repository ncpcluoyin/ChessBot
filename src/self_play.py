"""
Self-play game generation. Saves games to disk as .pt + .pgn.
Press Ctrl+C to stop gracefully.
"""

import gc, os, signal, sys, time, hashlib, json, glob, threading
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
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

_raw_model = None  # for raw-policy mode


def play_one_game_raw(mcts_or_model, config, stop_event=None):
    """Raw NN policy + temperature sampling (no MCTS). Much faster."""
    global _raw_model
    model = _raw_model if _raw_model is not None else mcts_or_model
    board = chess.Board()
    samples = []
    moves_history = []
    move_count = 0

    while not board.is_game_over() and move_count < MAX_MOVES:
        if stop_event and stop_event.is_set():
            break

        t = board_to_tensor(board).unsqueeze(0).cuda()
        with torch.no_grad():
            pol_log, v = model(t)
        pol = pol_log.squeeze(0).exp().cpu().numpy()
        val = float(v.squeeze(-1).cpu().item())

        samples.append({
            'tensor': board_to_tensor(board),
            'policy': torch.from_numpy(pol.copy()).float(),
        })

        use_dd = move_count < TEMP_THRESHOLD
        if use_dd:
            move = _sample_move(pol, board)
        else:
            move = _greedy_move(pol, board)
        if move is None:
            break
        moves_history.append(move)
        board.push(move)
        move_count += 1
        if abs(val) > 5.0 and move_count > MIN_MOVES:
            break

    outcome = board.outcome()
    if outcome is None:
        white_result = max(-1.0, min(1.0, val))
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


def play_one_game_mcts(mcts, config, stop_event=None):
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


play_one_game = play_one_game_mcts  # default: MCTS


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

    # 王车易位偏置: 仅在合法时加概率
    for i, mv in enumerate(legals):
        uci = mv.uci()
        if uci in ('e1g1', 'e8g8'):
            probs[i] += 0.10
        elif uci in ('e1c1', 'e8c8'):
            probs[i] += 0.08
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
        try:
            samples, result, length, pgn_text = play_one_game(mcts, config, stop_event)
        except Exception as e:
            print(f"  [!] Game error: {e}", flush=True)
            continue
        if len(samples) < 5:
            continue

        # 每 10 局打印一次保活信号
        if count % 10 == 0 and verbose:
            elapsed = time.time() - t0
            rate = count / (elapsed + 1e-6) * 3600
            total = stats['wins'] + stats['losses'] + stats['draws']
            print(f"  [{count} games] W/B/D {stats['wins']}/{stats['losses']}/{stats['draws']}  "
                  f"{rate:.0f} games/hr  ({elapsed:.0f}s)", flush=True)

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
    p.add_argument("--raw-policy", action="store_true", help="Use raw NN + temperature (no MCTS)")
    p.add_argument("--output", default="data/self_play_games")
    args = p.parse_args()

    config = Config()
    config.self_play_simulations = args.sims
    config.num_mcts_workers = args.workers
    config.max_game_length = MAX_MOVES

    model = load_model(args.model, config).cuda().eval()
    print(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    if args.raw_policy:
        _raw_model = model
        play_one_game = play_one_game_raw
        engine = None
        print("Mode: raw policy + temperature (no MCTS)")
    else:
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
        if engine is not None:
            engine.shutdown()
