"""
Self-play game generation for RL training.

Generates games using MCTS + current model and saves them as training data.
Self-play is ONLY for data generation — training is separate (distill pipeline).

Key design:
  - Dirichlet noise at root for first 30 moves (exploration)
  - Temperature-based sampling (1.0 early, 0.0 late)
  - Each game saved independently as a .pt file
  - No random opening — MCTS from move 1 with noise is enough diversity
  - Repeat detection via position hash to avoid duplicate games

Usage:
  python -m src.self_play --model data/models/model_sf.pt --games 100 --sims 2000
"""

import gc, os, signal, sys, time, hashlib, json
import numpy as np
import torch
import chess

from src.board import board_to_tensor, move_to_index
from src.config import Config
from src.network import load_model
from src.mcts import get_mcts_engine


# ── Constants ────────────────────────────────────────────────────────

TEMP_THRESHOLD = 30       # First 30 moves use temperature + Dirichlet
TEMPERATURE = 1.0         # Softmax temperature for early moves
TOP_K_SAMPLE = 3          # Top-K for move sampling
DIR_ALPHA = 0.3           # Dirichlet noise alpha
DIR_EPSILON = 0.25        # Dirichlet noise weight
MIN_MOVES_BEFORE_EARLY_STOP = 20


# ── Self-play game ───────────────────────────────────────────────────

def play_one_game(mcts, config, stop_event=None):
    """Play one complete self-play game using MCTS.

    Returns:
        positions: list of (tensor_15x8x8, policy_4672, None)
        result: ±1 from White's perspective
        pgn: game PGN string (for debugging)
        length: number of moves played
    """
    board = chess.Board()
    positions = []
    move_count = 0

    game = chess.pgn.Game()
    node = game

    while not board.is_game_over() and move_count < config.max_game_length:
        if stop_event is not None and stop_event.is_set():
            break

        use_dd = move_count < TEMP_THRESHOLD
        ns = config.self_play_simulations

        # MCTS search
        result = mcts.search(board, num_simulations=ns, use_dirichlet=use_dd,
                              stop_event=stop_event)
        if result is None or result.policy is None:
            break

        # Record position + MCTS policy
        tensor = board_to_tensor(board)
        positions.append((tensor, result.policy.copy(), None))

        # Select move
        if move_count < TEMP_THRESHOLD:
            move = _sample_move(result.policy, board, temperature=TEMPERATURE, top_k=TOP_K_SAMPLE)
        else:
            move = _greedy_move(result.policy, board)

        if move is None:
            break

        # Record in PGN
        node = node.add_variation(move)
        board.push(move)
        move_count += 1

        # Early stop: one side has crushing advantage
        if abs(result.root_value) > 5.0 and move_count > MIN_MOVES_BEFORE_EARLY_STOP:
            break

    # Determine game result from White's perspective
    outcome = board.outcome()
    if outcome is None:
        # Game not over (early stop or max length): use search value
        if positions:
            final_val = result.root_value if result else 0.0
            white_result = final_val if board.turn == chess.WHITE else -final_val
            white_result = max(-1.0, min(1.0, white_result))
        else:
            white_result = 0.0
    elif outcome.winner == chess.WHITE:
        white_result = 1.0
    elif outcome.winner == chess.BLACK:
        white_result = -1.0
    else:
        white_result = 0.0

    # Build non-rotated PGN
    exporter = chess.pgn.StringExporter(headers=False, variations=False, comments=False)
    pgn_text = game.accept(exporter)

    return positions, white_result, pgn_text, move_count


# ── Move selection ───────────────────────────────────────────────────

def _sample_move(policy, board, temperature=1.0, top_k=3):
    """Sample a move from the policy distribution with temperature."""
    legals = list(board.legal_moves)
    if not legals:
        return None

    indices = [move_to_index(mv, board) for mv in legals]
    probs = np.array([max(policy[idx], 0.0) for idx in indices], dtype=np.float64)

    if top_k > 0 and top_k < len(legals):
        top_indices = np.argpartition(probs, -top_k)[-top_k:]
        mask = np.zeros_like(probs, dtype=bool)
        mask[top_indices] = True
        probs[~mask] = 0.0

    s = probs.sum()
    if s <= 0:
        return legals[np.random.randint(len(legals))]

    if temperature != 1.0:
        probs = probs ** (1.0 / temperature)

    probs /= probs.sum()
    idx = np.random.choice(len(legals), p=probs)
    return legals[idx]


def _greedy_move(policy, board):
    """Select the highest-probability legal move."""
    legals = list(board.legal_moves)
    if not legals:
        return None
    indices = [move_to_index(mv, board) for mv in legals]
    best = max(indices, key=lambda i: policy[i])
    return legals[indices.index(best)]


# ── Game hash (repeat detection) ─────────────────────────────────────

def _game_hash(positions):
    """Compute a hash of the game's position sequence to detect duplicates."""
    h = hashlib.md5()
    for tensor, _, _ in positions:
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()[:16]


# ── Batch generation ─────────────────────────────────────────────────

def generate_self_play_games(mcts, config, num_games, output_dir,
                              verbose=True, stop_event=None):
    """Generate self-play games and save them to disk.

    Each game is saved as a separate .pt file:
        game_0000.pt, game_0001.pt, ...

    File format: {
        'positions': [tensor_15x8x8, ...],
        'policies': [policy_4672, ...],
        'values': [float, ...],         # ±1 value labels per position
        'result': float,                # game result from White's perspective
        'pgn': str,                     # game PGN
        'length': int,                  # number of moves
        'hash': str,                    # game hash for dedup
    }
    """
    os.makedirs(output_dir, exist_ok=True)

    # Find existing game count for offset
    existing = [f for f in os.listdir(output_dir) if f.startswith('game_') and f.endswith('.pt')]
    start_idx = 0
    if existing:
        nums = []
        for f in existing:
            try:
                nums.append(int(f.replace('game_', '').replace('.pt', '')))
            except ValueError:
                pass
        if nums:
            start_idx = max(nums) + 1

    # Track hashes for repeat detection
    seen_hashes = set()
    stats = {'white_wins': 0, 'black_wins': 0, 'draws': 0, 'early_stops': 0, 'repeats': 0}
    total_positions = 0
    t0 = time.time()

    for i in range(num_games):
        if stop_event is not None and stop_event.is_set():
            break

        gc.collect()

        positions, white_result, pgn_text, length = play_one_game(
            mcts, config, stop_event=stop_event)

        if len(positions) < 5:
            continue  # too short, skip

        # Detect repeats
        gh = _game_hash(positions)
        if gh in seen_hashes:
            stats['repeats'] += 1
            if verbose:
                print(f"  [!] Repeat game detected, skipping (hash={gh})", flush=True)
            continue
        seen_hashes.add(gh)

        # Assign value labels
        tensors = []
        policies = []
        values = []
        for j, (t, p, _) in enumerate(positions):
            # Value from White's perspective
            if board_at_move_is_white(j):
                val = white_result
            else:
                val = -white_result
            tensors.append(t)
            policies.append(p)
            values.append(val)

        # Save game
        game_data = {
            'positions': tensors,
            'policies': [torch.from_numpy(p).float() for p in policies],
            'values': torch.tensor(values, dtype=torch.float32),
            'result': white_result,
            'pgn': pgn_text,
            'length': length,
            'hash': gh,
        }
        out_path = os.path.join(output_dir, f'game_{start_idx + i:04d}.pt')
        torch.save(game_data, out_path)

        # Stats
        total_positions += len(positions)
        if white_result > 0.5:
            stats['white_wins'] += 1
        elif white_result < -0.5:
            stats['black_wins'] += 1
        else:
            stats['draws'] += 1
        if length < MIN_MOVES_BEFORE_EARLY_STOP:
            stats['early_stops'] += 1

        if verbose:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed * 3600 if elapsed > 0 else 0
            avg_len = total_positions // (i + 1 - stats['repeats'])
            print(f"  [{i+1}/{num_games}] {len(positions):3d} pos  "
                  f"W/B/D {stats['white_wins']}/{stats['black_wins']}/{stats['draws']}  "
                  f"avg {avg_len} moves  {rate:.0f} games/hr  "
                  f"({elapsed:.0f}s)",
                  flush=True)

    # Save stats
    stats['total_positions'] = total_positions
    stats['total_games'] = num_games - stats['repeats']
    stats['time_seconds'] = time.time() - t0
    stats_path = os.path.join(output_dir, 'stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nSelf-play done: {stats['total_games']} games, "
          f"{total_positions} positions, {stats['repeats']} repeats filtered",
          flush=True)
    return stats


def board_at_move_is_white(move_index):
    """After `move_index` half-moves, who just moved? Alternates."""
    # Move 0 = White's first move, Move 1 = Black's first move, etc.
    # position[0] is before White's first move → White to move
    # So value for position[j] should be from the perspective of who is to move
    return (move_index % 2 == 0)  # White to move


# ── CLI entry point ──────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description="Self-play game generation")
    p.add_argument("--model", required=True, help="Model checkpoint path")
    p.add_argument("--games", type=int, default=100, help="Number of games to play")
    p.add_argument("--sims", type=int, default=2000, help="MCTS simulations per move")
    p.add_argument("--workers", type=int, default=12, help="MCTS workers")
    p.add_argument("--output", default="data/self_play_games", help="Output directory")
    p.add_argument("--max-moves", type=int, default=200, help="Max moves per game")
    p.add_argument("--verbose", action="store_true", default=True)
    args = p.parse_args()

    config = Config()
    config.self_play_simulations = args.sims
    config.num_mcts_workers = args.workers
    config.max_game_length = args.max_moves

    print(f"Loading model: {args.model}")
    model = load_model(args.model, config)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params/1e6:.1f}M  Device: {config.device}")

    print(f"Creating MCTS engine (workers={config.num_mcts_workers})...")
    engine = get_mcts_engine(model, config, force='gpu_batch')
    engine._ensure_pool()

    stop_evt = None
    def _on_int(sig, frame):
        print("\n[Interrupted]", flush=True)
        if stop_evt:
            stop_evt.set()
    import signal as _sig
    _sig.signal(_sig.SIGINT, _on_int)

    print(f"Generating {args.games} games (sims={args.sims}, max_moves={args.max_moves})...")
    try:
        generate_self_play_games(
            engine, config, args.games, args.output,
            verbose=args.verbose)
    finally:
        engine.shutdown()
        print("Done.", flush=True)
