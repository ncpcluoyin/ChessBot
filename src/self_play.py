"""
Self-play: generate games + train on them, alternating.

Each cycle:
  1. Generate N games via MCTS + current model
  2. Train on self-play data (policy + value)
  3. Update model, repeat

Usage:
  python -m src.self_play --model data/models/model_sf.pt --games 200 --sims 800 --train-epochs 5
"""

import gc, os, signal, sys, time, hashlib, json, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import chess

from src.board import board_to_tensor, move_to_index
from src.config import Config
from src.network import load_model, create_model, save_model
from src.mcts import get_mcts_engine

torch.set_float32_matmul_precision('high')

# ── Constants ────────────────────────────────────────────────────────

TEMP_THRESHOLD = 30
TEMPERATURE = 1.0
TOP_K_SAMPLE = 3
DIR_ALPHA = 0.3
DIR_EPSILON = 0.25
MIN_MOVES = 20
MAX_MOVES = 200

# ── Generate games ──────────────────────────────────────────────────

def play_one_game(mcts, config, stop_event=None):
    board = chess.Board()
    positions = []
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
        positions.append((tensor, result.policy.copy(), None))

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

    # Build STM-perspective value labels
    tensors, policies, values = [], [], []
    for j, (t, p, _) in enumerate(positions):
        stm_val = white_result if (j % 2 == 0) else -white_result
        tensors.append(t)
        policies.append(p)
        values.append(stm_val)

    return tensors, policies, values, white_result, move_count


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


def _game_hash(positions):
    h = hashlib.md5()
    for t, _, _ in positions:
        h.update(t.numpy().tobytes())
    return h.hexdigest()[:16]


# ── Training on self-play data ──────────────────────────────────────

def train_selfplay(model, games, config, epochs=5, batch_size=1024, lr=0.001):
    """Train model on self-play games. Returns avg loss."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Extract all samples
    samples = []
    for game in games:
        tensors = game['positions']
        policies = game['policies']
        values = game['values']
        for j in range(len(tensors)):
            samples.append((tensors[j], policies[j], float(values[j])))

    if len(samples) < batch_size:
        return 0.0

    indices = list(range(len(samples)))
    total_loss = 0.0
    n_batches = 0

    for _ in range(epochs):
        np.random.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            sel = [samples[i] for i in indices[start:start+batch_size]]
            inputs = torch.stack([s[0] for s in sel]).to(config.device)
            td = torch.tensor(np.array([s[1] for s in sel]), dtype=torch.float32).to(config.device)
            v_label = torch.tensor([s[2] for s in sel], dtype=torch.float32).to(config.device)

            optimizer.zero_grad()
            pol, v_pred = model(inputs)
            v_pred = v_pred.squeeze(-1)
            pol_loss = -(td * pol).sum(dim=-1).mean()
            val_loss = ((v_pred - v_label) ** 2).mean()
            loss = pol_loss + 12.0 * val_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss)
            n_batches += 1

    return total_loss / max(n_batches, 1)


# ── Main loop ───────────────────────────────────────────────────────

def generate_and_train(mcts, model, config, num_games, train_epochs,
                        output_dir, stop_event=None, verbose=True):
    """Generate games, train on them, return stats."""
    os.makedirs(output_dir, exist_ok=True)
    seen_hashes = set()

    # Generate
    t0 = time.time()
    games = []
    stats = {'wins': 0, 'losses': 0, 'draws': 0}
    for i in range(num_games):
        if stop_event and stop_event.is_set():
            break
        gc.collect()
        tensors, policies, values, result, length = play_one_game(mcts, config, stop_event)
        if len(tensors) < 5:
            continue
        gh = _game_hash(list(zip(tensors, policies, [None]*len(tensors))))
        if gh in seen_hashes:
            continue
        seen_hashes.add(gh)

        games.append({
            'positions': tensors, 'policies': policies,
            'values': torch.tensor(values, dtype=torch.float32),
            'result': result, 'length': length,
        })
        if result > 0.5: stats['wins'] += 1
        elif result < -0.5: stats['losses'] += 1
        else: stats['draws'] += 1

        if verbose:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{num_games}] {length:3d} moves  "
                  f"W/B/D {stats['wins']}/{stats['losses']}/{stats['draws']}  "
                  f"{elapsed:.0f}s", flush=True)

    gen_time = time.time() - t0

    # Train
    t0 = time.time()
    avg_loss = train_selfplay(model, games, config, epochs=train_epochs)
    train_time = time.time() - t0

    total_pos = sum(len(g['positions']) for g in games)
    print(f"  Trained: {len(games)} games, {total_pos} positions, "
          f"avg_loss={avg_loss:.4f}, gen={gen_time:.0f}s, train={train_time:.0f}s",
          flush=True)

    stats['games'] = len(games)
    stats['positions'] = total_pos
    stats['avg_loss'] = avg_loss
    stats['gen_time'] = gen_time
    stats['train_time'] = train_time
    return stats


# ── CLI ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description="Self-play: generate + train")
    p.add_argument("--model", required=True)
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--sims", type=int, default=800)
    p.add_argument("--train-epochs", type=int, default=5)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--output", default="data/self_play_games")
    p.add_argument("--lr", type=float, default=0.001)
    args = p.parse_args()

    config = Config()
    config.self_play_simulations = args.sims
    config.num_mcts_workers = args.workers
    config.max_game_length = MAX_MOVES

    print(f"Loading model: {args.model}")
    model = load_model(args.model, config).cuda()
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params/1e6:.1f}M")

    print(f"Creating MCTS (workers={config.num_mcts_workers}, sims={args.sims})...")
    engine = get_mcts_engine(model, config)
    engine._ensure_pool()

    stop_evt = None
    def _on_int(sig, frame):
        print("\n[Interrupted]", flush=True)
        if stop_evt: stop_evt.set()
    import signal as _sig
    _sig.signal(_sig.SIGINT, _on_int)

    print(f"Generating {args.games} games + training {args.train_epochs} epochs...")
    try:
        stats = generate_and_train(engine, model, config, args.games,
                                    args.train_epochs, args.output, stop_evt)
        # Save model
        save_model(model, args.model)
        print(f"Model saved to {args.model}")
    finally:
        engine.shutdown()
        print("Done.")
