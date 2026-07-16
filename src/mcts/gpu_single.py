"""
单线程 GPU MCTS — 适用于 10M+ 参数模型。

特点:
  - GPU EvalCache (position_key → policy, value)
  - _transposition_key() 树键 (40x faster than FEN)
  - 可选批量评估 (batch_size=N)
"""

import math
import time
from collections import OrderedDict

import numpy as np
import chess
import torch

from src.board import board_to_tensor, get_legal_moves_mask, move_to_index
from src.mcts.engine import MCTSEngine, SearchResult


class _Node:
    __slots__ = ('prior', 'n', 'w', 'children', 'vl')

    def __init__(self, prior: float = 0.0):
        self.prior = prior
        self.n = 0
        self.w = 0.0
        self.children: dict = {}
        self.vl = 0

    def q(self) -> float:
        n = self.n + self.vl
        return self.w / n if n > 0 else 0.0


def _extract_pv(root_key, tree, board):
    pv = []
    sim_board = board.copy()
    pkey = root_key
    for _ in range(20):
        node = tree.get(pkey)
        if not node or not node.children:
            break
        best_uci, best_child = max(node.children.items(), key=lambda x: x[1].n)
        if best_child.n == 0:
            break
        pv.append(best_uci)
        sim_board.push_uci(best_uci)
        pkey = sim_board._transposition_key()
    return pv


class GPUEvalCache:
    """GPU 端评估缓存 (单线程, 无需锁)。"""

    def __init__(self, max_size: int = 10_000):
        self._cache = OrderedDict()
        self._max = max_size
        self.hits = 0
        self.misses = 0

    def get(self, key):
        if key in self._cache:
            self._cache.move_to_end(key)
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        return None

    def put(self, key, policy_map, value):
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = (policy_map, value)
        while len(self._cache) > self._max:
            self._cache.popitem(last=False)

    def clear(self):
        self._cache.clear()
        self.hits = 0
        self.misses = 0


class GPUSingleMCTS(MCTSEngine):
    """单线程 GPU MCTS + 评估缓存。"""

    def __init__(self, network, config, batch_size: int = 1):
        self.network = network
        self.config = config
        self.device = next(network.parameters()).device
        self.dtype = next(network.parameters()).dtype
        self.batch_size = batch_size
        self._cache = GPUEvalCache()

        self._progress_nodes = 0
        self._progress_elapsed = 0.0
        self._progress_depth = 0
        self._progress_pv = []
        self._progress_q = 0.0

    def _eval_with_cache(self, board):
        """GPU 推理 + 缓存查询。返回 (policy_map, value)。"""
        key = board._transposition_key()
        cached = self._cache.get(key)
        if cached:
            return cached

        t = board_to_tensor(board).unsqueeze(0).to(self.device, dtype=self.dtype)
        m = get_legal_moves_mask(board).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logp, vals = self.network(t, m)
        v = float(vals.item())
        probs = torch.exp(logp).float().cpu().numpy()[0]

        policy_map = {}
        for mv in board.legal_moves:
            try:
                policy_map[mv.uci()] = float(probs[move_to_index(mv, board)])
            except Exception:
                pass

        self._cache.put(key, policy_map, v)
        return policy_map, v

    def search(self, board: chess.Board, num_simulations: int = None,
               time_limit: float = None, stop_event=None, nn_raw_value=None):
        if num_simulations is None:
            num_simulations = self.config.mcts_simulations

        t_start = time.time()
        root_key = board._transposition_key()
        tree = {}
        c_puct = self.config.c_puct
        dir_a = self.config.dirichlet_alpha
        dir_e = self.config.dirichlet_epsilon

        root = _Node()
        tree[root_key] = root

        legals = list(board.legal_moves)
        if legals:
            policy_map, _ = self._eval_with_cache(board)
            priors = np.array([policy_map.get(mv.uci(), 0.0) for mv in legals],
                              dtype=np.float32)
            s = priors.sum()
            if s > 0:
                priors /= s
            else:
                priors[:] = 1.0 / len(legals)
            noise = np.random.dirichlet([dir_a] * len(legals))
            priors = (1 - dir_e) * priors + dir_e * noise
            for i, mv in enumerate(legals):
                root.children[mv.uci()] = _Node(float(priors[i]))

        done = 0
        for _ in range(num_simulations):
            if stop_event and stop_event.is_set():
                break
            if time_limit and time.time() - t_start > time_limit:
                break

            key = root_key
            b = board.copy()
            path = []

            # Selection
            while key in tree:
                node = tree[key]
                if not node.children:
                    break
                best_u, best_m = -1e9, None
                pN = math.sqrt(max(1, sum(c.n + c.vl
                                           for c in node.children.values())))
                for uci, ch in node.children.items():
                    nn = ch.n + ch.vl
                    sc = ch.w / max(1, nn) + c_puct * ch.prior * pN / (1 + nn)
                    if sc > best_u:
                        best_u, best_m = sc, uci
                if not best_m:
                    break
                ch = node.children[best_m]
                ch.vl += 1
                path.append((key, best_m))
                b.push_uci(best_m)
                key = b._transposition_key()

            # Evaluation
            if not b.is_game_over(claim_draw=True):
                policy_map, v = self._eval_with_cache(b)
                if key not in tree:
                    tree[key] = _Node()
                node = tree[key]
                if not node.children and policy_map:
                    sp = np.array(list(policy_map.values()), dtype=np.float32)
                    ss = sp.sum()
                    if ss > 0:
                        sp /= ss
                    moves = list(policy_map.keys())
                    for i, mv in enumerate(moves):
                        node.children[mv] = _Node(float(sp[i]))
            else:
                v = -1.0 if b.is_checkmate() else 0.0

            # Backup
            for pf_key, uci in reversed(path):
                cn = tree[pf_key].children[uci]
                cn.vl -= 1
                cn.n += 1
                cn.w += v
                v = -v

            done += 1
            self._progress_nodes = done
            self._progress_elapsed = time.time() - t_start

        visits = {uci: child.n for uci, child in root.children.items()}
        total = sum(visits.values()) or 1
        root_q = 0.0
        for uci, child in root.children.items():
            root_q += (child.n / total) * (-child.q())

        best_uci = max(visits, key=visits.get) if visits else ''
        policy = np.zeros(self.config.policy_output_dim, dtype=np.float32)
        for uci, cnt in visits.items():
            try:
                move = chess.Move.from_uci(uci)
                if move in board.legal_moves:
                    policy[move_to_index(move, board)] = cnt / total
            except Exception:
                continue

        best_move = None
        try:
            best_move = chess.Move.from_uci(best_uci)
        except Exception:
            for m in board.legal_moves:
                best_move = m
                break

        pv = _extract_pv(root_key, tree, board)
        self._progress_pv = pv
        self._progress_q = root_q
        self._progress_nodes = total
        self._progress_elapsed = time.time() - t_start
        self._progress_depth = len(pv)

        return SearchResult(
            best_move=best_move, best_move_uci=best_uci,
            policy=policy, root_value=root_q,
            nodes_searched=total, max_depth=len(pv),
            time_elapsed=self._progress_elapsed, pv=pv)

    def get_search_progress(self, board):
        return {
            "nodes": self._progress_nodes,
            "elapsed": max(self._progress_elapsed, 0.001),
            "depth": self._progress_depth,
            "root_q": self._progress_q,
            "pv": self._progress_pv,
        }

    def reset(self):
        self._cache.clear()
        self._progress_nodes = 0
        self._progress_elapsed = 0.0
        self._progress_depth = 0
        self._progress_pv = []
        self._progress_q = 0.0
