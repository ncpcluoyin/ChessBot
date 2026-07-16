"""
持久化 CPU Worker 池 — 进程常驻，消除 spawn/kill 开销。

每个 worker:
  - 加载模型到 CPU，torch.set_num_threads(2)
  - 维护本地评估缓存 (position_key → policy, value)
  - 通过双向 Pipe 接收命令、发送进度/结果
"""

import json
import math
import os
import sys
import time
import tempfile
import traceback
import signal as _signal
import multiprocessing as mp
from collections import OrderedDict

import chess
import numpy as np

from src.config import Config
from src.board import board_to_tensor, get_legal_moves_mask, move_to_index
from src.mcts.engine import MCTSEngine, SearchResult


# ── 评估缓存 ──

class EvalCache:
    """LRU 评估缓存: position_key → (policy_uci_map, value)."""

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

    def put(self, key, policy_uci_map, value):
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = (policy_uci_map, value)
        while len(self._cache) > self._max:
            self._cache.popitem(last=False)

    def clear(self):
        self._cache.clear()
        self.hits = 0
        self.misses = 0

    def stats(self) -> str:
        total = self.hits + self.misses
        rate = self.hits / max(1, total) * 100
        return f"cache={len(self._cache)} hits={self.hits} miss={self.misses} ({rate:.0f}%)"


# ── MCTS Node ──

class _Node:
    """轻量 MCTS 节点 (__slots__ 内存优化)。"""
    __slots__ = ('prior', 'n', 'w', 'children', 'vl')

    def __init__(self, p=0.0):
        self.prior = p
        self.n = 0
        self.w = 0.0
        self.children = {}   # uci → _Node
        self.vl = 0

    def q(self) -> float:
        n = self.n + self.vl
        return self.w / n if n > 0 else 0.0


# ── PV 提取 ──

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


# ── Worker 搜索逻辑 ──

def _do_search(worker_id, cmd_pipe, msg, model, eval_cache, c_puct, dir_a, dir_e):
    """Worker 内执行一次 MCTS 搜索。需要 torch 已在调用方导入。"""
    # torch 由 _worker_main 导入 (CUDA_VISIBLE_DEVICES='' 后)
    import torch
    board = chess.Board(msg["board_fen"])
    n_sims = msg["n_sims"]
    sid = msg.get("sid", 0)
    use_dirichlet = msg.get("use_dirichlet", False)
    root_key = board._transposition_key()

    root = _Node()
    tree = {root_key: root}

    legals = list(board.legal_moves)
    if legals:
        cached = eval_cache.get(root_key)
        if cached:
            policy_map, _ = cached
        else:
            t = board_to_tensor(board).unsqueeze(0)
            m = get_legal_moves_mask(board).unsqueeze(0)
            with torch.no_grad():
                logp, vals = model(t, m)
            probs = torch.exp(logp).numpy()[0]
            policy_map = {}
            for mv in legals:
                try:
                    policy_map[mv.uci()] = float(probs[move_to_index(mv, board)])
                except Exception:
                    pass

        priors = np.array([policy_map.get(mv.uci(), 0.0) for mv in legals],
                          dtype=np.float32)
        s = priors.sum()
        if s > 0:
            priors /= s
        else:
            priors[:] = 1.0 / len(legals)

        # Dirichlet 噪声 (仅 root, 探索多样性)
        if use_dirichlet:
            noise = np.random.dirichlet([dir_a] * len(legals)).astype(np.float32)
            priors = (1.0 - dir_e) * priors + dir_e * noise

        noise = np.random.dirichlet([dir_a] * len(legals))
        priors = (1 - dir_e) * priors + dir_e * noise
        for i, mv in enumerate(legals):
            root.children[mv.uci()] = _Node(float(priors[i]))

    done = 0
    for sim_i in range(n_sims):
        key = root_key
        b = board.copy()
        path = []

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

        if not b.is_game_over(claim_draw=True):
            cached = eval_cache.get(key)
            if cached:
                policy_map, v = cached
            else:
                t = board_to_tensor(b).unsqueeze(0)
                mk = get_legal_moves_mask(b).unsqueeze(0)
                with torch.no_grad():
                    logp, vals = model(t, mk)
                v = float(vals.squeeze(-1).item())
                probs = torch.exp(logp).numpy()[0]
                policy_map = {}
                sl = list(b.legal_moves)
                for mv in sl:
                    try:
                        policy_map[mv.uci()] = float(probs[move_to_index(mv, b)])
                    except Exception:
                        pass
                eval_cache.put(key, policy_map, v)

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

        for pf_key, uci in reversed(path):
            cn = tree[pf_key].children[uci]
            cn.vl -= 1
            cn.n += 1
            cn.w += v
            v = -v

        done += 1

        if done % 50 == 0:
            pv = _extract_pv(root_key, tree, board)
            total = sum(c.n for c in root.children.values()) or 0
            rq = 0.0
            if total > 0:
                for u, c in root.children.items():
                    rq += (c.n / total) * (-c.q())
            try:
                cmd_pipe.send({
                    "type": "progress",
                    "sid": sid,
                    "worker": worker_id,
                    "n": done,
                    "total": total,
                    "pv": pv,
                    "q": rq,
                })
            except (BrokenPipeError, OSError):
                break

    visits = {u: c.n for u, c in root.children.items()}
    total = sum(visits.values()) or 0
    root_q = 0.0
    if total > 0:
        for u, c in root.children.items():
            root_q += (c.n / total) * (-c.q())
    pv = _extract_pv(root_key, tree, board)

    try:
        cmd_pipe.send({
            "type": "result",
            "sid": sid,
            "worker": worker_id,
            "visits": visits,
            "total": total,
            "sims_done": done,
            "root_q": root_q,
            "pv": pv,
        })
    except (BrokenPipeError, OSError):
        pass


# ── Worker 主循环 ──

def _worker_main(worker_id, cmd_pipe, config_dict, model_path):
    """持久化 worker — 等待命令，执行搜索，返回结果。"""
    _signal.signal(_signal.SIGINT, _signal.SIG_IGN)

    # 禁止 CUDA（CPU worker 不需要）
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    torch.set_num_threads(2)

    from src.network import ChessNet as _ChessNet
    from src.config import Config as _Config

    try:
        _cfg = _Config()
        model = _ChessNet(_cfg)
        model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=False))
        model.eval()
    except Exception as e:
        try:
            cmd_pipe.send({"type": "error", "worker": worker_id,
                           "msg": f"Model load failed: {e}"})
        except Exception:
            pass
        return

    eval_cache = EvalCache(max_size=10_000)
    c_puct = config_dict.get('c_puct', 1.5)
    dir_a = config_dict.get('dirichlet_alpha', 0.25)
    dir_e = config_dict.get('dirichlet_epsilon', 0.03)

    running = True
    while running:
        try:
            if cmd_pipe.poll(0.5):
                msg = cmd_pipe.recv()
            else:
                continue
        except (EOFError, OSError):
            break

        cmd = msg.get("cmd")

        if cmd == "search":
            try:
                _do_search(worker_id, cmd_pipe, msg, model, eval_cache,
                           c_puct, dir_a, dir_e)
            except Exception as e:
                try:
                    cmd_pipe.send({
                        "type": "error",
                        "worker": worker_id,
                        "msg": f"Search crashed: {e}\n{traceback.format_exc()}",
                    })
                except Exception:
                    pass

        elif cmd == "newgame":
            eval_cache.clear()

        elif cmd == "quit":
            running = False

    cmd_pipe.close()


# ── 持久化 Worker 池 ──

class CPUWorkersMCTS(MCTSEngine):
    """持久化多进程 CPU MCTS 引擎。"""

    def __init__(self, network, config: Config, model_path: str = None):
        self.network = network
        self.config = config
        self.num_workers = config.num_mcts_workers
        self._model_path = model_path
        self._temp_model_path = None
        self._pool = None        # list of (process, pipe)
        self._pool_ready = False
        self._search_id = 0       # 搜索序列号，防旧结果污染

        self._progress_nodes = 0
        self._progress_elapsed = 0.0
        self._progress_depth = 0
        self._progress_pv = []
        self._progress_q = 0.0
        self._search_start = 0.0

    def _ensure_model_path(self):
        if self._model_path and os.path.exists(self._model_path):
            return self._model_path
        if self._temp_model_path is None:
            fd, self._temp_model_path = tempfile.mkstemp(suffix='.pt', prefix='chessbot_')
            os.close(fd)
            from src.network import save_model
            save_model(self.network, self._temp_model_path)
        return self._temp_model_path

    def _ensure_pool(self):
        """懒启动 worker 池（进程常驻）。"""
        if self._pool_ready:
            return

        model_path = self._ensure_model_path()
        cd = {
            'c_puct': self.config.c_puct,
            'dirichlet_alpha': self.config.dirichlet_alpha,
            'dirichlet_epsilon': self.config.dirichlet_epsilon,
        }

        self._pool = []
        for i in range(self.num_workers):
            parent_pipe, child_pipe = mp.Pipe(duplex=True)
            p = mp.Process(
                target=_worker_main,
                args=(i, child_pipe, cd, model_path),
                daemon=True,
            )
            p.start()
            child_pipe.close()
            self._pool.append((p, parent_pipe))

        self._pool_ready = True

    def search(self, board: chess.Board, num_simulations=None, time_limit=None,
               stop_event=None, use_dirichlet=False, nn_raw_value=None):
        self._ensure_pool()

        if num_simulations is None:
            num_simulations = self.config.mcts_simulations

        per_worker = num_simulations // self.num_workers
        remainder = num_simulations % self.num_workers
        board_fen = board.fen()

        self._search_start = time.time()
        self._progress_nodes = 0
        self._progress_elapsed = 0.001
        self._progress_depth = 0
        self._progress_pv = []
        self._progress_q = 0.0

        # 递增搜索 ID (防止旧结果污染)
        self._search_id += 1
        sid = self._search_id

        # 分发搜索任务
        for i, (_, pipe) in enumerate(self._pool):
            n_sims = per_worker + (1 if i < remainder else 0)
            try:
                pipe.send({
                    "cmd": "search",
                    "sid": sid,
                    "board_fen": board_fen,
                    "n_sims": n_sims,
                    "use_dirichlet": use_dirichlet and i == 0,
                })
            except (BrokenPipeError, OSError):
                pass

        # 收集结果
        merged = {}
        total_nodes = 0
        root_q_sum = 0.0
        q_count = 0
        worker_pvs = []
        pending = set(range(self.num_workers))

        while pending:
            # 检查 stop
            if stop_event and stop_event.is_set():
                for i, (_, pipe) in enumerate(self._pool):
                    if i in pending:
                        try:
                            pipe.send({"cmd": "stop"})
                        except Exception:
                            pass
                break
            if time_limit is not None and time.time() - self._search_start > time_limit:
                for i, (_, pipe) in enumerate(self._pool):
                    if i in pending:
                        try:
                            pipe.send({"cmd": "stop"})
                        except Exception:
                            pass
                break

            # 高效多路复用
            pipes = [self._pool[i][1] for i in pending]
            ready = mp.connection.wait(pipes, timeout=0.05) if pipes else []

            for conn in ready:
                try:
                    msg = conn.recv()
                except (EOFError, OSError):
                    # 找到对应索引
                    for i in list(pending):
                        if self._pool[i][1] is conn:
                            pending.discard(i)
                            break
                    continue

                msg_type = msg.get("type")

                if msg.get("sid") != sid:
                    continue  # 旧搜索结果，丢弃

                if msg_type == "error":
                    wid = msg.get("worker", -1)
                    sys.stderr.write(f"info string Worker {wid} error: {msg.get('msg', 'unknown')}\n")
                    sys.stderr.flush()
                    pending.discard(wid)
                    continue

                if msg_type == "progress":
                    wid = msg.get("worker", -1)
                    if wid in pending:
                        n_done = msg.get("n", 0)
                        wtotal = msg.get("total", 0)
                        if wtotal > 0 and n_done > 0:
                            self._progress_nodes = max(self._progress_nodes, n_done * self.num_workers)
                        pv = msg.get("pv", [])
                        if len(pv) > self._progress_depth:
                            self._progress_depth = len(pv)
                            self._progress_pv = pv
                        if 'q' in msg:
                            self._progress_q = msg['q']

                elif msg_type == "result":
                    wid = msg.get("worker", -1)
                    total_nodes += msg.get('total', 0)
                    for uci, cnt in msg.get('visits', {}).items():
                        merged[uci] = merged.get(uci, 0) + cnt
                    if 'root_q' in msg:
                        root_q_sum += msg['root_q']
                        q_count += 1
                    if msg.get('pv'):
                        wbest = max(msg.get('visits', {}),
                                    key=msg['visits'].get) if msg.get('visits') else ''
                        worker_pvs.append((wbest, msg['pv']))
                    pending.discard(wid)

                self._progress_elapsed = time.time() - self._search_start

        # 清空管道残留 (超时后 worker 可能还在发送旧结果)
        if pending:
            for i in list(pending):
                try:
                    while self._pool[i][1].poll(0.001):
                        self._pool[i][1].recv()
                except Exception:
                    pass

        if not merged:
            # 所有 worker 失败 — 紧急回退
            return self._fallback_result(board)

        best_uci = max(merged, key=merged.get)
        total = sum(merged.values()) or 1
        policy = np.zeros(self.config.policy_output_dim, dtype=np.float32)
        for uci, cnt in merged.items():
            try:
                move = chess.Move.from_uci(uci)
                if move in board.legal_moves:
                    policy[move_to_index(move, board)] = cnt / total
            except Exception:
                continue

        root_q = root_q_sum / max(1, q_count)
        pv = [best_uci]
        best_len = 0
        for wbest, wpv in worker_pvs:
            if wbest == best_uci and len(wpv) > best_len:
                pv = wpv
                best_len = len(wpv)

        self._progress_pv = pv
        self._progress_q = root_q
        self._progress_nodes = total_nodes
        self._progress_elapsed = time.time() - self._search_start
        self._progress_depth = len(pv)

        best_move = None
        try:
            best_move = chess.Move.from_uci(best_uci)
        except Exception:
            for m in board.legal_moves:
                best_move = m
                break

        return SearchResult(
            best_move=best_move, best_move_uci=best_uci,
            policy=policy, root_value=root_q,
            nodes_searched=total_nodes, max_depth=len(pv), pv=pv,
            time_elapsed=self._progress_elapsed)

    def _fallback_result(self, board):
        """紧急回退：单次 CPU 评估选最优合法走法。"""
        import os as _os
        _os.environ['CUDA_VISIBLE_DEVICES'] = ''
        import torch
        torch.set_num_threads(2)
        t = board_to_tensor(board).unsqueeze(0)
        m = get_legal_moves_mask(board).unsqueeze(0)
        with torch.no_grad():
            logp, vals = self.network.cpu()(t, m)
        probs = torch.exp(logp).numpy()[0]
        value = float(vals.item())

        best_move = None
        best_p = -1.0
        for mv in board.legal_moves:
            try:
                p = float(probs[move_to_index(mv, board)])
                if p > best_p:
                    best_p = p
                    best_move = mv
            except Exception:
                pass

        # 更新进度信息 (让 info 输出有内容)
        self._progress_nodes = 1
        self._progress_elapsed = 0.005
        self._progress_depth = 1
        self._progress_q = value
        self._progress_pv = [best_move.uci()] if best_move else []

        _pv = [best_move.uci()] if best_move else []
        if best_move:
            return SearchResult(
                best_move=best_move, best_move_uci=best_move.uci(),
                policy=np.zeros(self.config.policy_output_dim, dtype=np.float32),
                root_value=value, nodes_searched=1, max_depth=1,
                time_elapsed=0.005, pv=_pv)
        return self._empty_result(board)

    def _empty_result(self, board):
        return SearchResult(
            best_move=None, best_move_uci='',
            policy=np.zeros(self.config.policy_output_dim, dtype=np.float32),
            root_value=0.0, nodes_searched=0, max_depth=0,
            time_elapsed=0, pv=[])

    def reset(self):
        """通知所有 worker 清空缓存。"""
        if self._pool:
            for _, pipe in self._pool:
                try:
                    pipe.send({"cmd": "newgame"})
                except Exception:
                    pass
        self._progress_nodes = 0
        self._progress_elapsed = 0.0
        self._progress_depth = 0
        self._progress_pv = []
        self._progress_q = 0.0

    def shutdown(self):
        """关闭 worker 池。"""
        if self._pool:
            for _, pipe in self._pool:
                try:
                    pipe.send({"cmd": "quit"})
                except Exception:
                    pass
            for p, _ in self._pool:
                p.join(timeout=2)
                if p.is_alive():
                    p.kill()
            self._pool = None
            self._pool_ready = False
        self._cleanup_temp()

    def _cleanup_temp(self):
        if self._temp_model_path is not None:
            try:
                if os.path.exists(self._temp_model_path):
                    os.unlink(self._temp_model_path)
            except Exception:
                pass
            self._temp_model_path = None

    def __del__(self):
        self.shutdown()

    def get_search_progress(self, board):
        return {
            "nodes": self._progress_nodes,
            "elapsed": max(self._progress_elapsed, 0.001),
            "depth": self._progress_depth,
            "root_q": self._progress_q,
            "pv": self._progress_pv,
        }
