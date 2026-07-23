"""
GPU Batch MCTS — Decoupled producer-consumer
==============================================

Architecture:
  - 1 GPU inference process (CUDA, batched inference, max_batch=4096, 5ms window)
  - 4 persistent MCTS workers (CPU, speculative expansion, independent result queues)
  - Workers stay alive between searches → zero spawn overhead
  - Decoupled: CPU workers speculatively expand without waiting for GPU
  - Each worker has its own result queue (prevents result stealing)

Performance (18.4M model, RTX 5070):
  4 workers:  2398 nps  ← sweet spot
  8 workers:  1232 nps  (overhead > parallelism)
  CPU 24:      371 nps  (baseline)
"""

import os
import sys
import math
import time
import queue
import multiprocessing as mp
import chess
import numpy as np
import torch

from src.config import Config
from src.mcts.engine import MCTSEngine, SearchResult
from src.board import board_to_tensor, get_legal_moves_mask, move_to_index


# ════════════════════════════════════════════════
# GPU Inference Process
# ════════════════════════════════════════════════

def _gpu_inference_loop(req_q, res_qs, model_path, config_dict, max_batch, collect_ms):
    """GPU inference process: batches requests, routes results to per-worker queues.

    req_q receives (rid, wid, tensor_np, mask_np) tuples.
    res_qs[wid] receives (rid, probs, value) for each worker.
    """
    import torch

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # 推理优化: TF32 + channels_last
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.allow_tf32 = True

    from src.network import ChessNet
    cfg = Config()
    for k, v in config_dict.items():
        setattr(cfg, k, v)
    model = ChessNet(cfg)
    model.load_state_dict(torch.load(model_path, map_location='cuda', weights_only=False))
    model = model.cuda().eval()
    amp_dtype = next(model.parameters()).dtype
    device = 'cuda'

    collect_sec = collect_ms / 1000.0
    running = True

    while running:
        batch = []
        deadline = time.time() + collect_sec

        while len(batch) < max_batch:
            timeout = max(deadline - time.time(), 0.0)
            if timeout <= 0:
                break
            try:
                msg = req_q.get(timeout=0.001)
                if msg is None:
                    running = False
                    break
                batch.append(msg)
            except queue.Empty:
                break

        if not batch:
            continue

        ids, wids, tensors, masks = [], [], [], []
        for rid, wid, t_np, m_np in batch:
            ids.append(rid)
            wids.append(wid)
            tensors.append(torch.from_numpy(t_np))
            masks.append(torch.from_numpy(m_np))

        t_batch = torch.stack(tensors).to(device, dtype=amp_dtype)
        m_batch = torch.stack(masks).to(device)

        with torch.inference_mode():
            logp, values = model(t_batch, m_batch)

        probs = torch.exp(logp).float().cpu().numpy()
        vals = values.float().cpu().numpy()

        for i, (rid, wid) in enumerate(zip(ids, wids)):
            if wid < len(res_qs) and res_qs[wid] is not None:
                res_qs[wid].put((rid, probs[i], float(vals[i][0])))

    for q in res_qs:
        if q is not None:
            try:
                q.put(None)
            except Exception:
                pass


# ════════════════════════════════════════════════
# MCTS Node
# ════════════════════════════════════════════════

class _Node:
    __slots__ = ('p', 'n', 'q', 'children', 'virtual_n', 'pending_rid', 'speculative')
    def __init__(self, p=0.0, speculative=False):
        self.p = p
        self.n = 0
        self.q = 0.0
        self.children = {}
        self.virtual_n = 0
        self.pending_rid = None
        self.speculative = speculative


# ════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════

def _get_top_k(priors, moves, depth):
    """动态 top-k: 覆盖 90% 概率质量, 深度衰减, 保底 3 个."""
    # 按概率排序
    idx = np.argsort(priors)[::-1]
    cum = 0.0
    base_k = 0
    for i, j in enumerate(idx):
        if priors[j] < 1e-6:  # 跳过接近 0 的走法
            break
        cum += priors[j]
        base_k = i + 1
        if cum >= 0.85:
            break
    base_k = max(4, min(18, base_k))
    k = int(base_k / (1 + 0.25 * depth))
    k = max(3, k)
    return idx[:k]


def _expand_with_policy(node, board, legals, probs, tree, depth=0):
    """Expand node with real NN policy, top-k pruning."""
    priors = np.zeros(len(legals), dtype=np.float32)
    for i, mv in enumerate(legals):
        try:
            priors[i] = float(probs[move_to_index(mv, board)])
        except Exception:
            priors[i] = 0.0
    s = priors.sum()
    if s > 0:
        priors /= s
    else:
        priors[:] = 1.0 / len(legals)

    top_idx = _get_top_k(priors, legals, depth)
    # 归一化前 k 个概率
    top_sum = priors[top_idx].sum()
    if top_sum > 0:
        priors[top_idx] /= top_sum
    else:
        priors[top_idx] = 1.0 / len(top_idx)

    for idx in top_idx:
        mv = legals[idx]
        b2 = board.copy()
        b2.push(mv)
        ck = b2._transposition_key()
        child = _Node(p=float(priors[idx]))
        if ck not in tree:
            tree[ck] = child
        node.children[mv.uci()] = child


def _merge_real_policy(node, board, legals, probs, tree, depth=0):
    """Merge real NN policy into node, preserving Q/N from speculative children.
    After merge, prune to top-k to limit tree width.
    """
    new_children = {}
    for mv in legals:
        uci = mv.uci()
        try:
            real_p = max(float(probs[move_to_index(mv, board)]), 0.0)
        except Exception:
            real_p = 0.0

        if uci in node.children:
            old = node.children[uci]
            old.p = real_p
            old.speculative = False
            new_children[uci] = old
        else:
            new_children[uci] = _Node(p=real_p)

    total = sum(c.p for c in new_children.values())
    if total > 0:
        for c in new_children.values():
            c.p /= total
    else:
        u = 1.0 / max(len(new_children), 1)
        for c in new_children.values():
            c.p = u

    # 按概率排序, 取 top-k
    sorted_children = sorted(new_children.items(), key=lambda x: x[1].p, reverse=True)
    priors = np.array([c.p for _, c in sorted_children], dtype=np.float32)
    top_idx = _get_top_k(priors, [c[0] for c in sorted_children], depth)
    pruned = {sorted_children[i][0]: sorted_children[i][1] for i in top_idx}
    # 归一化保留的节点
    top_sum = sum(c.p for c in pruned.values())
    if top_sum > 0:
        for c in pruned.values():
            c.p /= top_sum

    node.children = pruned
    for mv in legals:
        b2 = board.copy()
        b2.push(mv)
        ck = b2._transposition_key()
        if ck not in tree:
            tree[ck] = pruned.get(mv.uci())


# ════════════════════════════════════════════════
# Persistent MCTS Worker (runs in a Process)
# ════════════════════════════════════════════════

def _mcts_worker_loop(req_q, res_q, cmd_q, progress_q, wid, stop_evt=None):
    """Long-running MCTS worker: waits for search commands, runs search, loops.

    req_q: shared request queue for GPU
    res_q: THIS WORKER'S dedicated result queue (not shared!)
    cmd_q: command queue (search/stop)
    progress_q: shared progress/result reporting queue

    cmd_q receives:
        ('search', board_fen, n_sims, c_puct, search_id, first_rid_offset)
        ('stop',) → exit

    progress_q sends:
        ('progress', wid, search_id, sims_done, n_sims, total_nodes)
        ('result',  wid, search_id, visits_dict, pv_list, best_q, total_n)
    """
    import chess as _chess
    import time as _time
    import queue as _queue
    from src.board import board_to_tensor as _b2t, get_legal_moves_mask as _glm, move_to_index as _m2i

    while True:
        try:
            cmd = cmd_q.get()
        except Exception:
            break

        if cmd[0] == 'stop':
            break

        _, board_fen, n_sims, c_puct, virtual_loss, search_id, first_rid_offset = cmd

        board = _chess.Board(board_fen)
        root_key = board._transposition_key()
        root = _Node()
        tree = {root_key: root}
        MAX_TREE_NODES = 200000  # Hard cap: 5000 sims x 40 legal moves ≈ 200K nodes

        legals = list(board.legal_moves)
        if not legals:
            progress_q.put(('result', wid, search_id, {}, [], 0.0, 0))
            continue

        # Drain any stale results from previous search
        while True:
            try:
                res_q.get_nowait()
            except _queue.Empty:
                break

        # ── Root eval (synchronous) ──
        rid_root = first_rid_offset + wid
        req_q.put((rid_root, wid, _b2t(board).numpy(), _glm(board).numpy()))

        root_result = None
        _root_deadline = _time.time() + 10.0
        while root_result is None and _time.time() < _root_deadline:
            try:
                msg = res_q.get(timeout=0.01)
                if msg is None:
                    return
                if msg[0] == rid_root:
                    root_result = (msg[1], msg[2])
            except _queue.Empty:
                continue
        if root_result is None:
            # Root eval timed out - send empty result and continue
            progress_q.put(('result', wid, search_id, {}, [], 0.0, 0))
            continue

        _expand_with_policy(root, board, legals, root_result[0], tree)

        # ── Async search ──
        sims_done = 0
        pending = {}
        next_rid = first_rid_offset + 100
        report_interval = max(3, n_sims // 30)  # Frequent updates for smooth UCI
        last_report_time = _time.time()
        last_report_sims = 0

        while sims_done < n_sims:
            if stop_evt is not None and stop_evt.is_set():
                break
            # Drain completed evals
            while True:
                try:
                    msg = res_q.get_nowait()
                except _queue.Empty:
                    break
                if msg is None:
                    break
                rid, rprobs, rvalue = msg
                if rid in pending:
                    leaf_node, path_nodes, leaf_board, leaf_legals = pending.pop(rid)
                    leaf_node.pending_rid = None
                    _merge_real_policy(leaf_node, leaf_board, leaf_legals, rprobs, tree, depth=len(path_nodes)-1)
                    val = rvalue
                    for node in reversed(path_nodes):
                        node.n += 1
                        node.virtual_n = max(0, node.virtual_n - 1)
                        node.q = val if node.n == 1 else max(node.q, val)
                        val = -val
                    sims_done += 1

            # PUCT select
            b = board.copy()
            path = [root]
            current = root
            root_n = root.n + root.virtual_n
            log_sqrt = math.sqrt(math.log(root_n + 2.0))  # AlphaZero 风格: sqrt(log(N+1)), N=0 时 = sqrt(log(2)) ≈ 0.83

            while current.children:
                best_uci = None
                best_score = -1e9
                for uci, child in current.children.items():
                    eff_n = child.n + child.virtual_n
                    instant_q = (child.q * child.n - virtual_loss * child.virtual_n) / max(eff_n, 1)
                    sc = instant_q + c_puct * log_sqrt / math.sqrt(1 + eff_n)
                    if sc > best_score:
                        best_score = sc
                        best_uci = uci
                if best_uci is None:
                    break
                mv = _chess.Move.from_uci(best_uci)
                b.push(mv)
                child_node = current.children[best_uci]
                path.append(child_node)
                current = child_node

            # Handle leaf
            legals_leaf = list(b.legal_moves)

            if not legals_leaf:
                leaf_value = 1.0 if b.is_checkmate() else 0.0
                val = leaf_value
                for node in reversed(path):
                    node.n += 1
                    node.q = val if node.n == 1 else max(node.q, val)
                    val = -val
                sims_done += 1
                continue

            if current.pending_rid is not None:
                for node in path:
                    if node.virtual_n < 50:
                        node.virtual_n += 1
                continue

            # Speculative expansion
            if len(tree) >= MAX_TREE_NODES:
                # Tree too large — apply virtual loss but don't expand further
                for node in path:
                    if node.virtual_n < 50:
                        node.virtual_n += 1
                continue

            t_leaf = _b2t(b).numpy()
            m_leaf = _glm(b).numpy()
            rid = next_rid
            next_rid += 1
            req_q.put((rid, wid, t_leaf, m_leaf))
            current.pending_rid = rid
            pending[rid] = (current, path, b.copy(), legals_leaf)

            u_prior = 1.0 / len(legals_leaf)
            for mv in legals_leaf:
                b2 = b.copy()
                b2.push(mv)
                ck = b2._transposition_key()
                child = _Node(p=u_prior, speculative=True)
                if ck not in tree:
                    tree[ck] = child
                current.children[mv.uci()] = child

            for node in path:
                node.virtual_n += 1

            # ── Progress: report frequently (by sims or by time) ──
            now = _time.time()
            if (sims_done - last_report_sims >= report_interval or
                    (sims_done > last_report_sims and now - last_report_time > 0.05) or
                    sims_done >= n_sims):
                try:
                    # Full PV from most-visited path
                    _pv = []
                    _cur = root
                    while _cur.children:
                        _best = max(_cur.children.items(), key=lambda x: x[1].n)
                        _pv.append(_best[0])
                        _cur = _best[1]
                    _depth = len(_pv)
                    # Weighted average root Q (使用实际访问量, 排除虚拟损失)
                    _root_n = sum(c.n for c in root.children.values())
                    _root_q = 0.0
                    if _root_n > 0:
                        _root_q = sum(-c.q * (c.n / _root_n)
                                      for c in root.children.values())
                    progress_q.put_nowait(
                        ('progress', wid, search_id, sims_done, n_sims,
                         _root_n, _depth, _pv,
                         _root_q))
                    last_report_sims = sims_done
                    last_report_time = now
                except _queue.Full:
                    pass

        # Results - skip drain, main thread clears res_qs between searches
        visits = {uci: c.n for uci, c in root.children.items()}
        pv = []
        cur = root
        while cur.children:
            best = max(cur.children.items(), key=lambda x: x[1].n)
            pv.append(best[0])
            cur = best[1]
        _root_n = sum(c.n for c in root.children.values())
        if _root_n > 0:
            best_q = sum(-c.q * (c.n / _root_n)
                         for c in root.children.values())
        else:
            best_q = 0.0
        # Send result (non-blocking, avoid deadlock if queue is full)
        deadline = _time.time() + 5.0
        while _time.time() < deadline:
            try:
                progress_q.put_nowait(('result', wid, search_id, visits, pv, best_q, root.n))
                break
            except _queue.Full:
                _time.sleep(0.001)

        # Free memory between searches
        tree.clear()
        import gc as _gc
        _gc.collect()


# ════════════════════════════════════════════════
# Batch GPU Engine (Persistent Workers)
# ════════════════════════════════════════════════

class BatchGPUEngine(MCTSEngine):
    """MCTS engine: N persistent workers + 1 GPU process, fully decoupled."""

    def __init__(self, model, config: Config):
        self.model = model
        self.config = config
        # 推理优化
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.allow_tf32 = True
        # GPU batch optimal: 4 workers for 18.4M model (2398 nps)
        # More workers adds overhead without improving throughput
        self.num_workers = min(config.num_mcts_workers, 8)
        self._gpu_collect_ms = 5  # Batch collection window (ms)
        self._temp_path = None

        self._gpu_proc = None
        self._req_q = None
        self._res_qs = []

        self._workers = []
        self._cmd_qs = []
        self._progress_q = None  # Will be replaced by mp.Queue in _ensure_pool
        self._search_id = 0
        self._pool_ready = False
        self._stop_evt = None  # mp.Event for stopping workers mid-search

        self._progress_nodes = 0
        self._progress_elapsed = 0.001
        self._progress_depth = 0
        self._progress_pv = []
        self._progress_root_q = 0.0
        self._search_start = 0.0

    def _ensure_gpu(self):
        if self._temp_path is not None:
            return

        import tempfile
        fd, self._temp_path = tempfile.mkstemp(suffix='.pt', prefix='gpubatch_')
        os.close(fd)
        torch.save(self.model.state_dict(), self._temp_path)

        self._req_q = mp.Queue(maxsize=50000)
        # Per-worker queues created later in _ensure_pool

    def _ensure_pool(self):
        """Compatibility with UCI isready warmup."""
        self._ensure_gpu()
        if self._pool_ready:
            return

        # Create per-worker result queues (GPU routes results by wid)
        for wid in range(self.num_workers):
            self._res_qs.append(mp.Queue(maxsize=50000))

        # Start GPU process AFTER queues exist
        config_dict = {
            'num_filters': self.config.num_filters,
            'num_input_planes': self.config.num_input_planes,
            'policy_output_dim': self.config.policy_output_dim,
        }
        self._gpu_proc = mp.Process(
            target=_gpu_inference_loop,
            args=(self._req_q, self._res_qs, self._temp_path, config_dict, 4096, self._gpu_collect_ms),
            daemon=True, name="GPU-Batch"
        )
        self._gpu_proc.start()

        self._progress_q = mp.Queue(maxsize=max(10000, self.num_workers * 2000))
        self._stop_evt = mp.Event()

        for wid in range(self.num_workers):
            cmd_q = mp.Queue()
            p = mp.Process(
                target=_mcts_worker_loop,
                args=(self._req_q, self._res_qs[wid], cmd_q, self._progress_q, wid, self._stop_evt),
                daemon=True, name=f"BatchMCTS-{wid}"
            )
            p.start()
            self._cmd_qs.append(cmd_q)
            self._workers.append(p)

        # Quick warmup: send tiny searches to all workers concurrently, then drain
        warm_sid = -1
        for wid in range(self.num_workers):
            self._cmd_qs[wid].put(
                ('search', chess.STARTING_FEN, 3, self.config.c_puct * min(1.0, 3/400.0), self.config.virtual_loss, warm_sid, wid * 50000))

        # Drain all warmup results from progress_q
        warm_remaining = set(range(self.num_workers))
        warm_deadline = time.time() + 60.0
        while warm_remaining and time.time() < warm_deadline:
            try:
                item = self._progress_q.get(timeout=0.1)
                if item[0] == 'result' and item[2] == warm_sid:
                    warm_remaining.discard(item[1])
                    # Also give workers time to send results
                    time.sleep(0.001)
            except queue.Empty:
                continue

        # Drain leftover warmup progress messages
        while True:
            try:
                self._progress_q.get_nowait()
            except queue.Empty:
                break

        self._pool_ready = True

    def search(self, board, num_simulations=None, time_limit=None,
               stop_event=None, use_dirichlet=False,
               nn_raw_value=None):
        self._ensure_gpu()
        self._ensure_pool()

        if num_simulations is None:
            num_simulations = self.config.mcts_simulations

        self._search_start = time.time()
        self._progress_nodes = 0
        self._progress_elapsed = 0.001
        self._progress_depth = 0
        self._progress_pv = []
        self._progress_root_q = 0.0
        self._search_id += 1
        sid = self._search_id

        # Clear stop event from previous interrupted search
        if self._stop_evt is not None:
            self._stop_evt.clear()

        # Drain stale state from previous search (req_q + res_qs)
        import sys as _sys
        if self._gpu_proc is not None and not self._gpu_proc.is_alive():
            
            _sys.stderr.flush()
            self._pool_ready = False
            self._ensure_pool()
        for wid, w in enumerate(self._workers):
            if not w.is_alive():
                
                _sys.stderr.flush()
        _stale = 0
        while True:
            try:
                self._req_q.get_nowait()
                _stale += 1
            except:
                break
        for _wq in self._res_qs:
            while True:
                try:
                    _wq.get_nowait()
                except:
                    break
        if _stale > 0:
            
            _sys.stderr.flush()

        # 动态 c_puct: 低搜索量时更 exploit
        c_puct_eff = self.config.c_puct * min(1.0, num_simulations / 400.0)

        board_fen = board.fen()
        per_worker = num_simulations // self.num_workers
        remainder = num_simulations % self.num_workers
        offset = sid * 10000

        # Send search commands to all workers
        for wid in range(self.num_workers):
            n_sims = per_worker + (1 if wid < remainder else 0)
            self._cmd_qs[wid].put(
                ('search', board_fen, n_sims, c_puct_eff, self.config.virtual_loss, sid, offset))
        import sys as _sys2
        _sys2.stderr.flush()

        # Collect results
        merged = {}
        total_nodes = 0
        q_sum = 0.0
        q_cnt = 0
        worker_pvs = []
        pending_workers = set(range(self.num_workers))
        worker_done = {}  # wid → last reported sims_done for accurate progress

        _worker_root_q = {}
        _safety_deadline = self._search_start + (time_limit if time_limit is not None else 120.0)
        while pending_workers:
            if stop_event and stop_event.is_set():
                break
            if time.time() > _safety_deadline:
                break

            # Drain ALL available messages (not one per poll)
            got_any = False
            while True:
                try:
                    item = self._progress_q.get_nowait()
                    got_any = True
                except queue.Empty:
                    break

                msg_sid = item[2]
                if msg_sid != sid:
                    continue

                msg_type = item[0]

                if msg_type == 'progress':
                    _, wid_, _, sims_done, n_sims, total_n, depth, pv_list, root_q = item
                    worker_done[wid_] = sims_done
                    self._progress_nodes = sum(worker_done.values())
                    self._progress_elapsed = time.time() - self._search_start
                    if depth > self._progress_depth:
                        self._progress_depth = depth
                    if pv_list and len(pv_list) > len(self._progress_pv):
                        self._progress_pv = pv_list
                    _worker_root_q[wid_] = root_q
                    self._progress_root_q = sum(_worker_root_q.values()) / max(1, len(_worker_root_q))
                elif msg_type == 'result':
                    _, wid_, _, visits, pv, best_q, rt_n = item
                    for uci, cnt in visits.items():
                        merged[uci] = merged.get(uci, 0) + cnt
                    total_nodes += sum(visits.values())
                    if best_q != 0.0:
                        q_sum += best_q
                        q_cnt += 1
                    if visits:
                        worker_pvs.append((max(visits, key=visits.get), pv))
                    pending_workers.discard(wid_)

            if not got_any:
                time.sleep(0.01)
                if not any(p.is_alive() for p in self._workers):
                    import sys as _sys
                    _sys.stderr.write(f"[DBG] sid={sid} workers ALL DEAD, "
                                      f"pending={pending_workers}\n")
                    _sys.stderr.flush()
                    break

        # Signal workers to stop (timeout/interrupt)
        # Keep event set until next search clears it
        if pending_workers and self._stop_evt is not None:
            self._stop_evt.set()

        import sys as _sys
        # Drain any leftover stale messages from progress_q
        while True:
            try:
                self._progress_q.get_nowait()
            except queue.Empty:
                break

        # Drain res_qs (stale eval results between searches)
        self._progress_depth_saved = self._progress_depth
        self._progress_nodes_saved = self._progress_nodes
        self._progress_root_q_saved = self._progress_root_q
        for _wq in self._res_qs:
            while True:
                try:
                    _wq.get_nowait()
                except queue.Empty:
                    break

        if not merged:
            import sys as _sys
            _sys.stderr.write(f"[DBG] sid={sid} no merged results, "
                              f"pending={pending_workers} "
                              f"alive={[p.is_alive() for p in self._workers]}\n")
            _sys.stderr.flush()
            return self._fallback(board)

        best_uci = max(merged, key=merged.get)
        total = sum(merged.values()) or 1

        policy = np.zeros(self.config.policy_output_dim, dtype=np.float32)
        for uci, cnt in merged.items():
            try:
                mv = chess.Move.from_uci(uci)
                if mv in board.legal_moves:
                    policy[move_to_index(mv, board)] = cnt / total
            except Exception:
                continue

        # root.q = best child value (minimax)
        root_q = (max(-c.q for c in root.children if c.n > 0)
                  if any(c.n > 0 for c in root.children) else 0.0)
        # 混入 NN 原始评估做保险
        if nn_raw_value is not None:
            root_q = 0.7 * root_q + 0.3 * nn_raw_value

        best_move = None
        try:
            best_move = chess.Move.from_uci(best_uci)
            if best_move not in board.legal_moves:
                import sys as _sys
                _sys.stderr.write(f"[DBG] bad best_uci={best_uci}, "
                                  f"merged keys={list(merged.keys())[:10]}\n")
                _sys.stderr.flush()
                raise ValueError("not legal")
        except Exception:
            for m in board.legal_moves:
                best_move = m
                best_uci = m.uci()
                break

        pv = [best_uci]
        best_len = 0
        for wbest, wpv in worker_pvs:
            if wbest == best_uci and len(wpv) > best_len:
                pv = wpv
                best_len = len(wpv)

        self._progress_nodes = total_nodes
        self._progress_depth = len(pv)
        self._progress_pv = pv
        self._progress_root_q = root_q
        self._progress_elapsed = time.time() - self._search_start

        return SearchResult(
            best_move=best_move, best_move_uci=best_uci,
            policy=policy, root_value=root_q,
            nodes_searched=total_nodes, max_depth=len(pv),
            time_elapsed=self._progress_elapsed, pv=pv)

    def _fallback(self, board):
        t = board_to_tensor(board).numpy()
        m = get_legal_moves_mask(board).numpy()
        rid = 99999999
        # Use self as wid=0, ensure res_qs[0] exists for the response
        if len(self._res_qs) > 0:
            self._req_q.put((rid, 0, t, m))
            result = None
            _fb_deadline = time.time() + 5.0
            while result is None and time.time() < _fb_deadline:
                try:
                    msg = self._res_qs[0].get(timeout=0.01)
                    if msg is None:
                        break
                    if msg[0] == rid:
                        result = (msg[1], msg[2])
                except queue.Empty:
                    pass
            if result is None:
                # Fallback GPU timeout - use CPU direct inference
                import torch
                t_t = torch.from_numpy(t).unsqueeze(0).to(self.config.device)
                m_t = torch.from_numpy(m).unsqueeze(0).to(self.config.device)
                with torch.no_grad():
                    logp, val = self.model(t_t, m_t.bool())
                result = (torch.exp(logp).float().cpu().numpy()[0],
                          float(val.cpu().item()))
        else:
            # No workers yet, do direct inference
            import torch
            t_t = torch.from_numpy(t).unsqueeze(0).cuda()
            m_t = torch.from_numpy(m).unsqueeze(0).cuda()
            with torch.inference_mode():
                logp, val = self.model(t_t, m_t.bool())
            result = (torch.exp(logp).float().cpu().numpy()[0],
                      float(val.cpu().item()))

        probs, value = result
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
        # Guarantee a move (pick first legal if NN gave nothing)
        if best_move is None:
            for mv in board.legal_moves:
                best_move = mv
                break
        self._progress_elapsed = time.time() - self._search_start
        self._progress_nodes = max(getattr(self, '_progress_nodes_saved', 1), 1)
        self._progress_depth = max(getattr(self, '_progress_depth_saved', 0), 1)
        self._progress_pv = [best_move.uci()]
        self._progress_root_q = getattr(self, '_progress_root_q_saved', value)

        return SearchResult(
            best_move=best_move, best_move_uci=best_move.uci(),
            policy=np.zeros(self.config.policy_output_dim, dtype=np.float32),
            root_value=value, nodes_searched=self._progress_nodes,
            max_depth=self._progress_depth,
            time_elapsed=self._progress_elapsed, pv=[best_move.uci()])

    def get_search_progress(self, board=None):
        elapsed = max(self._progress_elapsed, 0.001)
        nps = int(self._progress_nodes / elapsed) if self._progress_nodes else 0
        return {
            "nodes": self._progress_nodes, "elapsed": elapsed,
            "depth": self._progress_depth, "root_q": self._progress_root_q,
            "pv": self._progress_pv, "nps": nps
        }

    def reset(self):
        pass

    def shutdown(self):
        # Stop workers
        for cmd_q in self._cmd_qs:
            try:
                cmd_q.put(('stop',))
            except Exception:
                pass
        for p in self._workers:
            if p.is_alive():
                p.join(timeout=2.0)
                if p.is_alive():
                    p.terminate()
        self._workers.clear()
        self._cmd_qs.clear()
        self._pool_ready = False

        # Clear per-worker queues
        for q in self._res_qs:
            try:
                while q.get_nowait() is not None:
                    pass
            except Exception:
                pass
        self._res_qs.clear()

        # Stop GPU
        if self._gpu_proc and self._gpu_proc.is_alive():
            try:
                self._req_q.put(None)
            except Exception:
                pass
            self._gpu_proc.join(timeout=3.0)
            if self._gpu_proc.is_alive():
                self._gpu_proc.terminate()
        self._gpu_proc = None

        if self._temp_path and os.path.exists(self._temp_path):
            try:
                os.unlink(self._temp_path)
            except Exception:
                pass
