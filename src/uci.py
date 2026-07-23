"""
UCI 协议接口 — 通过 stdin/stdout 与象棋 GUI 通信。

支持命令: uci, setoption, isready, ucinewgame, position, go, stop, quit
UCI 选项:
  - MCTSEngine   (combo, default cpu)     引擎: cpu/gpu/auto
  - BatchSize    (spin, default 1, 1-4096) GPU 批量大小
  - ModelPath    (string, default data/models/model_sf.pt) 模型路径
"""

import os
import sys
import time
import threading

import chess
import torch

from src.config import Config
from src.mcts.engine import SearchResult


class ChessBotEngine:
    def __init__(self, network, config: Config = None,
                 initial_options: dict = None):
        self.network = network
        self.config = config or Config()

        # UCI 选项默认值
        self._opt_model_path = "data/models/model_sf.pt"
        self._opt_batch_size = 1
        self._opt_num_workers = config.num_mcts_workers
        self._opt_simulations = config.mcts_simulations

        self._opt_intuition = False

        # 应用 CLI 预设选项
        if initial_options:
            if initial_options.get("intuition"):
                self._opt_intuition = True

        self._apply_options()  # 构建初始 MCTS
        self.board = chess.Board()
        self._history_boards = []
        self._stop_event = threading.Event()
        self._search_start = 0.0

    # ── MCTS 引擎构建 ──

    def _build_mcts(self):
        """根据当前选项构建 MCTS 引擎。"""
        force = "gpu_batch"

        # 将 UCI 选项同步到 config
        self.config.num_mcts_workers = self._opt_num_workers
        self.config.mcts_simulations = self._opt_simulations

        from src.mcts import get_mcts_engine
        return get_mcts_engine(self.network, self.config, force=force)

    def _apply_options(self):
        """应用选项：重建 MCTS 引擎。"""
        # 关闭旧引擎 (如果存在)
        mcts_old = getattr(self, 'mcts', None)
        if mcts_old is not None and hasattr(mcts_old, 'shutdown'):
            mcts_old.shutdown()

        self.mcts = self._build_mcts()

    # ── 主循环 ──

    def run(self):
        """进入 UCI 主循环。"""
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            self._dispatch(line)

    def _dispatch(self, line: str):
        cmd = line.split()[0] if line else ""
        if cmd == "uci":
            self._cmd_uci()
        elif cmd == "setoption":
            self._cmd_setoption(line)
        elif cmd == "isready":
            self._cmd_isready()
        elif cmd == "ucinewgame":
            self._cmd_ucinewgame()
        elif cmd == "position":
            self._cmd_position(line)
        elif cmd == "go":
            self._cmd_go(line)
        elif cmd == "stop":
            self._stop_event.set()
        elif cmd == "quit":
            if hasattr(self.mcts, 'shutdown'):
                self.mcts.shutdown()
            sys.exit(0)

    # ── UCI 命令实现 ──

    def _cmd_uci(self):
        nl = chr(10)
        sys.stdout.write('id name ChessBot' + nl)
        sys.stdout.write('id author ChessBot' + nl)
        sys.stdout.write('option name BatchSize type spin default 1 min 1 max 4096' + nl)
        sys.stdout.write(f'option name NumWorkers type spin default {self.config.num_mcts_workers} min 1 max 64' + nl)
        sys.stdout.write(f'option name Simulations type spin default {self.config.mcts_simulations} min 100 max 50000' + nl)
        sys.stdout.write('option name IntuitionMode type check default false' + nl)
        sys.stdout.write('option name ModelPath type string default data/models/model_sf.pt' + nl)
        sys.stdout.write('uciok' + nl)
        sys.stdout.flush()

    def _cmd_setoption(self, line: str):
        """解析 setoption name <name> value <value>。"""
        parts = line.split()
        try:
            name_idx = parts.index("name")
            value_idx = parts.index("value")
        except ValueError:
            return

        name = " ".join(parts[name_idx + 1:value_idx])
        value = " ".join(parts[value_idx + 1:])

        changed = False
        if name == "BatchSize":
            try:
                val = int(value)
                val = max(1, min(4096, val))
                if val != self._opt_batch_size:
                    self._opt_batch_size = val
                    changed = True
            except ValueError:
                pass
        elif name == "NumWorkers":
            try:
                val = int(value)
                val = max(1, min(64, val))
                if val != self._opt_num_workers:
                    self._opt_num_workers = val
                    changed = True
            except ValueError:
                pass
        elif name == "Simulations":
            try:
                val = int(value)
                val = max(100, min(100000, val))
                if val != self._opt_simulations:
                    self._opt_simulations = val
                    changed = True
            except ValueError:
                pass
        elif name == "ModelPath":
            if os.path.exists(value):
                self._opt_model_path = value
                changed = True
                # Load new model
                from src.network import load_model as _lm
                self.network = _lm(value, self.config)
                print('info string Loaded model:', value, flush=True)
            else:
                print('info string Model not found:', value, flush=True)
        elif name == "IntuitionMode":
            self._opt_intuition = value.lower() in ("true", "1", "yes")
            changed = True

        if changed:
            self._apply_options()

    def _scan_models(self):
        """Scan data/models/ for .pt files, return as 'var val1 var val2' string."""
        import glob as _g, os as _o
        base = _o.path.join('data', 'models')
        files = sorted(_g.glob(_o.path.join(base, '*.pt')))
        # Exclude checkpoints and temp files
        files = [f for f in files if not any(s in f for s in ('_ckpt', '_checkpoint', 'temp', '_rl'))]
        if not files:
            files = [_o.path.join(base, 'model_sf.pt')]
        return ' var '.join(files)

    def _cmd_isready(self):
        if hasattr(self.mcts, '_ensure_pool'):
            self.mcts._ensure_pool()  # 预热 CPU worker 池
        sys.stdout.write("readyok\n")
        sys.stdout.flush()

    def _cmd_ucinewgame(self):
        self.board = chess.Board()
        self._history_boards = []
        self.mcts.reset()  # 清空评估缓存 + 重置进度

    def _cmd_position(self, line: str):
        parts = line.split()
        if len(parts) < 2:
            return

        # 解析起始局面
        if parts[1] == "startpos":
            self.board = chess.Board()
            # 查找 moves 关键字 (兼容有无 moves 关键字两种格式)
            try:
                moves_idx = parts.index("moves") + 1
            except ValueError:
                moves_idx = 2 if len(parts) > 2 else -1
        elif parts[1] == "fen":
            # FEN 可能含空格 (如 "r nbqkbnr/pppppppp/... w KQkq - 0 1")
            try:
                moves_idx = parts.index("moves") + 1
            except ValueError:
                moves_idx = -1
            if moves_idx > 0:
                fen = " ".join(parts[2:moves_idx - 1])
            else:
                fen = " ".join(parts[2:])
            try:
                self.board = chess.Board(fen)
            except ValueError:
                return
        else:
            return

        self._history_boards = []

        # 应用走法序列
        if moves_idx > 0 and moves_idx < len(parts):
            for uci in parts[moves_idx:]:
                try:
                    move = chess.Move.from_uci(uci)
                    if move in self.board.legal_moves:
                        self._history_boards.insert(0, self.board.copy())
                        self.board.push(move)
                except ValueError:
                    pass

        if len(self._history_boards) > 7:
            self._history_boards = self._history_boards[:7]

    def _cmd_go(self, line: str):
        time_limit = self._parse_time(line)
        num_simulations = self._parse_depth_nodes(line)

        if time_limit is None and num_simulations is None:
            num_simulations = self.config.mcts_simulations

        # NN 直觉预测 (MCTS 前原始值头输出)
        from src.board import board_to_tensor
        _t = board_to_tensor(self.board).unsqueeze(0).to(
            self.config.device, dtype=next(self.network.parameters()).dtype)
        from src.board import get_legal_moves_mask
        _legal = get_legal_moves_mask(self.board).to(self.config.device)
        with torch.inference_mode():
            _pol, _raw_v = self.network(_t, _legal)
        _raw_v = float(_raw_v.item())
        self._nn_raw_v = _raw_v                                   # 保存供 _output_info 使用
        _cp = self.config.adj_to_cp(_raw_v)
        sys.stdout.write(f"info string raw_nn value={_raw_v:+.4f} cp={_cp}\n")
        sys.stdout.flush()

        self._stop_event.clear()
        self._search_start = time.time()

        # 直觉模式: 跳过 MCTS, 直接走策略头 top1
        if self._opt_intuition:
            from src.board import index_to_move
            top_idx = int(_pol[0].argmax().item())
            try:
                best = index_to_move(top_idx, self.board)
            except Exception:
                best = None
            elapsed = time.time() - self._search_start
            sys.stdout.write(
                f"info depth 1 "
                f"nodes 0 "
                f"nps 0 "
                f"time {int(max(elapsed,0)*1000)} "
                f"score cp {_cp} ")
            if best:
                sys.stdout.write(f"pv {best.uci()} ")
            sys.stdout.write("\n")
            if best:
                sys.stdout.write(f"bestmove {best.uci()}\n")
            else:
                sys.stdout.write("bestmove 0000\n")
            sys.stdout.flush()
            return

        result_container = []

        def do_search():
            result = self.mcts.search(
                self.board,
                num_simulations=num_simulations,
                time_limit=time_limit,
                stop_event=self._stop_event,
                nn_raw_value=_raw_v,
            )
            result_container.append(result)

        thread = threading.Thread(target=do_search, daemon=True)
        thread.start()

        while thread.is_alive():
            thread.join(timeout=0.05)
            self._output_info()

        if result_container:
            res = result_container[0]
            # 最后一轮 info（用 SearchResult 字段, 比 progress_q 可靠）
            nodes = res.nodes_searched or 0
            depth = max(res.max_depth, 1)
            elapsed = max(res.time_elapsed, 0.001)
            nps = int(nodes / elapsed) if nodes > 0 else 0
            # cp: 用搜索 root.q (minimax backup)
            nn_v = res.root_value
            cp = self.config.adj_to_cp(nn_v)
            # Use progress values if fallback was called
            _progress = self.mcts.get_search_progress(self.board)
            if _progress:
                if _progress.get("depth", 0) > depth:
                    depth = _progress["depth"]
                if _progress.get("nodes", 0) > nodes:
                    nodes = _progress["nodes"]
            pv_list = res.pv if hasattr(res, 'pv') and res.pv else []
            sys.stdout.write(

                f"info depth {depth} "
                f"nodes {nodes} "
                f"nps {nps} "
                f"time {int(elapsed * 1000)} "
                f"score cp {cp} ")
            if pv_list:
                sys.stdout.write(f"pv {' '.join(pv_list[:10])} ")
            sys.stdout.write("\n")
            sys.stdout.flush()
            if res.best_move:
                sys.stdout.write(f"bestmove {res.best_move.uci()}\n")
            else:
                sys.stdout.write("bestmove 0000\n")
        else:
            sys.stdout.write("bestmove 0000\n")
        sys.stdout.flush()

    # ── 时间解析 ──

    def _parse_time(self, line: str) -> float | None:
        parts = line.split()

        if "movetime" in parts:
            idx = parts.index("movetime")
            return float(parts[idx + 1]) / 1000.0 - self.config.movetime_safety_margin

        if "infinite" in parts or "depth" in parts or "nodes" in parts:
            return None

        wtime = btime = winc = binc = None
        if "wtime" in parts:
            wtime = float(parts[parts.index("wtime") + 1])
        if "btime" in parts:
            btime = float(parts[parts.index("btime") + 1])
        if "winc" in parts:
            winc = float(parts[parts.index("winc") + 1])
        if "binc" in parts:
            binc = float(parts[parts.index("binc") + 1])

        if wtime is not None or btime is not None:
            my_time = wtime if self.board.turn == chess.WHITE else btime
            my_inc = winc if self.board.turn == chess.WHITE else binc
            if my_time is None:
                return None
            return self.config.time_for_move(my_time, my_inc or 0.0)

        return None

    def _parse_depth_nodes(self, line: str) -> int | None:
        parts = line.split()
        if "depth" in parts:
            return None  # depth is handled as simulations=config default
        if "nodes" in parts:
            return int(parts[parts.index("nodes") + 1])
        return None

    # ── info 输出 ──

    def _output_info(self):
        progress = self.mcts.get_search_progress(self.board)
        if progress is None:
            return

        nodes = progress["nodes"]
        if nodes == 0:
            return

        elapsed = max(progress["elapsed"], 0.001)
        nps = int(nodes / elapsed)
        # 用搜索 root.q (minimax backup)
        q = progress.get("root_q", getattr(self, '_nn_raw_v', 0.0))
        cp = self.config.adj_to_cp(q)
        depth = max(progress["depth"], 1)
        sys.stdout.write(
            f"info depth {depth} "
            f"nodes {nodes} "
            f"nps {nps} "
            f"time {int(elapsed * 1000)} "
            f"score cp {cp} ")
        pv = progress.get("pv") or []
        if pv:
            sys.stdout.write(f"pv {' '.join(pv[:10])} ")
        sys.stdout.write("\n")
        sys.stdout.flush()
