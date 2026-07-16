"""
MCTS 引擎接口 — 所有 MCTS 实现的统一抽象。
"""

from dataclasses import dataclass
import chess
import numpy as np
import threading


@dataclass
class SearchResult:
    """一次搜索的完整结果。"""
    best_move: chess.Move | None
    best_move_uci: str
    policy: np.ndarray          # (4672,) 策略分布
    root_value: float           # 根节点 Q 值
    nodes_searched: int         # 总搜索节点数
    max_depth: int              # 最大搜索深度
    time_elapsed: float         # 搜索用时 (秒)
    pv: list = None             # PV 走法列表 (UCI 字符串)


class MCTSEngine:
    """MCTS 引擎抽象基类。"""

    def reset(self):
        """重置引擎状态（新棋局开始时调用）。"""
        pass

    def search(
        self,
        board: chess.Board,
        num_simulations: int | None = None,
        time_limit: float | None = None,
        stop_event: threading.Event | None = None,
        nn_raw_value: float | None = None,
    ) -> SearchResult:
        """执行 MCTS 搜索，返回最佳走法。"""
        raise NotImplementedError

    def get_search_progress(self, board: chess.Board) -> dict:
        """返回当前搜索进度（用于 UCI info 输出）。"""
        return {
            "nodes": 0,
            "elapsed": 0.001,
            "depth": 0,
            "root_q": 0.0,
            "pv": [],
        }
