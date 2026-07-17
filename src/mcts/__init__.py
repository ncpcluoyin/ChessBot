"""
MCTS 引擎工厂 — 仅 GPU Batch。
"""

from src.mcts.engine import MCTSEngine, SearchResult


def get_mcts_engine(network, config, force: str = "auto",
                    model_path: str = None) -> MCTSEngine:
    from src.mcts.gpu_batch import BatchGPUEngine
    return BatchGPUEngine(network, config)
