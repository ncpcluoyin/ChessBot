"""
MCTS 引擎工厂 — 自动选择最优实现。

选择逻辑:
  --engine cpu       → CPUWorkersMCTS   (多进程 CPU, 18.4M 模型 ~370 nps)
  --engine gpu       → GPUSingleMCTS    (单线程 GPU, 18.4M 模型 ~50 nps)
  --engine gpu_batch → BatchGPUEngine   (批量 GPU, 18.4M 模型 ~2400 nps)
  --engine auto      → 自动选择 (≥10M params + CUDA → gpu_batch, 否则 cpu)
"""

from src.mcts.engine import MCTSEngine, SearchResult


def get_mcts_engine(network, config, force: str = "auto",
                    model_path: str = None) -> MCTSEngine:
    """根据参数自动选择/强制指定 MCTS 引擎。"""
    import torch

    total_params = sum(p.numel() for p in network.parameters())

    if force == "cpu":
        use = "cpu"
    elif force == "gpu":
        use = "gpu"
    elif force == "gpu_batch":
        use = "gpu_batch"
    else:
        if torch.cuda.is_available() and total_params >= 10_000_000:
            use = "gpu_batch"
        else:
            use = "cpu"

    if use == "cpu":
        from src.mcts.cpu_workers import CPUWorkersMCTS
        return CPUWorkersMCTS(network, config, model_path=model_path)
    elif use == "gpu_batch":
        from src.mcts.gpu_batch import BatchGPUEngine
        return BatchGPUEngine(network, config)
    elif use == "gpu":
        from src.mcts.gpu_single import GPUSingleMCTS
        return GPUSingleMCTS(network, config, batch_size=1)
    else:
        from src.mcts.cpu_workers import CPUWorkersMCTS
        return CPUWorkersMCTS(network, config, model_path=model_path)
