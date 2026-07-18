import os
import sys
import math
from dataclasses import dataclass


def _default_sf_binary() -> str:
    """根据平台返回默认的 Stockfish 二进制文件名。"""
    if sys.platform == "win32":
        return "stockfish-windows-x86-64-avxvnni.exe"
    else:
        return "stockfish-ubuntu-x86-64-avx2"


def _default_device() -> str:
    """自动检测 CUDA 可用性。"""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


@dataclass
class Config:
    # --- 网络结构 ---
    num_filters: int = 512
    num_res_blocks: int = 10
    head_channels: int = 64       # 策略/价值头 conv1×1 的输出通道
    policy_fc_hidden: int = 512   # 策略头 FC 隐藏层维度
    value_fc_hidden: int = 256    # 价值头第二FC隐藏层维度 (第一FC硬编码512)
    value_fc_hidden2: int = 256   # 价值头第二 FC 隐藏层维度
    num_input_planes: int = 16
    num_history_frames: int = 1
    planes_per_frame: int = 8
    policy_output_dim: int = 4672
    bn_momentum: float = 0.01

    # --- MCTS ---
    mcts_simulations: int = 2000
    c_puct: float = 2.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    virtual_loss: float = 0.5
    num_mcts_workers: int = 12  # 默认 12 (平衡速度/内存), 可设 4-24
    temperature_threshold: int = 16

    # --- 训练 ---

    batch_size: int = 512
    rl_batch_size: int = 2048
    learning_rate: float = 0.002
    rl_learning_rate: float = 0.001
    weight_decay: float = 1e-4
    replay_buffer_size: int = 200_000
    train_epochs: int = 1
    device: str = "cpu"  # __post_init__ 会自动检测为 cuda（若有 GPU）
    value_label_scale: float = 1.0
    use_amp: bool = True

    # --- 自对弈 ---
    num_self_play_games: int = 100
    max_game_length: int = 200
    self_play_simulations: int = 800
    rl_parallel_games: int = 4
    temperature_start: float = 1.0
    temperature_decay: float = 0.8
    top_k_sampling: int = 3

    # --- Stockfish 蒸馏 ---
    sf_data_dir: str = "data/hf_supervised_samples"
    sf_binary: str = "stockfish-ubuntu-x86-64-avx2"  # 运行时会被 platform_defaults 覆盖
    sf_depth: int = 20
    sf_num_workers: int = 24

    # --- 时间管理 ---
    movetime_default: float = 1.0
    movetime_safety_margin: float = 0.05

    # --- 路径 ---
    model_dir: str = "data/models"

    def __post_init__(self):
        """初始化后自动设置平台相关默认值。"""
        self.sf_binary = _default_sf_binary()
        self.device = _default_device()

    @staticmethod
    def adj_to_cp(adj: float) -> int:
        """零中心值 [-1,1] 转 logit cp, clamp [-1000, 1000]."""
        adj = min(max(adj, -0.9999), 0.9999)
        win_prob = (adj + 1.0) / 2.0
        cp = 271.6 * math.log(win_prob / (1.0 - win_prob + 1e-30))
        return int(min(max(cp, -10000), 10000))

    def time_for_move(self, remaining_ms: float, increment_ms: float = 0.0) -> float:
        t = remaining_ms / 1000.0
        inc = increment_ms / 1000.0
        return max(min(t / 30.0 + inc, t / 5.0), 0.1)
