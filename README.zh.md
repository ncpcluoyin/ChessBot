# ChessBot

纯 CNN + MCTS 国际象棋引擎，通过 Stockfish 蒸馏训练。

## 网络架构

| 组件 | 详情 |
|------|------|
| 主干 | 10 层 InceptionResBlock, 512 滤波器 |
| 卷积类型 | 3×2（水平）+ 2×3（垂直），隔层交替 |
| 参数量 | 22.63M |
| 棋盘编码 | 19 平面，rank-flip（走棋方视角） |
| 策略头 | Conv1×1 512→64 → FC 4096→512 → 4672, log_softmax |
| 价值头 | Conv1×1 512→64 → FC 4096→512 → 256 → 1, tanh |
| MCTS | GPU 批量推理，默认 2000 模拟，12 工作线程 |
| 直觉模式 | 纯 NN 不搜索，走法经 legal mask 过滤 |

## 文件结构

- `src/network.py` — ChessNet 模型定义
- `src/board.py` — 19 平面 rank-flip 编码，4672 走法编解码
- `src/train.py` — 蒸馏训练循环（EMA、余弦学习率、平衡采样）
- `src/sf_dataset.py` — 在线 63-sq → rank-flip 转换 + 正负平衡采样
- `src/mcts/` — GPU 批量 MCTS 引擎，持久工作线程池
- `src/uci.py` — UCI 协议处理，支持 IntuitionMode 选项
- `scripts/download_hf_dataset.py` — 下载 HuggingFace 监督数据集

## 训练

```bash
distill_daemon.bat
```

默认使用 HuggingFace Stockfish 数据（80 万局），训练 1800 epoch，batch_size=512。

## 运行

```bash
# UCI 模式（MCTS 搜索）
chessbot_fp.bat
```

## UCI 选项

- `IntuitionMode` — true/false，跳过 MCTS 直接使用 NN 策略
- `BatchSize` — GPU 批大小（默认 32）
- `Simulations` — MCTS 模拟次数（默认 2000）
- `NumWorkers` — MCTS 工作线程数（默认 12）

## 检查点

- `model_sf.pt` — 最新权重
- `model_sf_ema.pt` — 指数移动平均权重（decay=0.999）
- `model_sf_checkpoint.pt` — 完整训练状态（优化器、epoch 数等）
