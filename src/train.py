"""
训练管线 — 自对弈强化学习 (AlphaZero 风格)
"""

import os
import gc
import math
import random
import signal
import time
import copy
from collections import deque

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from src.config import Config
from src.network import create_model, save_model, load_model
from src.board import board_to_tensor
from src.sf_dataset import SFDistillDataset, collate_fn_distill


def _worker_init_fn(worker_id):
    """DataLoader worker 的 SIGINT 屏蔽 (pickle-safe 模块级函数)。"""
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)


class ReplayBuffer:
    def __init__(self, max_size: int = 200_000):
        self.buffer = deque(maxlen=max_size)

    def add(self, samples: list):
        self.buffer.extend(samples)

    def sample(self, batch_size: int) -> list:
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        return random.sample(list(self.buffer), batch_size)

    def __len__(self):
        return len(self.buffer)


def collate_fn_selfplay(batch):
    inputs = torch.stack([s[0] for s in batch])
    policies = torch.from_numpy(np.stack([s[1] for s in batch])).float()
    values = torch.tensor([s[2] for s in batch], dtype=torch.float32)
    return inputs, policies, values


# ═══════════════════════════════════════════════════════════════════
# SF 蒸馏训练
# ═══════════════════════════════════════════════════════════════════

def train_distill(config: Config, data_dir: str, epochs: int = 100,
                  model_path: str = None, num_workers: int = 0,
                  resume: bool = False, max_games: int = 0,
                  game_offset: int = 0):
    if model_path is None:
        model_path = os.path.join(config.model_dir, "model_sf.pt")

    start_epoch = 0
    opt_state = None
    sched_state = None
    total_epochs = epochs

    if resume and os.path.exists(model_path):
        model = load_model(model_path, config)
        model.train()

        # 初始化 EMA 参数 (与模型结构绑定, 在加载 checkpoint 前)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        ema_params = [p.detach().clone() for p in trainable_params]
        ema_loaded = False

        cp_path = model_path.replace(".pt", "_checkpoint.pt")
        if os.path.exists(cp_path):
            ckpt = torch.load(cp_path, map_location="cpu", weights_only=False)
            start_epoch = ckpt.get("epoch", 0)
            total_epochs = ckpt.get("total_epochs", epochs)
            opt_state = ckpt.get("optimizer")
            sched_state = ckpt.get("scheduler")
            # 加载 EMA 参数
            ema_ckpt = ckpt.get("ema_params")
            if ema_ckpt is not None and len(ema_ckpt) == len(ema_params):
                skipped_ema = 0
                for dst, src in zip(ema_params, ema_ckpt):
                    if dst.shape == src.shape:
                        dst.copy_(src)
                    else:
                        skipped_ema += 1
                if skipped_ema:
                    print(f"  EMA 跳过 {skipped_ema} 个形状不匹配的参数")
                ema_loaded = True
                print(f"  EMA 已加载 ({len(ema_ckpt)} 参数)")
            # 如果 CLI --epochs 大于 checkpoint 的 total_epochs, 扩展周期
            if epochs > total_epochs:
                total_epochs = epochs
                sched_state = None
                print(f"  调度器重置: T_max={total_epochs}")
        print(f"从 epoch {start_epoch} 续训训练")
    else:
        model = create_model(config).to(config.device)
        model.train()
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        ema_params = [p.detach().clone() for p in trainable_params]

    total_games = SFDistillDataset(data_dir, max_games=0, game_offset=0).total_games
    # 启用 TensorFloat-32 (RTX 30xx+ Tensor Core), fp32 精度 2x 加速
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.allow_tf32 = True

    print(f"SF 蒸馏数据: {data_dir}  ({total_games} 局)")
    print(f"  epochs={total_epochs}  batch_size={config.batch_size}  lr={config.learning_rate}")
    print(f"  max_games={max_games or '全部'}  start_offset={game_offset}  workers={num_workers}")
    print(f"  device={config.device}")

    # ── 骨干+策略头 与 值头 分离优化器 ──
    backbone_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and not n.startswith('value_')]
    value_params = [p for n, p in model.named_parameters()
                    if p.requires_grad and n.startswith('value_')]
    optimizer = torch.optim.AdamW(
        backbone_params, lr=config.learning_rate, weight_decay=config.weight_decay)
    value_optimizer = torch.optim.AdamW(
        value_params, lr=config.learning_rate, weight_decay=config.weight_decay)

    # 余弦衰减 (手动计算每个 epoch 的 LR)
    _lr_min = config.learning_rate * 0.01
    _lr_range = config.learning_rate - _lr_min

    if sched_state:
        # 兼容旧 checkpoint (单优化器含全部参数 vs 双优化器)
        try:
            optimizer.load_state_dict(opt_state)
        except ValueError as e:
            print(f"  optimizer 兼容性跳过: {e}")
        if 'value_optimizer' in ckpt:
            try:
                value_optimizer.load_state_dict(ckpt['value_optimizer'])
            except ValueError as e:
                print(f"  value_optimizer 兼容性跳过: {e}")
    if sched_state:
        # 忽略旧调度器状态, 手动余弦衰减
        pass

    # 调度器重置时, 按当前 epoch 位置计算初始 LR, 不从头爬
    if sched_state is None and start_epoch > 0:
        _init_lr = _lr_min + 0.5 * _lr_range * (1 + math.cos(math.pi * start_epoch / total_epochs))
        for g in optimizer.param_groups:
            g['lr'] = _init_lr
        for g in value_optimizer.param_groups:
            g['lr'] = _init_lr
        print(f"  初始 LR: {_init_lr:.6f} (epoch {start_epoch}/{total_epochs})")

    _interrupted = False
    def _on_sigint(sig, frame):
        nonlocal _interrupted
        _interrupted = True
    old = signal.signal(signal.SIGINT, _on_sigint)

    try:
        ema_decay = 0.999
        for epoch_idx in range(start_epoch, epochs):
            if _interrupted:
                break
            epoch = epoch_idx

            # ── 手动余弦 LR ──
            cos_val = 0.5 * (1 + math.cos(math.pi * epoch / total_epochs))
            _cos_lr = _lr_min + _lr_range * cos_val
            for g in optimizer.param_groups:
                g['lr'] = _cos_lr
            for g in value_optimizer.param_groups:
                g['lr'] = _cos_lr

            for p in model.parameters():
                p.requires_grad = True

            if max_games > 0:
                max_offset = max(0, total_games - max_games)
                offset = random.randint(0, max_offset)
            else:
                offset = 0

            _epoch_t0 = time.perf_counter()
            dataset = SFDistillDataset(data_dir, max_games=max_games,
                                        game_offset=offset,
                                        batch_size=config.batch_size)
            _epoch_ds = time.perf_counter()
            dataloader = None  # 直接迭代 dataset (预组装批)

            total_loss = torch.tensor(0.0, device=config.device)
            total_policy_loss = torch.tensor(0.0, device=config.device)
            total_value_loss = torch.tensor(0.0, device=config.device)
            n_batches = 0
            _batch_load_t = 0.0
            _batch_forward_t = 0.0
            _batch_backward_t = 0.0

            for batch_input, batch_target_dist, batch_value in dataset:
                if _interrupted:
                    break
                _t0 = time.perf_counter()
                # 锁页 + 异步传输
                batch_input = batch_input.pin_memory().to(config.device, non_blocking=True)
                batch_target_dist = batch_target_dist.pin_memory().to(config.device, non_blocking=True)
                batch_value = batch_value.pin_memory().to(config.device, non_blocking=True)
                optimizer.zero_grad()
                value_optimizer.zero_grad()
                _t1 = time.perf_counter()

                policy_log_probs, value_pred = model(batch_input)
                value_pred = value_pred.squeeze(-1)

                # ── 平滑策略目标 ──
                with torch.no_grad():
                    from src.board import move_index_to_uci
                    model_probs = policy_log_probs.exp()  # (B, 4672)
                    sf_labels = batch_target_dist.argmax(dim=-1)  # (B,)
                    model_top1 = model_probs.argmax(dim=-1)
                    match = (model_top1 == sf_labels)  # (B,) bool

                    k = 10
                    alpha = 0.35
                    smooth_targets = []
                    for i in range(model_probs.shape[0]):
                        probs = model_probs[i].cpu().numpy()
                        label_idx = int(sf_labels[i])

                        if match[i]:
                            # 正确: 算锐度, 高才平滑
                            idx = np.argsort(probs)[::-1][:k]
                            top_p = np.maximum(probs[idx], 1e-10)
                            if label_idx not in idx:
                                idx = np.append(idx[:k-1], label_idx)
                                top_p = np.maximum(probs[idx], 1e-10)

                            # 锐度 = top-1 概率 (越高越尖锐)
                            sharpness = top_p[0]
                            if sharpness > 0.5:
                                # 高锐度: 幂变换平滑 (易位 ×2 在变换前)
                                uci = move_index_to_uci(label_idx)
                                if uci in ('e1g1', 'e8g8', 'e1c1', 'e8c8'):
                                    pos = np.where(idx == label_idx)[0]
                                    if len(pos) > 0:
                                        top_p[pos[0]] *= 2.0
                                q = top_p ** alpha
                                q /= q.sum()
                            else:
                                # 已够平滑, 保持原分布 (仅归一化)
                                q = top_p / top_p.sum()
                            out = np.zeros(4672, dtype=np.float32)
                            out[idx] = q
                        else:
                            # 错误: 标签 50%, 模型 top-5 等差 14/12/10/8/6
                            order = np.argsort(probs)[::-1]
                            model_top = [j for j in order if j != label_idx][:5]
                            out = np.zeros(4672, dtype=np.float32)
                            out[label_idx] = 0.50
                            for rank, j in enumerate(model_top):
                                out[j] = 0.16 - rank * 0.02  # 14%, 12%, 10%, 8%, 6%
                        smooth_targets.append(out)
                    smooth_targets = torch.from_numpy(np.stack(smooth_targets)).to(config.device)

                policy_loss = -(smooth_targets * policy_log_probs).sum(dim=-1).mean()

                # 策略头熵正则: 鼓励平滑分布
                with torch.no_grad():
                    policy_probs = policy_log_probs.exp()
                    policy_entropy = -(policy_probs * policy_log_probs).sum(dim=-1).mean()
                policy_loss = policy_loss - config.policy_entropy_weight * policy_entropy

                # 三分类 CE
                thr = config.value_class_threshold
                v_label_raw = (batch_value * config.value_label_scale).clamp(-1, 1)
                v_class = torch.full_like(v_label_raw, 1, dtype=torch.long)
                v_class[v_label_raw > thr] = 2
                v_class[v_label_raw < -thr] = 0
                v_logits = model._last_value_logits
                value_loss = F.cross_entropy(v_logits, v_class)
                loss = policy_loss + 3.0 * value_loss

                _t2 = time.perf_counter()
                _batch_load_t += _t1 - _t0
                _batch_forward_t += _t2 - _t1

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                value_optimizer.step()
                # ── EMA 更新 ──
                with torch.no_grad():
                    for p, ema_p in zip(trainable_params, ema_params):
                        ema_p.mul_(ema_decay).add_(p.detach(), alpha=1 - ema_decay)
                _t3 = time.perf_counter()
                _batch_backward_t += _t3 - _t2

                total_loss += loss.detach()
                total_policy_loss += policy_loss.detach()
                total_value_loss += value_loss.detach()
                n_batches += 1

            # 只在 epoch 结束时同步一次
            avg_loss = (total_loss / n_batches).item()
            avg_p_loss = (total_policy_loss / n_batches).item()
            avg_v_loss = (total_value_loss / n_batches).item()
            _epoch_t = time.perf_counter() - _epoch_t0
            _epoch_ds_t = _epoch_ds - _epoch_t0
            # Python 开销 = 总时间 - ds - load - forward - backward - save
            _epoch_save_t0 = time.perf_counter()
            print(f"Epoch {epoch + 1}/{epochs}  "
                  f"loss={avg_loss:.4f}  policy={avg_p_loss:.4f}  value={avg_v_loss:.4f}  "
                  f"lr={optimizer.param_groups[0]['lr']:.6f}  "
                  f"time={_epoch_t:.1f}s "
                  f"[ds={_epoch_ds_t:.1f}s load={_batch_load_t:.1f}s "
                  f"fwd={_batch_forward_t:.1f}s bwd={_batch_backward_t:.1f}s]")

            if _interrupted:
                break

            save_model(model, model_path)
            cp_path = model_path.replace(".pt", "_checkpoint.pt")
            torch.save({
                "epoch": epoch + 1,
                "total_epochs": total_epochs,
                "optimizer": optimizer.state_dict(),
                "value_optimizer": value_optimizer.state_dict(),
                "loss": avg_loss,
                "ema_params": [p.clone() for p in ema_params],
            }, cp_path)

            # 保存 EMA 模型 (不影响原 model_path)
            ema_path = model_path.replace(".pt", "_ema.pt")
            _save_ema_model(model, trainable_params, ema_params, ema_path)

            # 每20 epoch 备份到 D:\models_bak
            if (epoch + 1) % 20 == 0:
                bak_dir = "D:/models_bak"
                os.makedirs(bak_dir, exist_ok=True)
                bak_path = os.path.join(bak_dir, f"model_sf_epoch_{epoch+1:04d}.pt")
                save_model(model, bak_path)
                # EMA 备份
                _save_ema_model(model, trainable_params, ema_params,
                                bak_path.replace(".pt", "_ema.pt"))
                # 同时复制 checkpoint (含优化器状态, 可续训)
                bak_cp = os.path.join(bak_dir, f"model_sf_epoch_{epoch+1:04d}_checkpoint.pt")
                torch.save({
                    "epoch": epoch + 1,
                    "total_epochs": total_epochs,
                    "optimizer": optimizer.state_dict(),
                    "value_optimizer": value_optimizer.state_dict(),
                    "loss": avg_loss,
                    "ema_params": [p.clone() for p in ema_params],
                }, bak_cp)
                print(f"  [备份] {bak_path}")

            _save_t = time.perf_counter() - _epoch_save_t0
            _py_t = _epoch_t - _epoch_ds_t - _batch_load_t - _batch_forward_t - _batch_backward_t - _save_t
            print(f"  save={_save_t:.1f}s py={_py_t:.1f}s", flush=True)

            del dataloader, dataset
            gc.collect()

    finally:
        signal.signal(signal.SIGINT, old)

    if _interrupted and n_batches > 0:
        save_model(model, model_path)
        cp_path = model_path.replace(".pt", "_checkpoint.pt")
        torch.save({
            "epoch": epoch,
            "total_epochs": total_epochs,
            "optimizer": optimizer.state_dict(),
            "value_optimizer": value_optimizer.state_dict(),
            "loss": total_loss / n_batches,
            "ema_params": [p.clone() for p in ema_params],
        }, cp_path)
        _save_ema_model(model, trainable_params, ema_params,
                        model_path.replace(".pt", "_ema.pt"))
        print(f"\n收到中断，已保存 (epoch {epoch})")

    gc.collect()
    print(f"模型已保存到 {model_path}")

def _save_ema_model(model, trainable_params, ema_params, ema_path):
    """将 EMA 权重写入模型副本并保存。"""
    # 清理可能挂着的计算图张量
    for attr in ['_last_value_logits']:
        if hasattr(model, attr):
            setattr(model, attr, None)
    model_cpu = copy.deepcopy(model).cpu()
    state = model_cpu.state_dict()
    ema_idx = 0
    for name, param in model_cpu.named_parameters():
        if ema_idx < len(ema_params):
            state[name].copy_(ema_params[ema_idx])
            ema_idx += 1
    model_cpu.load_state_dict(state)
    torch.save(model_cpu.state_dict(), ema_path)
    del model_cpu
