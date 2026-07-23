"""
价值头三分类恢复训练 (交叉熵)
冻结策略头+骨干网, BN 分别处理 (eval/train)
"""
import os, sys, glob, gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
from torch.utils.data import DataLoader, IterableDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.config import Config
from src.network import load_model
from src.board import board_to_tensor

torch.set_float32_matmul_precision('high')
device = 'cuda'

# ─── 配置 ────────────────────────────────────────────
MODEL_PATH = "data/models/model_sf.pt"
CHECKPOINT_PATH = "data/models/model_sf_checkpoint.pt"
DATA_DIR = "data/hf_supervised_samples"

BATCH_SIZE = 512
LR = 3e-4
EPOCHS = 10
WEIGHT_DECAY = 1e-2           # L2 正则加强
VALUE_LOSS_WEIGHT = 24.0      # 交叉熵 loss 倍率
CLASS_THRESHOLD = 0.2          # |eval| ≤ 0.2 → 和棋, 之外用硬标签
MAX_FILES = 7900

# ─── 标签转换 ────────────────────────────────────────
def value_to_3class(values, threshold=CLASS_THRESHOLD):
    """values: (N,) float in [-1,1]
    0=黑胜(硬), 1=和棋(|eval|≤thr), 2=白胜(硬)"""
    c = torch.full_like(values, 1, dtype=torch.long)  # 默认和棋
    c[values > threshold] = 2
    c[values < -threshold] = 0
    return c

# ─── 数据加载 ────────────────────────────────────────
class HFValueDataset(IterableDataset):
    def __init__(self, config, max_files=MAX_FILES):
        self.config = config
        self.files = sorted(glob.glob(os.path.join(DATA_DIR, "hf_batch_*.pt")))
        self.rng = np.random.default_rng(42)
        self.rng.shuffle(self.files)
        self.files = self.files[:max_files]

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        files = self.files
        if worker_info is not None:
            files = files[worker_info.id::worker_info.num_workers]

        for f in files:
            try:
                data = torch.load(f, map_location='cpu', weights_only=True)
            except:
                continue
            items = data['data']
            idx = self.rng.choice(len(items), min(len(items), 128), replace=False)

            batch_x, batch_c = [], []
            for i in idx:
                fen, _, value_raw = items[i]
                try:
                    board = chess.Board(fen)
                    val = -value_raw if board.turn == chess.BLACK else value_raw
                    val = max(-1.0, min(1.0, val))
                    batch_x.append(board_to_tensor(board))
                    batch_c.append(value_to_3class(torch.tensor([val]))[0].item())
                except:
                    continue
            if not batch_x:
                continue
            yield torch.stack(batch_x), torch.tensor(batch_c, dtype=torch.long)

# ─── 模型修改 ────────────────────────────────────────
def patch_model_3class(model):
    """替换 value_fc2 1→3, 打 patch 去掉 tanh"""
    model.value_fc2 = nn.Linear(256, 3)
    nn.init.normal_(model.value_fc2.weight, mean=0, std=0.01)
    nn.init.zeros_(model.value_fc2.bias)

    orig_fwd = model.forward
    def patched_fwd(self, x, legal_mask=None):
        x = self.conv_input(x)
        for blk in self.res_blocks:
            x = blk(x)
        # policy (unchanged)
        p = self.policy_conv(x)
        p = p.reshape(p.size(0), -1)
        p = F.gelu(self.policy_fc1(p))
        logits = self.policy_fc2(p)
        if legal_mask is not None:
            logits = logits.masked_fill(~legal_mask, -1e4)
        p_out = F.log_softmax(logits, dim=-1)
        # value: 3-class softmax (硬标签: 负/和/胜)
        v = self.value_conv(x)
        v = self.value_reduce(v)
        v = F.gelu(self.value_fc1(v.flatten(1)))
        v = F.gelu(self.value_fc_hidden(v))
        v_logits = self.value_fc2(v)  # [B, 3] raw logits
        v_probs = F.softmax(v_logits, dim=1)  # [p_loss, p_draw, p_win]
        q = v_probs[:, 2] - v_probs[:, 0]    # win - loss → scalar [-1,1]
        # 存 logits 给训练用 CE
        self._last_v_logits = v_logits.detach() if not self.training else v_logits
        return p_out, q  # policy + scalar q (MCTS 兼容)

    model.forward = patched_fwd.__get__(model, type(model))
    return model

def freeze_non_value(model):
    """冻结非价值头, BN 骨干→eval, 价值→train"""
    for n, p in model.named_parameters():
        p.requires_grad = n.startswith('value_')
    n_t = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable: {n_t:,} params (value head only)")

    for n, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm2d,)):
            if n.startswith('value_'):
                m.train()
            else:
                m.eval()

# ─── EMA ─────────────────────────────────────────────
@torch.no_grad()
def ema_update(ema, params, decay=0.999):
    for e, p in zip(ema, params):
        e.copy_(e * decay + p * (1 - decay))

# ─── 训练 ────────────────────────────────────────────
def run():
    print(f"Device: {device}")
    print(f"Batch={BATCH_SIZE}  LR={LR}  Epochs={EPOCHS}")
    print(f"ValueLossWeight={VALUE_LOSS_WEIGHT}  WD={WEIGHT_DECAY}  Thr={CLASS_THRESHOLD} (硬标签)")

    config = Config()
    model = load_model(MODEL_PATH, config).to(device)
    tot = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Loaded: {tot:.1f}M params")

    model = patch_model_3class(model)
    freeze_non_value(model)

    value_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(value_params, lr=LR, weight_decay=WEIGHT_DECAY)
    ema_p = [p.detach().clone() for p in value_params]

    # checkpoint resume
    start_epoch = 0
    if os.path.exists(CHECKPOINT_PATH):
        ck = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
        # strict=False because value_fc2 changed shape
        model.load_state_dict(ck['model'], strict=False)
        opt.load_state_dict(ck['optimizer'])
        ema_p = [x.to(device) for x in ck['ema']]
        start_epoch = ck['epoch'] + 1
        print(f"Resumed epoch {start_epoch}")

    dataset = HFValueDataset(config, max_files=MAX_FILES)
    loader = DataLoader(dataset, batch_size=None, num_workers=0)

    for epoch in range(start_epoch, EPOCHS):
        # Re-apply freeze (model.train() would turn all BN→train, we undo it)
        freeze_non_value(model)
        model.train()

        total, ce_sum, acc_sum = 0.0, 0.0, 0.0
        n_b = 0

        for inputs, classes in loader:
            inputs = inputs.to(device, non_blocking=True)
            classes = classes.to(device, non_blocking=True)

            opt.zero_grad()
            _, _ = model(inputs)  # forward, q scalar stored in _last_v_logits
            v_logits = model._last_v_logits
            ce = F.cross_entropy(v_logits, classes)
            loss = ce * VALUE_LOSS_WEIGHT
            loss.backward()
            opt.step()
            ema_update(ema_p, value_params)

            acc = (v_logits.argmax(dim=1) == classes).float().mean()
            total += loss.item()
            ce_sum += ce.item()
            acc_sum += acc.item()
            n_b += 1

            if n_b % 200 == 0:
                print(f"  ep{epoch} b{n_b:4d}  loss={total/n_b:.4f}  ce={ce_sum/n_b:.4f}  acc={acc_sum/n_b*100:.1f}%")

        n = max(n_b, 1)
        print(f"Epoch {epoch}:  loss={total/n:.4f}  ce={ce_sum/n:.4f}  acc={acc_sum/n*100:.1f}%")

        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': opt.state_dict(),
            'ema': [x.cpu() for x in ema_p],
        }, CHECKPOINT_PATH)
        print(f"  Checkpoint saved")

        gc.collect()
        torch.cuda.empty_cache()

    # 保存 (保存 state_dict, 后续推理需同样 patch)
    torch.save(model.state_dict(), "data/models/model_sf_3class.pt")
    print("Done → data/models/model_sf_3class.pt")

if __name__ == '__main__':
    run()
