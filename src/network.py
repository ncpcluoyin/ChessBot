"""
ChessNet — 纯 CNN 架构.

输入: (batch, num_input_planes, 8, 8)
输出: (policy_log_probs, value)

结构:
  conv_input: num_input_planes → num_filters, k=3, BN, GELU
  ResBlocks ×num_res_blocks: num_filters → num_filters, k=3, BN, GELU, shortcut
  ── 策略头 ──
    conv1×1: num_filters → head_channels, BN, GELU
    FC: head_channels*64 → policy_fc_hidden, GELU
    FC: policy_fc_hidden → policy_output_dim, log_softmax
  ── 价值头 ──
    conv1×1: num_filters → head_channels, BN, GELU
    FC: head_channels*64 → value_fc_hidden, GELU
    FC: value_fc_hidden → value_fc_hidden2, GELU
    FC: value_fc_hidden2 → 1, tanh

提供 create_model() 和 detect_architecture() 实现灵活的模型创建/加载。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

from src.config import Config


class ChessNet(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        nf = config.num_filters
        hc = config.head_channels
        fc_h = config.policy_fc_hidden
        v_h = config.value_fc_hidden
        v_h2 = config.value_fc_hidden2
        n_blocks = config.num_res_blocks

        # CNN backbone
        self.conv_input = nn.Sequential(
            nn.Conv2d(config.num_input_planes, nf, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
        )

        self.res_blocks = nn.ModuleList(
            [InceptionResBlock(nf, i) for i in range(n_blocks)]
        )

        # ── 策略头 ──
        self.policy_conv = nn.Sequential(
            nn.Conv2d(nf, hc, kernel_size=1, bias=False),
            nn.BatchNorm2d(hc),
            nn.GELU(),
        )
        self.policy_fc1 = nn.Linear(hc * 8 * 8, fc_h)
        self.policy_fc2 = nn.Linear(fc_h, config.policy_output_dim)

        # ── 价值头 ──
        self.value_conv = nn.Sequential(
            nn.Conv2d(nf, hc, kernel_size=1, bias=False),
            nn.BatchNorm2d(hc),
            nn.GELU(),
        )
        self.value_fc1 = nn.Linear(hc * 8 * 8, 512)
        self.value_fc_hidden = nn.Linear(512, v_h)
        self.value_fc2 = nn.Linear(v_h, 1)

        self._init_weights()
        nn.init.normal_(self.value_fc2.weight, mean=0, std=0.01)
        nn.init.zeros_(self.value_fc2.bias)

    def _init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) and 'value_fc2' not in name:
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, legal_mask: torch.Tensor = None):
        x = self.conv_input(x)
        for block in self.res_blocks:
            x = block(x)

        # 策略头
        p = self.policy_conv(x)
        p = p.reshape(p.size(0), -1)
        p = F.gelu(self.policy_fc1(p))
        policy_logits = self.policy_fc2(p)

        if legal_mask is not None:
            policy_logits = policy_logits.masked_fill(~legal_mask, -1e4)

        policy_log_probs = F.log_softmax(policy_logits, dim=-1)

        # 价值头
        v = self.value_conv(x)
        v = v.reshape(v.size(0), -1)
        v = F.gelu(self.value_fc1(v))
        v = F.gelu(self.value_fc_hidden(v))
        value = torch.tanh(self.value_fc2(v))

        return policy_log_probs, value


class InceptionResBlock(nn.Module):
    """并联双路径, 各半通道. 隔层交换 3×2/2×3 顺序.
    偶层: Path A=3×2, Path B=2×3
    奇层: Path A=2×3, Path B=3×2
    """
    def __init__(self, num_filters: int, layer_idx: int):
        super().__init__()
        n = num_filters
        nc = n // 2
        ks_a = (3, 2) if layer_idx % 2 == 0 else (2, 3)
        ks_b = (2, 3) if layer_idx % 2 == 0 else (3, 2)
        self.conv_a = nn.Conv2d(n, nc, kernel_size=ks_a, bias=False)
        self.bn_a = nn.BatchNorm2d(nc)
        self.conv_b = nn.Conv2d(n, n - nc, kernel_size=ks_b, bias=False)
        self.bn_b = nn.BatchNorm2d(n - nc)
        self.out_bn = nn.BatchNorm2d(n)
        self.ks_a = ks_a
        self.ks_b = ks_b

    @staticmethod
    def _pad(x, ks):
        k_h, k_w = ks
        l = (k_w - 1) // 2; r = k_w - 1 - l
        t = (k_h - 1) // 2; b = k_h - 1 - t
        return F.pad(x, (l, r, t, b))

    def forward(self, x):
        residual = x
        pa = F.gelu(self.bn_a(self.conv_a(self._pad(x, self.ks_a))))
        pb = F.gelu(self.bn_b(self.conv_b(self._pad(x, self.ks_b))))
        out = torch.cat([pa, pb], dim=1)
        out = F.gelu(self.out_bn(out))
        return out + residual


def create_model(config: Config = None) -> ChessNet:
    if config is None:
        config = Config()
    return ChessNet(config)


def save_model(model: ChessNet, path: str):
    torch.save(model.state_dict(), path)


def load_model(path: str, config: Config = None) -> ChessNet:
    """加载模型权重, 自动跳过形状不匹配的键。"""
    state = torch.load(path, map_location="cpu", weights_only=False)

    if config is None:
        config = Config()

    model = ChessNet(config)
    model_state = model.state_dict()

    filtered = {}
    skipped = []
    for k, v in state.items():
        if k in model_state and model_state[k].shape == v.shape:
            filtered[k] = v
        else:
            skipped.append(k)

    model.load_state_dict(filtered, strict=False)
    if skipped:
        warnings.warn(f"Skipped {len(skipped)} incompatible keys: {skipped[:5]}...")

    model = model.to(config.device)
    model.eval()
    return model
