"""将 checkpoint 中的 BN 融合到卷积权重, 适配无 BN 架构"""
import torch, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
cfg = Config()

# 1. 构造旧架构 (带 BN) 加载权重
from src.network_old import ChessNetWithBN
old_ckpt = torch.load('D:/models_bak/model_sf_epoch_0500.pt', map_location='cpu')
old_model = ChessNetWithBN(cfg)
old_model.load_state_dict(old_ckpt, strict=False)
old_model.eval()

# 2. 构造新架构 (无 BN)
from src.network import ChessNet
new_model = ChessNet(cfg)
new_state = {}

# 融合 conv_input: conv(15,256) + BN
conv_w = old_model.conv_input[0].weight.data  # (256,15,3,3)
bn = old_model.conv_input[1]
scale = bn.weight / (bn.running_var + bn.eps).sqrt()
new_w = conv_w * scale.view(-1, 1, 1, 1)
new_state['conv_input.0.weight'] = new_w

# 融合 policy_conv: conv(256,64) + BN
conv_w = old_model.policy_conv[0].weight.data
bn = old_model.policy_conv[1]
scale = bn.weight / (bn.running_var + bn.eps).sqrt()
new_w = conv_w * scale.view(-1, 1, 1, 1)
new_state['policy_conv.0.weight'] = new_w

# 融合 value_conv: conv(256,64) + BN (已无 BN, 直接拷贝)
try:
    conv_w = old_model.value_conv[0].weight.data
    if len(old_model.value_conv) > 1:
        bn = old_model.value_conv[1]
        scale = bn.weight / (bn.running_var + bn.eps).sqrt()
        new_w = conv_w * scale.view(-1, 1, 1, 1)
    else:
        new_w = conv_w
except:
    conv_w = old_model.value_conv[0].weight.data
    new_w = conv_w
new_state['value_conv.0.weight'] = new_w

# 融合 ResBlocks: conv1 + bn1, conv2 + bn2
for i in range(12):
    for ci in [1, 2]:
        conv = getattr(old_model.res_blocks[i], f'conv{ci}')
        bn = getattr(old_model.res_blocks[i], f'bn{ci}')
        conv_w = conv.weight.data
        scale = bn.weight / (bn.running_var + bn.eps).sqrt()
        new_w = conv_w * scale.view(-1, 1, 1, 1)
        new_state[f'res_blocks.{i}.conv{ci}.weight'] = new_w

# FC 层直接拷贝
for name in ['policy_fc1.weight', 'policy_fc1.bias',
             'policy_fc2.weight', 'policy_fc2.bias',
             'value_fc1.weight', 'value_fc1.bias',
             'value_fc_hidden.weight', 'value_fc_hidden.bias',
             'value_fc2.weight', 'value_fc2.bias']:
    if name in old_ckpt:
        new_state[name] = old_ckpt[name]

# 3. 加载到新模型并保存
new_model.load_state_dict(new_state, strict=False)
torch.save(new_model.state_dict(), 'data/models/model_sf.pt')
print(f"融合完成, 已保存到 data/models/model_sf.pt")
print(f"权重数: {len(new_state)}")
