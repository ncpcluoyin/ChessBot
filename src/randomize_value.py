"""
Load model_sf_ema.pt, randomize value head, save as temp, run test_nn_vs_mcts.py.
"""

import os, sys, tempfile
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import Config
from src.network import load_model, save_model


def randomize_value_head(model):
    """Re-init all value head conv/fc layers."""
    for name, mod in model.named_modules():
        if 'value_head' in name and isinstance(mod, (torch.nn.Conv2d, torch.nn.Linear)):
            torch.nn.init.orthogonal_(mod.weight, gain=1.0)
            if mod.bias is not None:
                torch.nn.init.zeros_(mod.bias)
        elif 'value_head' in name and isinstance(mod, torch.nn.BatchNorm2d):
            mod.reset_running_stats()
            torch.nn.init.constant_(mod.weight, 1.0)
            torch.nn.init.zeros_(mod.bias)
    print("Value head randomized.")


if __name__ == '__main__':
    src = "data/models/model_sf_ema.pt"
    dst = "data/models/model_sf_ema_random_val.pt"

    config = Config()
    model = load_model(src, config).cuda().eval()
    randomize_value_head(model)
    save_model(model, dst)
    print(f"Saved to {dst}")
    print(f"\nNow run:\n  .venv311\\Scripts\\python.exe -u tests\\test_nn_vs_mcts.py --model {dst}")
