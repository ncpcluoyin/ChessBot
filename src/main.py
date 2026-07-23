"""
ChessBot — UCI engine or distill training

Usage:
    python -m src.main uci [model_path]       # UCI engine
    python -m src.main distill --data DIR ... # SF distill training
"""

import argparse
import os
import sys

from src.config import Config
from src.network import ChessNet, create_model, load_model
from src.uci import ChessBotEngine
from src.train import train_distill


def main():
    parser = argparse.ArgumentParser(description="ChessBot")
    subparsers = parser.add_subparsers(dest="command")

    uci_parser = subparsers.add_parser("uci", help="UCI engine mode")
    uci_parser.add_argument("model", nargs="?", default=None, help="Model weights path")
    uci_parser.add_argument("--intuition", action="store_true",
                            help="Intuition mode: NN only, no MCTS")

    dist_parser = subparsers.add_parser("distill", help="SF distill training")
    dist_parser.add_argument("--data", required=True, help="SF 蒸馏数据目录")
    dist_parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    dist_parser.add_argument("--model", default=None, help="输出模型路径")
    dist_parser.add_argument("--workers", type=int, default=0, help="数据加载线程")
    dist_parser.add_argument("--resume", action="store_true", help="从已有模型续训")
    dist_parser.add_argument("--max-games", type=int, default=0, help="每 epoch 对局数 (0=全部)")
    dist_parser.add_argument("--game-offset", type=int, default=0, help="起始对局偏移")
    dist_parser.add_argument('--freeze', action='store_true', help='freeze backbone+policy, train value only')
    dist_parser.add_argument('--recover', action='store_true', help='freeze backbone, train new heads only')
    dist_parser.add_argument('--dual-lr', action='store_true', help='value head high LR, backbone low LR')
    dist_parser.add_argument('--castling-ratio', type=float, default=0.08,
                            help='每批中易位样本过采样比例 (默认 0.08)')

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return

    config = Config()

    if args.command == "uci":
        _run_uci(config, args.model, intuition=getattr(args, 'intuition', False))
    elif args.command == "distill":
        train_distill(config, data_dir=args.data, epochs=args.epochs,
                      model_path=args.model, num_workers=args.workers,
                      resume=args.resume, max_games=args.max_games,
                      game_offset=args.game_offset,
                      castling_ratio=args.castling_ratio)


def _run_uci(config: Config, model_path: str = None, intuition: bool = False):
    if model_path is None:
        paths = [p for p in glob.glob(os.path.join(config.model_dir, "model_*.pt"))
                 if "_checkpoint" not in p]
        if paths:
            model_path = max(paths, key=os.path.getmtime)
            print(f"info string auto selected: {model_path}", flush=True, file=sys.stderr)

    if model_path and os.path.exists(model_path):
        print(f"info string loading model {model_path}...", flush=True, file=sys.stderr)
        network = load_model(model_path, config)
    else:
        print("info string creating new model", flush=True, file=sys.stderr)
        network = create_model(config)

    network.eval()

    initial_opts = {}
    if intuition:
        initial_opts["intuition"] = True
    engine = ChessBotEngine(network, config, initial_options=initial_opts)
    engine.run()


if __name__ == "__main__":
    main()
