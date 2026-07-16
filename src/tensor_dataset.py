"""
Batch dataset - reads original .pt files, converts FEN->tensor, yields batches.
"""

import glob
import os
import random
import chess
import torch

from src.board import board_to_tensor


class TensorBatchDataset:
    def __init__(self, data_dir: str, batch_size: int = 4096,
                 max_games: int = 0, game_offset: int = 0,
                 shuffle: bool = True):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle

        orig_files = sorted(glob.glob(os.path.join(data_dir, "*_batch_*.pt")))
        orig_files = [f for f in orig_files
                      if not any(f.endswith(s) for s in
                                 ("_tensor.pt", "_stacked.pt", "_final.pt"))]
        if not orig_files:
            raise FileNotFoundError(f"No sf_batch_*.pt files in {data_dir}")

        self._file_info = []
        all_games = []
        for fi, fpath in enumerate(orig_files):
            data = torch.load(fpath, map_location="cpu", weights_only=False)
            gls = list(data["game_lens"])
            del data
            cum = [0]
            for gl in gls:
                cum.append(cum[-1] + gl)
            self._file_info.append((fpath, gls, cum))
            for gi, gl in enumerate(gls):
                if gl > 0:
                    all_games.append((fi, gi))

        if not all_games:
            raise ValueError(f"No valid games in {data_dir}")
        self.total_games = len(all_games)
        self.all_games = all_games

        if game_offset > 0:
            self.all_games = self.all_games[game_offset:]
        if max_games > 0:
            self.all_games = self.all_games[:max_games]

    def __len__(self):
        return sum(self._file_info[fi][1][gi] for fi, gi in self.all_games)

    def __iter__(self):
        games = list(self.all_games)
        if self.shuffle:
            random.shuffle(games)

        file_groups = {}
        for fi, gi in games:
            file_groups.setdefault(fi, []).append(gi)
        file_order = sorted(file_groups.keys())
        random.shuffle(file_order)

        buf_t, buf_p, buf_v = [], [], []
        for fi in file_order:
            fpath, gls, cum = self._file_info[fi]
            data = torch.load(fpath, map_location="cpu", weights_only=False)["data"]
            for gi in file_groups[fi]:
                gl = gls[gi]
                s = cum[gi]
                for i in range(gl):
                    fen, moves_probs, value = data[s + i]
                    board = chess.Board(fen)
                    tensor = board_to_tensor(board)
                    buf_t.append(tensor)
                    buf_p.append(moves_probs)
                    buf_v.append(value)
                    if len(buf_t) >= self.batch_size:
                        yield self._build_batch(buf_t, buf_p, buf_v)
                        buf_t, buf_p, buf_v = [], [], []
            del data
        if buf_t:
            yield self._build_batch(buf_t, buf_p, buf_v)

    @staticmethod
    def _build_batch(buf_t, buf_p, buf_v):
        inputs = torch.stack(buf_t)
        B = len(buf_p)
        target_dist = torch.zeros(B, 4672)
        for i, pol in enumerate(buf_p):
            for move_idx, prob in pol:
                if move_idx is not None and 0 <= move_idx < 4672:
                    target_dist[i, move_idx] = prob
        values = torch.tensor(buf_v, dtype=torch.float32)
        return inputs, target_dist, values
