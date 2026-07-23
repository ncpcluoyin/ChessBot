"""
SF 蒸馏数据集 — 流式加载，逐文件读入，不爆内存

样本格式: dict {"data": [(fen, [(move_idx, prob), ...], value), ...], "game_lens": [n1, ...]}
"""

import glob
import os
import random
import chess
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from src.board import board_to_tensor


# ========== 旧 63-sq 转新 rank-flip 编码 ==========
_QUEEN_DIRS = [(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1)]
_KNIGHT_OFFS = [(-1,-2),(-1,2),(-2,-1),(-2,1),(1,-2),(1,2),(2,-1),(2,1)]
_UNDERPROS = [(-1,chess.KNIGHT),(-1,chess.BISHOP),(-1,chess.ROOK),
              (0,chess.KNIGHT),(0,chess.BISHOP),(0,chess.ROOK),
              (1,chess.KNIGHT),(1,chess.BISHOP),(1,chess.ROOK)]

def _old_idx2move(index, board):
    """63-sq 索引 -> UCI 走法 (硬编码)"""
    from_sq = index // 73; mt = index % 73
    fi = chess.square_file(from_sq); ri = chess.square_rank(from_sq)
    if mt < 56:
        d = mt // 7; dist = (mt % 7) + 1
        ddx,ddy = _QUEEN_DIRS[d]
        tf = fi + ddx*dist; tr = ri + ddy*dist
        promo = None
        if ri == 6 and tr == 7:
            real_from = chess.square(fi, ri)
            if board.turn == chess.BLACK: real_from = 63 - real_from
            p = board.piece_at(real_from)
            if p and p.piece_type == chess.PAWN: promo = chess.QUEEN
        rf = chess.square(fi, ri); rt = chess.square(tf, tr)
        if board.turn == chess.BLACK: rf,rt = 63-rf, 63-rt
        return chess.Move(rf, rt, promotion=promo)
    elif mt < 64:
        k = mt - 56; ddx,ddy = _KNIGHT_OFFS[k]
        rf = chess.square(fi, ri); rt = chess.square(fi+ddx, ri+ddy)
        if board.turn == chess.BLACK: rf,rt = 63-rf, 63-rt
        return chess.Move(rf, rt)
    else:
        u = mt - 64; dx = u // 3 - 1
        promo = {0:chess.KNIGHT,1:chess.BISHOP,2:chess.ROOK}[u % 3]
        rf = chess.square(fi, ri); rt = chess.square(fi+dx, ri+1)
        if board.turn == chess.BLACK: rf,rt = 63-rf, 63-rt
        return chess.Move(rf, rt, promotion=promo)

def _new_move2idx(move, board):
    """UCI 走法 -> rank-flip 索引 (硬编码)"""
    from_sq = move.from_square; to_sq = move.to_square
    if board.turn == chess.BLACK:
        from_sq = chess.square(chess.square_file(from_sq), 7 - chess.square_rank(from_sq))
        to_sq = chess.square(chess.square_file(to_sq), 7 - chess.square_rank(to_sq))
    dx = chess.square_file(to_sq) - chess.square_file(from_sq)
    dy = chess.square_rank(to_sq) - chess.square_rank(from_sq)
    def _qi():
        for d,(ddx,ddy) in enumerate(_QUEEN_DIRS):
            dist=max(abs(dx),abs(dy))
            if dist==0 or dist>7: continue
            if ddx*dist==dx and ddy*dist==dy: return d*7+dist-1
        return None
    def _ki():
        for k,(kdx,kdy) in enumerate(_KNIGHT_OFFS):
            if kdx==dx and kdy==dy: return 56+k
        return None
    def _ui(promo):
        for u,(udx,upromo) in enumerate(_UNDERPROS):
            if udx==dx and upromo==promo: return 64+u
        return None
    if move.promotion == chess.QUEEN:
        qi = _qi()
        if qi is not None: return from_sq*73+qi
    if move.promotion and move.promotion in (chess.KNIGHT,chess.BISHOP,chess.ROOK):
        ui = _ui(move.promotion)
        if ui is not None: return from_sq*73+ui
    qi = _qi()
    if qi is not None: return from_sq*73+qi
    ki = _ki()
    if ki is not None: return from_sq*73+ki
    return 0  # fallback

def _convert_moves(fen, moves_probs):
    """(old_63sq_idx, prob) -> [(new_rankflip_idx, prob)]"""
    board = chess.Board(fen)
    result = []
    for old_idx, prob in moves_probs:
        if prob <= 0: continue
        try:
            move = _old_idx2move(old_idx, board)
            new_idx = _new_move2idx(move, board)
            if 0 <= new_idx < 4672:
                result.append((new_idx, prob))
        except Exception:
            pass
    return result


def collate_fn_distill(batch):
    inputs = torch.stack([item[0] for item in batch])
    values = torch.tensor([item[2] for item in batch], dtype=torch.float32)

    target_dist = torch.zeros(len(batch), 4672)
    for i, item in enumerate(batch):
        for move_idx, prob in item[1]:
            if move_idx is not None and 0 <= move_idx < 4672:
                target_dist[i, move_idx] = prob

    return inputs, target_dist, values


# ── 模块级元数据缓存 ──
_META_CACHE = {}

def _build_meta(data_dir: str):
    """扫描 .pt 文件, 构建元数据缓存 (只做一次, 持久化到磁盘)。"""
    if data_dir in _META_CACHE:
        return _META_CACHE[data_dir]

    cache_path = os.path.join(data_dir, "_meta_cache.pt")

    # 尝试加载磁盘缓存
    if os.path.exists(cache_path):
        try:
            cm = torch.load(cache_path, map_location="cpu", weights_only=False)
            if cm.get("_version") == 2 and len(cm.get("file_paths", [])) > 0:
                _META_CACHE[data_dir] = cm
                return cm
        except Exception:
            pass

    # 扫描文件构建
    files = sorted(glob.glob(os.path.join(data_dir, "*_batch_*.pt")))
    if not files:
        raise FileNotFoundError(f"未找到 *_batch_*.pt 文件于 {data_dir}")

    game_lens = []
    game_cum  = []
    game_index = []

    for file_idx, f in enumerate(files):
        batch = torch.load(f, map_location="cpu", weights_only=False)
        gls = list(batch["game_lens"])
        del batch

        cum = [0]
        for gl in gls:
            cum.append(cum[-1] + gl)
        game_lens.append(gls)
        game_cum.append(cum)

        for game_idx, gl in enumerate(gls):
            if gl > 0:
                game_index.append((file_idx, game_idx))

    if not game_index:
        raise ValueError(f"数据目录 {data_dir} 中没有有效对局")

    meta = {
        "_version": 2,
        "file_paths": files,
        "game_lens": game_lens,
        "game_cum": game_cum,
        "game_index": game_index,
    }

    # 保存磁盘缓存
    torch.save(meta, cache_path)

    _META_CACHE[data_dir] = meta
    return meta


class SFDistillDataset(IterableDataset):
    def __init__(self, data_dir: str, max_games: int = 0, game_offset: int = 0,
                 shuffle: bool = True, batch_size: int = 512,
                 castling_ratio: float = 0.08):
        self.data_dir = data_dir
        self.shuffle = shuffle
        self._batch_size = batch_size
        self.castling_ratio = castling_ratio  # 每批中易位样本占比

        # 从模块级缓存读取元数据 (只加载一次文件)
        meta = _build_meta(data_dir)
        self._file_paths = meta["file_paths"]
        self._game_lens = meta["game_lens"]
        self._game_cum = meta["game_cum"]
        game_index_all = list(meta["game_index"])  # 全部对局
        self.total_games = len(game_index_all)

        # 按文件分组 (全部对局, 非切片)
        file_groups_full = {}
        for fi, gi in game_index_all:
            file_groups_full.setdefault(fi, []).append(gi)

        # 按文件采样: shuffle 文件列表, 逐个文件全部取, 凑够 max_games
        self._shuffle_group = shuffle

        if shuffle and max_games > 0:
            file_idxs = list(file_groups_full.keys())
            random.shuffle(file_idxs)
            self._file_game_map = {}
            remaining = max_games
            max_per_file = max(1, max_games // 80)  # 每文件最多取 ~80 局, 保证覆盖足够文件
            for fi in file_idxs:
                gis = file_groups_full[fi]
                # 每文件最多取 max_per_file 局, 确保多样性
                take = min(len(gis), max_per_file, remaining)
                if take > 0:
                    chosen = random.sample(gis, take)
                    random.shuffle(chosen)
                    self._file_game_map[fi] = chosen
                    remaining -= take
                if remaining <= 0:
                    break
        else:
            self._file_game_map = file_groups_full

        # 重建 game_index 用于 __len__
        self._game_index = [(fi, gi) for fi, gis in self._file_game_map.items() for gi in gis]

    def __len__(self):
        return sum(self._game_lens[fi][gi] for fi, gi in self._game_index)

    def __iter__(self):
        worker_info = get_worker_info()
        file_game_map = self._file_game_map
        file_idxs = sorted(file_game_map.keys())

        if worker_info is not None:
            per_worker = len(file_idxs) // worker_info.num_workers
            start = worker_info.id * per_worker
            end = start + per_worker if worker_info.id < worker_info.num_workers - 1 else len(file_idxs)
            file_idxs = file_idxs[start:end]

        batch_size = self._batch_size
        pos_buffer = []
        neg_buffer = []
        castling_buffer = []  # 易位样本单独缓存

        for file_idx in file_idxs:
            batch = torch.load(self._file_paths[file_idx],
                               map_location="cpu", weights_only=False)
            current_data = batch["data"]

            game_idxs = list(file_game_map[file_idx])
            if self._shuffle_group:
                random.shuffle(game_idxs)

            for game_idx in game_idxs:
                gl = self._game_lens[file_idx][game_idx]
                sample_start = self._game_cum[file_idx][game_idx]

                for i in range(gl):
                    entry = current_data[sample_start + i]
                    fen, moves_probs, value = entry[0], entry[1], entry[2]

                    board = chess.Board(fen)
                    tensor = board_to_tensor(board)
                    converted = _convert_moves(fen, moves_probs)
                    val = -value if fen.split()[1] == 'b' else value

                    # 判断是否易位: 检查旧 63-sq 索引
                    is_castling = False
                    if moves_probs and self.castling_ratio > 0:
                        try:
                            old_idx = moves_probs[0][0]
                            move = _old_idx2move(old_idx, board)
                            is_castling = board.is_castling(move)
                        except:
                            pass

                    sample = (tensor, converted, val)
                    if is_castling:
                        castling_buffer.append(sample)
                    elif val >= 0:
                        pos_buffer.append(sample)
                    else:
                        neg_buffer.append(sample)

                    # 缓冲区够大时平衡采样
                    need = batch_size
                    n_castle = min(len(castling_buffer), int(need * self.castling_ratio))
                    n_rest = need - n_castle
                    n_pos = min(len(pos_buffer), n_rest // 2)
                    n_neg = min(len(neg_buffer), n_rest - n_pos)
                    if n_castle + n_pos + n_neg >= need:
                        sel = (random.sample(castling_buffer, n_castle) +
                               random.sample(pos_buffer, n_pos) +
                               random.sample(neg_buffer, n_neg))
                        random.shuffle(sel)
                        inputs = torch.stack([s[0] for s in sel]).float()
                        target_dist = torch.zeros(batch_size, 4672, dtype=torch.float32)
                        rows, cols, vals = [], [], []
                        for j, s in enumerate(sel):
                            for move_idx, prob in s[1]:
                                if move_idx is not None and 0 <= move_idx < 4672:
                                    rows.append(j); cols.append(move_idx); vals.append(prob)
                        if rows:
                            target_dist[rows, cols] = torch.tensor(vals, dtype=torch.float32)
                        values = torch.tensor([s[2] for s in sel], dtype=torch.float32)
                        yield inputs, target_dist, values
                        # 不从 buffer 删除已用的, 允许重复采样(过采样)

        # 最后一批
        all_s = pos_buffer + neg_buffer + castling_buffer
        if all_s:
            random.shuffle(all_s)
            take = min(len(all_s), batch_size)
            sel = all_s[:take]
            if sel:
                inputs = torch.stack([s[0] for s in sel]).float()
                target_dist = torch.zeros(take, 4672, dtype=torch.float32)
                rows, cols, vals = [], [], []
                for j, s in enumerate(sel):
                    for move_idx, prob in s[1]:
                        if move_idx is not None and 0 <= move_idx < 4672:
                            rows.append(j); cols.append(move_idx); vals.append(prob)
                if rows:
                    target_dist[rows, cols] = torch.tensor(vals, dtype=torch.float32)
                values = torch.tensor([s[2] for s in sel], dtype=torch.float32)
                yield inputs, target_dist, values
