"""
将 63-sq (180°旋转+文件交换) 编码迁移到 rank-flip (仅翻转行, 保留列)。

流程:
  1. 用旧 63-sq 函数解码每个走法索引 → UCI 字符串
  2. 用新 rank-flip 函数将 UCI 字符串重新编码为新索引
  3. 覆盖原文件
"""

import chess
import torch
import glob
import os
import sys
import time

# ─── 编码常数 ───
QUEEN_DIRECTIONS = [(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1)]
KNIGHT_OFFSETS = [(-1,-2),(-1,2),(-2,-1),(-2,1),(1,-2),(1,2),(2,-1),(2,1)]
UNDERPROMOTIONS = [(-1, chess.KNIGHT), (-1, chess.BISHOP), (-1, chess.ROOK),
                   (0, chess.KNIGHT), (0, chess.BISHOP), (0, chess.ROOK),
                   (1, chess.KNIGHT), (1, chess.BISHOP), (1, chess.ROOK)]


def _queen_idx(dx, dy):
    for d, (ddx, ddy) in enumerate(QUEEN_DIRECTIONS):
        dist = max(abs(dx), abs(dy))
        if dist == 0 or dist > 7: continue
        if ddx * dist == dx and ddy * dist == dy:
            return d * 7 + dist - 1
    return None

def _knight_idx(dx, dy):
    for k, (kdx, kdy) in enumerate(KNIGHT_OFFSETS):
        if kdx == dx and kdy == dy: return 56 + k
    return None

def _underpromo_idx(dx, promo):
    for u, (udx, upromo) in enumerate(UNDERPROMOTIONS):
        if udx == dx and upromo == promo: return 64 + u
    return None


# ==================== 旧 63-sq 解码 ====================

def old_index_to_move(index, board):
    """63-sq 索引 → UCI 走法。硬编码, 不依赖 board.py。"""
    from_sq = index // 73
    move_type = index % 73
    fi = chess.square_file(from_sq)
    ri = chess.square_rank(from_sq)

    if move_type < 56:
        d = move_type // 7
        dist = (move_type % 7) + 1
        ddx, ddy = QUEEN_DIRECTIONS[d]
        tf = fi + ddx * dist
        tr = ri + ddy * dist

        promo = None
        if ri == 6 and tr == 7:
            # 视角第6排走向第7排 = 升变
            real_from = chess.square(fi, ri)
            if board.turn == chess.BLACK:
                real_from = 63 - real_from
            p = board.piece_at(real_from)
            if p and p.piece_type == chess.PAWN:
                promo = chess.QUEEN

        rf = chess.square(fi, ri)
        rt = chess.square(tf, tr)
        if board.turn == chess.BLACK:
            rf, rt = 63 - rf, 63 - rt
        return chess.Move(rf, rt, promotion=promo)

    elif move_type < 64:
        k = move_type - 56
        ddx, ddy = KNIGHT_OFFSETS[k]
        rf = chess.square(fi, ri)
        rt = chess.square(fi + ddx, ri + ddy)
        if board.turn == chess.BLACK:
            rf, rt = 63 - rf, 63 - rt
        return chess.Move(rf, rt)

    else:
        u = move_type - 64
        dx = u // 3 - 1
        promo_map = {0: chess.KNIGHT, 1: chess.BISHOP, 2: chess.ROOK}
        promo = promo_map[u % 3]
        rf = chess.square(fi, ri)
        rt = chess.square(fi + dx, ri + 1)
        if board.turn == chess.BLACK:
            rf, rt = 63 - rf, 63 - rt
        return chess.Move(rf, rt, promotion=promo)


# ==================== 新 rank-flip 编码 ====================

def new_move_to_index(move, board):
    """UCI 走法 → rank-flip 索引。硬编码, 不依赖 board.py。"""
    from_sq = move.from_square
    to_sq = move.to_square
    if board.turn == chess.BLACK:
        from_sq = chess.square(chess.square_file(from_sq), 7 - chess.square_rank(from_sq))
        to_sq = chess.square(chess.square_file(to_sq), 7 - chess.square_rank(to_sq))
    dx = chess.square_file(to_sq) - chess.square_file(from_sq)
    dy = chess.square_rank(to_sq) - chess.square_rank(from_sq)

    if move.promotion == chess.QUEEN:
        qi = _queen_idx(dx, dy)
        if qi is not None: return from_sq * 73 + qi
    if move.promotion and move.promotion in (chess.KNIGHT, chess.BISHOP, chess.ROOK):
        ui = _underpromo_idx(dx, move.promotion)
        if ui is not None: return from_sq * 73 + ui
    qi = _queen_idx(dx, dy)
    if qi is not None: return from_sq * 73 + qi
    ki = _knight_idx(dx, dy)
    if ki is not None: return from_sq * 73 + ki
    raise ValueError(f'无法编码 {move.uci()} (dx={dx}, dy={dy})')


# ==================== 迁移单个文件 ====================

def migrate_file(file_path):
    batch = torch.load(file_path, map_location='cpu', weights_only=False)
    data = batch['data']
    modified = 0
    errors = 0

    for i in range(len(data)):
        fen, moves, value = data[i]
        board = chess.Board(fen)
        new_moves = []
        for old_idx, prob in moves:
            try:
                move = old_index_to_move(old_idx, board)
                new_idx = new_move_to_index(move, board)
                new_moves.append((new_idx, prob))
            except Exception as e:
                errors += 1
                continue
        if new_moves:
            data[i] = (fen, new_moves, value)
            modified += 1

    torch.save(batch, file_path)
    return modified, errors


# ==================== 主入口 ====================

def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'data/hf_supervised_samples'
    files = sorted(glob.glob(os.path.join(data_dir, '*_batch_*.pt')))
    if not files:
        print(f'未找到 *_batch_*.pt 于 {data_dir}')
        return

    print(f'找到 {len(files)} 个文件, 开始迁移...')
    t0 = time.time()
    total_modified = 0
    total_errors = 0

    for fi, fp in enumerate(files):
        mod, err = migrate_file(fp)
        total_modified += mod
        total_errors += err
        if (fi + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f'  [{fi+1}/{len(files)}] {total_modified} 样本, {total_errors} 错误, {elapsed:.0f}s')

    elapsed = time.time() - t0
    print(f'\n完成! {len(files)} 文件, {total_modified} 样本, {total_errors} 错误, {elapsed:.0f}s')

    # 清理元数据缓存
    cache = os.path.join(data_dir, '_meta_cache.pt')
    if os.path.exists(cache):
        os.remove(cache)
        print(f'已清理 {cache}')


if __name__ == '__main__':
    main()
