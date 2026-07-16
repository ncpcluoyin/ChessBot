"""
棋盘状态 → 神经网络输入张量，以及 4672 维走法编解码。

输入张量 (16, 8, 8):
  - 我方棋子 (0-5): P,N,B,R,Q,K, 0/1
  - 对方棋子 (6-11): P,N,B,R,Q,K, 0/1
  - 易位权 (12-15): 我方王翼/后翼, 对方王翼/后翼
  - rank-flip 编码, STM 视角

走法编码 (4672 = 64 × 73):
  - 每格 73 种走法:
      [0:56]  后走法 (8方向 × 7距离)
      [56:64] 马走法 (8方向)
      [64:73] 低升变 (3方向 × 3升变类型: N,B,R)
"""

import chess
import numpy as np
import torch

# ─── 走法编码常数 (从当前走棋方视角) ───

QUEEN_DIRECTIONS = [
    (0, 1),    # N  (前)
    (1, 1),    # NE
    (1, 0),    # E
    (1, -1),   # SE
    (0, -1),   # S  (后)
    (-1, -1),  # SW
    (-1, 0),   # W
    (-1, 1),   # NW
]

KNIGHT_OFFSETS = [
    (-1, -2), (-1, 2), (-2, -1), (-2, 1),
    (1, -2),  (1, 2),  (2, -1),  (2, 1),
]

# 低升变: (dx, promotion_piece)
UNDERPROMOTIONS = [
    (-1, chess.KNIGHT), (-1, chess.BISHOP), (-1, chess.ROOK),
    (0, chess.KNIGHT),  (0, chess.BISHOP),  (0, chess.ROOK),
    (1, chess.KNIGHT),  (1, chess.BISHOP),  (1, chess.ROOK),
]

PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]


# ═══════════════════════════════════════════════════════════════════════════════
# 棋盘 → 张量
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# 数据增强
# ═══════════════════════════════════════════════════════════════════════════════

# 预计算增强走法映射表
# horizontal[old_idx] = new_idx (水平镜像)
# swap_color[old_idx] = new_idx (颜色交换)
# rot180[old_idx] = new_idx (180度旋转 + 颜色交换)

# ═══════════════════════════════════════════════════════════════════════════════
# 棋盘 → 张量
# ═══════════════════════════════════════════════════════════════════════════════

def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """
    将棋盘转换为 (16, 8, 8) 浮点张量。
    从 STM 视角编码 (黑方走棋时 180°旋转棋盘)。
    我方棋子 (0-5), 对方棋子 (6-11), 走棋方由空间布局隐式指示。
    """
    perspective = board.turn
    planes = []
    for pt in PIECE_TYPES:
        planes.append(_squares_to_plane(board.pieces(pt, perspective), perspective))
    for pt in PIECE_TYPES:
        planes.append(_squares_to_plane(board.pieces(pt, not perspective), perspective))
    planes.extend(_global_planes(board, perspective))
    tensor = torch.from_numpy(np.stack(planes)).float()
    return tensor


def _global_planes(board: chess.Board, perspective: bool) -> list:
    """生成 4 个全局平面 (从 STM 视角)。"""
    planes = []
    our_color, their_color = perspective, not perspective
    planes.append(np.full((8, 8), 1.0 if board.has_kingside_castling_rights(our_color) else 0.0, dtype=np.float32))
    planes.append(np.full((8, 8), 1.0 if board.has_queenside_castling_rights(our_color) else 0.0, dtype=np.float32))
    planes.append(np.full((8, 8), 1.0 if board.has_kingside_castling_rights(their_color) else 0.0, dtype=np.float32))
    planes.append(np.full((8, 8), 1.0 if board.has_queenside_castling_rights(their_color) else 0.0, dtype=np.float32))
    return planes


def _squares_to_plane(squares: chess.SquareSet, perspective: bool) -> np.ndarray:
    """将 square 集合转为 (8,8) 平面 (黑方视角时 180°旋转)。"""
    plane = np.zeros((8, 8), dtype=np.float32)
    for sq in squares:
        if perspective == chess.BLACK:
            sq = chess.square(chess.square_file(sq), 7 - chess.square_rank(sq))
        r, f = chess.square_rank(sq), chess.square_file(sq)
        plane[r, f] = 1.0
    return plane


# ═══════════════════════════════════════════════════════════════════════════════
# 走法编解码 (4672 = 64 × 73)
# ═══════════════════════════════════════════════════════════════════════════════

def move_to_index(move: chess.Move, board: chess.Board) -> int:
    """将 chess.Move 转为 0~4671 的整数索引 (从当前走棋方视角)。"""
    from_sq = move.from_square
    to_sq = move.to_square

    if board.turn == chess.BLACK:
        from_sq = chess.square(chess.square_file(from_sq), 7 - chess.square_rank(from_sq))
        to_sq = chess.square(chess.square_file(to_sq), 7 - chess.square_rank(to_sq))

    dx = chess.square_file(to_sq) - chess.square_file(from_sq)
    dy = chess.square_rank(to_sq) - chess.square_rank(from_sq)

    # 升变为后也算后走法
    if move.promotion == chess.QUEEN:
        qi = _queen_index(dx, dy)
        if qi is not None:
            return from_sq * 73 + qi

    # 低升变
    if move.promotion and move.promotion in (chess.KNIGHT, chess.BISHOP, chess.ROOK):
        ui = _underpromotion_index(dx, move.promotion)
        if ui is not None:
            return from_sq * 73 + ui

    # 后走法
    qi = _queen_index(dx, dy)
    if qi is not None:
        return from_sq * 73 + qi

    # 马走法
    ki = _knight_index(dx, dy)
    if ki is not None:
        return from_sq * 73 + ki

    raise ValueError(f"无法编码走法 {move.uci()} (dx={dx}, dy={dy})")


def index_to_move(index: int, board: chess.Board) -> chess.Move:
    """将 0~4671 索引解码为 chess.Move (从当前走棋方视角还原到真实棋盘)。"""
    from_sq = index // 73
    move_type = index % 73
    from_file = chess.square_file(from_sq)
    from_rank = chess.square_rank(from_sq)
    to_file = to_rank = 0

    def _real(f, t, promotion=None):
        """将视角坐标还原为真实坐标。"""
        rf, rt = chess.square(f[0], f[1]), chess.square(t[0], t[1])
        if board.turn == chess.BLACK:
            rf = chess.square(chess.square_file(rf), 7 - chess.square_rank(rf))
            rt = chess.square(chess.square_file(rt), 7 - chess.square_rank(rt))
        return chess.Move(rf, rt, promotion=promotion)

    if move_type < 56:
        # 后走法
        d = move_type // 7
        dist = (move_type % 7) + 1
        ddx, ddy = QUEEN_DIRECTIONS[d]
        to_file = from_file + ddx * dist
        to_rank = from_rank + ddy * dist
        actual_from = (from_file, from_rank)
        actual_to = (to_file, to_rank)

        # 判断是否需要升变为后
        promotion = None
        if board.turn == chess.WHITE and from_rank == 6 and to_rank == 7:
            # 需要验证起始格是否有白兵
            rf = chess.square(from_file, from_rank)
            p = board.piece_at(rf)
            if p and p.piece_type == chess.PAWN and p.color == chess.WHITE:
                promotion = chess.QUEEN
        elif board.turn == chess.BLACK and from_rank == 6 and to_rank == 7:
            # 黑方视角: 视角第6排 = 真实第1排
            real_from_sq = chess.square(from_file, 7 - from_rank)
            p = board.piece_at(real_from_sq)
            if p and p.piece_type == chess.PAWN and p.color == chess.BLACK:
                promotion = chess.QUEEN

        return _real(actual_from, actual_to, promotion)

    elif move_type < 64:
        # 马走法
        k = move_type - 56
        ddx, ddy = KNIGHT_OFFSETS[k]
        to_file = from_file + ddx
        to_rank = from_rank + ddy
        return _real((from_file, from_rank), (to_file, to_rank))

    else:
        # 低升变
        u = move_type - 64
        dx_idx = u // 3
        promo_idx = u % 3
        dx = dx_idx - 1
        promo_map = {0: chess.KNIGHT, 1: chess.BISHOP, 2: chess.ROOK}
        promotion = promo_map[promo_idx]
        to_file = from_file + dx
        to_rank = from_rank + 1  # 总是向前
        return _real((from_file, from_rank), (to_file, to_rank), promotion)


def _queen_index(dx: int, dy: int) -> int | None:
    for d, (ddx, ddy) in enumerate(QUEEN_DIRECTIONS):
        dist = max(abs(dx), abs(dy))
        if dist == 0 or dist > 7:
            continue
        if ddx * dist == dx and ddy * dist == dy:
            return d * 7 + dist - 1
    return None


def _knight_index(dx: int, dy: int) -> int | None:
    for k, (kdx, kdy) in enumerate(KNIGHT_OFFSETS):
        if kdx == dx and kdy == dy:
            return 56 + k
    return None


def _underpromotion_index(dx: int, promotion: int) -> int | None:
    for u, (udx, upromo) in enumerate(UNDERPROMOTIONS):
        if udx == dx and upromo == promotion:
            return 64 + u
    return None


def move_index_to_uci(index: int, perspective: bool = True) -> str:
    """仅用于调试: 将 0~4671 索引转为可读的 UCI 字符串。"""
    from_sq = index // 73
    move_type = index % 73
    from_file = chess.square_file(from_sq)
    from_rank = chess.square_rank(from_sq)

    if move_type < 56:
        d = move_type // 7
        dist = (move_type % 7) + 1
        ddx, ddy = QUEEN_DIRECTIONS[d]
        to_file = from_file + ddx * dist
        to_rank = from_rank + ddy * dist
        promotion = "q" if (from_rank == 6 and to_rank == 7) else ""
    elif move_type < 64:
        k = move_type - 56
        ddx, ddy = KNIGHT_OFFSETS[k]
        to_file = from_file + ddx
        to_rank = from_rank + ddy
        promotion = ""
    else:
        u = move_type - 64
        dx_idx = u // 3
        promo_idx = u % 3
        dx = dx_idx - 1
        to_file = from_file + dx
        to_rank = from_rank + 1
        promo_map = {0: "n", 1: "b", 2: "r"}
        promotion = promo_map[promo_idx]

    f = chess.square_name(chess.square(from_file, from_rank))
    t = chess.square_name(chess.square(to_file, to_rank))
    return f + t + promotion


# ═══════════════════════════════════════════════════════════════════════════════
# 合法走法掩码
# ═══════════════════════════════════════════════════════════════════════════════

def get_legal_moves_mask(board: chess.Board) -> torch.Tensor:
    """返回 (4672,) 布尔掩码，True 表示合法走法。"""
    mask = torch.zeros(4672, dtype=torch.bool)
    for move in board.legal_moves:
        try:
            idx = move_to_index(move, board)
            mask[idx] = True
        except ValueError:
            continue
    return mask
