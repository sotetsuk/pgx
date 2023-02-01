#! python3

import jax
import jax.numpy as jnp
from flax.serialization import to_bytes, from_bytes
from pgx.animal_shogi import _init_legal_actions

# BLACK/WHITE/(NONE)_○○_MOVEは22にいるときの各駒の動き
# 端にいる場合は対応するところに0をかけていけないようにする
BLACK_PAWN_MOVE = jnp.bool_([[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
WHITE_PAWN_MOVE = jnp.bool_([[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]])
BLACK_GOLD_MOVE = jnp.bool_([[1, 1, 0, 0], [1, 0, 1, 0], [1, 1, 0, 0]])
WHITE_GOLD_MOVE = jnp.bool_([[0, 1, 1, 0], [1, 0, 1, 0], [0, 1, 1, 0]])
ROOK_MOVE = jnp.bool_([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 0]])
BISHOP_MOVE = jnp.bool_([[1, 0, 1, 0], [0, 0, 0, 0], [1, 0, 1, 0]])
KING_MOVE = jnp.bool_([[1, 1, 1, 0], [1, 0, 1, 0], [1, 1, 1, 0]])


#  上下左右の辺に接しているかどうか
#  接している場合は後の関数で行ける場所を制限する
@jax.jit
def _is_side(
    point,
):
    is_up = point % 4 == 0
    is_down = point % 4 == 3
    is_left = point >= 8
    is_right = point <= 3
    return is_up, is_down, is_left, is_right


# はみ出す部分をカットする
@jax.jit
def _cut_outside(array, point):
    u, d, l, r = _is_side(point)
    array = jax.lax.cond(
        u, lambda: array.at[:3, 0].set(False), lambda: array
    )
    array = jax.lax.cond(
        d, lambda: array.at[:3, 2].set(False), lambda: array
    )
    array = jax.lax.cond(
        r, lambda: array.at[0, :4].set(False), lambda: array
    )
    array = jax.lax.cond(
        l, lambda: array.at[2, :4].set(False), lambda: array
    )
    return array


@jax.jit
def _action_board(array, point):
    # point(0~11)を座標((0, 0)~(2, 3))に変換
    y, t = point // 4, point % 4
    array = _cut_outside(array, point)
    return jnp.roll(array, (y - 1, t - 1), axis=(0, 1))


#  座標と駒の種類から到達できる座標を列挙
POINT_MOVES = jnp.zeros((12, 11, 3, 4), dtype=jnp.bool_)
for i in range(12):
    POINT_MOVES = POINT_MOVES.at[i, 1].set(_action_board(BLACK_PAWN_MOVE, i))
    POINT_MOVES = POINT_MOVES.at[i, 2].set(_action_board(ROOK_MOVE, i))
    POINT_MOVES = POINT_MOVES.at[i, 3].set(_action_board(BISHOP_MOVE, i))
    POINT_MOVES = POINT_MOVES.at[i, 4].set(_action_board(KING_MOVE, i))
    POINT_MOVES = POINT_MOVES.at[i, 5].set(_action_board(BLACK_GOLD_MOVE, i))
    POINT_MOVES = POINT_MOVES.at[i, 6].set(_action_board(WHITE_PAWN_MOVE, i))
    POINT_MOVES = POINT_MOVES.at[i, 7].set(_action_board(ROOK_MOVE, i))
    POINT_MOVES = POINT_MOVES.at[i, 8].set(_action_board(BISHOP_MOVE, i))
    POINT_MOVES = POINT_MOVES.at[i, 9].set(_action_board(KING_MOVE, i))
    POINT_MOVES = POINT_MOVES.at[i, 10].set(_action_board(WHITE_GOLD_MOVE, i))


b = to_bytes(POINT_MOVES)
print(b)

legal_action_masks = _init_legal_actions()
b = to_bytes(legal_action_masks)
print(b)
