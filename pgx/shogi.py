"""Shogi.

piece_board (81,):
  -1 空白
   0 歩
   1 香車
   2 桂馬
   3 銀
   4 角
   5 飛車
   6 金
   7 玉
   8 と
   9 成香
  10 成桂
  11 成銀
  12 馬
  13 龍
  14 相手歩
  15 相手香車
  16 相手桂馬
  17 相手銀
  18 相手角
  19 相手飛車
  20 相手金
  21 相手玉
  22 相手と
  23 相手成香
  24 相手成桂
  25 相手成銀
  26 相手馬
  27 相手龍
"""

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp

import pgx.core as core
from pgx.cache import (
    load_shogi_is_on_the_way,
    load_shogi_legal_from_mask,
    load_shogi_raw_effect_boards,
)
from pgx.flax.struct import dataclass

TRUE = jnp.bool_(True)
FALSE = jnp.bool_(False)
ZERO = jnp.int8(0)
ONE = jnp.int8(1)
TWO = jnp.int8(2)

# Pieces
EMPTY = jnp.int8(-1)
PAWN = jnp.int8(0)
LANCE = jnp.int8(1)
KNIGHT = jnp.int8(2)
SILVER = jnp.int8(3)
BISHOP = jnp.int8(4)
ROOK = jnp.int8(5)
GOLD = jnp.int8(6)
KING = jnp.int8(7)
PRO_PAWN = jnp.int8(8)
PRO_LANCE = jnp.int8(9)
PRO_KNIGHT = jnp.int8(10)
PRO_SILVER = jnp.int8(11)
HORSE = jnp.int8(12)
DRAGON = jnp.int8(13)
OPP_PAWN = jnp.int8(14)
OPP_LANCE = jnp.int8(15)
OPP_KNIGHT = jnp.int8(16)
OPP_SILVER = jnp.int8(17)
OPP_BISHOP = jnp.int8(18)
OPP_ROOK = jnp.int8(19)
OPP_GOLD = jnp.int8(20)
OPP_KING = jnp.int8(21)
OPP_PRO_PAWN = jnp.int8(22)
OPP_PRO_LANCE = jnp.int8(23)
OPP_PRO_KNIGHT = jnp.int8(24)
OPP_PRO_SILVER = jnp.int8(25)
OPP_HORSE = jnp.int8(26)
OPP_DRAGON = jnp.int8(27)


# fmt: off
INIT_PIECE_BOARD = jnp.int8([[15, -1, 14, -1, -1, -1, 0, -1, 1],  # noqa: E241
                             [16, 18, 14, -1, -1, -1, 0,  5, 2],  # noqa: E241
                             [17, -1, 14, -1, -1, -1, 0, -1, 3],  # noqa: E241
                             [20, -1, 14, -1, -1, -1, 0, -1, 6],  # noqa: E241
                             [21, -1, 14, -1, -1, -1, 0, -1, 7],  # noqa: E241
                             [20, -1, 14, -1, -1, -1, 0, -1, 6],  # noqa: E241
                             [17, -1, 14, -1, -1, -1, 0, -1, 3],  # noqa: E241
                             [16, 19, 14, -1, -1, -1, 0,  4, 2],  # noqa: E241
                             [15, -1, 14, -1, -1, -1, 0, -1, 1]]).flatten()  # noqa: E241
# fmt: on

# Can <piece,14> reach from <from,81> to <to,81> ignoring pieces on board?
RAW_EFFECT_BOARDS = load_shogi_raw_effect_boards()  # bool (14, 81, 81)
# When <lance/bishop/rook/horse/dragon,5> moves from <from,81> to <to,81>,
# is <point,81> on the way between two points?
IS_ON_THE_WAY = load_shogi_is_on_the_way()  # bool (5, 81, 81, 81)
# Give <dir,10> and <to,81>, return the legal <from,81> mask
# E.g. LEGAL_FROM_MASK[Up right, to=19]
# x x x x x x x
# x x x x t x x
# x x x o x x x
# x x o x x x x
LEGAL_FROM_MASK = load_shogi_legal_from_mask()


@dataclass
class State(core.State):
    curr_player: jnp.ndarray = jnp.int8(0)
    reward: jnp.ndarray = jnp.float32([0.0, 0.0])
    terminated: jnp.ndarray = FALSE
    legal_action_mask: jnp.ndarray = jnp.zeros(27 * 81, dtype=jnp.bool_)
    # --- Shogi specific ---
    turn: jnp.ndarray = jnp.int8(0)  # 0 or 1
    piece_board: jnp.ndarray = INIT_PIECE_BOARD  # (81,) 後手のときにはflipする
    hand: jnp.ndarray = jnp.zeros((2, 7), dtype=jnp.int8)  # 後手のときにはflipする


class Shogi(core.Env):
    def __init__(self):
        super().__init__()

    def init(self, rng: jax.random.KeyArray) -> State:
        return init(rng)

    def _step(self, state: core.State, action: jnp.ndarray) -> State:
        assert isinstance(state, State)
        return step(state, action)

    def observe(
        self, state: core.State, player_id: jnp.ndarray
    ) -> jnp.ndarray:
        assert isinstance(state, State)
        return observe(state, player_id)

    def num_players(self) -> int:
        return 2


def init(rng):
    state = _init()
    rng, subkey = jax.random.split(rng)
    curr_player = jnp.int8(jax.random.bernoulli(subkey))
    return state.replace(curr_player=curr_player)


def _init():
    """Initialize Shogi State.
    >>> s = _init()
    >>> s.piece_board.reshape((9, 9))
    Array([[15, -1, 14, -1, -1, -1,  0, -1,  1],
           [16, 18, 14, -1, -1, -1,  0,  5,  2],
           [17, -1, 14, -1, -1, -1,  0, -1,  3],
           [20, -1, 14, -1, -1, -1,  0, -1,  6],
           [21, -1, 14, -1, -1, -1,  0, -1,  7],
           [20, -1, 14, -1, -1, -1,  0, -1,  6],
           [17, -1, 14, -1, -1, -1,  0, -1,  3],
           [16, 19, 14, -1, -1, -1,  0,  4,  2],
           [15, -1, 14, -1, -1, -1,  0, -1,  1]], dtype=int8)
    >>> jnp.rot90(s.piece_board.reshape((9, 9)), k=3)
    Array([[15, 16, 17, 20, 21, 20, 17, 16, 15],
           [-1, 19, -1, -1, -1, -1, -1, 18, -1],
           [14, 14, 14, 14, 14, 14, 14, 14, 14],
           [-1, -1, -1, -1, -1, -1, -1, -1, -1],
           [-1, -1, -1, -1, -1, -1, -1, -1, -1],
           [-1, -1, -1, -1, -1, -1, -1, -1, -1],
           [ 0,  0,  0,  0,  0,  0,  0,  0,  0],
           [-1,  4, -1, -1, -1, -1, -1,  5, -1],
           [ 1,  2,  3,  6,  7,  6,  3,  2,  1]], dtype=int8)
    """
    state = State()
    legal_actions = _legal_actions(state)
    return state.replace(  # type: ignore
        legal_action_mask=_to_direction(legal_actions)
    )


# 指し手のdataclass
@dataclass
class Action:
    """
    direction (from github.com/TadaoYamaoka/cshogi)

     0 Up
     1 Up left
     2 Up right
     3 Left
     4 Right
     5 Down
     6 Down left
     7 Down right
     8 Up2 left
     9 Up2 right
    10 Promote +  Up
    11 Promote +  Up left
    12 Promote +  Up right
    13 Promote +  Left
    14 Promote +  Right
    15 Promote +  Down
    16 Promote +  Down left
    17 Promote +  Down right
    18 Promote +  Up2 left
    19 Promote +  Up2 right
    20 Drop 歩
    21 Drop 香車
    22 Drop 桂馬
    23 Drop 銀
    24 Drop 角
    25 Drop 飛車
    26 Drop 金
    """

    # 駒打ちかどうか
    is_drop: jnp.ndarray
    # 動かした(打った)駒の種類/打つ前が成っていなければ成ってない
    piece: jnp.ndarray
    # 移動後の座標
    to: jnp.ndarray
    # --- Optional (only for move action) ---
    # 移動前の座標
    from_: jnp.ndarray = jnp.int8(0)
    # 駒を成るかどうかの判定
    is_promotion: jnp.ndarray = FALSE

    @staticmethod
    def make_move(piece, from_, to, is_promotion=FALSE):
        return Action(
            is_drop=False,
            piece=piece,
            from_=from_,
            to=to,
            is_promotion=is_promotion,
        )

    @staticmethod
    def make_drop(piece, to):
        return Action(is_drop=True, piece=piece, to=to)

    @staticmethod
    def from_dlshogi_action(state: State, action: jnp.ndarray):
        # NOTE: action (e.g., 2000) is bigger than int8
        action = jnp.int32(action)
        direction, to = jnp.int8(action // 81), jnp.int8(action % 81)
        is_drop = direction >= 20
        # Compute <from> from <dir, to>
        #
        # LEGAL_FROM_MASK[UP, 18]  # to = 81
        # x x x x t x x
        # x x x x o x x
        # x x x x o x x
        # x x x x o x x
        # x x x x o x x
        #
        # legal_moves[:, 18]  # to = 18
        # x x o x t x x
        # x x x x x x x
        # x x o x o x x
        # x x x x x x x
        # x x x x x x x
        mask1 = LEGAL_FROM_MASK[direction, to]  # (81,)
        legal_moves, _, _ = _legal_actions(state)  # TODO: cache legal moves
        mask2 = legal_moves[:, to]  # (81,)
        from_ = jnp.nonzero(mask1 & mask2, size=1)[0][0]

        piece = jax.lax.cond(
            is_drop,
            lambda: direction - 20,
            lambda: state.piece_board[from_],
        )
        is_promotion = (10 <= direction) & (direction < 20)
        return Action(is_drop=is_drop, piece=piece, to=to, from_=from_, is_promotion=is_promotion)  # type: ignore


def step(state: State, action: jnp.ndarray) -> State:
    # Note: Assume that illegal action is already filtered by Env.step
    return _step(state, Action.from_dlshogi_action(state, action))


def observe(state: State, player_id: jnp.ndarray) -> jnp.ndarray:
    # TODO: write me
    return jnp.zeros(100)


def _step(state: State, action: Action) -> State:
    # apply move/drop action
    state = jax.lax.cond(
        action.is_drop, _step_drop, _step_move, *(state, action)
    )
    # flip state
    state = _flip(state)
    state = state.replace(  # type: ignore
        curr_player=(state.curr_player + 1) % 2, turn=(state.turn + 1) % 2
    )
    legal_actions = _legal_actions(state)
    legal_action_mask = _to_direction(legal_actions)
    terminated = ~legal_action_mask.any()
    reward = jax.lax.cond(
        terminated,
        lambda: jnp.float32([-1.0, 1.0]),
        lambda: jnp.float32([0.0, 0.0]),
    )
    reward = jax.lax.cond(
        state.curr_player != 0, lambda: reward[::-1], lambda: reward
    )
    return state.replace(  # type: ignore
        reward=reward,
        terminated=terminated,
        legal_action_mask=legal_action_mask,
    )


def _step_move(state: State, action: Action) -> State:
    pb = state.piece_board
    # remove piece from the original position
    pb = pb.at[action.from_].set(EMPTY)
    # capture the opponent if exists
    captured = pb[action.to]  # suppose >= OPP_PAWN, -1 if EMPTY
    hand = jax.lax.cond(
        captured == EMPTY,
        lambda: state.hand,
        # add captured piece to my hand after
        #   (1) tuning opp piece into mine by (x + 14) % 28, and
        #   (2) filtering promoted piece by x % 8
        lambda: state.hand.at[0, ((captured + 14) % 28) % 8].add(1),
    )
    # promote piece
    piece = jax.lax.cond(
        action.is_promotion,
        lambda: _promote(action.piece),
        lambda: action.piece,
    )
    # set piece to the target position
    pb = pb.at[action.to].set(piece)
    return state.replace(piece_board=pb, hand=hand)  # type: ignore


def _step_drop(state: State, action: Action) -> State:
    # add piece to board
    pb = state.piece_board.at[action.to].set(action.piece)
    # remove piece from hand
    hand = state.hand.at[0, action.piece].add(-1)
    return state.replace(piece_board=pb, hand=hand)  # type: ignore


def _legal_actions(state: State):
    effect_boards = _apply_effects(state)
    legal_moves = _pseudo_legal_moves(state, effect_boards)
    legal_drops = _pseudo_legal_drops(state, effect_boards)

    # Prepare necessary materials
    flipped_state = _flip(state)
    flipped_effect_boards = _apply_effects(flipped_state)
    checking_point_board, check_defense_board = _check_info(
        state, flipped_state, flipped_effect_boards
    )
    is_pinned = _find_pinned_pieces(state, flipped_state)

    # Filter illegal moves
    legal_moves = _filter_suicide_moves(
        state, legal_moves, flipped_effect_boards, is_pinned
    )
    legal_moves = _filter_ignoring_check_moves(
        state, legal_moves, checking_point_board, check_defense_board
    )

    # Filter illegal drops
    legal_drops = _filter_pawn_drop_mate(
        state, legal_drops, effect_boards, flipped_effect_boards
    )
    legal_drops = _filter_ignoring_check_drops(
        legal_drops, checking_point_board, check_defense_board
    )

    # Generate legal promotion
    legal_promotion = _legal_promotion(state, legal_moves)

    return legal_moves, legal_promotion, legal_drops


def _pseudo_legal_moves(
    state: State, effect_boards: jnp.ndarray
) -> jnp.ndarray:
    """Filter (81, 81) effects and return legal moves (81, 81)"""

    # filter the destinations where my piece exists
    is_my_piece = (PAWN <= state.piece_board) & (state.piece_board < OPP_PAWN)
    effect_boards = jnp.where(is_my_piece, FALSE, effect_boards)

    return effect_boards


def _filter_suicide_moves(
    state: State, legal_moves: jnp.ndarray, flipped_effect_boards, is_pinned
) -> jnp.ndarray:
    """Filter suicide action
     - King moves into the effected area
     - Pinned piece moves
    A piece is pinned when
     - it exists between king and (Lance/Bishop/Rook/Horse/Dragon)
     - no other pieces exist on the way to king
    """
    # king cannot move into the effected area
    opp_effect_boards = jnp.flip(flipped_effect_boards)  # (81,)
    king_mask = state.piece_board == KING
    mask = king_mask.reshape(81, 1) * opp_effect_boards.any(axis=0).reshape(
        1, 81
    )
    legal_moves = jnp.where(mask, FALSE, legal_moves)

    # pinned piece cannot move
    legal_moves = jnp.where(is_pinned.reshape(81, 1), FALSE, legal_moves)

    return legal_moves


def _find_pinned_pieces(state, flipped_state):
    flipped_opp_raw_effect_boards = _apply_raw_effects(flipped_state)
    flipped_king_pos = (
        80 - jnp.nonzero(state.piece_board == KING, size=1)[0][0]
    )
    flipped_effecting_mask = flipped_opp_raw_effect_boards[
        :, flipped_king_pos
    ]  # (81,) 王に遮蔽無視して聞いている駒の位置

    @jax.vmap
    def pinned_piece_mask(p, f):
        # fにあるpから王までの間にある駒が1枚だけの場合、そこをマスクして返す
        mask = IS_ON_THE_WAY[p, f, flipped_king_pos, :] & (
            flipped_state.piece_board != EMPTY
        )
        return jax.lax.cond(
            mask.sum() == 1, lambda: mask, lambda: jnp.zeros_like(mask)
        )

    from_ = jnp.arange(81)
    large_piece = _to_large_piece_ix(flipped_state.piece_board)
    # 利いてないところからの結果は無視する
    flipped_is_pinned = jnp.where(
        flipped_effecting_mask.reshape(81, 1),
        pinned_piece_mask(large_piece, from_),
        FALSE,
    ).any(axis=0)
    is_pinned = flipped_is_pinned[::-1]  # (81,)

    return is_pinned


def _filter_ignoring_check_moves(
    state: State,
    legal_moves: jnp.ndarray,
    checking_point_board,
    check_defense_board,
) -> jnp.ndarray:
    """Filter moves which ignores check

    Legal moves are one of
      - King escapes from the check to non-effected place (including taking the checking piece)
      - Capturing the checking piece by the other pieces
      - Move the other piece between King and checking piece
    """
    leave_check_mask = jnp.zeros_like(legal_moves, dtype=jnp.bool_)

    # King escapes (i.e., Only King can move)
    king_mask = state.piece_board == KING
    king_escape_mask = jnp.tile(king_mask, reps=(81, 1)).transpose()
    leave_check_mask |= king_escape_mask

    # Capture the checking piece
    capturing_mask = jnp.tile(checking_point_board, reps=(81, 1))
    leave_check_mask |= capturing_mask

    # 駒を動かして合駒をする
    leave_check_mask |= check_defense_board  # filter target

    # 両王手の場合、王が避ける以外ない
    num_checks = checking_point_board.sum()
    is_double_checked = num_checks > 1
    leave_check_mask = jax.lax.cond(
        is_double_checked, lambda: king_escape_mask, lambda: leave_check_mask
    )

    # 王手がかかってないなら王手放置は考えなくてよい
    is_not_checked = num_checks == 0  # scalar
    leave_check_mask |= is_not_checked

    # filter by leave check mask
    legal_moves = jnp.where(leave_check_mask, legal_moves, FALSE)
    return legal_moves


def _legal_promotion(state: State, legal_moves: jnp.ndarray) -> jnp.ndarray:
    """Generate legal promotion (81, 81)
    0 = cannot promote
    1 = can promote (from or to opp area)
    2 = have to promote (get stuck)
    """
    promotion = legal_moves.astype(jnp.int8)
    # mask where piece cannot promote
    in_opp_area = jnp.arange(81) % 9 < 3
    tgt_in_opp_area = jnp.tile(in_opp_area, reps=(81, 1))
    src_in_opp_area = tgt_in_opp_area.transpose()
    mask = src_in_opp_area | tgt_in_opp_area
    promotion = jnp.where(mask, promotion, ZERO)
    # mask where piece have to promote
    is_line1 = jnp.tile(jnp.arange(81) % 9 == 0, reps=(81, 1))
    is_line2 = jnp.tile(jnp.arange(81) % 9 == 1, reps=(81, 1))
    where_pawn_or_lance = (state.piece_board == PAWN) | (
        state.piece_board == LANCE
    )
    where_knight = state.piece_board == KNIGHT
    is_stuck = jnp.tile(where_pawn_or_lance, (81, 1)).transpose() & is_line1
    is_stuck |= jnp.tile(where_knight, (81, 1)).transpose() & (
        is_line1 | is_line2
    )
    promotion = jnp.where((promotion != 0) & is_stuck, TWO, promotion)
    return promotion


def _pseudo_legal_drops(
    state: State, effect_boards: jnp.ndarray
) -> jnp.ndarray:
    """Return (7, 81) boolean array

    >>> s = _init()
    >>> s = s.replace(piece_board=s.piece_board.at[15].set(EMPTY))
    >>> s = s.replace(hand=s.hand.at[0].set(1))
    >>> effect_boards = _apply_effects(s)
    >>> _rotate(_pseudo_legal_drops(s, effect_boards)[PAWN])
    Array([[False, False, False, False, False, False, False, False, False],
           [False, False, False, False, False, False, False, False, False],
           [False, False, False, False, False, False, False, False, False],
           [False, False, False, False, False, False, False,  True, False],
           [False, False, False, False, False, False, False,  True, False],
           [False, False, False, False, False, False, False,  True, False],
           [False, False, False, False, False, False, False,  True, False],
           [False, False, False, False, False, False, False, False, False],
           [False, False, False, False, False, False, False, False, False]],      dtype=bool)
    """
    legal_drops = jnp.zeros((7, 81), dtype=jnp.bool_)

    # is the piece in my hand?
    is_in_my_hand = (state.hand[0] > 0).reshape(7, 1)
    legal_drops = jnp.where(is_in_my_hand, TRUE, legal_drops)

    # piece exists
    is_not_empty = state.piece_board != EMPTY
    legal_drops = jnp.where(is_not_empty, FALSE, legal_drops)

    # get stuck
    is_line1 = jnp.arange(81) % 9 == 0
    is_line2 = jnp.arange(81) % 9 == 1
    is_pawn_or_lance = jnp.arange(7) <= LANCE
    is_knight = jnp.arange(7) == KNIGHT
    mask_pawn_lance = is_pawn_or_lance.reshape(7, 1) * is_line1.reshape(1, 81)
    mask_knight = is_knight.reshape(7, 1) * (is_line1 | is_line2).reshape(
        1, 81
    )
    legal_drops = jnp.where(mask_pawn_lance, FALSE, legal_drops)
    legal_drops = jnp.where(mask_knight, FALSE, legal_drops)

    # double pawn
    has_pawn = (state.piece_board == PAWN).reshape(9, 9).any(axis=1)
    has_pawn = jnp.tile(has_pawn, reps=(9, 1)).transpose().flatten()
    legal_drops = jnp.where(has_pawn, FALSE, legal_drops)

    return legal_drops


def _filter_pawn_drop_mate(
    state: State,
    legal_drops: jnp.ndarray,
    effect_boards: jnp.ndarray,
    flipped_effect_boards,
) -> jnp.ndarray:
    """打ち歩詰

    避け方は次の3通り
    - (1) 頭の歩を王で取る
    - (2) 王が逃げる
    - (3) 頭の歩を王以外の駒で取る
    (1)と(2)はようするに、今王が逃げられるところがあるか（利きがないか）ということでまとめて処理できる
    """

    pb = state.piece_board
    opp_king_pos = jnp.nonzero(pb == OPP_KING, size=1)[0][0]
    opp_king_head_pos = (
        opp_king_pos + 1
    )  # NOTE: 王が一番下の段にいるとき間違っているが、その場合は使われないので問題ない
    can_check_by_pawn_drop = opp_king_pos % 9 != 8

    # 王が利きも味方の駒もないところへ逃げられるか
    king_escape_mask = RAW_EFFECT_BOARDS[KING, opp_king_pos, :]  # (81,)
    king_escape_mask &= ~(
        (OPP_PAWN <= pb) & (pb <= OPP_DRAGON)
    )  # 味方駒があり、逃げられない
    king_escape_mask &= ~effect_boards.any(axis=0)  # 利きがあり逃げられない
    can_king_escape = king_escape_mask.any()

    # 反転したボードで処理していることに注意
    flipped_opp_king_head_pos = 80 - opp_king_head_pos
    can_capture_pawn = (
        flipped_effect_boards[:, flipped_opp_king_head_pos].sum() > 1
    )  # 自分以外の利きがないといけない

    legal_drops = jax.lax.cond(
        (can_check_by_pawn_drop & (~can_king_escape) & (~can_capture_pawn)),
        lambda: legal_drops.at[PAWN, opp_king_head_pos].set(FALSE),
        lambda: legal_drops,
    )
    return legal_drops


def _check_info(state, flipped_state, flipped_effect_boards):
    flipped_king_pos = (
        80 - jnp.nonzero(state.piece_board == KING, size=1)[0][0]
    )
    flipped_effecting_mask = flipped_effect_boards[
        :, flipped_king_pos
    ]  # (81,) 王に利いている駒の位置

    @jax.vmap
    def between_king(p, f):
        return IS_ON_THE_WAY[p, f, flipped_king_pos, :]

    from_ = jnp.arange(81)
    large_piece = _to_large_piece_ix(flipped_state.piece_board)
    flipped_between_king_mask = between_king(large_piece, from_)  # (81, 81)
    # 王手してない駒からのマスクは外す
    flipped_aigoma_area_boards = jnp.where(
        flipped_effecting_mask.reshape(81, 1),
        flipped_between_king_mask,
        jnp.zeros_like(flipped_between_king_mask),
    )
    aigoma_area_boards = jnp.flip(flipped_aigoma_area_boards).any(
        axis=0
    )  # (81,)

    return jnp.flip(flipped_effecting_mask), aigoma_area_boards


def _filter_ignoring_check_drops(
    legal_drops: jnp.ndarray,
    checking_piece_board,
    check_defense_board,
):
    num_checks = checking_piece_board.sum()

    # 合駒（王手放置）
    is_not_checked = num_checks == 0
    legal_drops &= is_not_checked | check_defense_board

    # 両王手の場合、合駒は無駄
    is_double_checked = num_checks > 1
    legal_drops &= ~is_double_checked

    return legal_drops


def _flip(state: State):
    empty_mask = state.piece_board == EMPTY
    pb = (state.piece_board + 14) % 28
    pb = jnp.where(empty_mask, EMPTY, pb)
    pb = pb[::-1]
    return state.replace(  # type: ignore
        piece_board=pb, hand=state.hand[jnp.int8((1, 0))]
    )


def _promote(piece: jnp.ndarray) -> jnp.ndarray:
    return piece + 8


def _apply_raw_effects(state: State) -> jnp.ndarray:
    """Obtain raw effect boards from piece board by batch.

    >>> s = _init()
    >>> jnp.rot90(_apply_raw_effects(s).any(axis=0).reshape(9, 9), k=3)
    Array([[ True, False, False, False, False, False, False,  True,  True],
           [ True, False, False, False, False, False, False,  True,  True],
           [ True, False, False, False, False, False,  True,  True,  True],
           [ True, False, False, False, False,  True, False,  True,  True],
           [ True, False, False, False,  True, False, False,  True,  True],
           [ True,  True,  True,  True,  True,  True,  True,  True,  True],
           [ True, False,  True, False, False, False,  True,  True,  True],
           [ True,  True,  True,  True,  True,  True,  True,  True,  True],
           [ True, False,  True,  True,  True,  True,  True,  True, False]],      dtype=bool)
    """
    from_ = jnp.arange(81)
    pieces = state.piece_board  # include -1

    # fix to and apply (pieces, from_) by batch
    @jax.vmap
    def _raw_effect_boards(p, f):
        return RAW_EFFECT_BOARDS[p, f, :]  # (81,)

    mask = ((0 <= pieces) & (pieces < 14)).reshape(81, 1)
    raw_effect_boards = _raw_effect_boards(pieces, from_)
    raw_effect_boards = jnp.where(mask, raw_effect_boards, FALSE)
    return raw_effect_boards  # (81, 81)


def _to_large_piece_ix(piece):
    # Filtering only Lance(0), Bishop(1), Rook(2), Horse(3), and Dragon(4)
    # NOTE: last 14th -1 is sentinel for avoid accessing via -1
    return jnp.int8([-1, 0, -1, -1, 1, 2, -1, -1, -1, -1, -1, -1, 3, 4, -1])[
        piece
    ]


def _apply_effect_filter(state: State) -> jnp.ndarray:
    """
    >>> s = _init()
    >>> jnp.rot90(_apply_effect_filter(s).any(axis=0).reshape(9, 9), k=3)
    Array([[ True, False, False, False, False, False, False,  True,  True],
           [ True, False, False, False, False, False, False,  True,  True],
           [ True, False, False, False, False, False,  True,  True,  True],
           [ True, False, False, False, False,  True, False,  True,  True],
           [ True, False, False, False,  True, False, False,  True,  True],
           [ True, False, False,  True, False, False, False,  True,  True],
           [False, False, False, False, False, False, False, False, False],
           [ True, False, False, False, False, False, False, False, False],
           [False, False, False, False, False, False, False, False, False]],      dtype=bool)

    """
    pieces = state.piece_board
    large_pieces = jax.vmap(_to_large_piece_ix)(pieces)
    from_ = jnp.arange(81)
    to = jnp.arange(81)

    def func1(p, f, t):
        # (piece, from, to) を固定したとき、pieceがfromからtoへ妨害されずに到達できるか否か
        # True = 途中でbrockされ、到達できない
        return (IS_ON_THE_WAY[p, f, t, :] & (state.piece_board >= 0)).any()

    def func2(p, f):
        # (piece, from) を固定してtoにバッチで適用
        return jax.vmap(partial(func1, p=p, f=f))(t=to)

    # (piece, from) にバッチで適用
    filter_boards = jax.vmap(func2)(large_pieces, from_)  # (81,81)

    mask = (large_pieces >= 0).reshape(81, 1)
    filter_boards = jnp.where(mask, filter_boards, FALSE)

    return filter_boards  # (81=from, 81=to)


def _apply_effects(state: State):
    """
    >>> s = _init()
    >>> jnp.rot90(_apply_effects(s)[8].reshape(9, 9), k=3)  # 香
    Array([[False, False, False, False, False, False, False, False, False],
           [False, False, False, False, False, False, False, False, False],
           [False, False, False, False, False, False, False, False, False],
           [False, False, False, False, False, False, False, False, False],
           [False, False, False, False, False, False, False, False, False],
           [False, False, False, False, False, False, False, False, False],
           [False, False, False, False, False, False, False, False,  True],
           [False, False, False, False, False, False, False, False,  True],
           [False, False, False, False, False, False, False, False, False]],      dtype=bool)
    >>> jnp.rot90(_apply_effects(s)[16].reshape(9, 9), k=3)  # 飛
    Array([[False, False, False, False, False, False, False, False, False],
           [False, False, False, False, False, False, False, False, False],
           [False, False, False, False, False, False, False, False, False],
           [False, False, False, False, False, False, False, False, False],
           [False, False, False, False, False, False, False, False, False],
           [False, False, False, False, False, False, False, False, False],
           [False, False, False, False, False, False, False,  True, False],
           [False,  True,  True,  True,  True,  True,  True, False,  True],
           [False, False, False, False, False, False, False,  True, False]],      dtype=bool)
    """
    raw_effect_boards = _apply_raw_effects(state)
    effect_filter_boards = _apply_effect_filter(state)
    return raw_effect_boards & ~effect_filter_boards


def _to_direction(legal_actions: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]):
    # legal_moves から legal_action_mask を作る。toを固定して、
    #
    # legal_from = legal_moves[:, 18]  # to = 18
    # x x o x t x x
    # x x x x x x x
    # x x o x o x x
    # x x x x x x x
    # x x x x x x x
    #
    # があたえられたとき、 (10, 81, 81) の
    #
    # LEGAL_FROM_MASK[UP, 18]  # to = 81
    # x x x x t x x
    # x x x x o x x
    # x x x x o x x
    # x x x x o x x
    # x x x x o x x
    #
    # とandを取って、anyを取れば、(dir=UP, to=18)がTrueか否かわかる
    legal_moves, legal_promotions, legal_drops = legal_actions
    dir_ = jnp.arange(10)
    to = jnp.arange(81)

    def f(t):
        return (
            legal_moves[:, t]
            & (legal_promotions[:, t] != 2)
            & LEGAL_FROM_MASK[dir_, t, :]
        ).any(axis=1)

    def g(t):
        return (
            legal_moves[:, t]
            & (legal_promotions[:, t] != 0)
            & LEGAL_FROM_MASK[dir_, t, :]
        ).any(axis=1)

    legal_action_mask_wo_promotion = jax.vmap(f)(to).transpose()
    legal_action_mask_w_promotion = jax.vmap(g)(to).transpose()
    legal_action_mask = jnp.concatenate(
        [
            legal_action_mask_wo_promotion,
            legal_action_mask_w_promotion,
            legal_drops,
        ]
    )
    return legal_action_mask.flatten()


def _rotate(board: jnp.ndarray) -> jnp.ndarray:
    return jnp.rot90(board.reshape(9, 9), k=3)


def to_sfen(state: State):
    """Convert state into sfen expression.

    - 歩:P 香車:L 桂馬:N 銀:S 角:B 飛車:R 金:G 王:K
    - 成駒なら駒の前に+をつける（と金なら+P）
    - 先手の駒は大文字、後手の駒は小文字で表現
    - 空白の場合、連続する空白の数を入れて次の駒にシフトする。歩空空空飛ならP3R
    - 左上から開始して右に見ていく
    - 段が変わるときは/を挿入
    - 盤面の記入が終わったら手番（b/w）
    - 持ち駒は先手の物から順番はRBGSNLPの順
    - 最後に手数（1で固定）

    >>> s = _init()
    >>> to_sfen(s)
    'lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1'
    """

    pb = jnp.rot90(state.piece_board.reshape((9, 9)), k=3)
    sfen = ""
    # fmt: off
    board_char_dir = ["", "P", "L", "N", "S", "B", "R", "G", "K", "+P", "+L", "+N", "+S", "+B", "+R", "p", "l", "n", "s", "b", "r", "g", "k", "+p", "+l", "+n", "+s", "+b", "+r"]
    hand_char_dir = ["P", "L", "N", "S", "B", "R", "G", "p", "l", "n", "s", "b", "r", "g"]
    hand_dir = [5, 4, 6, 3, 2, 1, 0, 12, 11, 13, 10, 9, 8, 7]
    # fmt: on
    # 盤面
    for i in range(9):
        space_length = 0
        for j in range(9):
            piece = pb[i, j] + 1
            if piece == 0:
                space_length += 1
            elif space_length != 0:
                sfen += str(space_length)
                space_length = 0
            if piece != 0:
                sfen += board_char_dir[piece]
        if space_length != 0:
            sfen += str(space_length)
        if i != 8:
            sfen += "/"
        else:
            sfen += " "
    # 手番
    if state.turn == 0:
        sfen += "b "
    else:
        sfen += "w "
    # 持ち駒
    if jnp.all(state.hand == 0):
        sfen += "- 1"
    else:
        for i in range(2):
            for j in range(7):
                piece_type = hand_dir[i * 7 + j]
                num_piece = state.hand.flatten()[piece_type]
                if num_piece == 0:
                    continue
                if num_piece >= 2:
                    sfen += str(num_piece)
                sfen += hand_char_dir[piece_type]
        sfen += " 1"
    return sfen


def _cshogi_board_to_state(board):
    """Convert cshogi (github.com/TadaoYamaoka/cshogi) board into Pgx state.

    board.pieces_in_hand: 歩香桂銀[金]角飛 金のindexが違う
    """
    pb = jnp.zeros(81, dtype=jnp.int8)
    hand = jnp.zeros((2, 7), dtype=jnp.int8)
    # fmt: off
    board_piece_dir = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0, 0, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
    # fmt: on
    hand_piece_dir = [0, 1, 2, 3, 5, 6, 4]
    pieces = board.pieces
    pieces_in_hand = board.pieces_in_hand
    for i in range(81):
        pb = pb.at[i].set(board_piece_dir[pieces[i]])
    for i in range(2):
        for j in range(7):
            hand = hand.at[i, j].set(pieces_in_hand[i][hand_piece_dir[j]])
    return State(turn=board.turn, piece_board=pb, hand=hand)
