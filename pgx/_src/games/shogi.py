# Copyright 2023 The Pgx Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import NamedTuple
from functools import partial

import numpy as np
import jax
from jax import Array
import jax.numpy as jnp


MAX_TERMINATION_STEPS = 512  # From AZ paper

TRUE = jnp.bool_(True)
FALSE = jnp.bool_(False)


EMPTY = -1  # 空白
PAWN = 0  # 歩
LANCE = 1  # 香
KNIGHT = 2  # 桂
SILVER = 3  # 銀
BISHOP = 4  # 角
ROOK = 5  # 飛
GOLD = 6  # 金
KING = 7  # 玉
PRO_PAWN = 8  # と
PRO_LANCE = 9  # 成香
PRO_KNIGHT = 10  # 成桂
PRO_SILVER = 11  # 成銀
HORSE = 12  # 馬
DRAGON = 13  # 龍


# fmt: off
INIT_PIECE_BOARD = jnp.int32([[15, -1, 14, -1, -1, -1, 0, -1, 1],  # noqa: E241
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
def can_move_to(piece, from_, to):
    """Can <piece> move from <from_> to <to>?"""
    if from_ == to:
        return False
    x0, y0 = from_ // 9, from_ % 9
    x1, y1 = to // 9, to % 9
    dx = x1 - x0
    dy = y1 - y0
    if piece == PAWN:
        if dx == 0 and dy == -1:
            return True
        else:
            return False
    elif piece == LANCE:
        if dx == 0 and dy < 0:
            return True
        else:
            return False
    elif piece == KNIGHT:
        if dx in (-1, 1) and dy == -2:
            return True
        else:
            return False
    elif piece == SILVER:
        if dx in (-1, 0, 1) and dy == -1:
            return True
        elif dx in (-1, 1) and dy == 1:
            return True
        else:
            return False
    elif piece == BISHOP:
        if dx == dy or dx == -dy:
            return True
        else:
            return False
    elif piece == ROOK:
        if dx == 0 or dy == 0:
            return True
        else:
            return False
    if piece in (GOLD, PRO_PAWN, PRO_LANCE, PRO_KNIGHT, PRO_SILVER):
        if dx in (-1, 0, 1) and dy in (0, -1):
            return True
        elif dx == 0 and dy == 1:
            return True
        else:
            return False
    elif piece == KING:
        if abs(dx) <= 1 and abs(dy) <= 1:
            return True
        else:
            return False
    elif piece == HORSE:
        if abs(dx) <= 1 and abs(dy) <= 1:
            return True
        elif dx == dy or dx == -dy:
            return True
        else:
            return False
    elif piece == DRAGON:
        if abs(dx) <= 1 and abs(dy) <= 1:
            return True
        if dx == 0 or dy == 0:
            return True
        else:
            return False
    else:
        assert False


def is_on_the_way(piece, from_, to, point):
    if to == point:
        return False
    if piece not in (LANCE, BISHOP, ROOK, HORSE, DRAGON):
        return False
    if not can_move_to(piece, from_, to):
        return False
    if not can_move_to(piece, from_, point):
        return False

    x0, y0 = from_ // 9, from_ % 9
    x1, y1 = to // 9, to % 9
    x2, y2 = point // 9, point % 9
    dx1, dy1 = x1 - x0, y1 - y0
    dx2, dy2 = x2 - x0, y2 - y0

    def sign(d):
        if d == 0:
            return 0
        return d > 0

    if (sign(dx1) != sign(dx2)) or (sign(dy1) != sign(dy2)):
        return False

    return abs(dx2) <= abs(dx1) and abs(dy2) <= abs(dy1)


CAN_MOVE = np.zeros((14, 81, 81), dtype=jnp.bool_)
for piece in range(14):
    for from_ in range(81):
        for to in range(81):
            CAN_MOVE[piece, from_, to] = can_move_to(piece, from_, to)

assert CAN_MOVE.sum() == 8228
CAN_MOVE = jnp.array(CAN_MOVE)


# When <lance/bishop/rook/horse/dragon,5> moves from <from,81> to <to,81>,
# is <point,81> on the way between two points?
BETWEEN = np.zeros((5, 81, 81, 81), dtype=np.bool_)
for i, piece in enumerate((LANCE, BISHOP, ROOK, HORSE, DRAGON)):
    for from_ in range(81):
        for to in range(81):
            for p in range(81):
                BETWEEN[i, from_, to, p] = is_on_the_way(piece, from_, to, p)

BETWEEN = jnp.array(BETWEEN)
assert BETWEEN.sum() == 10564


# Give <dir,10> and <to,81>, return the legal <from> idx
# E.g. LEGAL_FROM_IDX[Up, to=19] = [20, 21, ..., -1] (filled by -1)
# Used for computing dlshogi action
#
#  dir, to, from
#  (10, 81, 81)
#
#  0 Up
#  1 Up left
#  2 Up right
#  3 Left
#  4 Right
#  5 Down
#  6 Down left
#  7 Down right
#  8 Up2 left
#  9 Up2 right

LEGAL_FROM_IDX = -np.ones((10, 81, 8), dtype=jnp.int32)  # type: ignore

for dir_ in range(10):
    for to in range(81):
        x, y = to // 9, to % 9
        if dir_ == 0:  # Up
            dx, dy = 0, +1
        elif dir_ == 1:  # Up left
            dx, dy = -1, +1
        elif dir_ == 2:  # Up right
            dx, dy = +1, +1
        elif dir_ == 3:  # Left
            dx, dy = -1, 0
        elif dir_ == 4:  # Right
            dx, dy = +1, 0
        elif dir_ == 5:  # Down
            dx, dy = 0, -1
        elif dir_ == 6:  # Down left
            dx, dy = -1, -1
        elif dir_ == 7:  # Down right
            dx, dy = +1, -1
        elif dir_ == 8:  # Up2 left
            dx, dy = -1, +2
        elif dir_ == 9:  # Up2 right
            dx, dy = +1, +2
        for i in range(8):
            x += dx
            y += dy
            if x < 0 or 8 < x or y < 0 or 8 < y:
                break
            LEGAL_FROM_IDX[dir_, to, i] = x * 9 + y
            if dir_ == 8 or dir_ == 9:
                break

LEGAL_FROM_IDX = jnp.array(LEGAL_FROM_IDX)  # type: ignore


@jax.jit
@jax.vmap
def can_move_any_ix(from_):
    return jnp.nonzero(
        (CAN_MOVE[:, from_, :] | CAN_MOVE[:, :, from_]).any(axis=0),
        size=36,
        fill_value=-1,
    )[0]


@jax.jit
@jax.vmap
def neighbour_ix(from_):
    return jnp.nonzero(
        (CAN_MOVE[7, from_, :] | CAN_MOVE[2, :, from_]),
        size=10,
        fill_value=-1,
    )[0]


NEIGHBOUR_IX = neighbour_ix(jnp.arange(81))


def between_ix(p, from_, to):
    return jnp.nonzero(BETWEEN[p, from_, to], size=8, fill_value=-1)[0]


BETWEEN_IX = jax.jit(
    jax.vmap(
        jax.vmap(jax.vmap(between_ix, (None, None, 0)), (None, 0, None)),
        (0, None, None),
    )
)(jnp.arange(5), jnp.arange(81), jnp.arange(81))


CAN_MOVE_ANY = can_move_any_ix(jnp.arange(81))  # (81, 36)


def _around(c):
    x, y = c // 9, c % 9
    dx = jnp.int32([-1, -1, 0, +1, +1, +1, 0, -1])
    dy = jnp.int32([0, -1, -1, -1, 0, +1, +1, +1])

    def f(i):
        new_x, new_y = x + dx[i], y + dy[i]
        return jax.lax.select(
            (new_x < 0) | (new_x >= 9) | (new_y < 0) | (new_y >= 9),
            -1,
            new_x * 9 + new_y,
        )

    return jax.vmap(f)(jnp.arange(8))


AROUND_IX = jax.vmap(_around)(jnp.arange(81))


EMPTY = jnp.int32(-1)  # 空白
PAWN = jnp.int32(0)  # 歩
LANCE = jnp.int32(1)  # 香
KNIGHT = jnp.int32(2)  # 桂
SILVER = jnp.int32(3)  # 銀
BISHOP = jnp.int32(4)  # 角
ROOK = jnp.int32(5)  # 飛
GOLD = jnp.int32(6)  # 金
KING = jnp.int32(7)  # 玉
PRO_PAWN = jnp.int32(8)  # と
PRO_LANCE = jnp.int32(9)  # 成香
PRO_KNIGHT = jnp.int32(10)  # 成桂
PRO_SILVER = jnp.int32(11)  # 成銀
HORSE = jnp.int32(12)  # 馬
DRAGON = jnp.int32(13)  # 龍
# --- opponent pieces ---
OPP_PAWN = jnp.int32(14)  # 歩
OPP_LANCE = jnp.int32(15)  # 香
OPP_KNIGHT = jnp.int32(16)  # 桂
OPP_SILVER = jnp.int32(17)  # 銀
OPP_BISHOP = jnp.int32(18)  # 角
OPP_ROOK = jnp.int32(19)  # 飛
OPP_GOLD = jnp.int32(20)  # 金
OPP_KING = jnp.int32(21)  # 玉
OPP_PRO_PAWN = jnp.int32(22)  # と
OPP_PRO_LANCE = jnp.int32(23)  # 成香
OPP_PRO_KNIGHT = jnp.int32(24)  # 成桂
OPP_PRO_SILVER = jnp.int32(25)  # 成銀
OPP_HORSE = jnp.int32(26)  # 馬
OPP_DRAGON = jnp.int32(27)  # 龍

ALL_SQ = jnp.arange(81)


class GameState(NamedTuple):
    turn: Array = jnp.int32(0)  # 0 or 1
    board: Array = INIT_PIECE_BOARD  # (81,) flip in turn
    hand: Array = jnp.zeros((2, 7), dtype=jnp.int32)  # flip in turn
    # cache
    # Redundant information used only in _is_checked for speeding-up
    cache_m2b: Array = -jnp.ones(8, dtype=jnp.int32)
    cache_king: Array = jnp.int32(44)


class Game:
    def init(self) -> GameState:
        return GameState()

    def step(self, state: GameState, action: Array) -> GameState:
        return _step(state, action)

    def observe(self, state: GameState) -> Array:
        return _observe(state, False)

    def legal_action_mask(self, state: GameState) -> Array:
        return _legal_action_mask(state)


class Action(NamedTuple):
    is_drop: Array
    piece: Array
    to: Array
    # --- Optional (only for move action) ---
    from_: Array = jnp.int32(0)
    is_promotion: Array = FALSE

    @staticmethod
    def make_move(piece, from_, to, is_promotion=FALSE):
        return Action(
            is_drop=FALSE,
            piece=piece,
            from_=from_,
            to=to,
            is_promotion=is_promotion,
        )

    @staticmethod
    def make_drop(piece, to):
        return Action(is_drop=TRUE, piece=piece, to=to)

    @staticmethod
    def _from_dlshogi_action(state: GameState, action: Array):
        """Direction (from github.com/TadaoYamaoka/cshogi)

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
        action = jnp.int32(action)
        direction, to = jnp.int32(action // 81), jnp.int32(action % 81)
        is_drop = direction >= 20
        is_promotion = (10 <= direction) & (direction < 20)
        # LEGAL_FROM_IDX[UP, 19] = [20, 21, ... -1]
        legal_from_idx = LEGAL_FROM_IDX[direction % 10, to]  # (81,)
        from_cand = state.board[legal_from_idx]  # (8,)
        mask = (legal_from_idx >= 0) & (PAWN <= from_cand) & (from_cand < OPP_PAWN)
        i = jnp.argmax(mask)
        from_ = jax.lax.select(is_drop, 0, legal_from_idx[i])
        piece = jax.lax.select(is_drop, direction - 20, state.board[from_])
        return Action(is_drop=is_drop, piece=piece, to=to, from_=from_, is_promotion=is_promotion)


def _init_board():
    """Initialize Shogi State."""
    return State()


def _step(state: GameState, action: Array) -> GameState:
    a = Action._from_dlshogi_action(state, action)
    # apply move/drop action
    state = jax.lax.cond(a.is_drop, _step_drop, _step_move, *(state, a))
    # flip state
    state = _flip(state)
    return state._replace(turn=(state.turn + 1) % 2)


def _step_move(state: GameState, action: Action) -> GameState:
    pb = state.board
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
    piece = jax.lax.select(action.is_promotion, action.piece + 8, action.piece)
    # set piece to the target position
    pb = pb.at[action.to].set(piece)
    # apply piece moves
    return state._replace(board=pb, hand=hand)  # type: ignore


def _step_drop(state: GameState, action: Action) -> GameState:
    # add piece to board
    pb = state.board.at[action.to].set(action.piece)
    # remove piece from hand
    hand = state.hand.at[0, action.piece].add(-1)
    return state._replace(board=pb, hand=hand)  # type: ignore


def _set_cache(state: GameState):
    return state._replace(  # type: ignore
        cache_m2b=jnp.nonzero(jax.vmap(_is_major_piece)(state.board), size=8, fill_value=-1)[0],
        cache_king=jnp.argmin(jnp.abs(state.board - KING)),
    )


def _legal_action_mask(state: GameState):
    # update cache
    state = _set_cache(state)

    a = jax.vmap(partial(Action._from_dlshogi_action, state=state))(action=jnp.arange(27 * 81))

    @jax.vmap
    def is_legal_move_wo_pro(i):
        return _is_legal_move_wo_pro(a.from_[i], a.to[i], state)

    @jax.vmap
    def is_legal_drop_wo_piece(to):
        return _is_legal_drop_wo_piece(to, state)

    pseudo_legal_moves = is_legal_move_wo_pro(jnp.arange(10 * 81))
    pseudo_legal_drops = is_legal_drop_wo_piece(jnp.arange(81))

    @jax.vmap
    def is_legal_move(i):
        return pseudo_legal_moves[i % (10 * 81)] & jax.lax.cond(
            a.is_promotion[i], _is_promotion_legal, _is_no_promotion_legal, *(a.from_[i], a.to[i], state)
        )

    @jax.vmap
    def is_legal_drop(i):
        return pseudo_legal_drops[i % 81] & _is_legal_drop_wo_ignoring_check(a.piece[i], a.to[i], state)

    legal_action_mask = jnp.hstack(
        (
            is_legal_move(jnp.arange(20 * 81)),
            is_legal_drop(jnp.arange(20 * 81, 27 * 81)),
        )
    )  # (27 * 81)

    # check drop pawn mate
    is_drop_pawn_mate, to = _is_drop_pawn_mate(state)
    direction = 20
    can_drop_pawn = legal_action_mask[direction * 81 + to]  # current
    can_drop_pawn &= ~is_drop_pawn_mate
    legal_action_mask = legal_action_mask.at[direction * 81 + to].set(can_drop_pawn)

    return legal_action_mask


def _is_drop_pawn_mate(state: GameState):
    # check pawn drop mate
    opp_king_pos = jnp.argmin(jnp.abs(state.board - OPP_KING))
    to = opp_king_pos + 1
    flip_state = _flip(state._replace(board=state.board.at[to].set(PAWN)))
    # Not checkmate if
    #   (1) can capture checking pawn, or
    #   (2) king can escape
    # fmt: off
    flipped_to = 80 - to
    flip_state = _set_cache(flip_state)
    can_capture_pawn = jax.vmap(partial(
        _is_legal_move_wo_pro, to=flipped_to, state=flip_state
    ))(from_=CAN_MOVE_ANY[flipped_to]).any()
    from_ = 80 - opp_king_pos
    can_king_escape = jax.vmap(
        partial(_is_legal_move_wo_pro, from_=from_, state=flip_state)
    )(to=AROUND_IX[from_]).any()
    is_pawn_mate = ~(can_capture_pawn | can_king_escape)
    # fmt: on
    return is_pawn_mate, to


def _is_legal_drop_wo_piece(to: Array, state: GameState):
    is_illegal = state.board[to] != EMPTY
    is_illegal |= _is_checked(state._replace(board=state.board.at[to].set(PAWN)))
    return ~is_illegal


def _is_legal_drop_wo_ignoring_check(piece: Array, to: Array, state: GameState):
    is_illegal = state.board[to] != EMPTY
    # don't have the piece
    is_illegal |= state.hand[0, piece] <= 0
    # double pawn
    is_illegal |= (piece == PAWN) & ((state.board == PAWN).reshape(9, 9).sum(axis=1) > 0)[to // 9]
    # get stuck
    is_illegal |= ((piece == PAWN) | (piece == LANCE)) & (to % 9 == 0)
    is_illegal |= (piece == KNIGHT) & (to % 9 < 2)
    return ~is_illegal


def _is_legal_move_wo_pro(
    from_: Array,
    to: Array,
    state: GameState,
):
    ok = _is_pseudo_legal_move(from_, to, state)
    ok &= ~_is_checked(
        state._replace(
            board=state.board.at[from_].set(EMPTY).at[to].set(state.board[from_]),
            cache_king=jax.lax.select(  # update cache
                state.board[from_] == KING,
                jnp.int32(to),
                state.cache_king,
            )
        )
    )
    return ok


def _is_pseudo_legal_move(
    from_: Array,
    to: Array,
    state: GameState,
):
    ok = _is_pseudo_legal_move_wo_obstacles(from_, to, state)
    # there is an obstacle between from_ and to
    i = _major_piece_ix(state.board[from_])
    between_ix = BETWEEN_IX[i, from_, to, :]
    is_illegal = (i >= 0) & ((between_ix >= 0) & (state.board[between_ix] != EMPTY)).any()
    return ok & ~is_illegal


def _is_pseudo_legal_move_wo_obstacles(
    from_: Array,
    to: Array,
    state: GameState,
):
    board = state.board
    # source is not my piece
    piece = board[from_]
    is_illegal = (from_ < 0) | ~((PAWN <= piece) & (piece < OPP_PAWN))
    # destination is my piece
    is_illegal |= (PAWN <= board[to]) & (board[to] < OPP_PAWN)
    # piece cannot move like that
    is_illegal |= ~CAN_MOVE[piece, from_, to]
    return ~is_illegal


def _is_no_promotion_legal(
    from_: Array,
    to: Array,
    state: GameState,
):
    # source is not my piece
    piece = state.board[from_]
    # promotion
    is_illegal = ((piece == PAWN) | (piece == LANCE)) & (to % 9 == 0)  # Must promote
    is_illegal |= (piece == KNIGHT) & (to % 9 < 2)  # Must promote
    return ~is_illegal


def _is_promotion_legal(
    from_: Array,
    to: Array,
    state: GameState,
):
    # source is not my piece
    piece = state.board[from_]
    # promotion
    is_illegal = (GOLD <= piece) & (piece <= DRAGON)  # Pieces cannot promote
    is_illegal |= (from_ % 9 >= 3) & (to % 9 >= 3)  # irrelevant to the opponent's territory
    return ~is_illegal


def _is_checked(state: GameState):
    # Use cached king position, simpler implementation is:
    # jnp.argmin(jnp.abs(state.pieceboard - KING))
    king_pos = state.cache_king
    flipped_king_pos = 80 - king_pos

    @jax.vmap
    def can_capture_king(from_):
        return _is_pseudo_legal_move(from_=from_, to=flipped_king_pos, state=_flip(state))

    @jax.vmap
    def can_capture_king_local(from_):
        return _is_pseudo_legal_move_wo_obstacles(from_=from_, to=flipped_king_pos, state=_flip(state))

    # Simpler implementation without cache of major piece places
    # from_ = CAN_MOVE_ANY[flipped_king_pos]
    # return can_capture_king(from_).any()
    from_ = 80 - state.cache_m2b
    from_ = jnp.where(from_ == 81, -1, from_)
    neighbours = NEIGHBOUR_IX[flipped_king_pos]
    return can_capture_king(from_).any() | can_capture_king_local(neighbours).any()


def _flip_piece(piece):
    return jax.lax.select(piece >= 0, (piece + 14) % 28, piece)


def _rotate(board: Array) -> Array:
    return jnp.rot90(board.reshape(9, 9), k=3)


def _flip(state: GameState):
    empty_mask = state.board == EMPTY
    pb = (state.board + 14) % 28
    pb = jnp.where(empty_mask, EMPTY, pb)
    pb = pb[::-1]
    return state._replace(
        board=pb,
        hand=state.hand[jnp.int32((1, 0))],
    )


def _is_major_piece(piece):
    return (
        (piece == LANCE)
        | (piece == BISHOP)
        | (piece == ROOK)
        | (piece == HORSE)
        | (piece == DRAGON)
        | (piece == OPP_LANCE)
        | (piece == OPP_BISHOP)
        | (piece == OPP_ROOK)
        | (piece == OPP_HORSE)
        | (piece == OPP_DRAGON)
    )


def _major_piece_ix(piece):
    ixs = (
        (-jnp.ones(28, dtype=jnp.int32))
        .at[LANCE]
        .set(0)
        .at[BISHOP]
        .set(1)
        .at[ROOK]
        .set(2)
        .at[HORSE]
        .set(1)
        .at[DRAGON]
        .set(2)
    )
    return jax.lax.select(piece >= 0, ixs[piece], jnp.int32(-1))


def _observe(state: GameState, flip: bool = False) -> Array:
    state, flip_state = jax.lax.cond(
        flip,
        lambda: (state, _flip(state)),
        lambda: (_flip(state), state),
    )

    def pieces(state):
        # piece positions
        my_pieces = jnp.arange(OPP_PAWN)
        my_piece_feat = jax.vmap(lambda p: state.board == p)(my_pieces)
        return my_piece_feat

    def effect_all(state):
        def effect(from_, to):
            piece = state.board[from_]
            can_move = CAN_MOVE[piece, from_, to]
            major_piece_ix = _major_piece_ix(piece)
            between_ix = BETWEEN_IX[major_piece_ix, from_, to, :]
            has_obstacles = jax.lax.select(
                major_piece_ix >= 0,
                ((between_ix >= 0) & (state.board[between_ix] != EMPTY)).any(),
                FALSE,
            )
            return can_move & ~has_obstacles

        effects = jax.vmap(jax.vmap(effect, (None, 0)), (0, None))(ALL_SQ, ALL_SQ)
        mine = (PAWN <= state.board) & (state.board < OPP_PAWN)
        return jnp.where(mine.reshape(81, 1), effects, FALSE)

    def piece_and_effect(state):
        my_pieces = jnp.arange(OPP_PAWN)
        my_effect = effect_all(state)

        @jax.vmap
        def filter_effect(p):
            mask = state.board == p
            return jnp.where(mask.reshape(81, 1), my_effect, FALSE).any(axis=0)

        my_effect_feat = filter_effect(my_pieces)
        my_effect_sum = my_effect.sum(axis=0)

        @jax.vmap
        def effect_sum(n) -> Array:
            return my_effect_sum >= n  # type: ignore

        effect_sum_feat = effect_sum(jnp.arange(1, 4))
        return my_effect_feat, effect_sum_feat

    def num_hand(n, hand, p):
        return jnp.tile(hand[p] >= n, reps=(9, 9))

    def hand_feat(hand):
        # fmt: off
        pawn_feat = jax.vmap(partial(num_hand, hand=hand, p=PAWN))(jnp.arange(1, 9))
        lance_feat = jax.vmap(partial(num_hand, hand=hand, p=LANCE))(jnp.arange(1, 5))
        knight_feat = jax.vmap(partial(num_hand, hand=hand, p=KNIGHT))(jnp.arange(1, 5))
        silver_feat = jax.vmap(partial(num_hand, hand=hand, p=SILVER))(jnp.arange(1, 5))
        gold_feat = jax.vmap(partial(num_hand, hand=hand, p=GOLD))(jnp.arange(1, 5))
        bishop_feat = jax.vmap(partial(num_hand, hand=hand, p=BISHOP))(jnp.arange(1, 3))
        rook_feat = jax.vmap(partial(num_hand, hand=hand, p=ROOK))(jnp.arange(1, 3))
        return [pawn_feat, lance_feat, knight_feat, silver_feat, gold_feat, bishop_feat, rook_feat]
        # fmt: on

    my_piece_feat = pieces(state)
    my_effect_feat, my_effect_sum_feat = piece_and_effect(state)
    opp_piece_feat = pieces(flip_state)
    opp_effect_feat, opp_effect_sum_feat = piece_and_effect(flip_state)
    opp_piece_feat = opp_piece_feat[:, ::-1]
    opp_effect_feat = opp_effect_feat[:, ::-1]
    opp_effect_sum_feat = opp_effect_sum_feat[:, ::-1]
    my_hand_feat = hand_feat(state.hand[0])
    opp_hand_feat = hand_feat(state.hand[1])
    # NOTE: update cache
    checked = jnp.tile(_is_checked(_set_cache(state)), reps=(1, 9, 9))
    feat1 = [
        my_piece_feat.reshape(14, 9, 9),
        my_effect_feat.reshape(14, 9, 9),
        my_effect_sum_feat.reshape(3, 9, 9),
        opp_piece_feat.reshape(14, 9, 9),
        opp_effect_feat.reshape(14, 9, 9),
        opp_effect_sum_feat.reshape(3, 9, 9),
    ]
    feat2 = my_hand_feat + opp_hand_feat + [checked]
    feat = jnp.vstack(feat1 + feat2)
    return jnp.rot90(feat.transpose((1, 2, 0)), k=3)
