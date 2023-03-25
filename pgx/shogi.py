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


from functools import partial

import jax
import jax.numpy as jnp

import pgx.core as core
from pgx._flax.struct import dataclass
from pgx._shogi_utils import (
    BETWEEN_IX,
    CAN_MOVE,
    CAN_MOVE_ANY,
    INIT_PIECE_BOARD,
    LEGAL_FROM_IDX,
    _from_sfen,
    _to_sfen,
)

TRUE = jnp.bool_(True)
FALSE = jnp.bool_(False)

EMPTY = jnp.int8(-1)  # 空白
PAWN = jnp.int8(0)  # 歩
LANCE = jnp.int8(1)  # 香
KNIGHT = jnp.int8(2)  # 桂
SILVER = jnp.int8(3)  # 銀
BISHOP = jnp.int8(4)  # 角
ROOK = jnp.int8(5)  # 飛
GOLD = jnp.int8(6)  # 金
KING = jnp.int8(7)  # 玉
PRO_PAWN = jnp.int8(8)  # と
PRO_LANCE = jnp.int8(9)  # 成香
PRO_KNIGHT = jnp.int8(10)  # 成桂
PRO_SILVER = jnp.int8(11)  # 成銀
HORSE = jnp.int8(12)  # 馬
DRAGON = jnp.int8(13)  # 龍
OPP_PAWN = jnp.int8(14)  # 相手歩
OPP_LANCE = jnp.int8(15)  # 相手香
OPP_KNIGHT = jnp.int8(16)  # 相手桂
OPP_SILVER = jnp.int8(17)  # 相手銀
OPP_BISHOP = jnp.int8(18)  # 相手角
OPP_ROOK = jnp.int8(19)  # 相手飛
OPP_GOLD = jnp.int8(20)  # 相手金
OPP_KING = jnp.int8(21)  # 相手玉
OPP_PRO_PAWN = jnp.int8(22)  # 相手と
OPP_PRO_LANCE = jnp.int8(23)  # 相手成香
OPP_PRO_KNIGHT = jnp.int8(24)  # 相手成桂
OPP_PRO_SILVER = jnp.int8(25)  # 相手成銀
OPP_HORSE = jnp.int8(26)  # 相手馬
OPP_DRAGON = jnp.int8(27)  # 相手龍

ALL_SQ = jnp.arange(81)


@dataclass
class State(core.State):
    current_player: jnp.ndarray = jnp.int8(0)
    reward: jnp.ndarray = jnp.float32([0.0, 0.0])
    terminated: jnp.ndarray = FALSE
    truncated: jnp.ndarray = FALSE
    legal_action_mask: jnp.ndarray = jnp.zeros(81 * 81, dtype=jnp.bool_)
    observation: jnp.ndarray = jnp.zeros((119, 9, 9), dtype=jnp.bool_)
    _rng_key: jax.random.KeyArray = jax.random.PRNGKey(0)
    _step_count: jnp.ndarray = jnp.int32(0)
    # --- Shogi specific ---
    turn: jnp.ndarray = jnp.int8(0)  # 0 or 1
    piece_board: jnp.ndarray = INIT_PIECE_BOARD  # (81,) 後手のときにはflipする
    hand: jnp.ndarray = jnp.zeros((2, 7), dtype=jnp.int8)  # 後手のときにはflipする

    @staticmethod
    def _from_board(turn, piece_board: jnp.ndarray, hand: jnp.ndarray):
        """Mainly for debugging purpose.
        terminated, reward, and current_player are not changed"""
        state = State(turn=turn, piece_board=piece_board, hand=hand)  # type: ignore
        # fmt: off
        state = jax.lax.cond(turn % 2 == 1, lambda: _flip(state), lambda: state)
        # fmt: on
        return state.replace(legal_action_mask=_legal_action_mask(state))  # type: ignore

    @staticmethod
    def _from_sfen(sfen):
        turn, pb, hand, step_count = _from_sfen(sfen)
        return jax.jit(State._from_board)(turn, pb, hand).replace(  # type: ignore
            _step_count=jnp.int32(step_count)
        )

    def _to_sfen(self):
        state = self if self.turn % 2 == 0 else _flip(self)
        return _to_sfen(state)


class Shogi(core.Env):
    def __init__(self, max_termination_steps: int = 1000):
        super().__init__()
        self.max_termination_steps = max_termination_steps

    def _init(self, key: jax.random.KeyArray) -> State:
        state = _init_board()
        rng, subkey = jax.random.split(key)
        current_player = jnp.int8(jax.random.bernoulli(subkey))
        return state.replace(current_player=current_player)  # type: ignore

    def _step(self, state: core.State, action: jnp.ndarray) -> State:
        assert isinstance(state, State)
        # Note: Assume that illegal action is already filtered by Env.step
        state = _step(state, action)
        state = jax.lax.cond(
            (0 <= self.max_termination_steps)
            & (self.max_termination_steps <= state._step_count),
            # end with tie
            lambda: state.replace(terminated=TRUE),  # type: ignore
            lambda: state,
        )
        return state  # type: ignore

    def _observe(
        self, state: core.State, player_id: jnp.ndarray
    ) -> jnp.ndarray:
        assert isinstance(state, State)
        return _observe(state, player_id)

    @property
    def name(self) -> str:
        return "Shogi"

    @property
    def version(self) -> str:
        return "beta"

    @property
    def num_players(self) -> int:
        return 2


@dataclass
class Action:
    is_drop: jnp.ndarray
    piece: jnp.ndarray
    to: jnp.ndarray
    # --- Optional (only for move action) ---
    from_: jnp.ndarray = jnp.int8(0)
    is_promotion: jnp.ndarray = FALSE

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
    def _from_dlshogi_action(state: State, action: jnp.ndarray):
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
        direction, to = jnp.int8(action // 81), jnp.int8(action % 81)
        is_drop = direction >= 20
        is_promotion = (10 <= direction) & (direction < 20)
        # LEGAL_FROM_IDX[UP, 19] = [20, 21, ... -1]
        legal_from_idx = LEGAL_FROM_IDX[direction % 10, to]  # (81,)
        from_cand = state.piece_board[legal_from_idx]  # (8,)
        mask = (
            (legal_from_idx >= 0)
            & (PAWN <= from_cand)
            & (from_cand < OPP_PAWN)
        )
        i = jnp.nonzero(mask, size=1)[0][0]
        from_ = jax.lax.select(is_drop, 0, legal_from_idx[i])
        piece = jax.lax.select(
            is_drop, direction - 20, state.piece_board[from_]
        )
        return Action(is_drop=is_drop, piece=piece, to=to, from_=from_, is_promotion=is_promotion)  # type: ignore


def _init_board():
    """Initialize Shogi State."""
    state = State()
    return state.replace(legal_action_mask=_legal_action_mask(state))  # type: ignore


def _step(state: State, action: jnp.ndarray):
    a = Action._from_dlshogi_action(state, action)
    # apply move/drop action
    state = jax.lax.cond(a.is_drop, _step_drop, _step_move, *(state, a))
    # flip state
    state = _flip(state)
    state = state.replace(  # type: ignore
        current_player=(state.current_player + 1) % 2,
        turn=(state.turn + 1) % 2,
    )
    legal_action_mask = _legal_action_mask(state)
    terminated = ~legal_action_mask.any()
    # fmt: off
    reward = jax.lax.select(
        terminated,
        jnp.ones(2, dtype=jnp.float32).at[state.current_player].set(-1),
        jnp.zeros(2, dtype=jnp.float32),
    )
    # fmt: on
    return state.replace(  # type: ignore
        legal_action_mask=legal_action_mask,
        terminated=terminated,
        reward=reward,
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
    piece = jax.lax.select(action.is_promotion, action.piece + 8, action.piece)
    # set piece to the target position
    pb = pb.at[action.to].set(piece)
    # apply piece moves
    return state.replace(piece_board=pb, hand=hand)  # type: ignore


def _step_drop(state: State, action: Action) -> State:
    # add piece to board
    pb = state.piece_board.at[action.to].set(action.piece)
    # remove piece from hand
    hand = state.hand.at[0, action.piece].add(-1)
    return state.replace(piece_board=pb, hand=hand)  # type: ignore


def _legal_action_mask(state: State):
    checking_places = _checking_places(state.piece_board)

    @jax.vmap
    def is_legal(action):
        a = Action._from_dlshogi_action(state, action)
        return jax.lax.cond(
            a.is_drop,
            lambda: _is_legal_drop(
                state.hand, a.piece, a.to, state.piece_board, checking_places
            ),
            lambda: jax.lax.cond(
                a.from_ < 0,  # a is invalid. All LEGAL_FROM_IDX == -1
                lambda: FALSE,
                lambda: _is_legal_move(
                    a.from_ * 81 + a.to, a.is_promotion, state.piece_board
                ),
            ),
        )

    legal_action_mask = is_legal(jnp.arange(27 * 81))

    # check pawn drop mate
    direction = 20  # drop pawn
    opp_king_pos = jnp.nonzero(state.piece_board == OPP_KING, size=1)[0][0]
    to = opp_king_pos + 1
    flip_state = _flip(
        state.replace(piece_board=state.piece_board.at[to].set(PAWN))  # type: ignore
    )

    # 玉頭の歩を取るか玉が逃げられれば詰みでない
    # fmt: off
    vmap_is_legal_move = jax.vmap(jax.vmap(
        partial(_is_legal_move, board=flip_state.piece_board),
        (0, None)), (None, 0)
    )
    flipped_to = 80 - to
    can_capture_pawn = vmap_is_legal_move(ALL_SQ * 81 + flipped_to, jnp.bool_([False, True])).any()
    from_ = 80 - opp_king_pos
    can_king_escape = vmap_is_legal_move(from_ * 81 + _around(from_), jnp.bool_([False])).any()
    is_pawn_mate = ~(can_capture_pawn | can_king_escape)
    # fmt: on

    can_drop_pawn = legal_action_mask[direction * 81 + to]  # current
    has_no_pawn = state.hand[0, PAWN] <= 0
    is_occupied = state.piece_board[to] != EMPTY
    can_drop_pawn &= (
        ~(has_no_pawn | is_occupied | (to % 9 == 0)) & ~is_pawn_mate
    )

    return legal_action_mask.at[direction * 81 + to].set(can_drop_pawn)


def _around(x):
    y = jnp.int8([x - 1, x - 10, x - 9, x - 8, x + 1, x + 8, x + 9, x + 10])
    return jnp.where((y < 0) | (y >= 81), -1, y)


def _is_legal_drop(
    hand: jnp.ndarray,
    piece: jnp.ndarray,
    to: jnp.ndarray,
    board: jnp.ndarray,
    checking_places: jnp.ndarray,
):
    ok = _is_pseudo_legal_drop(hand, piece, to, board)

    ##################################################
    # Filter illegal moves
    ##################################################
    # Simple implementation is to
    #     1. actually drop, and
    #     2. check whether the king is checked:
    # but this is slow
    # > ok &= ~_is_checked(board.at[to].set(piece))
    num_checks = (checking_places != -1).sum()
    # num_checks >= 2
    ok &= num_checks < 2  # 両王手は合駒できない
    # num_checks == 1
    king_pos = jnp.nonzero(board == KING, size=1)[0][0]
    checking_place = checking_places[
        jnp.nonzero(checking_places != -1, size=1)[0][0]
    ]
    checking_piece = _flip_piece(board[checking_place])
    checking_major_piece = _major_piece_ix(checking_piece)
    is_on_the_way = (
        to == BETWEEN_IX[checking_major_piece, king_pos, checking_place]
    ).any()
    ok &= (num_checks == 0) | is_on_the_way

    return ok


def _is_legal_move(
    move: jnp.ndarray, is_promotion: jnp.ndarray, board: jnp.ndarray
):
    from_, to = move // 81, move % 81
    piece = board[from_]
    ok = _is_pseudo_legal_move(from_, to, is_promotion, board)
    # actually move
    board = board.at[from_].set(EMPTY).at[to].set(piece)
    # suicide move （王手放置、自殺手）
    is_illegal = _is_checked(board)
    return ok & ~is_illegal


def _is_pseudo_legal_drop(
    hand: jnp.ndarray, piece: jnp.ndarray, to: jnp.ndarray, board: jnp.ndarray
):
    """自殺手を無視した合法手"""
    # destination is not empty
    is_illegal = board[to] != EMPTY
    # don't have the piece
    is_illegal |= hand[0, piece] <= 0
    # double pawn
    is_illegal |= (piece == PAWN) & (
        (board == PAWN).reshape(9, 9).sum(axis=1) > 0
    )[to // 9]
    # get stuck
    is_illegal |= ((piece == PAWN) | (piece == LANCE)) & (to % 9 == 0)
    is_illegal |= (piece == KNIGHT) & (to % 9 < 2)
    return ~is_illegal


def _is_pseudo_legal_move(
    from_: jnp.ndarray,
    to: jnp.ndarray,
    is_promotion: jnp.ndarray,
    board: jnp.ndarray,
):
    """自殺手を無視した合法手"""
    # source is not my piece
    piece = board[from_]
    is_illegal = ~((PAWN <= piece) & (piece < OPP_PAWN))
    # destination is my piece
    is_illegal |= (PAWN <= board[to]) & (board[to] < OPP_PAWN)
    # piece cannot move like that
    is_illegal |= ~CAN_MOVE[piece, from_, to]
    # there is an obstacle between from_ and to
    i = _major_piece_ix(piece)
    between_ix = BETWEEN_IX[i, from_, to, :]
    is_illegal |= (i >= 0) & (
        (between_ix >= 0) & (board[between_ix] != EMPTY)
    ).any()
    # promotion
    is_illegal |= is_promotion & (GOLD <= piece) & (piece <= DRAGON)  # 成れない駒
    is_illegal |= is_promotion & (from_ % 9 >= 3) & (to % 9 >= 3)  # 相手陣地と関係がない
    is_illegal |= (
        ~is_promotion & ((piece == PAWN) | (piece == LANCE)) & (to % 9 == 0)
    )  # 必ず成る
    is_illegal |= (~is_promotion) & (piece == KNIGHT) & (to % 9 < 2)  # 必ず成る
    return ~is_illegal


def _checking_places(board):
    king_pos = jnp.nonzero(board == KING, size=1)[0][0]
    flipped_king_pos = 80 - king_pos
    flipped_board = jax.vmap(_flip_piece)(board)[::-1]

    @jax.vmap
    def can_capture_king(from_):
        return (from_ >= 0) & jax.vmap(
            partial(
                _is_pseudo_legal_move,
                from_=from_,
                to=flipped_king_pos,
                board=flipped_board,
            )
        )(is_promotion=jnp.bool_([False, True])).any()

    from_ = CAN_MOVE_ANY[flipped_king_pos]
    is_checking = can_capture_king(from_)
    checking_ix = jnp.nonzero(is_checking, size=2, fill_value=-1)[0]
    return jnp.where(checking_ix != -1, 80 - from_[checking_ix], -1)


def _is_checked(board):
    return (_checking_places(board) != -1).any()


def _flip_piece(piece):
    return jax.lax.select(piece >= 0, (piece + 14) % 28, piece)


def _rotate(board: jnp.ndarray) -> jnp.ndarray:
    return jnp.rot90(board.reshape(9, 9), k=3)


def _flip(state):
    empty_mask = state.piece_board == EMPTY
    pb = (state.piece_board + 14) % 28
    pb = jnp.where(empty_mask, EMPTY, pb)
    pb = pb[::-1]
    return state.replace(  # type: ignore
        piece_board=pb,
        hand=state.hand[jnp.int8((1, 0))],
    )


def _major_piece_ix(piece):
    ixs = (
        (-jnp.ones(28, dtype=jnp.int8))
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
    return jax.lax.select(piece >= 0, ixs[piece], jnp.int8(-1))


def _observe(state: State, player_id: jnp.ndarray) -> jnp.ndarray:
    state, flip_state = jax.lax.cond(
        state.current_player == player_id,
        lambda: (state, _flip(state)),
        lambda: (_flip(state), state),
    )

    def pieces(state):
        # 駒の場所
        my_pieces = jnp.arange(OPP_PAWN)
        my_piece_feat = jax.vmap(lambda p: state.piece_board == p)(my_pieces)
        return my_piece_feat

    def effect_all(state):
        def effect(from_, to):
            piece = state.piece_board[from_]
            can_move = CAN_MOVE[piece, from_, to]
            major_piece_ix = _major_piece_ix(piece)
            between_ix = BETWEEN_IX[major_piece_ix, from_, to, :]
            has_obstacles = jax.lax.select(
                major_piece_ix >= 0,
                (
                    (between_ix >= 0)
                    & (state.piece_board[between_ix] != EMPTY)
                ).any(),
                FALSE,
            )
            return can_move & ~has_obstacles

        effects = jax.vmap(jax.vmap(effect, (None, 0)), (0, None))(
            ALL_SQ, ALL_SQ
        )
        mine = (PAWN <= state.piece_board) & (state.piece_board < OPP_PAWN)
        return jnp.where(mine.reshape(81, 1), effects, FALSE)

    def piece_and_effect(state):
        my_pieces = jnp.arange(OPP_PAWN)
        my_effect = effect_all(state)

        @jax.vmap
        def filter_effect(p):
            mask = state.piece_board == p
            return jnp.where(mask.reshape(81, 1), my_effect, FALSE).any(axis=0)

        my_effect_feat = filter_effect(my_pieces)
        my_effect_sum = my_effect.sum(axis=0)

        @jax.vmap
        def effect_sum(n) -> jnp.ndarray:
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
    checked = jnp.tile(_is_checked(state.piece_board), reps=(1, 9, 9))
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
    return feat
