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

import pgx.core as v1
from pgx._src.struct import dataclass

TRUE = jnp.bool_(True)
FALSE = jnp.bool_(False)


@dataclass
class State(v1.State):
    current_player: jnp.ndarray = jnp.int8(0)
    observation: jnp.ndarray = jnp.zeros(34, dtype=jnp.int8)
    rewards: jnp.ndarray = jnp.float32([0.0, 0.0])
    terminated: jnp.ndarray = FALSE
    truncated: jnp.ndarray = FALSE
    # micro action = 6 * src + die
    legal_action_mask: jnp.ndarray = jnp.zeros(6 * 26, dtype=jnp.bool_)
    _step_count: jnp.ndarray = jnp.int32(0)
    # --- Backgammon specific ---
    # points(24) bar(2) off(2). black+, white-
    _board: jnp.ndarray = jnp.zeros(28, dtype=jnp.int8)
    _dice: jnp.ndarray = jnp.zeros(2, dtype=jnp.int16)  # 0~5: 1~6
    _playable_dice: jnp.ndarray = jnp.zeros(
        4, dtype=jnp.int16
    )  # playable dice -1 for empty
    _played_dice_num: jnp.ndarray = jnp.int16(0)  # the number of dice played
    _turn: jnp.ndarray = jnp.int8(1)  # black: 0 white:1

    @property
    def env_id(self) -> v1.EnvId:
        return "backgammon"


class Backgammon(v1.Env):
    def __init__(self):
        super().__init__()

    def _init(self, key: jax.random.KeyArray) -> State:
        return _init(key)

    def _step(self, state: v1.State, action: jnp.ndarray, key) -> State:
        assert isinstance(state, State)
        return _step(state, action, key)

    def _observe(self, state: v1.State, player_id: jnp.ndarray) -> jnp.ndarray:
        assert isinstance(state, State)
        return _observe(state, player_id)

    @property
    def id(self) -> v1.EnvId:
        return "backgammon"

    @property
    def version(self) -> str:
        return "v1"

    @property
    def num_players(self) -> int:
        return 2

    @property
    def _illegal_action_penalty(self) -> float:
        return -3.0


def _init(rng: jax.random.KeyArray) -> State:
    rng1, rng2, rng3 = jax.random.split(rng, num=3)
    current_player: jnp.ndarray = jax.random.bernoulli(rng1).astype(jnp.int8)
    board: jnp.ndarray = _make_init_board()
    terminated: jnp.ndarray = FALSE
    dice: jnp.ndarray = _roll_init_dice(rng2)
    playable_dice: jnp.ndarray = _set_playable_dice(dice)
    played_dice_num: jnp.ndarray = jnp.int16(0)
    turn: jnp.ndarray = _init_turn(dice)
    legal_action_mask: jnp.ndarray = _legal_action_mask(board, playable_dice)
    state = State(  # type: ignore
        current_player=current_player,
        _board=board,
        terminated=terminated,
        _dice=dice,
        _playable_dice=playable_dice,
        _played_dice_num=played_dice_num,
        _turn=turn,
        legal_action_mask=legal_action_mask,
    )
    return state


def _step(state: State, action: jnp.ndarray, key) -> State:
    """
    Step when not terminated
    """
    state = _update_by_action(state, action)
    return jax.lax.cond(
        _is_all_off(state._board),
        lambda: _winning_step(state),
        lambda: _no_winning_step(state, action, key),
    )


def _observe(state: State, player_id: jnp.ndarray) -> jnp.ndarray:
    """
    Return observation for player_id
    """
    board: jnp.ndarray = state._board
    playable_dice_count_vec: jnp.ndarray = _to_playable_dice_count(
        state._playable_dice
    )  # 6 dim vec which represents the count of playable die.
    return jax.lax.cond(
        player_id == state.current_player,
        lambda: jnp.concatenate((board, playable_dice_count_vec), axis=None),  # type: ignore
        lambda: jnp.concatenate(
            (board, jnp.zeros(6, dtype=jnp.int8)), axis=None  # type: ignore
        ),
    )


def _to_playable_dice_count(playable_dice: jnp.ndarray) -> jnp.ndarray:
    """
    Return 6 dim vec which represents the number of playable die
    Examples
    Playable dice: 2, 3
    Return: [0, 1, 1, 0, 0, 0]

    Playable dice: 4, 4, 4, 4
    Return: [0, 0, 0, 0, 4, 0]
    """
    dice_indices: jnp.ndarray = jnp.array(
        [0, 1, 2, 3], dtype=jnp.int8
    )  # maximum number of playable dice is 4

    def _insert_dice_num(
        idx: jnp.ndarray, playable_dice: jnp.ndarray
    ) -> jnp.ndarray:
        vec: jnp.ndarray = jnp.zeros(6, dtype=jnp.int8)
        return (playable_dice[idx] != -1) * vec.at[playable_dice[idx]].set(
            1
        ) + (playable_dice[idx] == -1) * vec

    return jax.vmap(_insert_dice_num)(
        dice_indices, jnp.tile(playable_dice, (4, 1))
    ).sum(axis=0, dtype=jnp.int8)


def _winning_step(
    state: State,
) -> State:
    """
    Step with winner
    """
    win_score = _calc_win_score(state._board)
    winner = state.current_player
    loser = 1 - winner
    reward = jnp.ones_like(state.rewards)
    reward = reward.at[winner].set(win_score)
    reward = reward.at[loser].set(-win_score)
    state = state.replace(terminated=TRUE)  # type: ignore
    return state.replace(rewards=reward)  # type: ignore


def _no_winning_step(state: State, action: jnp.ndarray, key) -> State:
    """
    Step with no winner. Change turn if turn end condition is satisfied.
    """
    return jax.lax.cond(
        (_is_turn_end(state) | (action // 6 == 0)),
        lambda: _change_turn(state, key),
        lambda: state,
    )


def _update_by_action(state: State, action: jnp.ndarray) -> State:
    """
    Update state by action
    """
    is_no_op = action // 6 == 0
    current_player: jnp.ndarray = state.current_player
    terminated: jnp.ndarray = state.terminated
    board: jnp.ndarray = _move(state._board, action)
    played_dice_num: jnp.ndarray = jnp.int16(state._played_dice_num + 1)
    playable_dice: jnp.ndarray = _update_playable_dice(
        state._playable_dice, state._played_dice_num, state._dice, action
    )
    legal_action_mask: jnp.ndarray = _legal_action_mask(board, playable_dice)
    return jax.lax.cond(
        is_no_op,
        lambda: state,
        lambda: state.replace(  # type: ignore
            current_player=current_player,
            terminated=terminated,
            _board=board,
            _turn=state._turn,
            _dice=state._dice,
            _playable_dice=playable_dice,
            _played_dice_num=played_dice_num,
            legal_action_mask=legal_action_mask,
        ),
    )  # no-opの時はupdateしない


def _flip_board(board):
    """
    Flip a board when turn changes. Multiply -1 to the board so that we can always consider the board from black's perspective.
    """
    _board = board
    board = board.at[:24].set(jnp.flip(_board[:24]))
    board = board.at[24:26].set(jnp.flip(_board[24:26]))
    board = board.at[26:28].set(jnp.flip(_board[26:28]))
    return -1 * board


def _make_init_board() -> jnp.ndarray:
    """
    Initialize the board based on black's perspective.
    """
    board: jnp.ndarray = jnp.array([2, 0, 0, 0, 0, -5, 0, -3, 0, 0, 0, 5, -5, 0, 0, 0, 3, 0, 5, 0, 0, 0, 0, -2, 0, 0, 0, 0], dtype=jnp.int8)  # type: ignore
    return board


def _is_turn_end(state: State) -> bool:
    """
    Turn will end if there is no playable dice or no legal action.
    """
    return state._playable_dice.sum() == -4  # type: ignore


def _change_turn(state: State, key) -> State:
    """
    Change turn and return new state.
    """
    board: jnp.ndarray = _flip_board(state._board)
    turn: jnp.ndarray = (state._turn + 1) % 2
    current_player: jnp.ndarray = (state.current_player + 1) % 2
    terminated: jnp.ndarray = state.terminated
    dice: jnp.ndarray = _roll_dice(key)
    playable_dice: jnp.ndarray = _set_playable_dice(dice)
    played_dice_num: jnp.ndarray = jnp.int16(0)
    legal_action_mask: jnp.ndarray = _legal_action_mask(board, dice)
    return state.replace(  # type: ignore
        current_player=current_player,
        _board=board,
        terminated=terminated,
        _turn=turn,
        _dice=dice,
        _playable_dice=playable_dice,
        _played_dice_num=played_dice_num,
        legal_action_mask=legal_action_mask,
    )


def _roll_init_dice(rng: jax.random.KeyArray) -> jnp.ndarray:
    """
    Roll till the dice are different.
    """

    init_dice_pattern: jnp.ndarray = jnp.array([[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [1, 0], [1, 2], [1, 3], [1, 4], [1, 5], [2, 0], [2, 1], [2, 3], [2, 4], [2, 5], [3, 0], [3, 1], [3, 2], [3, 4], [3, 5], [4, 0], [4, 1], [4, 2], [4, 3], [4, 5], [5, 0], [5, 1], [5, 2], [5, 3], [5, 4]], dtype=jnp.int16)  # type: ignore
    return jax.random.choice(rng, init_dice_pattern)


def _roll_dice(rng: jax.random.KeyArray) -> jnp.ndarray:
    roll: jnp.ndarray = jax.random.randint(
        rng, shape=(1, 2), minval=0, maxval=6, dtype=jnp.int16
    )
    return roll[0]


def _init_turn(dice: jnp.ndarray) -> jnp.ndarray:
    """
    Decide turn at the beginning of the game.
    Begin with those who have bigger dice
    """
    diff = dice[1] - dice[0]
    return jnp.int8(diff > 0)


def _set_playable_dice(dice: jnp.ndarray) -> jnp.ndarray:
    """
    -1 for empty
    """
    return (dice[0] == dice[1]) * jnp.array([dice[0]] * 4, dtype=jnp.int16) + (
        dice[0] != dice[1]
    ) * jnp.array([dice[0], dice[1], -1, -1], dtype=jnp.int16)


def _update_playable_dice(
    playable_dice: jnp.ndarray,
    played_dice_num: jnp.ndarray,
    dice: jnp.ndarray,
    action: jnp.ndarray,
) -> jnp.ndarray:
    _n = played_dice_num
    die_array = jnp.array([action % 6] * 4, dtype=jnp.int16)
    dice_indices: jnp.ndarray = jnp.array(
        [0, 1, 2, 3], dtype=jnp.int16
    )  # maximum number of playable dice is 4

    def _update_for_diff_dice(
        die: jnp.ndarray, idx: jnp.ndarray, playable_dice: jnp.ndarray
    ):
        return (die == playable_dice[idx]) * -1 + (
            die != playable_dice[idx]
        ) * playable_dice[idx]

    return (dice[0] == dice[1]) * playable_dice.at[3 - _n].set(-1) + (
        dice[0] != dice[1]
    ) * jax.vmap(_update_for_diff_dice)(
        die_array, dice_indices, jnp.tile(playable_dice, (4, 1))
    ).astype(
        jnp.int16
    )


def _home_board() -> jnp.ndarray:
    """
    black: [18~23], white: [0~5]: Always black's perspective
    """
    return jnp.arange(18, 24, dtype=jnp.int8)  # type: ignore


def _off_idx() -> int:
    """
    black: 26, white: 27: Always black's perspective
    """
    return 26  # type: ignore


def _bar_idx() -> int:
    """
    black: 24, white 25: Always black's perspective
    """
    return 24  # type: ignore


def _rear_distance(board: jnp.ndarray) -> jnp.ndarray:
    """
    The distance from the farthest checker to the goal: Always black's perspective
    """
    b = board[:24]
    exists = jnp.where((b > 0), size=24, fill_value=jnp.nan)[0]  # type: ignore
    return 24 - jnp.min(jnp.nan_to_num(exists, nan=jnp.int16(100)))


def _is_all_on_home_board(board: jnp.ndarray) -> bool:
    """
    One can bear off if all checkers are on home board.
    """
    home_board: jnp.ndarray = _home_board()
    on_home_board: int = jnp.clip(board[home_board], a_min=0, a_max=15).sum()
    off: int = board[_off_idx()]  # type: ignore
    return (15 - off) == on_home_board


def _is_open(board: jnp.ndarray, point: int) -> bool:
    """
    Check if the point is open for the current player: Always black's perspective
    One can move to the point if there is no more than one opponent's checker.
    """
    checkers = board[point]
    return checkers >= -1  # type: ignore


def _exists(board: jnp.ndarray, point: int) -> bool:
    """
    Check if the point has the current player's checker: Always black's perspective
    """
    checkers = board[point]
    return checkers >= 1  # type: ignore


def _calc_src(src: jnp.ndarray) -> int:
    """
    Translate src to board index.
    """
    return (src == 1) * jnp.int16(_bar_idx()) + (src != 1) * jnp.int16(
        src - 2
    )  # type: ignore


def _calc_tgt(src: int, die) -> int:
    """
    Translate tgt to board index.
    """
    return (src >= 24) * (jnp.int16(die) - 1) + (src < 24) * jnp.int16(
        _from_board(src, die)
    )  # type: ignore


def _from_board(src: int, die: int) -> int:
    _is_to_board = (src + die >= 0) & (src + die <= 23)
    return _is_to_board * jnp.int16(src + die) + ((~_is_to_board)) * jnp.int16(
        _off_idx()
    )  # type: ignore


def _decompose_action(action: jnp.ndarray):
    """
    Decompose action to src, die, tgt.
    action = src*6 + die
    """
    src = _calc_src(action // 6)  # 0~25
    die = action % 6 + 1  # 0~5 -> 1~6
    tgt = _calc_tgt(src, die)
    return src, die, tgt


def _is_action_legal(board: jnp.ndarray, action: jnp.ndarray) -> bool:
    """
    Check if the action is legal.
    action = src * 6 + die
    src = [no op., from bar, 0, .., 23]
    """
    src, die, tgt = _decompose_action(action)
    _is_to_point = (0 <= tgt) & (tgt <= 23) & (src >= 0)
    return _is_to_point & _is_to_point_legal(board, src, tgt) | (
        ~_is_to_point
    ) & _is_to_off_legal(
        board, src, tgt, die
    )  # type: ignore


def _distance_to_goal(src: int) -> int:
    """
    The distance from the src to the goal: Always black's perspective
    """
    return 24 - src  # type: ignore


def _is_to_off_legal(board: jnp.ndarray, src: int, tgt: int, die: int):
    """
    Check if the action is legal when the target is off.
    The conditions are:
    1. src has checkers.
    2. All checkers are on home board.
    3. The distance from the src to the goal is the same as the die or the src is the farthest checker and the die is bigger than the distance.
    """
    r = _rear_distance(board)
    d = _distance_to_goal(src)
    return (
        (src >= 0)
        & _exists(board, src)
        & _is_all_on_home_board(board)
        & ((d == die) | ((r <= die) & (r == d)))
    )  # type: ignore


def _is_to_point_legal(board: jnp.ndarray, src: int, tgt: int) -> bool:
    """
    Check if the action is legal when the target is point.
    """
    e = _exists(board, src)
    o = _is_open(board, tgt)
    return ((src >= 24) & e & o) | (
        (src < 24) & e & o & (board[_bar_idx()] == 0)
    )  # type: ignore


def _move(board: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
    """
    Move checkers based on the action.
    """
    src, _, tgt = _decompose_action(action)
    board = board.at[_bar_idx() + 1].add(
        -1 * (board[tgt] == -1)
    )  # If there is an opponent's checker on the target, hit it
    board = board.at[src].add(-1)
    board = board.at[tgt].add(
        1 + (board[tgt] == -1)
    )  # If hit, the sign changes, so add 1
    return board


def _is_all_off(board: jnp.ndarray) -> bool:
    """
    手番のプレイヤーのチェッカーが全てoffにあれば勝利となる. 常に黒視点
    If all checkers are off, the player wins. Always black's perspective.
    """
    return board[_off_idx()] == 15  # type: ignore


def _calc_win_score(board: jnp.ndarray) -> int:
    """
    Normal win: 1 point
    Gammon win: 2 points
    Backgammon win: 3 points
    """
    g = _is_gammon(board)
    return 1 + g + (g & _remains_at_inner(board))


def _is_gammon(board: jnp.ndarray) -> bool:
    """
    If there is no opponent's checker on off, the player wins gammon.
    """
    return board[_off_idx() + 1] == 0  # type: ignore


def _remains_at_inner(board: jnp.ndarray) -> bool:
    """
    (1) If there is no opponent's checker on off and (2) there is at least one opponent's checker on inner, the player wins backgammon.
    """
    return jnp.take(board, _home_board()).sum() != 0  # type: ignore


def _legal_action_mask(board: jnp.ndarray, dice: jnp.ndarray) -> jnp.ndarray:
    no_op_mask = jnp.zeros(26 * 6, dtype=jnp.bool_).at[0:6].set(TRUE)
    legal_action_mask = jax.vmap(
        partial(_legal_action_mask_for_single_die, board=board)
    )(die=dice).any(
        axis=0
    )  # (26 * 6)
    legal_action_exists = ~(legal_action_mask.sum() == 0)
    return (
        legal_action_exists * legal_action_mask
        + ~legal_action_exists * no_op_mask
    )  # if there is no legal action, no-op is legal


def _legal_action_mask_for_single_die(board: jnp.ndarray, die) -> jnp.ndarray:
    """
    Legal action mask for a single die.
    """
    return (die == -1) * jnp.zeros(26 * 6, dtype=jnp.bool_) + (
        die != -1
    ) * _legal_action_mask_for_valid_single_dice(board, die)


def _legal_action_mask_for_valid_single_dice(
    board: jnp.ndarray, die
) -> jnp.ndarray:
    """
    Legal action mask for a single die when the die is valid.
    """
    src_indices = jnp.arange(
        26, dtype=jnp.int16
    )  # calc legal action for all src indices

    def _is_legal(idx: jnp.ndarray):
        action = idx * 6 + die
        legal_action_mask = jnp.zeros(26 * 6, dtype=jnp.bool_)
        legal_action_mask = legal_action_mask.at[action].set(
            _is_action_legal(board, action)
        )
        return legal_action_mask

    legal_action_mask = jax.vmap(_is_legal)(src_indices).any(
        axis=0
    )  # (26 * 6)
    return legal_action_mask


def _get_abs_board(state: State) -> jnp.ndarray:
    """
    For visualization.
    """
    board: jnp.ndarray = state._board
    turn: jnp.ndarray = state._turn
    return jax.lax.cond(turn == 0, lambda: board, lambda: _flip_board(board))
