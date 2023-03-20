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

TRUE = jnp.bool_(True)
FALSE = jnp.bool_(False)


@dataclass
class State(core.State):
    current_player: jnp.ndarray = jnp.int8(0)
    observation: jnp.ndarray = jnp.zeros(34, dtype=jnp.int8)
    reward: jnp.ndarray = jnp.float32([0.0, 0.0])
    terminated: jnp.ndarray = FALSE
    truncated: jnp.ndarray = FALSE
    # micro action = 6 * src + die
    legal_action_mask: jnp.ndarray = jnp.zeros(6 * 26 + 6, dtype=jnp.bool_)
    _rng_key: jax.random.KeyArray = jax.random.PRNGKey(0)
    _step_count: jnp.ndarray = jnp.int32(0)
    # --- Backgammon specific ---
    # 各point(24) bar(2) off(2)にあるcheckerの数. 黒+, 白-
    board: jnp.ndarray = jnp.zeros(28, dtype=jnp.int8)
    # サイコロを振るたびにrngをsplitして更新する.
    rng: jax.random.KeyArray = jnp.zeros(2, dtype=jnp.uint16)
    # サイコロの出目 0~5: 1~6
    dice: jnp.ndarray = jnp.zeros(2, dtype=jnp.int16)
    # プレイできるサイコロの目
    playable_dice: jnp.ndarray = jnp.zeros(4, dtype=jnp.int16)
    # プレイしたサイコロの目の数
    played_dice_num: jnp.ndarray = jnp.int16(0)
    # 黒0, 白1
    turn: jnp.ndarray = jnp.int8(1)


class Backgammon(core.Env):
    def __init__(
        self,
    ):
        super().__init__()

    def _init(self, key: jax.random.KeyArray) -> State:
        return _init(key)

    def _step(self, state: core.State, action: jnp.ndarray) -> State:
        assert isinstance(state, State)
        return _step(state, action)

    def _observe(
        self, state: core.State, player_id: jnp.ndarray
    ) -> jnp.ndarray:
        assert isinstance(state, State)
        return _observe(state, player_id)

    @property
    def name(self) -> str:
        return "Backgammon"

    @property
    def version(self) -> str:
        return "beta"

    @property
    def num_players(self) -> int:
        return 2

    @property
    def _illegal_action_penalty(self) -> float:
        return -3.0


def _init(rng: jax.random.KeyArray) -> State:
    rng1, rng2, rng3 = jax.random.split(rng, num=3)
    current_player: jnp.ndarray = jax.random.bernoulli(rng1).astype(jnp.int8)
    board: jnp.ndarray = _make_init_board()  # 初期配置は対象なので, turnに関係
    terminated: jnp.ndarray = FALSE
    dice: jnp.ndarray = _roll_init_dice(rng2)
    playable_dice: jnp.ndarray = _set_playable_dice(dice)
    played_dice_num: jnp.ndarray = jnp.int16(0)
    turn: jnp.ndarray = _init_turn(dice)
    legal_action_mask: jnp.ndarray = _legal_action_mask(board, playable_dice)
    state = State(  # type: ignore
        current_player=current_player,
        rng=rng3,
        board=board,
        terminated=terminated,
        dice=dice,
        playable_dice=playable_dice,
        played_dice_num=played_dice_num,
        turn=turn,
        legal_action_mask=legal_action_mask,
    )
    return state


def _step(state: State, action: jnp.ndarray) -> State:
    """
    terminated していない場合のstep 関数.
    """
    state = _update_by_action(state, action)
    return jax.lax.cond(
        _is_all_off(state.board),
        lambda: _winning_step(state),
        lambda: _no_winning_step(state, action),
    )


def _observe(state: State, player_id: jnp.ndarray) -> jnp.ndarray:
    """
    手番のplayerに対する観測を返す.
    """
    board: jnp.ndarray = state.board
    zero_one_dice_vec: jnp.ndarray = _to_zero_one_dice_vec(state.playable_dice)
    return jax.lax.cond(
        player_id == state.current_player,
        lambda: jnp.concatenate((board, zero_one_dice_vec), axis=None),  # type: ignore
        lambda: jnp.concatenate(
            (board, jnp.zeros(6, dtype=jnp.int8)), axis=None  # type: ignore
        ),
    )


def _to_zero_one_dice_vec(playable_dice: jnp.ndarray) -> jnp.ndarray:
    """
    playできるサイコロを6次元の0-1ベクトルで返す.
    """
    dice_indices: jnp.ndarray = jnp.array(
        [0, 1, 2, 3], dtype=jnp.int8
    )  # サイコロの数は最大4

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
    勝利者がいる場合のstep.
    """
    win_score = _calc_win_score(state.board)
    winner = state.current_player
    loser = 1 - winner
    reward = jnp.ones_like(state.reward)
    reward = reward.at[winner].set(win_score)
    reward = reward.at[loser].set(-win_score)
    state = state.replace(terminated=TRUE)  # type: ignore
    return state.replace(reward=reward)  # type: ignore


def _no_winning_step(state: State, action: jnp.ndarray) -> State:
    """
    勝利者がいない場合のstep, ターン終了の条件を満たせばターンを変更する.
    """
    return jax.lax.cond(
        (_is_turn_end(state) | (action // 6 == 0)),
        lambda: _change_turn(state),
        lambda: state,
    )


def _update_by_action(state: State, action: jnp.ndarray) -> State:
    """
    行動を受け取って状態をupdate
    """
    is_no_op = action // 6 == 0
    rng = state.rng
    current_player: jnp.ndarray = state.current_player
    terminated: jnp.ndarray = state.terminated
    board: jnp.ndarray = _move(state.board, action)
    played_dice_num: jnp.ndarray = jnp.int16(state.played_dice_num + 1)
    playable_dice: jnp.ndarray = _update_playable_dice(
        state.playable_dice, state.played_dice_num, state.dice, action
    )
    legal_action_mask: jnp.ndarray = _legal_action_mask(board, playable_dice)
    return jax.lax.cond(
        is_no_op,
        lambda: state,
        lambda: state.replace(  # type: ignore
            current_player=current_player,
            rng=rng,
            terminated=terminated,
            board=board,
            turn=state.turn,
            dice=state.dice,
            playable_dice=playable_dice,
            played_dice_num=played_dice_num,
            legal_action_mask=legal_action_mask,
        ),
    )  # no-opの時はupdateしない


def _flip_board(board):
    """
    ターンが変わる際にボードを反転させ, -1をかける. そうすることで常に黒視点で考えることができる.
    """
    _board = board
    board = board.at[:24].set(jnp.flip(_board[:24]))
    board = board.at[24:26].set(jnp.flip(_board[24:26]))
    board = board.at[26:28].set(jnp.flip(_board[26:28]))
    return -1 * board


def _make_init_board() -> jnp.ndarray:
    """
    黒基準で初期化
    """
    board: jnp.ndarray = jnp.array([2, 0, 0, 0, 0, -5, 0, -3, 0, 0, 0, 5, -5, 0, 0, 0, 3, 0, 5, 0, 0, 0, 0, -2, 0, 0, 0, 0], dtype=jnp.int8)  # type: ignore
    return board


def _is_turn_end(state: State) -> bool:
    """
    play可能なサイコロ数が0の場合ないしlegal_actionがない場合交代
    """
    return state.playable_dice.sum() == -4  # type: ignore


def _change_turn(state: State) -> State:
    """
    ターンを変更して新しい状態を返す.
    """
    rng1, rng2 = jax.random.split(state.rng)
    board: jnp.ndarray = _flip_board(state.board)  # boardを反転させて黒視点に変える
    turn: jnp.ndarray = (state.turn + 1) % 2  # turnを変える
    current_player: jnp.ndarray = (state.current_player + 1) % 2
    terminated: jnp.ndarray = state.terminated
    dice: jnp.ndarray = _roll_dice(rng1)  # diceを振る
    playable_dice: jnp.ndarray = _set_playable_dice(dice)  # play可能なサイコロを初期化
    played_dice_num: jnp.ndarray = jnp.int16(0)
    legal_action_mask: jnp.ndarray = _legal_action_mask(board, dice)
    return state.replace(  # type: ignore
        current_player=current_player,
        rng=rng2,
        board=board,
        terminated=terminated,
        turn=turn,
        dice=dice,
        playable_dice=playable_dice,
        played_dice_num=played_dice_num,
        legal_action_mask=legal_action_mask,
    )


def _roll_init_dice(rng: jax.random.KeyArray) -> jnp.ndarray:
    """
    # 違う目が出るまで振り続ける.
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
    ゲーム開始時のターン決め.
    サイコロの目が大きい方が手番.
    """
    diff = dice[1] - dice[0]
    return jnp.int8(diff > 0)


def _set_playable_dice(dice: jnp.ndarray) -> jnp.ndarray:
    """
    -1でemptyを表す.
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
    )  # サイコロの数は最大4

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
    黒: [18~23], 白: [0~5]: 常に黒視点
    """
    return jnp.arange(18, 24, dtype=jnp.int8)  # type: ignore


def _off_idx() -> int:
    """
    黒: 26, 白: 27: 常に黒視点
    """
    return 26  # type: ignore


def _bar_idx() -> int:
    """
    黒: 24, 白 25: 常に黒視点
    """
    return 24  # type: ignore


def _rear_distance(board: jnp.ndarray) -> jnp.ndarray:
    """
    board上にあるcheckerについて, goal地点とcheckerの距離の最大値: 常に黒視点
    """
    b = board[:24]
    exists = jnp.where((b > 0), size=24, fill_value=jnp.nan)[0]  # type: ignore
    return 24 - jnp.min(jnp.nan_to_num(exists, nan=jnp.int16(100)))


def _is_all_on_home_board(board: jnp.ndarray) -> bool:
    """
    全てのcheckerがhome boardにあれば, bear offできる.
    """
    home_board: jnp.ndarray = _home_board()
    on_home_board: int = jnp.clip(board[home_board], a_min=0, a_max=15).sum()
    off: int = board[_off_idx()]  # type: ignore
    return (15 - off) == on_home_board


def _is_open(board: jnp.ndarray, point: int) -> bool:
    """
    手番のplayerにとって, pointが空いてるかを判定する.
    pointにある相手のcheckerの数が1以下なら自分のcheckerをそのpointにおける.
    """
    checkers = board[point]
    return checkers >= -1  # type: ignore


def _exists(board: jnp.ndarray, point: int) -> bool:
    """
    指定pointに手番のchckerが存在するか.
    """
    checkers = board[point]
    return checkers >= 1  # type: ignore


def _calc_src(src: jnp.ndarray) -> int:
    """
    boardのindexに合わせる.
    """
    return (src == 1) * jnp.int16(_bar_idx()) + (src != 1) * jnp.int16(
        src - 2
    )  # type: ignore


def _calc_tgt(src: int, die) -> int:
    """
    boardのindexに合わせる. actionは src*6 + dieの形になっている. targetは黒ならsrcからdie分+白ならdie分-(目的地が逆だから.)
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
    action(int)をsource, die, tagetに分解する.
    """
    src = _calc_src(action // 6)  # 0~25
    die = action % 6 + 1  # 0~5 -> 1~6
    tgt = _calc_tgt(src, die)
    return src, die, tgt


def _is_action_legal(board: jnp.ndarray, action: jnp.ndarray) -> bool:
    """
    micro actionの合法判定
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
    goal までの距離: 常に黒視点
    """
    return 24 - src  # type: ignore


def _is_to_off_legal(board: jnp.ndarray, src: int, tgt: int, die: int):
    """
    board外への移動についての合法判定
    条件は
    1. srcにcheckerがある
    2. 自身のcheckeが全てhomeboardにある.
    3. サイコロの目とgoalへの距離が同じ or srcが最後尾であり, サイコロの目がそれよりも大きい.
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
    tgtがpointの場合の合法手判定
    """
    e = _exists(board, src)
    o = _is_open(board, tgt)
    return ((src >= 24) & e & o) | (
        (src < 24) & e & o & (board[_bar_idx()] == 0)
    )  # type: ignore


def _move(board: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
    """
    micro actionに基づく状態更新: 常に黒視点
    """
    src, _, tgt = _decompose_action(action)
    board = board.at[_bar_idx() + 1].add(
        -1 * (board[tgt] == -1)
    )  # targetに相手のcheckerが一枚だけある時, それを相手のbarに移動
    board = board.at[src].add(-1)
    board = board.at[tgt].add(1 + (board[tgt] == -1))  # hitした際は符号が変わるので余分に+1
    return board


def _is_all_off(board: jnp.ndarray) -> bool:
    """
    手番のプレイヤーのチェッカーが全てoffにあれば勝利となる. 常に黒視点
    """
    return board[_off_idx()] == 15  # type: ignore


def _calc_win_score(board: jnp.ndarray) -> int:
    """
    通常勝ち: 1点
    gammon勝ち: 2点
    backgammon勝ち: 3点
    """
    g = _is_gammon(board)
    return 1 + g + (g & _remains_at_inner(board))


def _is_gammon(board: jnp.ndarray) -> bool:
    """
    相手のoffに一つもcheckerがなければgammon勝ち
    """
    return board[_off_idx() + 1] == 0  # type: ignore


def _remains_at_inner(board: jnp.ndarray) -> bool:
    """
    相手のoffに一つもcheckerがない && 相手のcheckerが一つでも自分のインナーに残っている
    => backgammon勝ち
    """
    return jnp.take(board, _home_board()).sum() != 0  # type: ignore


def _legal_action_mask(board: jnp.ndarray, dice: jnp.ndarray) -> jnp.ndarray:
    no_op_mask = jnp.zeros(26 * 6 + 6, dtype=jnp.bool_).at[0:6].set(TRUE)
    legal_action_mask = jax.vmap(
        partial(_legal_action_mask_for_single_die, board=board)
    )(die=dice).any(
        axis=0
    )  # (26*6 + 6)
    legal_action_exists = ~(legal_action_mask.sum() == 0)
    return (
        legal_action_exists * legal_action_mask
        + ~legal_action_exists * no_op_mask
    )  # legal_actionがなければ, np_op maskを返す


def _legal_action_mask_for_single_die(board: jnp.ndarray, die) -> jnp.ndarray:
    """
    一つのサイコロの目に対するlegal micro action
    """
    return (die == -1) * jnp.zeros(26 * 6 + 6, dtype=jnp.bool_) + (
        die != -1
    ) * _legal_action_mask_for_valid_single_dice(board, die)


def _legal_action_mask_for_valid_single_dice(
    board: jnp.ndarray, die
) -> jnp.ndarray:
    """
    -1以外のサイコロの目に対して合法判定
    """
    src_indices = jnp.arange(
        26, dtype=jnp.int16
    )  # 26パターンのsrcに対してlegal_actionを求める.

    def _is_legal(idx: jnp.ndarray):
        action = idx * 6 + die
        legal_action_mask = jnp.zeros(26 * 6 + 6, dtype=jnp.bool_)
        legal_action_mask = legal_action_mask.at[action].set(
            _is_action_legal(board, action)
        )
        return legal_action_mask

    legal_action_mask = jax.vmap(_is_legal)(src_indices).any(
        axis=0
    )  # (26*6 + 6)
    return legal_action_mask


def _get_abs_board(state: State) -> jnp.ndarray:
    """
    visualization用
    黒ならそのまま, 白なら反転して返す.
    """
    board: jnp.ndarray = state.board
    turn: jnp.ndarray = state.turn
    return jax.lax.cond(turn == 0, lambda: board, lambda: _flip_board(board))
