from dataclasses import dataclass
from random import randint
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np


@dataclass
class BackgammonState:

    # 各point(24) bar(2) off(2)にあるcheckerの数 負の値は白, 正の値は黒
    board: jnp.ndarray = jnp.zeros(28, dtype=jnp.int8)

    # サイコロの出目 0~5: 1~6
    dice: jnp.ndarray = jnp.zeros(2, dtype=jnp.int8)

    # プレイできるサイコロの目
    playable_dice: jnp.ndarray = jnp.zeros(4, dtype=jnp.int8)

    # プレイしたサイコロの目の数
    played_dice_num: jnp.int8 = jnp.int8(0)

    # 白なら-1, 黒なら1
    turn: jnp.int8 = jnp.int8(1)
    """
    合法手
    micro action = 6*src+die
    """
    legal_action_mask: jnp.ndarray = jnp.zeros(6 * 26 + 6, dtype=jnp.int8)


def init() -> BackgammonState:
    board: jnp.ndarray = _make_init_board()
    dice: jnp.ndarray = _roll_init_dice()
    playable_dice: jnp.ndarray = _set_playable_dice(dice)
    played_dice_num: jnp.int8 = jnp.int8(0)
    turn: jnp.int8 = _init_turn(dice)
    legal_action_mask: jnp.ndarray = jnp.zeros(6 * 26 + 6, dtype=jnp.int8)
    state = BackgammonState(
        board=board,
        dice=dice,
        playable_dice=playable_dice,
        played_dice_num=played_dice_num,
        turn=turn,
        legal_action_mask=legal_action_mask,
    )
    return state


def step(
    state: BackgammonState, action: int
) -> Tuple[BackgammonState, int, bool]:
    state.board = _move(state.board, state.turn, action)
    state.played_dice_num += jnp.int8(1)
    state.playable_dice = _update_playable_dice(
        state.playable_dice, state.played_dice_num, state.dice, action
    )
    state.legal_action_mask = _legal_action_mask(
        state.board, state.turn, state.playable_dice
    )  # legal micro actionを更新
    if _is_all_off(state.board, state.turn):  # 全てのcheckerがoffにあるとき手番のプレイヤーの勝利
        reward = _calc_win_score(
            state.board, state.turn
        )  # 相手のcheckerの進出具合によって受け取る報酬が変わる.
        return state, reward, True
    else:
        if _is_turn_end(state):
            state = _change_turn(state)
            if _is_turn_end(state):  # danceの場合
                state = _change_turn(state)
        return state, 0, False


def _make_init_board() -> jnp.ndarray:
    board: jnp.ndarray = jnp.zeros(28, dtype=jnp.int8)
    board[0] = 2
    board[5] = -5
    board[7] = -3
    board[11] = 5
    board[12] = -5
    board[16] = 3
    board[18] = 5
    board[23] = -2
    return board


def _is_turn_end(state: BackgammonState) -> bool:
    return (
        state.playable_dice.sum() == -4 or state.legal_action_mask.sum() == 0
    )  # play可能なサイコロ数が0の場合ないしlegal_actionがない場合交代


def _change_turn(state: BackgammonState) -> BackgammonState:
    state.turn = -1 * state.turn  # turnを変える
    state.dice = _roll_dice()  # diceを振る
    state.playable_dice = _set_playable_dice(state.dice)  # play可能なサイコロを初期化
    state.played_dice_num = jnp.int8(0)
    state.legal_action_mask = _legal_action_mask(
        state.board, state.turn, state.dice
    )
    return state


def _roll_init_dice() -> jnp.ndarray:
    roll = randint(0, 5), randint(0, 5)
    # 違う目が出るまで振り続ける.
    while roll[0] == roll[1]:
        roll = randint(0, 5), randint(0, 5)
    return jnp.array(roll, dtype=np.int8)


def _roll_dice() -> jnp.ndarray:
    roll = randint(0, 5), randint(0, 5)
    return jnp.array(roll, dtype=np.int8)


def _init_turn(dice: jnp.ndarray) -> jnp.int8:
    """
    ゲーム開始時のターン決め.
    サイコロの目が大きい方が手番.
    """
    diff = dice[1] - dice[0]
    turn = np.sign(diff)  # diff > 0 : turn=1, diff < 0: turn=-1
    return jnp.int8(turn)


def _set_playable_dice(dice: jnp.ndarray) -> jnp.ndarray:
    """
    -1でemptyを表す.
    """
    if dice[0] == dice[1]:
        return jnp.array([dice[0] * 4], dtype=np.int8)
    else:
        return jnp.array([dice[0], dice[1], -1, -1], dtype=np.int8)


def _update_playable_dice(
    playable_dice: jnp.ndarray,
    played_dice_num: jnp.int8,
    dice: jnp.ndarray,
    action: int,
) -> jnp.ndarray:
    _n = played_dice_num
    die = action % 6
    if dice[0] == dice[1]:
        playable_dice[3 - _n] = -1  # プレイしたサイコロの目を-1にする.
    else:
        playable_dice[playable_dice == die] = -1  # プレイしたサイコロの目を-1にする.
    return playable_dice


def _home_board(turn: jnp.int8) -> jnp.ndarray:
    """
    白: [18~23], 黒: [0~5]
    """
    fin_idx: int = 6 * jnp.clip(turn, a_min=0, a_max=1) + np.abs(
        24 * jnp.clip(turn, a_min=-1, a_max=0)
    )
    return np.arange(fin_idx - 6, fin_idx, dtype=np.int8)


def _off_idx(turn: jnp.int8) -> int:
    """
    白: 26, 黒: 27
    """
    return int(26 + jnp.clip(turn, a_min=0, a_max=1))


def _bar_idx(turn: jnp.int8) -> int:
    """
    白: 24, 黒 25
    """
    return int(24 + jnp.clip(turn, a_min=0, a_max=1))


def _rear_distance(board: jnp.ndarray, turn: jnp.int8) -> int:
    """
    board上にあるcheckerについて, goal地点とcheckerの距離の最大値
    """
    b = board[:24]
    exists: jnp.ndarray = jnp.where((b * turn > 0))[0]
    if turn == 1:
        return int(np.max(exists)) + 1
    else:
        return 24 - int(np.min(exists))


def _is_all_on_homeboad(board: jnp.ndarray, turn: jnp.int8) -> bool:
    """
    全てのcheckerがhome boardにあれば, bear offできる.
    """
    home_board: jnp.ndarray = _home_board(turn)
    on_home_board: int = jnp.clip(
        -1 * board[home_board], a_min=0, a_max=15
    ).sum()
    off: int = board[_off_idx(turn)] * turn
    return (15 - off) == on_home_board


def _is_open(board: jnp.ndarray, turn: jnp.int8, point: int) -> bool:
    """
    手番のplayerにとって, pointが空いてるかを判定する.
    pointにある相手のcheckerの数が1以下なら自分のcheckerをそのpointにおける.
    """
    checkers: int = board[point]
    return bool(turn * checkers >= -1)  # 黒と白のcheckerは異符号


def _exists(board: jnp.ndarray, turn: jnp.int8, point: int) -> bool:
    """
    指定pointに手番のchckerが存在するか.
    """
    checkers: int = board[point]
    return bool(turn * checkers >= 1)


def _calc_src(src: int, turn: jnp.int8) -> int:
    """
    boardのindexに合わせる.
    """
    if src == 1:  # srcがbarの時
        return _bar_idx(turn)
    else:  # point to point
        return src - 2


def _calc_tgt(src: int, turn: jnp.int8, die) -> int:
    """
    boardのindexに合わせる.
    """
    if src >= 24:  # srcがbarの時
        return jnp.clip(24 * turn, a_min=-1, a_max=24) + die * -turn
    elif src + die * -turn >= 24 or src + die * -turn <= -1:  # 行き先がoffの時
        return _off_idx(turn)
    else:  # point to point
        return src + die * -turn


def _decompose_action(action: int, turn: jnp.int8) -> Tuple:
    """
    action(int)をsource, die, tagetに分解する.
    """
    src = _calc_src(action // 6, turn)
    die = action % 6 + 1  # 0~5 -> 1~6
    tgt = _calc_tgt(src, turn, die)
    return src, die, tgt


def _is_no_op(action):
    """
    no operationかどうか判定する.
    """
    return action // 6 == 0


def _is_action_legal(board: jnp.ndarray, turn, action: int) -> bool:
    """
    micro actionの合法判定
    action = src * 6 + die
    src = [no op., from bar, 0, .., 23]
    """
    src, die, tgt = _decompose_action(action, turn)
    if src >= 24:  # barからpointへの移動
        return (_exists(board, turn, src)) and (_is_open(board, turn, tgt))
    elif src < 0:
        return False
    elif 0 <= tgt <= 23:  # pointからpointへの移動
        return (
            (_exists(board, turn, src))
            and (_is_open(board, turn, tgt))
            and (board[_bar_idx(turn)] == 0)
        )
    else:  # bear off
        return (
            _exists(board, turn, src)
            and _is_all_on_homeboad(board, turn)
            and (_rear_distance(board, turn) <= die)
        )


def _move(board: jnp.ndarray, turn: jnp.int8, action: int) -> jnp.ndarray:
    """
    micro actionに基づく状態更新
    """
    src, _, tgt = _decompose_action(action, turn)
    board[_bar_idx(-1 * turn)] = board[_bar_idx(-1 * turn)] + (
        (board[tgt] == -turn) * -turn
    )  # targetに相手のcheckerが一枚だけある時, それを相手のbarに移動

    board[src] = board[src] - turn
    board[tgt] = (
        board[tgt] + turn + (board[tgt] == -turn) * turn
    )  # hitした際は符号が変わるので余分に+1
    return board


def _is_all_off(board: jnp.ndarray, turn: jnp.int8) -> bool:
    """
    手番のプレイヤーのチェッカーが全てoffにあれば勝利となる.
    """
    return board[_off_idx(turn)] * turn == 15


def _calc_win_score(board: jnp.ndarray, turn: jnp.int8) -> int:
    if _is_backgammon(board, turn):
        return 3
    if _is_gammon(board, turn):
        return 2
    else:
        return 1


def _is_gammon(board: jnp.ndarray, turn: jnp.int8) -> bool:
    """
    相手のoffに一つもcheckerがなければgammon勝ち
    """
    return board[_off_idx(-1 * turn)] == 0


def _is_backgammon(board: jnp.ndarray, turn: jnp.int8) -> bool:
    """
    相手のoffに一つもcheckerがない && 相手のcheckerが一つでも自分のインナーに残っている
    => backgammon勝ち
    """
    return (
        board[_off_idx(-1 * turn)] == 0
        and board[_home_board(-1 * turn)].sum() != 0
    )


def _legal_action_mask(
    board: jnp.ndarray, turn: jnp.int8, dice: jnp.ndarray
) -> jnp.ndarray:
    legal_action_mask = jnp.zeros(26 * 6 + 6, dtype=np.int8)
    for die in dice:
        legal_action_mask = (
            legal_action_mask
            | _legal_action_mask_for_single_die(board, turn, die)
        )  # play可能なサイコロごとのlegal micro actionの論理和
    return legal_action_mask


def _legal_action_mask_for_single_die(
    board: jnp.ndarray, turn: jnp.int8, die: int
) -> jnp.ndarray:
    """
    一つのサイコロの目に対するlegal micro action
    """
    legal_action_mask = jnp.zeros(26 * 6 + 6, dtype=np.int8)
    if die == -1:
        return legal_action_mask
    for i in range(26):
        action = i * 6 + die
        if _is_action_legal(board, turn, action):
            legal_action_mask[action] = 1
    return legal_action_mask
