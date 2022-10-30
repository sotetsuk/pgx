from dataclasses import dataclass
from random import randint
from typing import Tuple

import numpy as np


@dataclass
class BackgammonState:

    # 各point(24) bar(2) off(2)にあるcheckerの数 負の値は白, 正の値は黒
    board: np.ndarray = np.zeros(28, dtype=np.int8)

    # サイコロの出目 0~5: 1~6
    dice: np.ndarray = np.zeros(2, dtype=np.int8)

    # プレイできるサイコロの目
    playable_dice: np.ndarray = np.zeros(4, dtype=np.int8)

    # プレイしたサイコロの目の数
    played_dice_num: np.ndarray = np.zeros(1, dtype=np.int8)

    # 白なら-1, 黒なら1
    turn: np.ndarray = np.zeros(0, dtype=np.int8)
    """
    合法手
    micro action = 6*src+die
    """
    legal_micro_action_mask: np.ndarray = np.zeros(6 * 26 + 6, dtype=np.int8)


def init() -> BackgammonState:
    board: np.ndarray = _make_init_board()
    dice: np.ndarray = _roll_init_dice()
    playable_dice: np.ndarray = _set_playable_dice(dice)
    played_dice_num: np.ndarray = np.zeros(1, np.int8)
    turn: np.ndarray = _init_turn(dice)
    legal_micro_action_mask: np.ndarray = np.zeros(6 * 26 + 6, dtype=np.int8)
    state = BackgammonState(
        board=board,
        dice=dice,
        playable_dice=playable_dice,
        played_dice_num=played_dice_num,
        turn=turn,
        legal_micro_action_mask=legal_micro_action_mask,
    )
    return state


def step(
    state: BackgammonState, micro_action: int
) -> Tuple[BackgammonState, int, bool]:
    state.board = _micro_move(state.board, state.turn, micro_action)
    state.played_dice_num += 1
    state.playable_dice = _update_playable_dice(
        state.playable_dice, state.played_dice_num, state.dice, micro_action
    )
    state.legal_micro_action_mask = _legal_micro_action_mask(
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


def _make_init_board() -> np.ndarray:
    board: np.ndarray = np.zeros(28, dtype=np.int8)
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
        state.playable_dice.sum() == -4
        or state.legal_micro_action_mask.sum() == 0
    )  # play可能なサイコロ数が0の場合ないしlegal_actionがない場合交代


def _change_turn(state: BackgammonState) -> BackgammonState:
    state.turn = -1 * state.turn  # turnを変える
    state.dice = _roll_dice()  # diceを振る
    state.playable_dice = _set_playable_dice(state.dice)  # play可能なサイコロを初期化
    state.played_dice_num = np.zeros(1, np.int8)
    state.legal_micro_action_mask = _legal_micro_action_mask(
        state.board, state.turn, state.dice
    )
    return state


def _roll_init_dice() -> np.ndarray:
    roll = randint(0, 5), randint(0, 5)
    # 違う目が出るまで振り続ける.
    while roll[0] == roll[1]:
        roll = randint(0, 5), randint(0, 5)
    return np.array(roll, dtype=np.int8)


def _roll_dice() -> np.ndarray:
    roll = randint(0, 5), randint(0, 5)
    return np.array(roll, dtype=np.int8)


def _init_turn(dice: np.ndarray) -> np.ndarray:
    """
    ゲーム開始時のターン決め.
    サイコロの目が大きい方が手番.
    """
    diff = dice[1] - dice[0]
    turn = np.sign(diff)  # diff > 0 : turn=1, diff < 0: turn=-1
    return np.array([turn], dtype=np.int8)


def _set_playable_dice(dice: np.ndarray) -> np.ndarray:
    """
    -1でemptyを表す.
    """
    if dice[0] == dice[1]:
        return np.array([dice[0] * 4], dtype=np.int8)
    else:
        return np.array([dice[0], dice[1], -1, -1], dtype=np.int8)


def _update_playable_dice(
    playable_dice: np.ndarray,
    played_dice_num: np.ndarray,
    dice: np.ndarray,
    micro_action: int,
) -> np.ndarray:
    _n = played_dice_num[0]
    die = micro_action % 6
    if dice[0] == dice[1]:
        playable_dice[3 - _n] = -1  # プレイしたサイコロの目を-1にする.
    else:
        playable_dice[playable_dice == die] = -1  # プレイしたサイコロの目を-1にする.
    return playable_dice


def _home_board(turn: np.ndarray) -> np.ndarray:
    """
    白: [18~23], 黒: [0~5]
    """
    t = turn[0]
    fin_idx: int = 6 * np.clip(t, a_min=0, a_max=1) + np.abs(
        24 * np.clip(t, a_min=-1, a_max=0)
    )
    return np.array([i for i in range(fin_idx - 6, fin_idx)], dtype=np.int8)


def _off_idx(turn: np.ndarray) -> int:
    """
    白: 26, 黒: 27
    """
    t: int = turn[0]
    return 26 + np.clip(t, a_min=0, a_max=1)


def _bar_idx(turn: np.ndarray) -> int:
    """
    白: 24, 黒 25
    """
    t: int = turn[0]
    return 24 + np.clip(t, a_min=0, a_max=1)


def _rear_distance(board: np.ndarray, turn: np.ndarray) -> int:
    """
    board上にあるcheckerについて, goal地点とcheckerの距離の最大値
    """
    t: int = turn[0]
    b = board[:24]
    exists: np.ndarray = np.where((b * t > 0))[0]
    if turn == 1:
        return int(np.max(exists)) + 1
    else:
        return 24 - int(np.min(exists))


def _is_all_on_homeboad(board: np.ndarray, turn: np.ndarray) -> bool:
    """
    全てのcheckerがhome boardにあれば, bear offできる.
    """
    t: int = turn[0]
    home_board: np.ndarray = _home_board(turn)
    on_home_boad: int = np.array(
        [board[i] * t for i in home_board if board[i] * t >= 0], dtype=np.int8
    ).sum()
    off: int = board[_off_idx(turn)] * t
    return (15 - off) == on_home_boad


def _is_open(board: np.ndarray, turn: np.ndarray, point: int) -> bool:
    """
    手番のplayerにとって, pointが空いてるかを判定する.
    pointにある相手のcheckerの数が1以下なら自分のcheckerをそのpointにおける.
    """
    t: int = turn[0]
    checkers: int = board[point]
    return t * checkers >= -1  # 黒と白のcheckerは異符号


def _exists(board: np.ndarray, turn: np.ndarray, point: int) -> bool:
    """
    指定pointに手番のchckerが存在するか.
    """
    t: int = turn[0]
    checkers: int = board[point]
    return t * checkers >= 1


def _calc_src(src: int, turn: np.ndarray) -> int:
    """
    boardのindexに合わせる.
    """
    if src == 1:  # srcがbarの時
        return _bar_idx(turn)
    else:  # point to point
        return src - 2


def _calc_tgt(src: int, turn: np.ndarray, die) -> int:
    """
    boardのindexに合わせる.
    """
    t = turn[0]
    if src >= 24:  # srcがbarの時
        return np.clip(24 * t, a_min=-1, a_max=24) + die * -turn[0]
    elif src + die * -turn[0] >= 24 or src + die * -turn[0] <= -1:  # 行き先がoffの時
        return _off_idx(turn)
    else:  # point to point
        return src + die * -turn[0]


def _decompose_micro_action(micro_action: int, turn: np.ndarray) -> Tuple:
    """
    micro_action(int)をsource, die, tagetに分解する.
    """
    src = _calc_src(micro_action // 6, turn)
    die = micro_action % 6 + 1  # 0~5 -> 1~6
    tgt = _calc_tgt(src, turn, die)
    return src, die, tgt


def _is_no_op(micro_action):
    """
    no operationかどうか判定する.
    """
    return micro_action // 6 == 0


def _is_micro_action_legal(board: np.ndarray, turn, micro_action: int) -> bool:
    """
    micro actionの合法判定
    micro_action = src * 6 + die
    src = [no op., from bar, 0, .., 23]
    """
    src, die, tgt = _decompose_micro_action(micro_action, turn)
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


def _micro_move(
    board: np.ndarray, turn: np.ndarray, micro_action: int
) -> np.ndarray:
    t = turn[0]
    src, _, tgt = _decompose_micro_action(micro_action, turn)
    board[_bar_idx(-1 * turn)] = board[_bar_idx(-1 * turn)] + (
        (board[tgt] == -t) * -t
    )  # targetに相手のcheckerが一枚だけある時, それを相手のbarに移動

    board[src] = board[src] - t
    board[tgt] = (
        board[tgt] + t + (board[tgt] == -t) * t
    )  # hitした際は符号が変わるので余分に+1
    return board


def _is_all_off(board: np.ndarray, turn: np.ndarray) -> bool:
    """
    手番のプレイヤーのチェッカーが全てoffにあれば勝利となる.
    """
    return board[_off_idx(turn)] * turn[0] == 15


def _calc_win_score(board: np.ndarray, turn: np.ndarray) -> int:
    if _is_backgammon(board, turn):
        return 3
    if _is_gammon(board, turn):
        return 2
    else:
        return 1


def _is_gammon(board: np.ndarray, turn: np.ndarray) -> bool:
    return board[_off_idx(-1 * turn)] == 0


def _is_backgammon(board: np.ndarray, turn: np.ndarray) -> bool:
    return (
        board[_off_idx(-1 * turn)] == 0
        and board[_home_board(-1 * turn)].sum() != 0
    )


def _legal_micro_action_mask(
    board: np.ndarray, turn: np.ndarray, dice: np.ndarray
) -> np.ndarray:
    legal_action_mask = np.zeros(26 * 6 + 6, dtype=np.int8)
    for die in dice:
        if die != -1:
            legal_action_mask = (
                legal_action_mask
                | _legal_micro_action_mask_for_single_die(board, turn, die)
            )  # play可能なサイコロごとのlegal micro actionの論理和
    return legal_action_mask


def _legal_micro_action_mask_for_single_die(
    board: np.ndarray, turn: np.ndarray, die: int
) -> np.ndarray:
    """
    一つのサイコロの目に対するlegal action
    """
    legal_action_mask = np.zeros(26 * 6 + 6, dtype=np.int8)
    for i in range(26):
        micro_action = i * 6 + die
        if _is_micro_action_legal(board, turn, micro_action):
            legal_action_mask[micro_action] = 1
    return legal_action_mask
