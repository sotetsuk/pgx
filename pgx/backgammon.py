from dataclasses import dataclass
from random import randint
from typing import List

import numpy as np


@dataclass
class BackgammonState:

    # 各point(24) bar(2) off(2)にあるcheckerの数 負の値は白, 正の値は黒
    board: np.ndarray = np.zeros(28, dtype=np.int8)

    # ゲームの報酬. doublingが起こると変わる.
    game_reward: np.ndarray = np.zeros(0, dtype=np.int8)

    # doubligをしたかどうか.
    has_doubled: np.ndarray = np.zeros(2, dtype=np.int8)

    # bear offできるかどうか.
    is_bearable: np.ndarray = np.zeros(2, dtype=np.int8)

    # サイコロの出目
    dice: np.ndarray = np.zeros(2, dtype=np.int8)

    # 白なら-1, 黒なら1
    turn: np.ndarray = np.zeros(0, dtype=np.int8)

    # 合法手
    legal_action_musk: np.ndarray = np.zeros(
        (2, 4 * (6 * 26 + 6)), dtype=np.int8
    )


def init() -> BackgammonState:
    board: np.ndarray = _make_init_board()
    game_reward: np.ndarray = np.array([1])
    has_doubled: np.ndarray = np.zeros(2, dtype=np.int8)
    is_bearable: np.ndarray = np.zeros(2, dtype=np.int8)
    dice: np.ndarray = _roll_init_dice()
    turn: np.ndarray = _init_turn(dice)
    legal_action_musk: np.ndarray = np.zeros(
        (2, 4 * (6 * 26 + 6)), dtype=np.int8
    )
    state = BackgammonState(
        board=board,
        game_reward=game_reward,
        has_doubled=has_doubled,
        is_bearable=is_bearable,
        dice=dice,
        turn=turn,
        legal_action_musk=legal_action_musk,
    )
    return state


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


def _roll_init_dice() -> np.ndarray:
    roll = randint(1, 6), randint(1, 6)
    # 違う目が出るまで振り続ける.
    while roll[0] == roll[1]:
        roll = randint(1, 6), randint(1, 6)
    return np.array(roll, dtype=np.int8)


def _init_turn(dice: np.ndarray) -> np.ndarray:
    diff = dice[1] - dice[0]
    turn = np.sign(diff)  # diff > 0 : turn=1, diff < 0: turn=-1
    return np.array([turn], dtype=np.int8)


def _home_board(turn: int) -> np.ndarray:
    """
    白: [18~23], 黒: [0~5]
    """
    fin_idx: int = 6 * np.clip(turn, a_min=0, a_max=1) + np.abs(
        24 * np.clip(turn, a_min=-1, a_max=0)
    )
    return np.array([i for i in range(fin_idx - 6, fin_idx)], dtype=np.int8)


def _off(board: np.ndarray, turn: np.ndarray) -> np.ndarray:
    t: int = turn[0]
    off: int = board[26 + np.clip(t, a_min=0, a_max=1)]
    return np.array([off], dtype=np.int8)


def _bar(board: np.ndarray, turn: np.ndarray) -> np.ndarray:
    t: int = turn[0]
    bar: int = board[24 + np.clip(t, a_min=0, a_max=1)]
    return np.array([bar], dtype=np.int8)


def _can_bear_off(board: np.ndarray, turn: np.ndarray) -> bool:
    """
    全てのcheckerがhome boardにあれば bear offできる.
    """
    t: int = turn[0]
    home_board: np.ndarray = _home_board(t)
    on_home_boad: int = np.array(
        [board[i] * t for i in home_board if board[i] * t >= 0]
    ).sum()
    print(on_home_boad)
    off: int = _off(board, turn)[0] * t
    print(off)
    return (15 - off) == on_home_boad


def _is_open(board: np.ndarray, turn: np.ndarray, point: int) -> bool:
    """
    手番のplayerにとって, pointが空いてるかを判定する.
    pointにある相手のcheckerの数が1以下なら自分のcheckerをそのpointにおける.
    """
    t: int = turn[0]
    checkers: int = board[point]
    return t * checkers >= -1  # 黒と白のcheckerは異符号
