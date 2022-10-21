from dataclasses import dataclass
from random import randint

import numpy as np


@dataclass
class BackgammonState:

    # 各point(24) bar(2) off(2)にあるcheckerの数 正の値は白, 負の値は黒
    board: np.ndarray = np.zeros(28, dtype=int)

    # ゲームの報酬. doublingが起こると変わる.
    game_reward: int = 0

    # doubligをしたかどうか.
    has_doubled: np.ndarray = np.zeros(2, dtype=bool)

    # bear offできるかどうか.
    is_bearable: np.ndarray = np.zeros(2, dtype=bool)

    # サイコロの出目
    dice: np.ndarray = np.zeros(2, dtype=int)

    # 白なら0黒なら1
    turn: int = 0

    # 合法手
    legal_actions: np.ndarray = np.zeros((2, 4 * (6 * 26 + 6)), dtype=int)


def init() -> BackgammonState:
    board: np.ndarray = _make_init_board()
    game_reward: int = 1
    has_doubled: np.ndarray = np.zeros(2, dtype=int)
    is_bearable: np.ndarray = np.zeros(2, dtype=int)
    dice: np.ndarray = _roll_init_dice()
    turn: int = _init_turn(dice)
    legal_actions: np.ndarray = np.zeros((2, 4 * (6 * 26 + 6)), dtype=int)
    state = BackgammonState(
        board=board,
        game_reward=game_reward,
        has_doubled=has_doubled,
        is_bearable=is_bearable,
        dice=dice,
        turn=turn,
        legal_actions=legal_actions,
    )
    return state


def _make_init_board() -> np.ndarray:
    board: np.ndarray = np.zeros(28, dtype=int)
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
    return np.array(roll)


def _init_turn(dice: np.ndarray) -> int:
    diff = dice[1] - dice[0]
    turn = int((1 + np.sign(diff)) / 2)  # diff > 0 : turn=1, diff < 0: turn=0
    return turn
