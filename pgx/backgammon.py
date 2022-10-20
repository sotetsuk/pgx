from dataclasses import dataclass

import numpy as np


@dataclass
class BackgammonState:
    # 白なら0黒なら1
    turn: int = 0

    # 各point(24)にあるcheckerの数 [0]: 白, [1]: 黒
    borad: np.ndarray = np.zeros((2, 24), dtype=int)

    # barの上にあるcheckerの数
    bar: np.ndarray = np.zeros(2, dtype=int)

    # offにあるcheckerの数
    off: np.ndarray = np.zeros(2, dtype=int)

    # ゲームの報酬. doublingが起こると変わる.
    game_reward: int = 0

    # doubligをしたかどうか.
    has_doubled: np.ndarray = np.zeros(2, dtype=bool)

    # bear offできるかどうか.
    is_bearable: np.ndarray = np.zeros(2, dtype=bool)
