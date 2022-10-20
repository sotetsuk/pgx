from dataclasses import dataclass

import numpy as np


@dataclass
class ContractBridgeBiddingState:
    # 白なら0黒なら1
    turn: int = 0
    # 各point(24)にあるcheckerの数, 白と黒で二次元.
    borad: np.ndarray = np.zeros((2, 24), dtype=int)
    # barの上にあるcheckerの数
    bar: np.ndarray = np.zeros(2, dtype=int)
    # offにあるcheckerの数
    off: np.ndarray = np.zeros(2, dtype=int)
