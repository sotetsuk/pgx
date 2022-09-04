# import jax
import copy
from typing import Optional, Tuple

import numpy as np
from flax import struct

# from jax import numpy as jnp

BOARD_SIZE = 5

TO_CHAR = np.array(["0", "O", "+"], dtype=str)


@struct.dataclass
class MiniGoState:
    board: np.ndarray = np.full((BOARD_SIZE, BOARD_SIZE), 2)
    turn: np.ndarray = np.zeros(1)
    agehama: np.ndarray = np.zeros(
        2
    )  # agehama[0]: agehama earned by player(black), agehama[1]: agehama earned by player(white)


def init() -> MiniGoState:
    return MiniGoState()


def reset() -> MiniGoState:
    return MiniGoState()


def show(state: MiniGoState) -> None:
    board = state.board

    print("=======")
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            print(TO_CHAR[board[i][j]], end="")
        print("")


def step(
    state: MiniGoState, action: Optional[np.ndarray]
) -> Tuple[MiniGoState, int, bool]:
    new_state = copy.deepcopy(state)
    r = 0
    done = False
    if action is None:
        return new_state, r, done

    x = action[0]
    y = action[1]
    color = action[2]
    board = new_state.board
    if _can_set_stone(board, x, y, color):
        new_state.board[x, y] = color
    return new_state, r, done


def _can_set_stone(board: np.ndarray, x: int, y: int, color: int) -> bool:
    if board[x][y] == 2:
        return True
    return False


state = init()
show(state)
action = np.array([1, 1, 1])
state, _, _ = step(state=state, action=action)
action = np.array([1, 2, 0])
state, _, _ = step(state=state, action=action)
action = np.array([3, 1, 1])
state, _, _ = step(state=state, action=action)
action = None
state, _, _ = step(state=state, action=action)
show(state)
