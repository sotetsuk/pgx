# import jax
import copy
from typing import Optional, Tuple

import numpy as np
from flax import struct

# from jax import numpy as jnp

BOARD_SIZE = 5

# TODO: enum的にまとめたい
BLACK = 0
WHITE = 1
POINT = 2
TO_CHAR = np.array(["@", "O", "+"], dtype=str)


@struct.dataclass
class MiniGoState:
    board: np.ndarray = np.full((BOARD_SIZE, BOARD_SIZE), POINT)
    turn: np.ndarray = np.zeros(1, dtype=int)
    agehama: np.ndarray = np.zeros(
        2, dtype=int
    )  # agehama[0]: agehama earned by player(black), agehama[1]: agehama earned by player(white)
    passed = np.zeros(1, dtype=bool)


def init() -> MiniGoState:
    state = MiniGoState()
    return state


def reset() -> MiniGoState:
    return init()


def show(state: MiniGoState) -> None:
    board = state.board
    print()
    print("  [ ", end="")
    for i in range(BOARD_SIZE):
        print(f"{i%10} ", end="")
    print("]")
    for i in range(BOARD_SIZE):
        print(f"[{i%10}]", end="")
        for j in range(BOARD_SIZE):
            print(" " + TO_CHAR[board[i][j]], end="")
        print()


def step(
    state: MiniGoState, action: Optional[np.ndarray]
) -> Tuple[MiniGoState, int, bool]:
    """
    action: [x, y, color] | None
    """
    new_state = copy.deepcopy(state)
    r = 0
    done = False
    if action is None:
        if new_state.passed[0]:
            done = True
            # r = get_score()
            return new_state, r, done
        else:
            new_state.passed[0] = True
            return new_state, r, done

    new_state.passed[0] = False
    new_state.turn[0] = new_state.turn[0] + 1

    x = action[0]
    y = action[1]
    color = action[2]
    board = new_state.board
    if _can_set_stone(board, x, y, color):
        new_state.board[x, y] = color
    else:
        r = -100
        done = True
        return new_state, r, done

    # 囲んでいたら取る

    return new_state, r, done


def _can_set_stone(_board: np.ndarray, x: int, y: int, color: int) -> bool:
    board = _board.copy()

    if board[x][y] != 2:
        return False
    board[x][y] = color
    surrounded, _ = _is_surrounded(
        board, x, y, color, np.zeros_like(board, dtype=bool)
    )
    return not surrounded


def _is_surrounded(
    board: np.ndarray, x: int, y: int, color: int, _examined_stones: np.ndarray
) -> Tuple[bool, np.ndarray]:
    examined_stones = _examined_stones.copy()

    if x < 0 or BOARD_SIZE <= x or y < 0 or BOARD_SIZE <= y:
        return True, examined_stones

    if examined_stones[x][y]:
        return True, examined_stones
    else:
        examined_stones[x][y] = True

    if board[x][y] == color:
        _surrounded, examined_stones = _is_surrounded(
            board, x + 1, y, color, examined_stones
        )
        if not _surrounded:
            return False, examined_stones

        _surrounded, examined_stones = _is_surrounded(
            board, x, y - 1, color, examined_stones
        )
        if not _surrounded:
            return False, examined_stones

        _surrounded, examined_stones = _is_surrounded(
            board, x - 1, y, color, examined_stones
        )
        if not _surrounded:
            return False, examined_stones

        _surrounded, examined_stones = _is_surrounded(
            board, x, y + 1, color, examined_stones
        )
        if not _surrounded:
            return False, examined_stones
        else:
            return True, examined_stones

    elif board[x][y] == POINT:
        return False, examined_stones

    return True, examined_stones
