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
BLACK_CHAR = "@"
WHITE_CHAR = "O"
POINT_CHAR = "+"

TO_CHAR = np.array([BLACK_CHAR, WHITE_CHAR, POINT_CHAR], dtype=str)


@struct.dataclass
class MiniGoState:
    board: np.ndarray = np.full((BOARD_SIZE, BOARD_SIZE), POINT)
    turn: np.ndarray = np.zeros(1, dtype=int)
    agehama: np.ndarray = np.zeros(
        2, dtype=int
    )  # agehama[0]: agehama earned by player(black), agehama[1]: agehama earned by player(white)
    passed: np.ndarray = np.zeros(1, dtype=bool)


def init(init_board: Optional[np.ndarray]) -> MiniGoState:
    """
    ndarrayを渡して初期配置を指定できる
    ndarrayは(5, 5), dtype=int
    """
    if init_board is not None:
        assert init_board.shape == (BOARD_SIZE, BOARD_SIZE)
        # dataclassに初期値を与えるとmypyがよく分からんエラーを吐く
        # pgx/mini_go.py:36: error: Unexpected keyword argument "board" for "MiniGoState"
        # cf. https://github.com/python/mypy/issues/6239
        state = MiniGoState(board=init_board.copy())  # type: ignore
    else:
        state = MiniGoState()

    return state


def to_init_board(str: str) -> np.ndarray:
    """
    文字列から初期配置用のndarrayを生成する関数
    """
    assert len(str) == BOARD_SIZE * BOARD_SIZE
    init_board = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=int)
    for i in range(BOARD_SIZE * BOARD_SIZE):
        if str[i] == BLACK_CHAR:
            init_board[i] = BLACK
        elif str[i] == WHITE_CHAR:
            init_board[i] = WHITE
        elif str[i] == POINT_CHAR:
            init_board[i] = POINT
        else:
            assert False
    return init_board.reshape((BOARD_SIZE, BOARD_SIZE))


def reset() -> MiniGoState:
    return init(None)


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

    返り値
    (state, reward, done)
    """
    new_state = copy.deepcopy(state)
    r = 0
    done = False
    if action is None:
        if new_state.passed[0]:
            print("end by pass.")
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
        print("cannot set stone.")
        return new_state, r, done

    # TODO: 囲んでいたら取る

    return new_state, r, done


def _can_set_stone(_board: np.ndarray, x: int, y: int, color: int) -> bool:
    board = _board.copy()

    if board[x][y] != 2:
        return False
    board[x][y] = color
    surrounded = _is_surrounded(board, x, y, color)
    # TODO: 例外
    # 取れる場合は置ける
    return not surrounded


# TODO: C901 '_is_surrounded' is too complex (19) 後で改善したい
def _is_surrounded(  # noqa: C901
    _board: np.ndarray, _x: int, _y: int, color
) -> bool:
    board = _board.copy()
    LARGE_NUMBER = 361  # 361以上なら大丈夫なはず
    num_of_candidate = 0
    candidate_x: np.ndarray = np.zeros(LARGE_NUMBER, dtype=int)
    candidate_y: np.ndarray = np.zeros(LARGE_NUMBER, dtype=int)
    examined_stones: np.ndarray = np.zeros_like(board, dtype=bool)
    candidate_x[num_of_candidate] = _x
    candidate_y[num_of_candidate] = _y
    num_of_candidate += 1

    # 隣り合う石の座標を次々にcandidateへ放り込み、全て調べる作戦
    # 重複して調べるのを避けるため、既に調べた座標のリスト examined_stones も用意
    for _ in range(LARGE_NUMBER):
        if num_of_candidate == 0:
            break
        else:
            x = candidate_x[num_of_candidate - 1]
            y = candidate_y[num_of_candidate - 1]
            num_of_candidate -= 1

        # この座標は「既に調べたリスト」へ
        examined_stones[x][y] = True

        if y < BOARD_SIZE - 1:
            if not examined_stones[x][y + 1]:
                if board[x][y + 1] == POINT:
                    return False
                elif board[x][y + 1] == color:
                    candidate_x[num_of_candidate] = x
                    candidate_y[num_of_candidate] = y + 1
                    num_of_candidate += 1

        if x < BOARD_SIZE - 1:
            if not examined_stones[x + 1][y]:
                if board[x + 1][y] == POINT:
                    return False
                elif board[x + 1][y] == color:
                    candidate_x[num_of_candidate] = x + 1
                    candidate_y[num_of_candidate] = y
                    num_of_candidate += 1

        if 1 < y:
            if not examined_stones[x][y - 1]:
                if board[x][y - 1] == POINT:
                    return False
                elif board[x][y - 1] == color:
                    candidate_x[num_of_candidate] = x
                    candidate_y[num_of_candidate] = y - 1
                    num_of_candidate += 1

        if 1 < x:
            if not examined_stones[x - 1][y]:
                if board[x - 1][y] == POINT:
                    return False
                elif board[x - 1][y] == color:
                    candidate_x[num_of_candidate] = x - 1
                    candidate_y[num_of_candidate] = y
                    num_of_candidate += 1
    return True
