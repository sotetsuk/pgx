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
    kou: np.ndarray = np.zeros(1, dtype=bool)


def init() -> MiniGoState:
    return MiniGoState()


def to_init_board(str: str) -> np.ndarray:
    """
    文字列から初期配置用のndarrayを生成する関数

    ex.
    init_board = to_init_board("+@+++@O@++@O@+++@+++@++++")
    state = init(init_board)
    =>
      [ 0 1 2 3 4 ]
    [0] + @ + + +
    [1] @ O @ + +
    [2] @ O @ + +
    [3] + @ + + +
    [4] @ + + + +
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
    return init()


def show(state: MiniGoState) -> None:
    board = state.board
    _show_board(board)


def _show_board(board: np.ndarray) -> None:
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
    action: [x, y] | None

    返り値
    (state, reward, done)
    """
    new_state = copy.deepcopy(state)
    r = 0
    done = False

    # 2回連続でパスすると終局
    if action is None:
        if new_state.passed[0]:
            print("end by pass.")
            done = True
            # r = get_score()
            new_state.turn[0] = new_state.turn[0] + 1
            return new_state, r, done
        else:
            new_state.passed[0] = True
            new_state.turn[0] = new_state.turn[0] + 1
            return new_state, r, done

    new_state.passed[0] = False

    x = action[0]
    y = action[1]
    color = new_state.turn[0] % 2
    board = new_state.board

    # 合法手か確認
    if _can_set_stone(board, x, y, color):
        new_state.board[x, y] = color
    # 合法手でない場合負けとする
    else:
        r = -100
        done = True
        print("cannot set stone.")
        new_state.turn[0] = new_state.turn[0] + 1
        return new_state, r, done

    # 囲んでいたら取る
    surrounded_stones = _get_surrounded_stones(
        new_state.board, _opponent_color(color)
    )
    new_state.board[surrounded_stones] = POINT

    # 取った分だけアゲハマ増加
    agehama = np.count_nonzero(surrounded_stones)
    new_state.agehama[color] += agehama

    new_state.turn[0] = new_state.turn[0] + 1
    return new_state, r, done


def _can_set_stone(_board: np.ndarray, x: int, y: int, color: int) -> bool:
    board = _board.copy()

    if board[x][y] != 2:
        # 既に石があるならFalse
        return False

    # 試しに置いてみる
    board[x][y] = color
    surrounded = _is_surrounded_v2(board, x, y, color)
    opponent_surrounded = np.any(
        _get_surrounded_stones(board, _opponent_color(color))
    )
    # 着手禁止点はFalse
    # 着手禁止点でも相手の石を取れるならTrue
    can_set_stone = not surrounded or bool(surrounded and opponent_surrounded)

    return can_set_stone


def _is_surrounded(_board: np.ndarray, _x: int, _y: int, color: int) -> bool:
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
            return True
        else:
            x = candidate_x[num_of_candidate - 1]
            y = candidate_y[num_of_candidate - 1]
            num_of_candidate -= 1

        # この座標は「既に調べたリスト」へ
        if examined_stones[x][y]:
            continue
        examined_stones[x][y] = True

        if y < BOARD_SIZE - 1:
            if board[x][y + 1] == POINT:
                return False
            elif board[x][y + 1] == color:
                candidate_x[num_of_candidate] = x
                candidate_y[num_of_candidate] = y + 1
                num_of_candidate += 1

        if x < BOARD_SIZE - 1:
            if board[x + 1][y] == POINT:
                return False
            elif board[x + 1][y] == color:
                candidate_x[num_of_candidate] = x + 1
                candidate_y[num_of_candidate] = y
                num_of_candidate += 1

        if 1 < y:
            if board[x][y - 1] == POINT:
                return False
            elif board[x][y - 1] == color:
                candidate_x[num_of_candidate] = x
                candidate_y[num_of_candidate] = y - 1
                num_of_candidate += 1

        if 1 < x:
            if board[x - 1][y] == POINT:
                return False
            elif board[x - 1][y] == color:
                candidate_x[num_of_candidate] = x - 1
                candidate_y[num_of_candidate] = y
                num_of_candidate += 1
    return True


def _is_surrounded_v2(
    _board: np.ndarray, _x: int, _y: int, color: int
) -> bool:

    surrounded_stones = _get_surrounded_stones(_board, color)
    return surrounded_stones[_x][_y]


def _get_surrounded_stones(_board: np.ndarray, color: int):
    """
    _is_surrounded()と違うのは、
    1. 空点から調べ始める
    2. 囲まれた石を全て返す
    """
    # 1. boardの一番外側に1週分追加
    board = np.hstack(
        (
            np.full(BOARD_SIZE + 2, -1).reshape((BOARD_SIZE + 2, 1)),
            np.vstack(
                (
                    np.full(BOARD_SIZE, -1),
                    _board.copy(),
                    np.full(BOARD_SIZE, -1),
                )
            ),
            np.full(BOARD_SIZE + 2, -1).reshape((BOARD_SIZE + 2, 1)),
        )
    )
    # こうなる
    # [[-1 -1 -1 -1 -1 -1 -1]
    #  [-1  2  2  2  2  2 -1]
    #  [-1  2  2  2  2  2 -1]
    #  [-1  2  2  2  2  2 -1]
    #  [-1  2  2  2  2  2 -1]
    #  [-1  2  2  2  2  2 -1]
    #  [-1 -1 -1 -1 -1 -1 -1]]

    # 2. 空点に隣り合うcolorの石を取り除く
    LARGE_NUMBER = 361  # 361以上なら大丈夫なはず
    num_of_candidate = 0
    candidate_x: np.ndarray = np.zeros(LARGE_NUMBER, dtype=int)
    candidate_y: np.ndarray = np.zeros(LARGE_NUMBER, dtype=int)

    examined_stones: np.ndarray = np.zeros_like(board, dtype=bool)
    for _x in range(board.shape[0]):
        for _y in range(board.shape[1]):
            if board[_x][_y] == POINT:
                candidate_x[num_of_candidate] = _x
                candidate_y[num_of_candidate] = _y
                num_of_candidate += 1

    for _ in range(LARGE_NUMBER):
        if num_of_candidate == 0:
            break
        else:
            x = candidate_x[num_of_candidate - 1]
            y = candidate_y[num_of_candidate - 1]
            num_of_candidate -= 1

        # この座標は「既に調べたリスト」へ
        if examined_stones[x][y]:
            continue
        examined_stones[x][y] = True

        if board[x][y - 1] == color:
            board[x][y - 1] = POINT
        if board[x][y - 1] == POINT:
            candidate_x[num_of_candidate] = x
            candidate_y[num_of_candidate] = y - 1
            num_of_candidate += 1

        if board[x + 1][y] == color:
            board[x + 1][y] = POINT
        if board[x + 1][y] == POINT:
            candidate_x[num_of_candidate] = x + 1
            candidate_y[num_of_candidate] = y
            num_of_candidate += 1

        if board[x][y + 1] == color:
            board[x][y + 1] = POINT
        if board[x][y + 1] == POINT:
            candidate_x[num_of_candidate] = x
            candidate_y[num_of_candidate] = y + 1
            num_of_candidate += 1

        if board[x - 1][y] == color:
            board[x - 1][y] = POINT
        if board[x - 1][y] == POINT:
            candidate_x[num_of_candidate] = x - 1
            candidate_y[num_of_candidate] = y
            num_of_candidate += 1

    # 3. 増やした外側をカット
    board = np.delete(
        np.delete(arr=board, obj=[0, board.shape[0] - 1], axis=0),
        [0, board.shape[1] - 1],
        axis=1,
    )

    # 4. 囲まれた指定色の石をTrue、それ以外をFalseにして返す
    surrounded_stones = board == color

    return surrounded_stones


def _opponent_color(color: int) -> int:
    return (color + 1) % 2


def legal_actions(state: MiniGoState) -> np.ndarray:
    """
    全ての点に仮に置いた時を調べるのでかなり非効率的
    """
    board = copy.deepcopy(state.board)
    legal_actions: np.ndarray = np.zeros_like(board, dtype=bool)
    color = (state.turn[0]) % 2

    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            if _can_set_stone(board, x, y, color):
                legal_actions[x][y] = True

    return legal_actions
