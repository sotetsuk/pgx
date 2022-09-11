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
    board: np.ndarray = np.full((BOARD_SIZE, BOARD_SIZE), POINT, dtype=int)
    turn: np.ndarray = np.zeros(1, dtype=int)
    agehama: np.ndarray = np.zeros(
        2, dtype=int
    )  # agehama[0]: agehama earned by player(black), agehama[1]: agehama earned by player(white)
    passed: np.ndarray = np.zeros(1, dtype=bool)
    kou: np.ndarray = np.full(2, -1, dtype=int)  # コウによる着手禁止点, 無ければ(-1, -1)


def init() -> MiniGoState:
    return MiniGoState()


def to_init_board(str: str) -> np.ndarray:
    """
    文字列から初期配置用のndarrayを生成する関数

    ex.
    init_board = to_init_board("+@+++@O@++@O@+++@+++@++++")
    state = MiniGoState(board=init_board)
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
) -> Tuple[MiniGoState, np.ndarray, bool]:
    """
    action: [x, y] | None

    返り値
    (state, reward, done)
    """
    new_state = copy.deepcopy(state)
    r = np.array([0, 0])
    done = False

    # 2回連続でパスすると終局
    if action is None:
        if new_state.passed[0]:
            print("end by pass.")
            done = True
            r = _get_reward(new_state)
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

    # 合法手か確認
    if _can_set_stone(new_state, x, y, color):
        new_state.board[x, y] = color
    # 合法手でない場合負けとする
    else:
        r = np.array([1, 1])
        r[new_state.turn[0] % 2] = -1
        done = True
        print("cannot set stone.")
        new_state.turn[0] = new_state.turn[0] + 1
        return new_state, r, done

    new_state = _remove_stones_from_state(new_state, x, y)

    new_state.turn[0] = new_state.turn[0] + 1
    return new_state, r, done


def _can_set_stone(state: MiniGoState, x: int, y: int, color: int) -> bool:
    board = state.board.copy()

    if board[x][y] != 2:
        # 既に石があるならFalse
        return False
    kou = copy.deepcopy(state.kou)

    if x == kou[0] and y == kou[1]:
        # コウならFalse
        return False

    # 試しに置いてみる
    board[x][y] = color
    surrounded = _is_surrounded_v2(board, x, y, color)
    opponent_surrounded = np.any(
        _get_surrounded_stones(
            board, target_color=_opponent_color(color), surrounding_color=color
        )
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

    surrounded_stones = _get_surrounded_stones(
        _board, target_color=color, surrounding_color=_opponent_color(color)
    )
    return surrounded_stones[_x][_y]


def _get_surrounded_stones(
    _board: np.ndarray, target_color: int, surrounding_color: int
):
    """
    _is_surrounded()と違うのは、
    1. 空点から調べ始める
    2. 囲まれた石を全て返す
    """
    fill_color = 3 - target_color - surrounding_color

    # 1. boardの一番外側に1周分追加
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
    LARGE_NUMBER = 361 * 10
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

        if board[x][y - 1] == target_color:
            board[x][y - 1] = fill_color
        if board[x][y - 1] == fill_color and not examined_stones[x][y - 1]:
            candidate_x[num_of_candidate] = x
            candidate_y[num_of_candidate] = y - 1
            num_of_candidate += 1

        if board[x + 1][y] == target_color:
            board[x + 1][y] = fill_color
        if board[x + 1][y] == fill_color and not examined_stones[x + 1][y]:
            candidate_x[num_of_candidate] = x + 1
            candidate_y[num_of_candidate] = y
            num_of_candidate += 1

        if board[x][y + 1] == target_color:
            board[x][y + 1] = fill_color
        if board[x][y + 1] == fill_color and not examined_stones[x][y + 1]:
            candidate_x[num_of_candidate] = x
            candidate_y[num_of_candidate] = y + 1
            num_of_candidate += 1

        if board[x - 1][y] == target_color:
            board[x - 1][y] = fill_color
        if board[x - 1][y] == fill_color and not examined_stones[x - 1][y]:
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
    surrounded_stones = board == target_color

    return surrounded_stones


def _opponent_color(color: int) -> int:
    return (color + 1) % 2


def _remove_stones_from_state(
    state: MiniGoState, x: int, y: int
) -> MiniGoState:
    new_state = copy.deepcopy(state)
    color = new_state.turn[0] % 2
    op_color = _opponent_color(color)

    # 囲んでいたら取る
    surrounded_stones = _get_surrounded_stones(
        new_state.board, target_color=op_color, surrounding_color=color
    )
    new_state.board[surrounded_stones] = POINT

    # 取った分だけアゲハマ増加
    agehama = np.count_nonzero(surrounded_stones)
    new_state.agehama[color] += agehama

    # コウの確認
    if _check_kou(state, x, y, op_color) and agehama == 1:
        _removed_stone = np.where(surrounded_stones)
        new_state.kou[0] = _removed_stone[0][0]
        new_state.kou[1] = _removed_stone[1][0]
    else:
        new_state.kou[0] = -1
        new_state.kou[1] = -1

    return new_state


def _check_kou(state: MiniGoState, x, y, op_color):
    board = state.board
    return (
        (x < 0 or board[x - 1][y] == op_color)
        and (x >= BOARD_SIZE - 1 or board[x + 1][y] == op_color)
        and (y < 0 or board[x][y - 1] == op_color)
        and (y >= BOARD_SIZE - 1 or board[x][y + 1] == op_color)
    )


def legal_actions(state: MiniGoState) -> np.ndarray:
    """
    全ての点に仮に置いた時を調べるのでかなり非効率的
    """
    legal_actions: np.ndarray = np.zeros_like(state.board, dtype=bool)
    color = (state.turn[0]) % 2

    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            legal_actions[x][y] = _can_set_stone(state, x, y, color)

    return legal_actions


def _get_reward(_state: MiniGoState) -> np.ndarray:
    state = copy.deepcopy(_state)

    b_score = _count_ji(state.board, BLACK) - state.agehama[WHITE]
    w_score = _count_ji(state.board, WHITE) - state.agehama[BLACK]
    if w_score < b_score:
        return np.array([1, -1], dtype=int)
    elif w_score > b_score:
        return np.array([-1, 1], dtype=int)
    return np.array([0, 0], dtype=int)


def _count_ji(_board: np.ndarray, _color: int) -> int:
    board = _board.copy()
    ji = _get_surrounded_stones(
        _board=board, target_color=POINT, surrounding_color=_color
    )
    return np.count_nonzero(ji)
