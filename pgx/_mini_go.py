# import jax
import copy
from typing import Tuple

import numpy as np
from flax import struct

# from jax import numpy as jnp

BOARD_SIZE = 5

BLACK = 0
WHITE = 1
POINT = 2
BLACK_CHAR = "@"
WHITE_CHAR = "O"
POINT_CHAR = "+"

NSEW = np.array([[-1, 0], [1, 0], [0, 1], [0, -1]])


@struct.dataclass
class MiniGoState:
    ren_id_board: np.ndarray = np.full(
        (2, BOARD_SIZE * BOARD_SIZE), -1, dtype=int
    )
    available_ren_id: np.ndarray = np.ones(  # n番目の連idが使えるか
        (2, BOARD_SIZE * BOARD_SIZE), dtype=bool
    )
    liberty: np.ndarray = np.zeros(
        (2, BOARD_SIZE * BOARD_SIZE, BOARD_SIZE * BOARD_SIZE), dtype=bool
    )

    turn: np.ndarray = np.zeros(1, dtype=int)
    agehama: np.ndarray = np.zeros(2, dtype=int)  # [0]: 黒の得たアゲハマ, [1]: 白の方
    passed: np.ndarray = np.zeros(1, dtype=bool)  # 直前のactionがパスだとTrue
    kou: np.ndarray = np.full(2, -1, dtype=int)  # コウによる着手禁止点, 無ければ(-1, -1)


def init() -> MiniGoState:
    return MiniGoState()


def step(
    state: MiniGoState, action: int
) -> Tuple[MiniGoState, np.ndarray, bool]:
    state = copy.deepcopy(state)

    if action < 0:
        result = _pass_move(state)
    else:
        result = _not_pass_move(state, action)

    return result


def _pass_move(_state: MiniGoState) -> Tuple[MiniGoState, np.ndarray, bool]:
    state = copy.deepcopy(_state)

    if state.passed[0]:  # 2回連続でパスすると終局
        result = (_add_turn(state), _get_reward(state), True)
    else:
        result = (_add_turn(_add_pass(state)), np.array([0, 0]), False)
    return result


def _add_turn(_state: MiniGoState) -> MiniGoState:
    state = copy.deepcopy(_state)
    state.turn[0] = state.turn[0] + 1
    return state


def _add_pass(_state: MiniGoState) -> MiniGoState:
    state = copy.deepcopy(_state)
    state.passed[0] = True
    return state


def _not_pass_move(
    _state: MiniGoState, _action: int
) -> Tuple[MiniGoState, np.ndarray, bool]:
    state = copy.deepcopy(_state)
    state.passed[0] = False
    xy = copy.deepcopy(_action)
    x = xy // BOARD_SIZE
    y = xy % BOARD_SIZE

    my_color = state.turn[0] % 2
    oppo_color = _opponent_color(my_color)

    ren_id_board = state.ren_id_board[my_color]
    oppo_ren_id_board = state.ren_id_board[oppo_color]
    available_ren_id = state.available_ren_id[my_color]
    new_id = int(np.argmax(available_ren_id))  # 最初にTrueになったindex
    # liberty = state.liberty[my_color]

    pos = np.array([x, y])
    agehama = 0
    a_removed_stone_xy = -1

    # 石を置く
    if (
        ren_id_board[xy] != -1
        or oppo_ren_id_board[xy] != -1
        or (x == state.kou[0] and y == state.kou[1])
    ):  # 既に他の石が置かれている or コウ
        # print("ilegal")
        return _ilegal_move(state)

    ren_id_board[xy] = new_id
    available_ren_id[new_id] = False
    is_kou = _check_kou(state, x, y, oppo_color)

    # 周囲の連を数える
    for nsew in NSEW:
        around_mypos = pos + nsew
        if _is_off_board(around_mypos):
            continue

        around_xy = _pos_to_xy(around_mypos)
        if ren_id_board[around_xy] != -1:  # 既に連が作られていた場合
            (state, new_id) = _merge_ren(
                state,
                my_color,
                new_id,
                xy,
                around_xy,
            )

        elif oppo_ren_id_board[around_xy] != -1:  # 敵の連が作られていた場合
            oppo_ren_id = oppo_ren_id_board[around_xy]
            oppo_liberty = state.liberty[oppo_color]
            oppo_liberty[oppo_ren_id][xy] = False
            if np.count_nonzero(oppo_liberty[oppo_ren_id]) == 0:
                # 石を取る
                (state, a_removed_stone_xy, agehama) = _remove_stones(
                    state,
                    my_color,
                    oppo_ren_id,
                    agehama,
                    around_xy,
                )

        else:
            state.liberty[my_color][new_id][around_xy] = True

    if np.count_nonzero(state.liberty[my_color][new_id]) == 0:
        # 自殺手
        return _ilegal_move(state)

    # コウの確認
    if agehama == 1 and is_kou:
        state.kou[0], state.kou[1] = (
            a_removed_stone_xy // BOARD_SIZE,
            a_removed_stone_xy % BOARD_SIZE,
        )
    else:
        state.kou[0], state.kou[1] = (-1, -1)

    state.agehama[my_color] += agehama
    state = _add_turn(state)

    return state, np.array([0, 0]), False


def _ilegal_move(_state: MiniGoState) -> Tuple[MiniGoState, np.ndarray, bool]:
    state = copy.deepcopy(_state)
    r = np.array([1, 1])
    r[state.turn[0] % 2] = -1
    state = _add_turn(state)
    return state, r, True


def _merge_ren(
    _state: MiniGoState,
    _my_color: int,
    _new_id: int,
    xy: int,
    xy_around_mypos: int,
):
    state = copy.deepcopy(_state)
    ren_id_board = state.ren_id_board[_my_color]
    available_ren_id = state.available_ren_id[_my_color]
    liberty = state.liberty[_my_color]
    new_id = _new_id

    old_id = ren_id_board[xy_around_mypos]
    if old_id == new_id:
        return state, new_id

    small_id = min(old_id, new_id)
    large_id = max(old_id, new_id)
    # 大きいidの連を消し、小さいidの連と繋げる

    available_ren_id[large_id] = True
    ren_id_board[ren_id_board == large_id] = small_id

    liberty[large_id][xy] = liberty[small_id][xy] = False
    liberty[small_id] = liberty[small_id] | liberty[large_id]
    liberty[large_id] = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=bool)

    new_id = small_id

    return state, new_id


def _remove_stones(
    _state: MiniGoState,
    _my_color: int,
    _oppo_ren_id: int,
    _agehama: int,
    _around_xy,
):
    state = copy.deepcopy(_state)
    ren_id_board = state.ren_id_board[_my_color]
    oppo_color = _opponent_color(_my_color)
    oppo_ren_id_board = state.ren_id_board[oppo_color]
    oppo_available_ren_id = state.available_ren_id[oppo_color]
    liberty = state.liberty[_my_color]
    agehama = _agehama

    surrounded_stones = oppo_ren_id_board == _oppo_ren_id  # 呼吸点0の連の位置情報
    agehama += np.count_nonzero(surrounded_stones)  # その石の数
    oppo_ren_id_board[surrounded_stones] = -1  # ren_id_boardから削除
    oppo_available_ren_id[_oppo_ren_id] = True  # available_ren_idに追加
    a_removed_stone_xy = _around_xy  # コウのために取った位置を記憶

    # 空けたところを自軍の呼吸点に追加
    liberty[:] = _add_removed_pos_to_liberty(
        ren_id_board, liberty, surrounded_stones
    )

    return (
        state,
        a_removed_stone_xy,
        agehama,
    )


def _add_removed_pos_to_liberty(
    _ren_id_board: np.ndarray,
    _liberty: np.ndarray,
    _surrounded_stones: np.ndarray,
):
    ren_id_board = _ren_id_board.copy()
    liberty = _liberty.copy()
    for _xy in range(BOARD_SIZE * BOARD_SIZE):
        for _nsew in NSEW:
            _around_rmstone_pos = _xy_to_pos(_xy) + _nsew
            if _is_off_board(_around_rmstone_pos):
                continue
            _around_rmstone_xy = _to_xy(
                _around_rmstone_pos[0], _around_rmstone_pos[1]
            )
            if (
                ren_id_board[_around_rmstone_xy] != -1
                and _surrounded_stones[_xy]
            ):
                # adj_ren_id = ren_id_board[_around_rmstone_xy]
                # liberty[adj_ren_id][_xy] = True
                liberty[ren_id_board[_around_rmstone_xy]][_xy] = True

    return liberty


def legal_actions(state: MiniGoState) -> np.ndarray:
    legal_action = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=bool)
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            _, _, is_ilegal = step(state, _to_xy(i, j))
            legal_action[_to_xy(i, j)] = not is_ilegal
    return legal_action


def get_board(state: MiniGoState) -> np.ndarray:
    board = np.full(BOARD_SIZE * BOARD_SIZE, 2)
    board[state.ren_id_board[BLACK] != -1] = 0
    board[state.ren_id_board[WHITE] != -1] = 1
    return board


def show(state: MiniGoState) -> None:
    print("===========")

    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if state.ren_id_board[BLACK][_to_xy(i, j)] != -1:
                print(" " + BLACK_CHAR, end="")
            elif state.ren_id_board[WHITE][_to_xy(i, j)] != -1:
                print(" " + WHITE_CHAR, end="")
            else:
                print(" " + POINT_CHAR, end="")
        print("")


def _show_details(state: MiniGoState) -> None:
    show(state)
    print(state.ren_id_board[BLACK].reshape((5, 5)))
    print(state.ren_id_board[WHITE].reshape((5, 5)))


def _is_off_board(pos: np.ndarray) -> bool:
    return (
        pos[0] < 0
        or BOARD_SIZE <= pos[0]
        or pos[1] < 0
        or BOARD_SIZE <= pos[1]
    )


def _pos_to_xy(pos: np.ndarray) -> int:
    return pos[0] * BOARD_SIZE + pos[1]


def _to_xy(x, y) -> int:
    return x * BOARD_SIZE + y


def _xy_to_pos(xy):
    return np.array([xy // BOARD_SIZE, xy % BOARD_SIZE])


def _opponent_color(color: int) -> int:
    return (color + 1) % 2


def _check_kou(state: MiniGoState, x, y, oppo_color) -> bool:
    return (
        (x < 0 or state.ren_id_board[oppo_color][_to_xy(x - 1, y)] != -1)
        and (
            x >= BOARD_SIZE - 1
            or state.ren_id_board[oppo_color][_to_xy(x + 1, y)] != -1
        )
        and (y < 0 or state.ren_id_board[oppo_color][_to_xy(x, y - 1)] != -1)
        and (
            y >= BOARD_SIZE - 1
            or state.ren_id_board[oppo_color][_to_xy(x, y + 1)] != -1
        )
    )


def _get_reward(state: MiniGoState) -> np.ndarray:
    b = _count_ji(state)[BLACK] - state.agehama[WHITE]
    w = _count_ji(state)[WHITE] - state.agehama[BLACK]
    r = np.array([-1, 1])
    if b == w:
        r = np.array([0, 0])
    if b > w:
        r = np.array([1, -1])
    return r


def _count_ji(state: MiniGoState) -> np.ndarray:
    board = get_board(state)
    ji_id_board = _get_ji_id_board(state)

    # -1:未確定 0:黒 1:白 2:どちらでもないことが確定
    color_of_ji = np.full((BOARD_SIZE * BOARD_SIZE), -1, dtype=int)

    for xy in range(BOARD_SIZE * BOARD_SIZE):
        ji_id = ji_id_board[xy]
        if ji_id_board[xy] == -1 or color_of_ji[ji_id] == 2:
            continue

        for nsew in NSEW:
            around_pos = _xy_to_pos(xy) + nsew
            around_xy = _pos_to_xy(around_pos)
            if _is_off_board(around_pos) or board[around_xy] == POINT:
                continue
            if color_of_ji[xy] == -1:
                color_of_ji[xy] = board[around_xy]
            elif color_of_ji[xy] == _opponent_color(board[around_xy]):
                color_of_ji[ji_id_board == ji_id] = 2

    b = np.count_nonzero(color_of_ji == BLACK)
    w = np.count_nonzero(color_of_ji == WHITE)

    return np.array([b, w])


def _get_ji_id_board(state: MiniGoState):
    board = get_board(state)
    ji_id_board: np.ndarray = np.full((BOARD_SIZE * BOARD_SIZE), -1, dtype=int)
    available_ji_id: np.ndarray = np.ones(
        (BOARD_SIZE * BOARD_SIZE), dtype=bool
    )
    for xy in range(BOARD_SIZE * BOARD_SIZE):
        if board[xy] != POINT:
            continue
        new_id = int(np.argmax(available_ji_id))  # 最初にTrueになったindex
        ji_id_board[xy] = new_id
        available_ji_id[new_id] = False

        for nsew in NSEW:
            around_mypos = _xy_to_pos(xy) + nsew
            if _is_off_board(around_mypos):
                continue

            xy_around_mypos = _pos_to_xy(around_mypos)
            if ji_id_board[xy_around_mypos] != -1:
                (ji_id_board[:], available_ji_id[:], new_id,) = _merge_points(
                    ji_id_board,
                    available_ji_id,
                    new_id,
                    xy,
                    xy_around_mypos,
                )

    return ji_id_board


def _merge_points(
    _ji_id_board: np.ndarray,
    _available_ji_id: np.ndarray,
    new_id: int,
    xy: int,
    xy_around_mypos: int,
):
    ji_id_board = _ji_id_board.copy()
    available_ji_id = _available_ji_id.copy()
    new_id = int(np.argmax(available_ji_id))  # 最初にTrueになったindex

    old_id = ji_id_board[xy_around_mypos]
    if old_id == new_id:
        return ji_id_board, available_ji_id, new_id

    small_id = min(old_id, new_id)
    large_id = max(old_id, new_id)
    # 大きいidの連を消し、小さいidの連と繋げる

    available_ji_id[large_id] = True
    ji_id_board[ji_id_board == large_id] = small_id

    new_id = small_id

    return ji_id_board, available_ji_id, new_id
