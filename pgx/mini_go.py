# import jax
import copy
from typing import Optional, Tuple

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
    state: MiniGoState, action: Optional[int]
) -> Tuple[MiniGoState, np.ndarray, bool]:
    state = copy.deepcopy(state)

    if action is None:
        step_result = _pass_move(state)
    else:
        step_result = _not_pass_move(state, action)

    return step_result


def _pass_move(_state: MiniGoState) -> Tuple[MiniGoState, np.ndarray, bool]:
    state = copy.deepcopy(_state)

    if state.passed[0]:  # 2回連続でパスすると終局
        step_result = (_add_turn(state), _get_reward(state), True)
    else:
        step_result = (_add_turn(_add_pass(state)), np.array([0, 0]), False)
    return step_result


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

    my_color = state.turn[0] % 2

    # 石を置く
    if (
        state.ren_id_board[my_color][xy] != -1
        or state.ren_id_board[_opponent_color(my_color)][xy] != -1
        or (xy == _pos_to_xy(state.kou))
    ):  # 既に他の石が置かれている or コウ
        step_result = _illegal_move(state)
    else:
        step_result = _not_duplicate_nor_kou(state, xy, my_color)

    return step_result


def _not_duplicate_nor_kou(
    _state: MiniGoState, _xy: int, _my_color
) -> Tuple[MiniGoState, np.ndarray, bool]:
    state = copy.deepcopy(_state)
    x = _xy // BOARD_SIZE
    y = _xy % BOARD_SIZE

    # 最初にTrueになったindexをidとする
    new_id = int(np.argmax(state.available_ren_id[_my_color]))

    agehama = 0
    a_removed_stone_xy = -1  # コウのために取った位置を記憶する
    state.ren_id_board[_my_color][_xy] = new_id
    state.available_ren_id[_my_color][new_id] = False
    is_kou = _check_kou(state, x, y, _opponent_color(_my_color))

    # 周囲の連を数える
    for nsew in NSEW:
        around_mypos = np.array([x, y]) + nsew
        if _is_off_board(around_mypos):
            state, new_id, agehama, a_removed_stone_xy = (
                state,
                new_id,
                agehama,
                a_removed_stone_xy,
            )
        else:
            state, new_id, agehama, a_removed_stone_xy = _check_around_stones(
                state,
                _xy,
                around_mypos,
                _my_color,
                new_id,
                agehama,
                a_removed_stone_xy,
            )
    if np.count_nonzero(state.liberty[_my_color][new_id]) == 0:  # 自殺手
        step_result = _illegal_move(state)
    else:
        step_result = _not_suicide(
            state, _my_color, agehama, is_kou, a_removed_stone_xy
        )

    return step_result


def _illegal_move(_state: MiniGoState) -> Tuple[MiniGoState, np.ndarray, bool]:
    state = copy.deepcopy(_state)
    r = np.array([1, 1])
    r[state.turn[0] % 2] = -1
    return _add_turn(state), r, True


def _check_around_stones(
    _state: MiniGoState,
    _xy: int,
    _around_mypos: np.ndarray,
    _my_color: int,
    _new_id: int,
    _agehama: int,
    _a_removed_stone_xy: int,
) -> Tuple[MiniGoState, int, int, int]:
    state = copy.deepcopy(_state)
    around_xy = _pos_to_xy(_around_mypos)
    oppo_color = _opponent_color(_my_color)

    # 既に自分の連が作られていた場合
    if state.ren_id_board[_my_color][around_xy] != -1:
        (state, new_id) = _merge_ren(
            state,
            _my_color,
            _new_id,
            _xy,
            around_xy,
        )
    else:
        state, new_id = state, _new_id

    # 敵の連が作られていた場合
    if state.ren_id_board[oppo_color][around_xy] != -1:
        state.liberty[oppo_color] = _update_liberty(
            state.liberty[oppo_color],
            state.ren_id_board[oppo_color][around_xy],
            _xy,
            False,
        )
    else:
        state.liberty[oppo_color] = state.liberty[oppo_color]

    # 敵の連を取れる場合
    if (
        state.ren_id_board[oppo_color][around_xy] != -1
        and np.count_nonzero(
            state.liberty[oppo_color][
                state.ren_id_board[oppo_color][around_xy]
            ]
        )
        == 0
    ):
        # 石を取る
        (state, a_removed_stone_xy, agehama) = _remove_stones(
            state,
            _my_color,
            state.ren_id_board[oppo_color][around_xy],
            _agehama,
            around_xy,
        )
    else:
        state, a_removed_stone_xy, agehama = (
            state,
            _a_removed_stone_xy,
            _agehama,
        )

    # どちらでもない場合
    if (
        not state.ren_id_board[_my_color][around_xy] != -1
        and not state.ren_id_board[oppo_color][around_xy] != -1
    ):
        # 呼吸点に追加
        state.liberty[_my_color] = _update_liberty(
            state.liberty[_my_color], new_id, around_xy
        )
    else:
        state.liberty[_my_color] = state.liberty[_my_color]

    return state, new_id, agehama, a_removed_stone_xy


def _merge_ren(
    _state: MiniGoState,
    _my_color: int,
    _new_id: int,
    _xy: int,
    _xy_around_mypos: int,
) -> Tuple[MiniGoState, int]:
    state = copy.deepcopy(_state)
    ren_id_board = state.ren_id_board[_my_color]
    new_id = _new_id

    old_id = ren_id_board[_xy_around_mypos]

    if old_id == new_id:  # 既に結合済みの場合
        state_and_new_id = state, new_id
    else:
        state_and_new_id = __merge_ren(state, _my_color, old_id, new_id, _xy)

    return state_and_new_id


def __merge_ren(
    _state: MiniGoState, _my_color: int, _old_id: int, _new_id: int, _xy: int
) -> Tuple[MiniGoState, int]:
    state = copy.deepcopy(_state)
    ren_id_board = state.ren_id_board[_my_color]
    available_ren_id = state.available_ren_id[_my_color]
    liberty = state.liberty[_my_color]

    small_id = min(_old_id, _new_id)
    large_id = max(_old_id, _new_id)

    # 大きいidの連を消し、小さいidの連と繋げる
    available_ren_id[large_id] = True
    ren_id_board[ren_id_board == large_id] = small_id

    liberty[large_id][_xy] = liberty[small_id][_xy] = False
    liberty[small_id] = liberty[small_id] | liberty[large_id]
    liberty[large_id] = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=bool)

    return state, small_id


def _remove_stones(
    _state: MiniGoState,
    _my_color: int,
    _oppo_ren_id: int,
    _agehama: int,
    _around_xy,
) -> Tuple[MiniGoState, int, int]:
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
    a_removed_stone_xy = _around_xy

    # 空けたところを自軍の呼吸点に追加
    liberty = _add_removed_pos_to_liberty(
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
) -> np.ndarray:
    ren_id_board = _ren_id_board.copy()
    liberty = _liberty.copy()
    for _xy in range(BOARD_SIZE * BOARD_SIZE):
        for _nsew in NSEW:
            _around_rmstone_pos = _xy_to_pos(_xy) + _nsew
            if (  # 取り除かれた石に隣接する連の場合
                not _is_off_board(_around_rmstone_pos)
                and ren_id_board[_pos_to_xy(_around_rmstone_pos)] != -1
                and _surrounded_stones[_xy]
            ):
                # 呼吸点を追加
                liberty = _update_liberty(
                    liberty, ren_id_board[_pos_to_xy(_around_rmstone_pos)], _xy
                )
            else:
                liberty = liberty

    return liberty


def _update_liberty(
    _liberty: np.ndarray, _id: int, _xy: int, _bool: bool = True
) -> np.ndarray:
    liberty = _liberty.copy()
    liberty[_id][_xy] = _bool
    return liberty


def _not_suicide(
    _state: MiniGoState,
    _my_color: int,
    _agehama: int,
    _is_kou: bool,
    _a_removed_stone_xy: int,
) -> Tuple[MiniGoState, np.ndarray, bool]:
    state = copy.deepcopy(_state)

    # コウの確認
    if _agehama == 1 and _is_kou:
        state.kou[0], state.kou[1] = (
            _a_removed_stone_xy // BOARD_SIZE,
            _a_removed_stone_xy % BOARD_SIZE,
        )
    else:
        state.kou[0], state.kou[1] = (-1, -1)

    state.agehama[_my_color] += _agehama
    state = _add_turn(state)

    return state, np.array([0, 0]), False


def legal_actions(state: MiniGoState) -> np.ndarray:
    legal_action = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=bool)
    for xy in range(BOARD_SIZE * BOARD_SIZE):
        _, _, is_illegal = step(state, xy)
        legal_action[xy] = not is_illegal
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


def _is_off_board(_pos: np.ndarray) -> bool:
    return (
        _pos[0] < 0
        or BOARD_SIZE <= _pos[0]
        or _pos[1] < 0
        or BOARD_SIZE <= _pos[1]
    )


def _pos_to_xy(_pos: np.ndarray) -> int:
    return _pos[0] * BOARD_SIZE + _pos[1]


def _to_xy(_x, _y) -> int:
    return _x * BOARD_SIZE + _y


def _xy_to_pos(_xy) -> np.ndarray:
    return np.array([_xy // BOARD_SIZE, _xy % BOARD_SIZE])


def _opponent_color(_color: int) -> int:
    return (_color + 1) % 2


def _check_kou(_state: MiniGoState, _x, _y, _oppo_color) -> bool:
    return (
        (_x < 0 or _state.ren_id_board[_oppo_color][_to_xy(_x - 1, _y)] != -1)
        and (
            _x >= BOARD_SIZE - 1
            or _state.ren_id_board[_oppo_color][_to_xy(_x + 1, _y)] != -1
        )
        and (
            _y < 0
            or _state.ren_id_board[_oppo_color][_to_xy(_x, _y - 1)] != -1
        )
        and (
            _y >= BOARD_SIZE - 1
            or _state.ren_id_board[_oppo_color][_to_xy(_x, _y + 1)] != -1
        )
    )


def _get_reward(_state: MiniGoState) -> np.ndarray:
    b = _count_ji(_state)[BLACK] - _state.agehama[WHITE]
    w = _count_ji(_state)[WHITE] - _state.agehama[BLACK]
    if b == w:
        r = np.array([0, 0])
    else:
        r = np.array([-1, 1])

    if b > w:
        r = np.array([1, -1])
    else:
        r = r

    return r


def _count_ji(_state: MiniGoState) -> np.ndarray:
    board = get_board(_state)
    ji_id_board = _get_ji_id_board(_state)

    # -1:未確定 0:黒 1:白 2:どちらでもないことが確定
    color_of_ji = np.full((BOARD_SIZE * BOARD_SIZE), -1, dtype=int)

    for xy in range(BOARD_SIZE * BOARD_SIZE):
        if ji_id_board[xy] == -1 or color_of_ji[ji_id_board[xy]] == 2:
            # その点(xy)が空点でなかったり、どちらの地でもないことが確定なら何もしない
            color_of_ji = color_of_ji
        else:
            color_of_ji = _check_around_ji(color_of_ji, board, ji_id_board, xy)

    b = np.count_nonzero(color_of_ji == BLACK)
    w = np.count_nonzero(color_of_ji == WHITE)

    return np.array([b, w])


def _check_around_ji(
    _color_of_ji: np.ndarray,
    _board: np.ndarray,
    _ji_id_board: np.ndarray,
    _xy: int,
) -> np.ndarray:
    color_of_ji = _color_of_ji.copy()
    board = _board.copy()
    ji_id_board = _ji_id_board.copy()

    # 周囲の石が白か黒か判断
    for nsew in NSEW:
        around_pos = _xy_to_pos(_xy) + nsew
        around_xy = _pos_to_xy(around_pos)
        if _is_off_board(around_pos) or board[around_xy] == POINT:  # 周囲に石なし
            color_of_ji = color_of_ji
        elif color_of_ji[_xy] == -1:  # 色が未知の場合、その色を登録
            color_of_ji = _update_color_of_ji(
                color_of_ji, _xy, board[around_xy]
            )
        elif color_of_ji[_xy] == _opponent_color(board[around_xy]):
            # 既に登録された色と異なる場合、どちらでもないことが確定
            # そのidの地を全て2に
            color_of_ji = _update_color_of_ji_by_neutral(
                color_of_ji, ji_id_board == ji_id_board[_xy]
            )

    return color_of_ji


def _update_color_of_ji(
    _color_of_ji: np.ndarray, _xy: int, _num: int
) -> np.ndarray:
    color_of_ji = _color_of_ji.copy()
    color_of_ji[_xy] = _num
    return color_of_ji


def _update_color_of_ji_by_neutral(
    _color_of_ji: np.ndarray, _cond: np.ndarray
) -> np.ndarray:
    color_of_ji = _color_of_ji.copy()
    color_of_ji[_cond] = 2
    return color_of_ji


# 以下の関数はstep()の_merge_ren()とほぼ同じことをしている
# 連ではなく地に対して同じようにidを振る
def _get_ji_id_board(_state: MiniGoState) -> np.ndarray:
    board = get_board(_state)
    ji_id_board: np.ndarray = np.full(BOARD_SIZE * BOARD_SIZE, -1, dtype=int)
    available_ji_id: np.ndarray = np.ones(BOARD_SIZE * BOARD_SIZE, dtype=bool)
    for xy in range(BOARD_SIZE * BOARD_SIZE):
        if board[xy] != POINT:
            ji_id_board, available_ji_id = ji_id_board, available_ji_id
        else:
            ji_id_board, available_ji_id = _check_around_points(
                ji_id_board, available_ji_id, xy
            )

    return ji_id_board


def _check_around_points(
    _ji_id_board: np.ndarray, _available_ji_id: np.ndarray, _xy: int
) -> Tuple[np.ndarray, np.ndarray]:
    ji_id_board = _ji_id_board.copy()
    available_ji_id = _available_ji_id.copy()
    new_id = int(np.argmax(available_ji_id))  # 最初にTrueになったindex
    ji_id_board[_xy] = new_id
    available_ji_id[new_id] = False

    for nsew in NSEW:
        around_mypos = _xy_to_pos(_xy) + nsew
        if (
            not _is_off_board(around_mypos)
            and ji_id_board[_pos_to_xy(around_mypos)] != -1
        ):
            (ji_id_board, available_ji_id, new_id) = _merge_points(
                ji_id_board,
                available_ji_id,
                new_id,
                _pos_to_xy(around_mypos),
            )
        else:
            ji_id_board, available_ji_id, new_id = (
                ji_id_board,
                available_ji_id,
                new_id,
            )

    return ji_id_board, available_ji_id


def _merge_points(
    _ji_id_board: np.ndarray,
    _available_ji_id: np.ndarray,
    _new_id: int,
    _xy_around_mypos: int,
):
    ji_id_board = _ji_id_board.copy()
    available_ji_id = _available_ji_id.copy()

    old_id = ji_id_board[_xy_around_mypos]
    if old_id == _new_id:  # 既に結合済みの場合
        board_and_available_and_id = ji_id_board, available_ji_id, _new_id
    else:
        board_and_available_and_id = __merge_points(
            ji_id_board, available_ji_id, old_id, _new_id
        )
    return board_and_available_and_id


def __merge_points(
    _ji_id_board: np.ndarray,
    _available_ji_id: np.ndarray,
    _old_id: int,
    _new_id: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    ji_id_board = _ji_id_board.copy()
    available_ji_id = _available_ji_id.copy()

    small_id = min(_old_id, _new_id)
    large_id = max(_old_id, _new_id)
    # 大きいidの連を消し、小さいidの連と繋げる

    available_ji_id[large_id] = True
    ji_id_board[ji_id_board == large_id] = small_id
    np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=bool)

    new_id = small_id

    return ji_id_board, available_ji_id, new_id
