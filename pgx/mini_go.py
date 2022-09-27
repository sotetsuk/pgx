import copy
from typing import Tuple

import jax
from flax import struct
from jax import numpy as jnp

BOARD_SIZE = 5

BLACK = 0
WHITE = 1
POINT = 2
BLACK_CHAR = "@"
WHITE_CHAR = "O"
POINT_CHAR = "+"

NSEW: jnp.ndarray = jnp.array(
    [[-1, 0], [1, 0], [0, 1], [0, -1]]
)  # type:ignore


@struct.dataclass
class MiniGoState:
    ren_id_board: jnp.ndarray = jnp.full(
        (2, BOARD_SIZE * BOARD_SIZE), -1, dtype=int
    )  # type:ignore
    available_ren_id: jnp.ndarray = jnp.ones(  # n番目の連idが使えるか
        (2, BOARD_SIZE * BOARD_SIZE), dtype=bool
    )
    liberty: jnp.ndarray = jnp.zeros(
        (2, BOARD_SIZE * BOARD_SIZE, BOARD_SIZE * BOARD_SIZE), dtype=bool
    )

    turn: jnp.ndarray = jnp.zeros(1, dtype=int)
    agehama: jnp.ndarray = jnp.zeros(2, dtype=int)  # [0]: 黒の得たアゲハマ, [1]: 白の方
    passed: jnp.ndarray = jnp.zeros(1, dtype=bool)  # 直前のactionがパスだとTrue
    kou: jnp.ndarray = jnp.full(  # コウによる着手禁止点, 無ければ(-1, -1)
        2, -1, dtype=int
    )  # type:ignore


@jax.jit
def init() -> MiniGoState:
    return MiniGoState()


@jax.jit
def step(
    state: MiniGoState, action: int
) -> Tuple[MiniGoState, jnp.ndarray, bool]:
    state = copy.deepcopy(state)

    step_result = jax.lax.cond(
        action < 0,
        lambda state, action: _pass_move(state),
        lambda state, action: _not_pass_move(state, action),
        state,
        action,
    )

    return step_result


@jax.jit
def _pass_move(_state: MiniGoState) -> Tuple[MiniGoState, jnp.ndarray, bool]:
    state = copy.deepcopy(_state)
    """
    if state.passed[0]:  # 2回連続でパスすると終局
        step_result = (_add_turn(state), _get_reward(state), True)
    else:
        step_result = (_add_turn(_add_pass(state)), jnp.array([0, 0]), False)
    """
    step_result = jax.lax.cond(
        state.passed[0],
        lambda state: (_add_turn(state), _get_reward(state), True),
        lambda state: (_add_pass(state), jnp.array([0, 0]), False),
        state,
    )
    return step_result


@jax.jit
def _add_turn(_state: MiniGoState) -> MiniGoState:
    state = MiniGoState(  # type:ignore
        ren_id_board=_state.ren_id_board,
        available_ren_id=_state.available_ren_id,
        liberty=_state.liberty,
        turn=_state.turn + 1,
        agehama=_state.agehama,
        passed=_state.passed,
        kou=_state.kou,
    )
    return state


@jax.jit
def _add_pass(_state: MiniGoState) -> MiniGoState:
    state = MiniGoState(  # type:ignore
        ren_id_board=_state.ren_id_board,
        available_ren_id=_state.available_ren_id,
        liberty=_state.liberty,
        turn=_state.turn + 1,
        agehama=_state.agehama,
        passed=_state.passed.at[0].set(True),
        kou=_state.kou,
    )
    return state


@jax.jit
def _not_pass_move(
    _state: MiniGoState, _action: int
) -> Tuple[MiniGoState, jnp.ndarray, bool]:
    state = MiniGoState(  # type:ignore
        ren_id_board=_state.ren_id_board,
        available_ren_id=_state.available_ren_id,
        liberty=_state.liberty,
        turn=_state.turn,
        agehama=_state.agehama,
        passed=_state.passed.at[0].set(False),
        kou=_state.kou,
    )

    xy = _action
    my_color = _state.turn[0] % 2

    """
    # 石を置く
    if (
        state.ren_id_board[my_color][xy] != -1
        or state.ren_id_board[_opponent_color(my_color)][xy] != -1
        or (xy == _pos_to_xy(state.kou))
    ):  # 既に他の石が置かれている or コウ
        step_result = _illegal_move(state)
    else:
        step_result = _not_duplicate_nor_kou(state, xy, my_color)
    """

    step_result = jax.lax.cond(
        jnp.logical_or(
            jnp.logical_or(
                state.ren_id_board[my_color][xy] != -1,
                state.ren_id_board[_opponent_color(my_color)][xy] != -1,
            ),
            (xy == _pos_to_xy(state.kou)),
        ),
        lambda state, xy, my_color: _illegal_move(state),
        lambda state, xy, my_color: _not_duplicate_nor_kou(
            state, xy, my_color
        ),
        state,
        xy,
        my_color,
    )

    return step_result


@jax.jit
def _not_duplicate_nor_kou(
    _state: MiniGoState, _xy: int, _my_color
) -> Tuple[MiniGoState, jnp.ndarray, bool]:
    x = _xy // BOARD_SIZE
    y = _xy % BOARD_SIZE

    # 最初にTrueになったindexをidとする
    new_id = jnp.argmax(_state.available_ren_id[_my_color])
    agehama = 0
    a_removed_stone_xy = -1  # コウのために取った位置を記憶する
    # 石を置く
    state = MiniGoState(  # type:ignore
        ren_id_board=_state.ren_id_board.at[_my_color, _xy].set(new_id),
        available_ren_id=_state.available_ren_id.at[_my_color, new_id].set(
            False
        ),
        liberty=_state.liberty,
        turn=_state.turn,
        agehama=_state.agehama,
        passed=_state.passed,
        kou=_state.kou,
    )

    is_kou = _check_kou(state, x, y, _opponent_color(_my_color))

    # 周囲の連を数える
    for nsew in NSEW:
        around_mypos = jnp.array([x, y]) + nsew
        """
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
        """
        state, new_id, agehama, a_removed_stone_xy = jax.lax.cond(
            _is_off_board(around_mypos),
            lambda state, _xy, around_mypos, _my_color, new_id, agehama, a_removed_stone_xy: (
                state,
                new_id,
                agehama,
                a_removed_stone_xy,
            ),
            lambda state, _xy, around_mypos, _my_color, new_id, agehama, a_removed_stone_xy: _check_around_stones(
                state,
                _xy,
                around_mypos,
                _my_color,
                new_id,
                agehama,
                a_removed_stone_xy,
            ),
            state,
            _xy,
            around_mypos,
            _my_color,
            new_id,
            agehama,
            a_removed_stone_xy,
        )
    """
    if jnp.count_nonzero(state.liberty[_my_color][new_id]) == 0:  # 自殺手
        step_result = _illegal_move(state)
    else:
        step_result = _not_suicide(
            state, _my_color, agehama, is_kou, a_removed_stone_xy
        )
    """
    step_result = jax.lax.cond(
        jnp.count_nonzero(state.liberty[_my_color][new_id]) == 0,
        lambda state, _my_color, agehama, is_kou, a_removed_stone_xy: _illegal_move(
            state
        ),
        lambda state, _my_color, agehama, is_kou, a_removed_stone_xy: _not_suicide(
            state, _my_color, agehama, is_kou, a_removed_stone_xy
        ),
        state,
        _my_color,
        agehama,
        is_kou,
        a_removed_stone_xy,
    )
    return step_result


@jax.jit
def _illegal_move(
    _state: MiniGoState,
) -> Tuple[MiniGoState, jnp.ndarray, bool]:
    r: jnp.ndarray = jnp.array([1, 1])  # type:ignore
    r = r.at[_state.turn[0] % 2].set(-1)
    return _add_turn(_state), r, True


@jax.jit
def _check_around_stones(
    _state: MiniGoState,
    _xy: int,
    _around_mypos: jnp.ndarray,
    _my_color: int,
    _new_id: int,
    _agehama: int,
    _a_removed_stone_xy: int,
) -> Tuple[MiniGoState, int, int, int]:
    state = copy.deepcopy(_state)
    around_xy = _pos_to_xy(_around_mypos)
    oppo_color = _opponent_color(_my_color)

    # 既に自分の連が作られていた場合
    """
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
    """
    state, new_id = jax.lax.cond(
        _state.ren_id_board[_my_color, _pos_to_xy(_around_mypos)] != -1,
        lambda state, _my_color, _new_id, _xy, around_xy: _merge_ren(
            state,
            _my_color,
            _new_id,
            _xy,
            around_xy,
        ),
        lambda state, _my_color, _new_id, _xy, around_xy: (state, _new_id),
        state,
        _my_color,
        _new_id,
        _xy,
        around_xy,
    )

    # 敵の連が作られていた場合
    """
    if state.ren_id_board[oppo_color][around_xy] != -1:
        state.liberty[oppo_color] = _update_liberty(
            state.liberty[oppo_color],
            state.ren_id_board[oppo_color][around_xy],
            _xy,
            False,
        )
    else:
        state.liberty[oppo_color] = state.liberty[oppo_color]
    """
    _liberty = jax.lax.cond(
        state.ren_id_board[oppo_color][around_xy] != -1,
        lambda state, oppo_color, around_xy, _xy: _update_liberty(
            state.liberty[oppo_color],
            state.ren_id_board[oppo_color][around_xy],
            _xy,
            False,
        ),
        lambda state, oppo_color, around_xy, _xy: state.liberty[oppo_color],
        state,
        oppo_color,
        around_xy,
        _xy,
    )
    state = MiniGoState(  # type:ignore
        ren_id_board=state.ren_id_board,
        available_ren_id=state.available_ren_id,
        liberty=state.liberty.at[oppo_color].set(_liberty),
        turn=state.turn,
        agehama=state.agehama,
        passed=state.passed,
        kou=state.kou,
    )

    # 敵の連を取れる場合
    """
    if (
        state.ren_id_board[oppo_color][around_xy] != -1
        and jnp.count_nonzero(
            state.liberty[oppo_color][
                state.ren_id_board[oppo_color][around_xy]
            ]
        )
        == 0
    ):
        # 石を取る
        (state, _a_removed_stone_xy, _agehama) = _remove_stones(
            state,
            _my_color,
            state.ren_id_board[oppo_color][around_xy],
            _agehama,
            around_xy,
        )
    else:
        state, _a_removed_stone_xy, _agehama = (
            state,
            _a_removed_stone_xy,
            _agehama,
        )
    """
    (state, _a_removed_stone_xy, _agehama) = jax.lax.cond(
        jnp.logical_and(
            state.ren_id_board[oppo_color][around_xy] != -1,
            jnp.count_nonzero(
                state.liberty[oppo_color][
                    state.ren_id_board[oppo_color][around_xy]
                ]
            )
            == 0,
        ),
        lambda state, _my_color, _agehama, around_xy, _a_removed_stone_xy: _remove_stones(
            state,
            _my_color,
            state.ren_id_board[oppo_color][around_xy],
            _agehama,
            around_xy,
        ),
        lambda state, _my_color, _agehama, around_xy, _a_removed_stone_xy: (
            state,
            _a_removed_stone_xy,
            _agehama,
        ),
        state,
        _my_color,
        _agehama,
        around_xy,
        _a_removed_stone_xy,
    )

    # どちらでもない場合
    """
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
    """
    _liberty = jax.lax.cond(
        jnp.logical_and(
            state.ren_id_board[_my_color][around_xy] == -1,
            state.ren_id_board[oppo_color][around_xy] == -1,
        ),
        lambda state, _my_color, new_id, around_xy: _update_liberty(
            state.liberty[_my_color], new_id, around_xy
        ),
        lambda state, _my_color, new_id, around_xy: state.liberty[_my_color],
        state,
        _my_color,
        new_id,
        around_xy,
    )
    state = MiniGoState(  # type:ignore
        ren_id_board=state.ren_id_board,
        available_ren_id=state.available_ren_id,
        liberty=state.liberty.at[_my_color].set(_liberty),
        turn=state.turn,
        agehama=state.agehama,
        passed=state.passed,
        kou=state.kou,
    )

    return state, new_id, _agehama, _a_removed_stone_xy


@jax.jit
def _merge_ren(
    _state: MiniGoState,
    _my_color: int,
    _new_id: int,
    _xy: int,
    _xy_around_mypos: int,
) -> Tuple[MiniGoState, int]:
    # sys.exit()
    state = copy.deepcopy(_state)
    ren_id_board = state.ren_id_board[_my_color]
    new_id = _new_id

    old_id = ren_id_board[_xy_around_mypos]

    """
    if old_id == new_id:  # 既に結合済みの場合
        state_and_new_id = state, new_id
    else:
        state_and_new_id = __merge_ren(state, _my_color, old_id, new_id, _xy)
    """
    state_and_new_id = jax.lax.cond(
        old_id == new_id,
        lambda state, _my_color, old_id, new_id, _xy: (state, new_id),
        lambda state, _my_color, old_id, new_id, _xy: __merge_ren(
            state, _my_color, old_id, new_id, _xy
        ),
        state,
        _my_color,
        old_id,
        new_id,
        _xy,
    )

    return state_and_new_id


@jax.jit
def __merge_ren(
    _state: MiniGoState, _my_color: int, _old_id: int, _new_id: int, _xy: int
) -> Tuple[MiniGoState, int]:
    state = copy.deepcopy(_state)
    ren_id_board = state.ren_id_board[_my_color]
    available_ren_id = state.available_ren_id[_my_color]
    liberty = state.liberty[_my_color]

    # small_id = jnp.min(_old_id, _new_id)
    # large_id = jnp.max(_old_id, _new_id)
    small_id, large_id = jax.lax.cond(
        _old_id < _new_id,
        lambda _: (_old_id, _new_id),
        lambda _: (_new_id, _old_id),
        0,
    )

    # 大きいidの連を消し、小さいidの連と繋げる
    available_ren_id = available_ren_id.at[large_id].set(True)
    # ren_id_board = ren_id_board.at[ren_id_board == large_id].set(small_id)
    ren_id_board = jnp.where(ren_id_board == large_id, small_id, ren_id_board)

    liberty = liberty.at[large_id, _xy].set(False)
    liberty = liberty.at[small_id, _xy].set(False)
    liberty = liberty.at[small_id].set(liberty[small_id] | liberty[large_id])
    liberty = liberty.at[large_id].set(
        jnp.zeros(BOARD_SIZE * BOARD_SIZE, dtype=bool)
    )

    state = MiniGoState(  # type:ignore
        ren_id_board=_state.ren_id_board.at[_my_color].set(ren_id_board),
        available_ren_id=_state.available_ren_id.at[_my_color].set(
            available_ren_id
        ),
        liberty=_state.liberty.at[_my_color].set(liberty),
        turn=_state.turn,
        agehama=_state.agehama,
        passed=_state.passed,
        kou=_state.kou,
    )

    return state, small_id


@jax.jit
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
    agehama += jnp.count_nonzero(surrounded_stones)  # その石の数
    oppo_ren_id_board = jnp.where(
        surrounded_stones, -1, oppo_ren_id_board
    )  # ren_id_boardから削除
    oppo_available_ren_id = oppo_available_ren_id.at[_oppo_ren_id].set(
        True
    )  # available_ren_idに追加
    a_removed_stone_xy = _around_xy

    # 空けたところを自軍の呼吸点に追加
    liberty = _add_removed_pos_to_liberty(
        ren_id_board, liberty, surrounded_stones
    )

    state = MiniGoState(  # type:ignore
        ren_id_board=_state.ren_id_board.at[oppo_color].set(oppo_ren_id_board),
        available_ren_id=_state.available_ren_id.at[oppo_color].set(
            oppo_available_ren_id
        ),
        liberty=_state.liberty.at[_my_color].set(liberty),
        turn=_state.turn,
        agehama=_state.agehama,
        passed=_state.passed,
        kou=_state.kou,
    )
    return (
        state,
        a_removed_stone_xy,
        agehama,
    )


@jax.jit
def _add_removed_pos_to_liberty(
    _ren_id_board: jnp.ndarray,
    _liberty: jnp.ndarray,
    _surrounded_stones: jnp.ndarray,
) -> jnp.ndarray:
    ren_id_board = _ren_id_board.copy()
    liberty = _liberty.copy()
    for _xy in range(BOARD_SIZE * BOARD_SIZE):
        for _nsew in NSEW:
            _around_rmstone_pos = _xy_to_pos(_xy) + _nsew
            """
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
            """
            liberty = jax.lax.cond(
                jnp.logical_and(
                    jnp.logical_and(
                        _is_on_board(_around_rmstone_pos),
                        ren_id_board[_pos_to_xy(_around_rmstone_pos)] != -1,
                    ),
                    _surrounded_stones[_xy],
                ),
                lambda liberty, ren_id_board, _around_rmstone_pos, _xy: _update_liberty(
                    liberty, ren_id_board[_pos_to_xy(_around_rmstone_pos)], _xy
                ),
                lambda liberty, ren_id_board, _around_rmstone_pos, _xy: liberty,
                liberty,
                ren_id_board,
                _around_rmstone_pos,
                _xy,
            )

    return liberty


@jax.jit
def _update_liberty(
    _liberty: jnp.ndarray, _id: int, _xy: int, _bool: bool = True
) -> jnp.ndarray:
    liberty = _liberty.copy()
    liberty = liberty.at[_id, _xy].set(_bool)
    return liberty


@jax.jit
def _not_suicide(
    _state: MiniGoState,
    _my_color: int,
    _agehama: int,
    _is_kou: bool,
    _a_removed_stone_xy: int,
) -> Tuple[MiniGoState, jnp.ndarray, bool]:
    state = copy.deepcopy(_state)

    # コウの確認
    kou = jax.lax.cond(
        jnp.logical_and(_agehama == 1, _is_kou),
        lambda _: jnp.array(
            [
                _a_removed_stone_xy // BOARD_SIZE,
                _a_removed_stone_xy % BOARD_SIZE,
            ]
        ),
        lambda _: jnp.array([-1, -1]),
        0,
    )

    agehama = state.agehama.at[_my_color].set(
        state.agehama[_my_color] + _agehama
    )
    state = MiniGoState(  # type:ignore
        ren_id_board=state.ren_id_board,
        available_ren_id=state.available_ren_id,
        liberty=state.liberty,
        turn=state.turn,
        agehama=agehama,
        passed=state.passed,
        kou=kou,
    )
    state = _add_turn(state)

    return state, jnp.array([0, 0]), False  # type:ignore


@jax.jit
def legal_actions(state: MiniGoState) -> jnp.ndarray:
    legal_action = jnp.ones(BOARD_SIZE * BOARD_SIZE, dtype=bool)
    for xy in range(BOARD_SIZE * BOARD_SIZE):
        _, _, is_illegal = step(state, xy)
        legal_action = legal_action.at[xy].set(is_illegal)
    return jnp.where(legal_action, False, True)  # type:ignore


@jax.jit
def get_board(state: MiniGoState) -> jnp.ndarray:
    board: jnp.ndarray = jnp.full(BOARD_SIZE * BOARD_SIZE, 2)  # type:ignore
    # board=board.at[state.ren_id_board[BLACK] != -1].set(0)
    # board=board.at[state.ren_id_board[WHITE] != -1].set(1)
    for xy in range(BOARD_SIZE * BOARD_SIZE):
        color = jax.lax.cond(
            state.ren_id_board[BLACK][xy] != -1,
            lambda _: 0,
            lambda _: jax.lax.cond(
                state.ren_id_board[WHITE][xy] != -1,
                lambda _: 1,
                lambda _: 2,
                0,
            ),
            0,
        )
        board = board.at[xy].set(color)

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


@jax.jit
def _is_off_board(_pos: jnp.ndarray) -> bool:
    return jnp.logical_not(_is_on_board(_pos))


@jax.jit
def _is_on_board(_pos: jnp.ndarray) -> bool:
    x = _pos[0]
    y = _pos[1]
    is_on_board = jax.lax.cond(
        jnp.logical_and(
            jnp.logical_and(x >= 0, BOARD_SIZE > x),
            jnp.logical_and(y >= 0, BOARD_SIZE > y),
        ),
        lambda _: True,
        lambda _: False,
        0,
    )
    return is_on_board


@jax.jit
def _pos_to_xy(_pos: jnp.ndarray) -> int:
    return _pos[0] * BOARD_SIZE + _pos[1]


@jax.jit
def _to_xy(_x, _y) -> int:
    return _x * BOARD_SIZE + _y


@jax.jit
def _xy_to_pos(_xy) -> jnp.ndarray:
    return jnp.array([_xy // BOARD_SIZE, _xy % BOARD_SIZE])  # type:ignore


@jax.jit
def _opponent_color(_color: int) -> int:
    return (_color + 1) % 2


@jax.jit
def _check_kou(_state: MiniGoState, _x, _y, _oppo_color) -> bool:
    return jnp.logical_and(
        jnp.logical_and(
            jnp.logical_and(
                jnp.logical_or(
                    _x < 0,
                    _state.ren_id_board[_oppo_color][_to_xy(_x - 1, _y)] != -1,
                ),
                jnp.logical_or(
                    _x >= BOARD_SIZE - 1,
                    _state.ren_id_board[_oppo_color][_to_xy(_x + 1, _y)] != -1,
                ),
            ),
            jnp.logical_or(
                _y < 0,
                _state.ren_id_board[_oppo_color][_to_xy(_x, _y - 1)] != -1,
            ),
        ),
        jnp.logical_or(
            _y >= BOARD_SIZE - 1,
            _state.ren_id_board[_oppo_color][_to_xy(_x, _y + 1)] != -1,
        ),
    )


@jax.jit
def _get_reward(_state: MiniGoState) -> jnp.ndarray:
    b = _count_ji(_state)[BLACK] - _state.agehama[WHITE]
    w = _count_ji(_state)[WHITE] - _state.agehama[BLACK]
    """
    if b == w:
        r = jnp.array([0, 0])
    else:
        r = jnp.array([-1, 1])

    if b > w:
        r = jnp.array([1, -1])
    else:
        r = r
    """
    r = jax.lax.cond(
        b == w, lambda _: jnp.array([0, 0]), lambda _: jnp.array([-1, 1]), 0
    )
    r = jax.lax.cond(b > w, lambda r: jnp.array([1, -1]), lambda r: r, r)

    return r


@jax.jit
def _count_ji(_state: MiniGoState) -> jnp.ndarray:
    board = get_board(_state)
    ji_id_board = _get_ji_id_board(_state)

    # -1:未確定 0:黒 1:白 2:どちらでもないことが確定
    color_of_ji: jnp.ndarray = jnp.full(
        (BOARD_SIZE * BOARD_SIZE), -1, dtype=int
    )  # type:ignore

    for xy in range(BOARD_SIZE * BOARD_SIZE):
        """
        if ji_id_board[xy] == -1 or color_of_ji[ji_id_board[xy]] == 2:
            # その点(xy)が空点でなかったり、どちらの地でもないことが確定なら何もしない
            color_of_ji = color_of_ji
        else:
            color_of_ji = _check_around_ji(color_of_ji, board, ji_id_board, xy)
        """
        color_of_ji = jax.lax.cond(
            jnp.logical_or(
                ji_id_board[xy] == -1, color_of_ji[ji_id_board[xy]] == 2
            ),
            lambda color_of_ji, board, ji_id_board, xy: color_of_ji,
            lambda color_of_ji, board, ji_id_board, xy: _check_around_ji(
                color_of_ji, board, ji_id_board, xy
            ),
            color_of_ji,
            board,
            ji_id_board,
            xy,
        )

    b = jnp.count_nonzero(color_of_ji == BLACK)
    w = jnp.count_nonzero(color_of_ji == WHITE)

    return jnp.array([b, w])  # type:ignore


@jax.jit
def _check_around_ji(
    _color_of_ji: jnp.ndarray,
    _board: jnp.ndarray,
    _ji_id_board: jnp.ndarray,
    _xy: int,
) -> jnp.ndarray:
    color_of_ji = _color_of_ji.copy()
    board = _board.copy()
    ji_id_board = _ji_id_board.copy()

    # 周囲の石が白か黒か判断
    for nsew in NSEW:
        around_pos = _xy_to_pos(_xy) + nsew
        around_xy = _pos_to_xy(around_pos)
        """
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
        """
        color_of_ji = jax.lax.cond(
            jnp.logical_or(
                _is_off_board(around_pos), board[around_xy] == POINT
            ),
            lambda color_of_ji, board, around_xy, ji_id_board, _xy: color_of_ji,
            lambda color_of_ji, board, around_xy, ji_id_board, _xy: jax.lax.cond(
                color_of_ji[_xy] == -1,
                lambda color_of_ji, board, around_xy, ji_id_board, _xy: _update_color_of_ji(
                    color_of_ji, _xy, board[around_xy]
                ),
                lambda color_of_ji, board, around_xy, ji_id_board, _xy: jax.lax.cond(
                    color_of_ji[_xy] == _opponent_color(board[around_xy]),
                    lambda color_of_ji, board, around_xy, ji_id_board, _xy: _update_color_of_ji_by_neutral(
                        color_of_ji, ji_id_board == ji_id_board[_xy]
                    ),
                    lambda color_of_ji, board, around_xy, ji_id_board, _xy: color_of_ji,
                    color_of_ji,
                    board,
                    around_xy,
                    ji_id_board,
                    _xy,
                ),
                color_of_ji,
                board,
                around_xy,
                ji_id_board,
                _xy,
            ),
            color_of_ji,
            board,
            around_xy,
            ji_id_board,
            _xy,
        )

    return color_of_ji


@jax.jit
def _update_color_of_ji(
    _color_of_ji: jnp.ndarray, _xy: int, _num: int
) -> jnp.ndarray:
    color_of_ji = _color_of_ji.copy()
    color_of_ji = color_of_ji.at[_xy].set(_num)
    return color_of_ji


@jax.jit
def _update_color_of_ji_by_neutral(
    _color_of_ji: jnp.ndarray, _cond: jnp.ndarray
) -> jnp.ndarray:
    # color_of_ji = _color_of_ji.copy()
    # color_of_ji = color_of_ji.at[_cond].set(2)
    return jnp.where(_cond, 2, _color_of_ji)  # type:ignore


# 以下の関数はstep()の_merge_ren()とほぼ同じことをしている
# 連ではなく地に対して同じようにidを振る
@jax.jit
def _get_ji_id_board(_state: MiniGoState) -> jnp.ndarray:
    board = get_board(_state)
    ji_id_board: jnp.ndarray = jnp.full(
        BOARD_SIZE * BOARD_SIZE, -1, dtype=int
    )  # type:ignore
    available_ji_id: jnp.ndarray = jnp.ones(
        BOARD_SIZE * BOARD_SIZE, dtype=bool
    )
    for xy in range(BOARD_SIZE * BOARD_SIZE):
        """
        if board[xy] != POINT:
            ji_id_board, available_ji_id = ji_id_board, available_ji_id
        else:
            ji_id_board, available_ji_id = _check_around_points(
                ji_id_board, available_ji_id, xy
            )
        """
        ji_id_board, available_ji_id = jax.lax.cond(
            board[xy] == POINT,
            lambda ji_id_board, available_ji_id, xy: _check_around_points(
                ji_id_board, available_ji_id, xy
            ),
            lambda ji_id_board, available_ji_id, xy: (
                ji_id_board,
                available_ji_id,
            ),
            ji_id_board,
            available_ji_id,
            xy,
        )

    return ji_id_board


@jax.jit
def _check_around_points(
    _ji_id_board: jnp.ndarray, _available_ji_id: jnp.ndarray, _xy: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    ji_id_board: jnp.ndarray = _ji_id_board.copy()
    available_ji_id: jnp.ndarray = _available_ji_id.copy()
    new_id = jnp.argmax(available_ji_id)  # 最初にTrueになったindex
    ji_id_board = ji_id_board.at[_xy].set(new_id)
    available_ji_id = available_ji_id.at[new_id].set(False)

    for nsew in NSEW:
        around_mypos = _xy_to_pos(_xy) + nsew
        """
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
        """
        (ji_id_board, available_ji_id, new_id) = jax.lax.cond(
            jnp.logical_and(
                _is_on_board(around_mypos),
                ji_id_board[_pos_to_xy(around_mypos)] != -1,
            ),
            lambda ji_id_board, available_ji_id, new_id, around_mypos: _merge_points(
                ji_id_board,
                available_ji_id,
                new_id,
                _pos_to_xy(around_mypos),
            ),
            lambda ji_id_board, available_ji_id, new_id, around_mypos: (
                ji_id_board,
                available_ji_id,
                new_id,
            ),
            ji_id_board,
            available_ji_id,
            new_id,
            around_mypos,
        )

    return ji_id_board, available_ji_id


@jax.jit
def _merge_points(
    _ji_id_board: jnp.ndarray,
    _available_ji_id: jnp.ndarray,
    _new_id: int,
    _xy_around_mypos: int,
):
    ji_id_board = _ji_id_board.copy()
    available_ji_id = _available_ji_id.copy()

    old_id = ji_id_board[_xy_around_mypos]
    """
    if old_id == _new_id:  # 既に結合済みの場合
        board_and_available_and_id = ji_id_board, available_ji_id, _new_id
    else:
        board_and_available_and_id = __merge_points(
            ji_id_board, available_ji_id, old_id, _new_id
        )
    """
    board_and_available_and_id = jax.lax.cond(
        old_id == _new_id,
        lambda ji_id_board, available_ji_id, old_id, _new_id: (
            ji_id_board,
            available_ji_id,
            _new_id,
        ),
        lambda ji_id_board, available_ji_id, old_id, _new_id: __merge_points(
            ji_id_board, available_ji_id, old_id, _new_id
        ),
        ji_id_board,
        available_ji_id,
        old_id,
        _new_id,
    )
    return board_and_available_and_id


@jax.jit
def __merge_points(
    _ji_id_board: jnp.ndarray,
    _available_ji_id: jnp.ndarray,
    _old_id: int,
    _new_id: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, int]:
    ji_id_board: jnp.ndarray = _ji_id_board.copy()
    available_ji_id: jnp.ndarray = _available_ji_id.copy()

    # small_id = jnp.min(_old_id, _new_id)
    # large_id = jnp.max(_old_id, _new_id)
    # 大きいidの連を消し、小さいidの連と繋げる
    small_id, large_id = jax.lax.cond(
        _old_id < _new_id,
        lambda _: (_old_id, _new_id),
        lambda _: (_new_id, _old_id),
        0,
    )

    available_ji_id = available_ji_id.at[large_id].set(True)
    # ji_id_board[ji_id_board == large_id] = small_id
    ji_id_board = jnp.where(
        ji_id_board == large_id, small_id, ji_id_board
    )  # type:ignore

    new_id = small_id

    return ji_id_board, available_ji_id, new_id
