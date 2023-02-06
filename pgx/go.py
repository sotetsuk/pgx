from functools import partial
from typing import Tuple

import jax
from flax import struct
from jax import numpy as jnp

BLACK = 0
WHITE = 1
POINT = 2
BLACK_CHAR = "@"
WHITE_CHAR = "O"
POINT_CHAR = "+"

dx = jnp.int32([-1, 0, +1, 0])
dy = jnp.int32([0, 1, 0, -1])
_dx = jnp.int32([-1, -2, -1, 0, +1, +2, +1, 0, -1])
_dy = jnp.int32([-1, 0, +1, +2, +1, 0, -1, -2, -1])

FALSE = jnp.bool_(False)
TRUE = jnp.bool_(True)


@struct.dataclass
class GoState:
    # 横幅, マスの数ではない
    size: jnp.ndarray = jnp.int32(19)  # type:ignore

    # 連の代表点（一番小さいマス目）のマス目の座標
    ren_id_board: jnp.ndarray = jnp.full(
        (2, 19 * 19), -1, dtype=jnp.int32
    )  # type:ignore

    # 連周りの情報 0:None 1:呼吸点 2:石
    liberty: jnp.ndarray = jnp.zeros((2, 19 * 19, 19 * 19), dtype=jnp.int32)

    # 設置可能なマスをTrueとしたマスク
    legal_action_mask: jnp.ndarray = jnp.ones(19 * 19 + 1, dtype=jnp.bool_)
    _legal_action_mask: jnp.ndarray = jnp.ones((2, 19 * 19), dtype=jnp.bool_)

    # 直近8回のログ
    game_log: jnp.ndarray = jnp.full(
        (8, 19 * 19), 2, dtype=jnp.int32
    )  # type:ignore

    # 経過ターン, 0始まり
    turn: jnp.ndarray = jnp.int32(0)  # type:ignore

    # プレイヤーID
    curr_player: jnp.ndarray = jnp.int32(0)  # type:ignore

    # [0]: 黒の得たアゲハマ, [1]: 白の方
    agehama: jnp.ndarray = jnp.zeros(2, dtype=jnp.int32)

    # 直前のactionがパスだとTrue
    passed: jnp.ndarray = jnp.bool_(False)  # type:ignore

    # コウによる着手禁止点(xy), 無ければ(-1)
    kou: jnp.ndarray = jnp.int32(-1)  # type:ignore

    # コミ
    komi: jnp.ndarray = jnp.float32(6.5)  # type:ignore

    # 終局判定
    terminated: jnp.ndarray = jnp.bool_(False)  # type:ignore

    data: jnp.ndarray = jnp.int32(-1)  # type:ignore


def observe(state: GoState, player_id, observe_all=False):
    return _get_alphazero_features(state, player_id, observe_all)


def _get_alphazero_features(state: GoState, player_id, observe_all):
    """
    17 x (size x size)
    0: player_idの石
    1: player_idの石(1手前)
    ...
    7: player_idの石(7手前)
    8: player_idの相手の石
    9: player_idの相手の石(1手前)
    ...
    15: player_idの石(7手前)
    16: player_idの色(黒:1, 白:0)

    e.g.
    size=5, player_id=0, white
    [[0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
     [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]
    """
    my_color = jax.lax.cond(
        player_id == state.curr_player,
        lambda: state.turn % 2,
        lambda: (state.turn + 1) % 2,
    )

    my_log = jax.lax.fori_loop(
        0,
        8,
        lambda i, boards: boards.at[i].set(
            jnp.where(boards[i] == my_color, 1, 0)
        ),
        state.game_log,
    )
    oppo_log = jax.lax.fori_loop(
        0,
        8,
        lambda i, boards: boards.at[i].set(
            jnp.where(boards[i] == (my_color + 1) % 2, 1, 0)
        ),
        state.game_log,
    )
    color = jnp.full_like(
        state.game_log[0], (my_color + 1) % 2
    )  # AlphaZeroでは黒だと1

    log = jnp.concatenate((my_log, oppo_log))
    return jnp.vstack([log, color])


def init(
    rng: jax.random.KeyArray, size: int = 19
) -> Tuple[jnp.ndarray, GoState]:
    curr_player = jnp.int32(jax.random.bernoulli(rng))
    return curr_player, GoState(  # type:ignore
        size=jnp.int32(size),  # type:ignore
        ren_id_board=jnp.full(
            (2, size * size), -1, dtype=jnp.int32
        ),  # type:ignore
        liberty=jnp.zeros((2, size * size, size * size), dtype=jnp.int32),
        legal_action_mask=jnp.ones(size * size + 1, dtype=jnp.bool_),
        _legal_action_mask=jnp.ones((2, size * size), dtype=jnp.bool_),
        game_log=jnp.full((8, size * size), 2, dtype=jnp.int32),  # type:ignore
        curr_player=curr_player,  # type:ignore
        data=-1,  # type:ignore
    )


def step(
    state: GoState, action: int, size: int = 19
) -> Tuple[jnp.ndarray, GoState, jnp.ndarray]:
    # update state
    _state, reward = _update_state_wo_legal_action(state, action, size)

    # add legal actions
    _state = _state.replace(  # type:ignore
        legal_action_mask=state.legal_action_mask.at[:-1].set(
            _legal_actions(_state, size)
        )
    )

    # update log
    new_log = jnp.roll(_state.game_log, size * size)
    new_log = new_log.at[0].set(get_board(_state))
    _state = _state.replace(game_log=new_log)  # type:ignore

    return (_state.curr_player, _state, reward)


def _update_state_wo_legal_action(
    _state: GoState, _action: int, _size: int
) -> Tuple[GoState, jnp.ndarray]:
    _state, _reward = jax.lax.cond(
        (_action < _size * _size),
        lambda: _not_pass_move(_state, _action),
        lambda: _pass_move(_state, _size),
    )

    # increase turn
    _state = _state.replace(turn=_state.turn + 1)  # type: ignore

    # change player
    _state = _state.replace(curr_player=(_state.curr_player + 1) % 2)  # type: ignore

    return _state, _reward


def _pass_move(_state: GoState, _size: int) -> Tuple[GoState, jnp.ndarray]:
    return jax.lax.cond(
        _state.passed,
        # 連続でパスならば終局
        lambda: (
            _state.replace(terminated=TRUE),  # type: ignore
            _get_reward(_state, _size),
        ),
        # 1回目のパスならばStateにパスを追加してそのまま続行
        lambda: (_state.replace(passed=True), jnp.array([0, 0])),  # type: ignore
    )


def _not_pass_move(
    _state: GoState, _action: int
) -> Tuple[GoState, jnp.ndarray]:
    state = _state.replace(passed=False)  # type: ignore
    xy = _action
    my_color = _my_color(state)
    agehama_before = state.agehama[my_color]
    is_illegal = _is_illegal_move(state, xy)  # 既に他の石が置かれている or コウ

    # 石を置く
    kou_occurred = _kou_occurred(state, xy)
    state = _set_stone(state, xy)

    # 周囲の連を調べ、libertyを更新
    state = jax.lax.fori_loop(
        0, 4, lambda i, s: _check_around_xy(i, s, xy), state
    )

    # legal_actionを更新
    state = _update_legal_action(state, xy)

    # 自殺手の判定
    put_ren_id = state.ren_id_board[my_color, xy]
    is_illegal = (
        jnp.count_nonzero(state.liberty[my_color, put_ren_id] == 1) == 0
    ) | is_illegal

    # コウの確認
    state = jax.lax.cond(
        kou_occurred & (state.agehama[my_color] - agehama_before == 1),
        lambda: state,
        lambda: state.replace(kou=-1),  # type:ignore
    )

    return jax.lax.cond(
        is_illegal,
        lambda: _illegal_move(state),
        lambda: (state, jnp.array([0, 0])),
    )


def _check_around_xy(i, state, xy):
    x = xy // state.size + dx[i]
    y = xy % state.size + dy[i]

    adj_xy = x * state.size + y
    is_off = _is_off_board(x, y, state.size)
    is_my_ren = state.ren_id_board[_my_color(state), adj_xy] != -1
    is_opp_ren = state.ren_id_board[_opponent_color(state), adj_xy] != -1
    replaced_state = state.replace(
        liberty=state.liberty.at[
            _my_color(state),
            state.ren_id_board[_my_color(state), xy],
            adj_xy,
        ].set(1)
    )  # type:ignore
    state = jax.lax.cond(
        ((~is_off) & (~is_my_ren) & (~is_opp_ren)),
        lambda: replaced_state,
        lambda: state,
    )
    state = jax.lax.cond(
        ((~is_off) & (~is_my_ren) & is_opp_ren),
        lambda: _set_stone_next_to_oppo_ren(state, xy, adj_xy),
        lambda: state,
    )
    state = jax.lax.cond(
        ((~is_off) & is_my_ren),
        lambda: _merge_ren(state, xy, adj_xy),
        lambda: state,
    )
    return state


def _is_illegal_move(_state: GoState, _xy):
    """
    既に石があるorコウ
    """
    my_color = _my_color(_state)
    oppo_color = _opponent_color(_state)
    return (
        (_state.ren_id_board[my_color, _xy] != -1)
        | (_state.ren_id_board[oppo_color, _xy] != -1)
        | (_xy == _state.kou)
    )


def _illegal_move(
    _state: GoState,
) -> Tuple[GoState, jnp.ndarray]:
    r: jnp.ndarray = jnp.array([1, 1])  # type:ignore
    return _state.replace(terminated=TRUE), r.at[_state.turn % 2].set(-1)  # type: ignore


def _set_stone(_state: GoState, _xy: int) -> GoState:
    return _state.replace(  # type:ignore
        ren_id_board=_state.ren_id_board.at[_my_color(_state), _xy].set(_xy),
    )


def _update_legal_action(_state: GoState, _xy: int) -> GoState:
    my_color = _my_color(_state)
    oppo_color = _opponent_color(_state)
    size = _state.size
    state = _state.replace(  # type:ignore
        _legal_action_mask=_state._legal_action_mask.at[:, _xy].set(FALSE)
    )

    # cf. #255
    # (A) 石を置くことで味方の自殺点が生じる場合
    put_ren_id = state.ren_id_board[my_color, _xy]
    state = jax.lax.cond(
        _is_one_liberty_ren(state, my_color, put_ren_id),
        lambda: _check_if_suicide_point_exist(state, my_color, put_ren_id),
        lambda: state,
    )

    # (B) 石を置くことで相手の自殺点が生じる場合
    # 1. 隣接する、既に存在する相手の連が呼吸点1つになる場合
    max_ren_num = size * size
    adj_stone = state.liberty[my_color, put_ren_id] == 2
    adj_ren_id = jnp.where(~adj_stone, -1, state.ren_id_board[oppo_color])
    state = jax.lax.fori_loop(
        0,
        max_ren_num,
        lambda i, state: _check_if_suicide_point_exist(
            state, oppo_color, adj_ren_id[i]
        ),
        state,
    )

    # 2. 空点の四方を囲む形になる場合
    x = _xy // size
    y = _xy % size
    state = jax.lax.fori_loop(
        0,
        4,
        lambda i, s: _check_atari(i, s, x, y, my_color),
        state,
    )

    return state


def _check_atari(i, state, x, y, my_color):
    oppo_color = (my_color + 1) % 2

    return jax.lax.cond(
        ~_is_off_board(x + dx[i], y + dy[i], state.size)
        & _is_point(state, x + dx[i], y + dy[i])
        & (
            _is_off_board(x + _dx[2 * i], y + _dy[2 * i], state.size)
            | _is_two_liberty_xy(
                state, x + _dx[2 * i], y + _dy[2 * i], my_color
            )
            | _is_one_liberty_xy(
                state, x + _dx[2 * i], y + _dy[2 * i], oppo_color
            )
        )
        & (
            _is_off_board(x + _dx[2 * i + 1], y + _dy[2 * i + 1], state.size)
            | _is_two_liberty_xy(
                state, x + _dx[2 * i + 1], y + _dy[2 * i + 1], my_color
            )
            | _is_one_liberty_xy(
                state, x + _dx[2 * i + 1], y + _dy[2 * i + 1], oppo_color
            )
        )
        & (
            _is_off_board(x + _dx[2 * i + 2], y + _dy[2 * i + 2], state.size)
            | _is_two_liberty_xy(
                state, x + _dx[2 * i + 2], y + _dy[2 * i + 2], my_color
            )
            | _is_one_liberty_xy(
                state, x + _dx[2 * i + 2], y + _dy[2 * i + 2], oppo_color
            )
        ),
        lambda: state.replace(
            _legal_action_mask=state._legal_action_mask.at[
                oppo_color, (x + dx[i]) * state.size + (y + dy[i])
            ].set(FALSE)
        ),
        lambda: state,
    )


def _is_one_liberty_ren(_state, _color, _ren_id):
    return (_ren_id >= 0) & (
        jnp.count_nonzero(_state.liberty[_color, _ren_id] == 1) == 1
    )


def _is_two_liberty_ren(_state, _color, _ren_id):
    return (_ren_id >= 0) & (
        jnp.count_nonzero(_state.liberty[_color, _ren_id] == 1) > 1
    )


def _is_one_liberty_xy(_state, x, y, color):
    ren_id = _state.ren_id_board[color, x * _state.size + y]
    return _is_one_liberty_ren(_state, color, ren_id)


def _is_two_liberty_xy(_state, x, y, color):
    ren_id = _state.ren_id_board[color, x * _state.size + y]
    return _is_two_liberty_ren(_state, color, ren_id)


def _is_point(_state, x, y):
    return (_state.ren_id_board[0, x * _state.size + y] == -1) & (
        _state.ren_id_board[1, x * _state.size + y] == -1
    )


def _check_if_suicide_point_exist(_state: GoState, _color, _id):
    # 置いた連の周りの呼吸点が1つの場合、その呼吸点(a)の周り四方を確認する
    # 四方に含まれる自分の連の呼吸点が全て1つだった場合、点(a)は自殺点
    liberty_points = _state.liberty[_color, _id] == 1
    is_one_liberty = jnp.count_nonzero(liberty_points) == 1
    # 呼吸点の位置(xy)
    liberty_xy = jnp.nonzero(liberty_points, size=1)[0]
    one_liberty_xy = liberty_xy[0]

    is_atari = is_one_liberty & _is_suicide_point(
        _state, _color, one_liberty_xy
    )
    _state = jax.lax.cond(
        (_id >= 0) & is_atari,
        lambda: _state.replace(  # type:ignore
            _legal_action_mask=_state._legal_action_mask.at[
                _color, one_liberty_xy
            ].set(FALSE)
        ),
        lambda: _state,
    )
    _state = jax.lax.cond(
        (_id >= 0) & ~is_atari,
        lambda: _state.replace(  # type:ignore
            _legal_action_mask=_state._legal_action_mask.at[_color].set(
                jnp.where(
                    liberty_points,
                    TRUE,
                    _state._legal_action_mask[_color],
                )
            )
        ),
        lambda: _state,
    )
    return _state


def _is_suicide_point(_state, _color, one_liberty_point):
    oppo_color = (_color + 1) % 2
    xy = one_liberty_point

    # 四方を確認する
    _is_suicide_point = TRUE
    _is_suicide_point = jax.lax.fori_loop(
        0,
        4,
        # 呼吸点 or 呼吸点2つ以上の味方連 or 呼吸点1つの相手連ならば自殺点ではない
        lambda i, is_suicide_point: jax.lax.cond(
            (
                ~_is_off_board(
                    xy // _state.size + dx[i],
                    xy % _state.size + dy[i],
                    _state.size,
                )
                & (
                    _is_point(
                        _state,
                        xy // _state.size + dx[i],
                        xy % _state.size + dy[i],
                    )
                    | _is_two_liberty_xy(
                        _state,
                        xy // _state.size + dx[i],
                        xy % _state.size + dy[i],
                        _color,
                    )
                    | _is_one_liberty_xy(
                        _state,
                        xy // _state.size + dx[i],
                        xy % _state.size + dy[i],
                        oppo_color,
                    )
                )
            ),
            lambda: FALSE,
            lambda: is_suicide_point,
        ),
        _is_suicide_point,
    )
    # is_suicide_point = TRUEならばその点は自殺点
    return _is_suicide_point


def _merge_ren(_state: GoState, _xy: int, _adj_xy: int):
    ren_id_board = _state.ren_id_board.at[_my_color(_state)].get()

    new_id = ren_id_board.at[_xy].get()
    adj_ren_id = ren_id_board.at[_adj_xy].get()

    small_id, large_id = jax.lax.cond(
        adj_ren_id < new_id,
        lambda: (adj_ren_id, new_id),
        lambda: (new_id, adj_ren_id),
    )
    # 大きいidの連を消し、小さいidの連と繋げる

    ren_id_board = jnp.where(ren_id_board == large_id, small_id, ren_id_board)

    liberty = _state.liberty.at[_my_color(_state)].get()
    liberty = liberty.at[large_id, _xy].set(0)
    liberty = liberty.at[small_id, _xy].set(0)
    liberty = liberty.at[small_id].set(
        jnp.maximum(liberty[small_id], liberty[large_id])
    )
    liberty = liberty.at[large_id, :].set(False)

    return jax.lax.cond(
        new_id == adj_ren_id,
        lambda: _state,
        lambda: _state.replace(  # type:ignore
            ren_id_board=_state.ren_id_board.at[_my_color(_state)].set(
                ren_id_board
            ),
            liberty=_state.liberty.at[_my_color(_state)].set(liberty),
        ),
    )


def _set_stone_next_to_oppo_ren(_state: GoState, _xy, _adj_xy):
    my_color = _my_color(_state)
    put_ren_id = _state.ren_id_board[my_color, _xy]
    oppo_ren_id = _state.ren_id_board.at[
        _opponent_color(_state), _adj_xy
    ].get()

    liberty = (
        _state.liberty.at[_opponent_color(_state), oppo_ren_id, _xy]
        .set(2)
        .at[
            my_color,
            put_ren_id,
            _adj_xy,
        ]
        .set(2)
    )

    state = _state.replace(liberty=liberty)  # type:ignore

    return jax.lax.cond(
        jnp.count_nonzero(
            state.liberty[_opponent_color(state), oppo_ren_id] == 1
        )
        == 0,
        lambda: _remove_stones(state, oppo_ren_id, _adj_xy),
        lambda: state,
    )


def _remove_stones(_state: GoState, _rm_ren_id, _rm_stone_xy) -> GoState:
    my_color = _my_color(_state)
    opp_color = _opponent_color(_state)
    surrounded_stones = _state.ren_id_board[opp_color] == _rm_ren_id
    agehama = jnp.count_nonzero(surrounded_stones)
    adj_stone = _state.liberty[opp_color, _rm_ren_id] == 2
    adj_ren_id = jnp.where(~adj_stone, -1, _state.ren_id_board[my_color])

    # 石を取り除く
    oppo_ren_id_board = jnp.where(
        surrounded_stones, -1, _state.ren_id_board[opp_color]
    )
    my_lib = _state.liberty[my_color]  # (2, 361, 361) => (361, 361)
    # surrounded_stones (361) => (my_lib > 0) & surrounded_stones (361, 361)
    liberty = jnp.where((my_lib > 0) & surrounded_stones, 1, my_lib)
    _state = _state.replace(  # type:ignore
        ren_id_board=_state.ren_id_board.at[_opponent_color(_state)].set(
            oppo_ren_id_board
        ),
        liberty=_state.liberty.at[_my_color(_state)]
        .set(liberty)
        .at[opp_color, _rm_ren_id, :]
        .set(0),
    )

    # 取り除かれた場所は、少なくとも自分は置ける
    max_ren_num = _state.size * _state.size
    # _state = jax.lax.fori_loop(
    #    0,
    #    max_ren_num,
    #    lambda i, state: _check_if_suicide_point_exist(
    #        state, my_color, adj_ren_id[i]
    #    ),
    #    _state,
    # )
    # 試しに取り除かれた場所は、自分：置ける、相手：基本ルールで置けない としてみる
    _my_legal_action = jax.lax.fori_loop(
        0,
        max_ren_num,
        lambda i, board: board | _state.liberty[my_color, adj_ren_id[i]],
        _state._legal_action_mask[my_color],
    )
    _oppo_legal_action = jax.lax.fori_loop(
        0,
        max_ren_num,
        lambda i, board: board & ~_state.liberty[my_color, adj_ren_id[i]],
        _state._legal_action_mask[opp_color],
    )

    # 取り除かれた位置はコウの候補となる
    return _state.replace(  # type:ignore
        _legal_action_mask=_state._legal_action_mask.at[my_color]
        .set(_state._legal_action_mask[my_color] | surrounded_stones)
        .at[opp_color]
        .set(_state._legal_action_mask[opp_color] | surrounded_stones),
        agehama=_state.agehama.at[my_color].add(agehama),
        kou=jnp.int32(_rm_stone_xy),
    )


def _legal_actions(_state: GoState, size) -> jnp.ndarray:
    def is_exception(xy, board):
        ren_id = _state.ren_id_board[_opponent_color(_state), xy]
        liberty = _state.liberty[_opponent_color(_state), ren_id, :] == 1
        exception_xy = jnp.nonzero(liberty, size=1)[0]
        return jax.lax.cond(
            (ren_id >= 0) & (jnp.count_nonzero(liberty) == 1),
            lambda: board.at[exception_xy].set(TRUE),
            lambda: board,
        )

    exception = jnp.zeros(size * size, dtype=jnp.bool_)
    exception = jax.lax.fori_loop(0, size * size, is_exception, exception)
    _legal_action_mask = (
        _state._legal_action_mask[_my_color(_state)] | exception
    )
    return jax.lax.cond(
        _state.kou == -1,
        lambda: _legal_action_mask,
        lambda: _legal_action_mask.at[_state.kou].set(False),
    )


def get_board(state: GoState) -> jnp.ndarray:
    board = jnp.full_like(state.ren_id_board[BLACK], 2)
    board = jnp.where(state.ren_id_board[BLACK] != -1, 0, board)
    board = jnp.where(state.ren_id_board[WHITE] != -1, 1, board)
    return board  # type:ignore


def show(state: GoState) -> None:
    print("===========")
    for xy in range(state.size * state.size):
        if state.ren_id_board[BLACK][xy] != -1:
            print(" " + BLACK_CHAR, end="")
        elif state.ren_id_board[WHITE][xy] != -1:
            print(" " + WHITE_CHAR, end="")
        else:
            print(" " + POINT_CHAR, end="")

        if xy % state.size == state.size - 1:
            print()


def _show_details(state: GoState) -> None:
    show(state)
    print(state.ren_id_board[BLACK].reshape((state.size, state.size)))
    print(state.ren_id_board[WHITE].reshape((state.size, state.size)))
    print(state.kou)


def _my_color(_state: GoState):
    return jnp.int32(_state.turn % 2)


def _opponent_color(_state: GoState):
    return jnp.int32((_state.turn + 1) % 2)


def _is_off_board(_x, _y, size) -> bool:
    return (_x < 0) | (size <= _x) | (_y < 0) | (size <= _y)


def _kou_occurred(_state: GoState, xy: int) -> jnp.ndarray:
    size = _state.size
    x = xy // size
    y = xy % size

    oppo_color = _opponent_color(_state)

    to_xy_batch = jax.vmap(partial(_to_xy, size=size))
    oob = jnp.bool_([x - 1 < 0, x + 1 >= size, y - 1 < 0, y + 1 >= size])
    xs = x + dx
    ys = y + dy
    is_occupied = _state.ren_id_board[oppo_color][to_xy_batch(xs, ys)] != -1
    return (oob | is_occupied).all()


def _to_xy(x, y, size) -> int:
    return x * size + y


def _get_reward(_state: GoState, _size: int) -> jnp.ndarray:
    def count_ji(color):
        return (
            _count_ji(_state, color, _size) - _state.agehama[(color + 1) % 2]
        )

    count_ji = jax.vmap(count_ji)
    score = count_ji(jnp.array([0, 1]))
    r = jax.lax.cond(
        score[0] - _state.komi > score[1],
        lambda: jnp.array([1, -1]),
        lambda: jnp.array([-1, 1]),
    )
    return r


def _count_ji(_state: GoState, _color, _size):
    board = get_board(_state)
    return jnp.count_nonzero(_get_ji(board, _color, _size))


@struct.dataclass
class JI:
    size: jnp.ndarray
    board: jnp.ndarray
    candidate_xy: jnp.ndarray
    examined_stones: jnp.ndarray
    color: jnp.ndarray


def _get_ji(_board: jnp.ndarray, color: int, size: int):
    BOARD_WIDTH = size
    # 1. boardの一番外側に1周分追加
    board = jnp.pad(
        _board.reshape((BOARD_WIDTH, BOARD_WIDTH)),
        1,
        "constant",
        constant_values=-1,
    )
    # こうなる
    # [[-1 -1 -1 -1 -1 -1 -1]
    #  [-1  2  2  2  2  2 -1]
    #  [-1  2  2  2  2  2 -1]
    #  [-1  2  2  2  2  2 -1]
    #  [-1  2  2  2  2  2 -1]
    #  [-1  2  2  2  2  2 -1]
    #  [-1 -1 -1 -1 -1 -1 -1]]
    board = board.ravel()

    # 2. oppo_colorに隣り合う空点をoppo_colorに置き換える
    candidate_xy = board == (color + 1) % 2
    examined_stones: jnp.ndarray = jnp.zeros_like(board, dtype=bool)

    ji = JI(
        jnp.array([size], dtype=int),  # type:ignore
        board,
        candidate_xy,
        examined_stones,
        jnp.array([color], dtype=int),  # type:ignore
    )

    ji = jax.lax.while_loop(
        lambda ji: jnp.count_nonzero(ji.candidate_xy) != 0, _count_ji_loop, ji
    )
    board = ji.board.reshape((BOARD_WIDTH + 2, BOARD_WIDTH + 2))

    # 3. 増やした外側をカットし、残った空点がcolorの地となる
    return board[1 : BOARD_WIDTH + 1, 1 : BOARD_WIDTH + 1] == POINT


def _count_ji_loop(_ji: JI) -> JI:
    size = _ji.size
    board = _ji.board
    xy = jnp.nonzero(_ji.candidate_xy, size=1)[0][0]
    candidate_xy = _ji.candidate_xy.at[xy].set(False)
    o_color = (_ji.color[0] + 1) % 2
    _BOARD_WIDTH = size[0] + 2

    # この座標は「既に調べたリスト」へ
    examined_stones = _ji.examined_stones.at[xy].set(True)

    board = board.at[xy - _BOARD_WIDTH].set(
        jax.lax.cond(
            board[xy - _BOARD_WIDTH] == POINT,
            lambda: o_color,
            lambda: board[xy - _BOARD_WIDTH],
        )
    )
    candidate_xy = candidate_xy.at[xy - _BOARD_WIDTH].set(
        jnp.logical_and(
            board[xy - _BOARD_WIDTH] == o_color,
            examined_stones[xy - _BOARD_WIDTH] is False,
        )
    )

    board = board.at[xy + _BOARD_WIDTH].set(
        jax.lax.cond(
            board[xy + _BOARD_WIDTH] == POINT,
            lambda: o_color,
            lambda: board[xy + _BOARD_WIDTH],
        )
    )
    candidate_xy = candidate_xy.at[xy + _BOARD_WIDTH].set(
        jnp.logical_and(
            board[xy + _BOARD_WIDTH] == o_color,
            examined_stones[xy + _BOARD_WIDTH] is False,
        )
    )

    board = board.at[xy - 1].set(
        jax.lax.cond(
            board[xy - 1] == POINT,
            lambda: o_color,
            lambda: board[xy - 1],
        )
    )
    candidate_xy = candidate_xy.at[xy - 1].set(
        jnp.logical_and(
            board[xy - 1] == o_color,
            examined_stones[xy - 1] is False,
        )
    )

    board = board.at[xy + 1].set(
        jax.lax.cond(
            board[xy + 1] == POINT,
            lambda: o_color,
            lambda: board[xy + 1],
        )
    )
    candidate_xy = candidate_xy.at[xy + 1].set(
        jnp.logical_and(
            board[xy + 1] == o_color,
            examined_stones[xy + 1] is False,
        )
    )
    return JI(
        size, board, candidate_xy, examined_stones, _ji.color
    )  # type:ignore
