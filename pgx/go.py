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
INVALID_POINT = jnp.int32(999)

dx = jnp.int32([-1, +1, 0, 0])
dy = jnp.int32([0, 0, -1, +1])

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

    num_pseudo: jnp.ndarray = jnp.zeros((2, 19 * 19), dtype=jnp.int32)
    # 最初のマスが0とならないよう、idx_sumとidx_squared_sumはマス目が1始まり
    idx_sum: jnp.ndarray = jnp.zeros((2, 19 * 19), dtype=jnp.int32)
    idx_squared_sum: jnp.ndarray = jnp.zeros((2, 19 * 19), dtype=jnp.int32)

    # 設置可能なマスをTrueとしたマスク
    legal_action_mask: jnp.ndarray = jnp.zeros(19 * 19 + 1, dtype=jnp.bool_)

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
            jnp.where(boards[i] == ((my_color + 1) % 2), 1, 0)
        ),
        state.game_log,
    )
    color = jnp.full_like(
        state.game_log[0], (my_color + 1) % 2
    )  # AlphaZeroでは黒だと1

    log = jnp.concatenate((my_log, oppo_log))
    return jnp.vstack([log, color])


def init(
    rng: jax.random.KeyArray, size: int = 5
) -> Tuple[jnp.ndarray, GoState]:
    curr_player = jnp.int32(jax.random.bernoulli(rng))
    return curr_player, GoState(  # type:ignore
        size=jnp.int32(size),  # type:ignore
        ren_id_board=jnp.full(
            (2, size**2), -1, dtype=jnp.int32
        ),  # type:ignore
        num_pseudo=jnp.zeros((2, size**2), dtype=jnp.int32),
        idx_sum=jnp.zeros((2, size**2), dtype=jnp.int32),
        idx_squared_sum=jnp.zeros((2, size**2), dtype=jnp.int32),
        legal_action_mask=jnp.ones(size**2 + 1, dtype=jnp.bool_),
        game_log=jnp.full((8, size**2), 2, dtype=jnp.int32),  # type:ignore
        curr_player=curr_player,  # type:ignore
    )


def step(
    state: GoState, action: int, size: int
) -> Tuple[jnp.ndarray, GoState, jnp.ndarray]:
    # update state
    _state, reward = _update_state_wo_legal_action(state, action, size)

    # add legal actions
    _state = _state.replace(  # type:ignore
        legal_action_mask=_state.legal_action_mask.at[:-1]
        .set(legal_actions(_state, size))
        .at[-1]
        .set(TRUE)
    )

    # update log
    new_log = jnp.roll(_state.game_log, size**2)
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
    agehama_before = state.agehama[_my_color(state)]
    is_illegal = _is_illegal_move(state, xy)  # 既に他の石が置かれている or コウ

    # 石を置く
    kou_occurred = _kou_occurred(state, xy)
    state = _set_stone(state, xy)

    # 周囲の連を調べる
    state = jax.lax.fori_loop(
        0, 4, lambda i, s: _check_around_xy(i, s, xy), state
    )

    # 取り除ける石は取り除く
    state = state.replace(
        ren_id_board=state.ren_id_board.at[_opponent_color(state)].set(
            jnp.where(
                state.ren_id_board[_opponent_color(state)] == INVALID_POINT,
                -1,
                state.ren_id_board[_opponent_color(state)],
            )
        ),
    )

    # 自殺手
    is_illegal = (
        state.num_pseudo[
            _my_color(state), state.ren_id_board[_my_color(state), xy]
        ]
        == 0
    ) | is_illegal

    # コウの確認
    kou = jax.lax.cond(
        kou_occurred & state.agehama[_my_color(state)] - agehama_before == 1,
        lambda: state.kou,
        lambda: jnp.int32(-1),
    )

    state = state.replace(kou=kou)

    return jax.lax.cond(
        is_illegal,
        lambda: _illegal_move(_set_stone(_state, xy)),  # 石くらいは置いておく
        lambda: (state, jnp.array([0, 0])),
    )


def _check_around_xy(i, state: GoState, xy):

    x = xy // state.size + dx[i]
    y = xy % state.size + dy[i]
    adj_xy = x * state.size + y

    put_ren_id = state.ren_id_board[_my_color(state), xy]

    is_off = _is_off_board(x, y, state.size) | (
        state.ren_id_board[_opponent_color(state), adj_xy] == INVALID_POINT
    )
    is_my_ren = state.ren_id_board[_my_color(state), adj_xy] != -1
    is_opp_ren = state.ren_id_board[_opponent_color(state), adj_xy] != -1
    replaced_state = state.replace(  # type:ignore
        num_pseudo=state.num_pseudo.at[_my_color(state), put_ren_id].add(1),
        idx_sum=state.idx_sum.at[_my_color(state), put_ren_id].add(adj_xy + 1),
        idx_squared_sum=state.idx_squared_sum.at[
            _my_color(state), put_ren_id
        ].add((adj_xy + 1) ** 2),
    )
    state = jax.lax.cond(
        ((~is_off) & (~is_my_ren) & (~is_opp_ren)),
        lambda: replaced_state,
        lambda: state,
    )
    state = jax.lax.cond(
        ((~is_off) & is_opp_ren),
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
    return (
        (_state.ren_id_board[_my_color(_state), _xy] != -1)
        | (_state.ren_id_board[_opponent_color(_state), _xy] != -1)
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
        num_pseudo=_state.num_pseudo.at[_my_color(_state), _xy].set(0),
        idx_sum=_state.idx_sum.at[_my_color(_state), _xy].set(0),
        idx_squared_sum=_state.idx_squared_sum.at[_my_color(_state), _xy].set(
            0
        ),
    )


def _is_atari(state: GoState, color, ren_id):
    """
    colorのren_idの連がアタリか判定する
    """
    return (state.idx_sum[color, ren_id] ** 2) == state.idx_squared_sum[
        color, ren_id
    ] * state.num_pseudo[color, ren_id]


def _single_liberty(state: GoState, color, ren_id):
    """
    アタリの点を返す
    アタリでない連に用いても無意味であることに注意
    """
    return (
        state.idx_squared_sum[color, ren_id] // state.idx_sum[color, ren_id]
    ) - 1


def _merge_ren(_state: GoState, _xy: int, _adj_xy: int):
    my_ren_id_board = _state.ren_id_board[_my_color(_state)]

    new_id = my_ren_id_board[_xy]
    adj_ren_id = my_ren_id_board[_adj_xy]

    small_id, large_id = jax.lax.cond(
        adj_ren_id < new_id,
        lambda: (adj_ren_id, new_id),
        lambda: (new_id, adj_ren_id),
    )
    # 大きいidの連を消し、小さいidの連と繋げる
    ren_id_board = jnp.where(
        my_ren_id_board == large_id, small_id, my_ren_id_board
    )

    _other_num_pseudo = _state.num_pseudo[_my_color(_state), large_id] - 1
    _other_idx_sum = _state.idx_sum[_my_color(_state), large_id] - (_xy + 1)
    _other_idx_squared_sum = (
        _state.idx_squared_sum[_my_color(_state), large_id] - (_xy + 1) ** 2
    )

    # return _state.replace(  # type:ignore
    #    ren_id_board=_state.ren_id_board.at[_my_color(_state)].set(
    #        ren_id_board
    #    ),
    #    num_pseudo=_state.num_pseudo.at[_my_color(_state), small_id].add(
    #        _other_num_pseudo
    #    ),
    #    idx_sum=_state.idx_sum.at[_my_color(_state), small_id].add(
    #        _other_idx_sum
    #    ),
    #    idx_squared_sum=_state.idx_squared_sum.at[
    #        _my_color(_state), small_id
    #    ].add(_other_idx_squared_sum),
    # )

    return jax.lax.cond(
        new_id == adj_ren_id,
        lambda: _state.replace(  # type:ignore
            ren_id_board=_state.ren_id_board.at[_my_color(_state)].set(
                ren_id_board
            ),
            num_pseudo=_state.num_pseudo.at[_my_color(_state), small_id].set(
                _state.num_pseudo[_my_color(_state), small_id] - 1
            ),
            idx_sum=_state.idx_sum.at[_my_color(_state), small_id].set(
                _state.idx_sum[_my_color(_state), small_id] - (_xy + 1)
            ),
            idx_squared_sum=_state.idx_squared_sum.at[
                _my_color(_state), small_id
            ].set(
                _state.idx_squared_sum[_my_color(_state), small_id]
                - (_xy + 1) ** 2
            ),
        ),
        lambda: _state.replace(  # type:ignore
            ren_id_board=_state.ren_id_board.at[_my_color(_state)].set(
                ren_id_board
            ),
            num_pseudo=_state.num_pseudo.at[_my_color(_state), small_id]
            .add(_other_num_pseudo)
            .at[_my_color(_state), large_id]
            .set(0),
            idx_sum=_state.idx_sum.at[_my_color(_state), small_id]
            .add(_other_idx_sum)
            .at[_my_color(_state), large_id]
            .set(0),
            idx_squared_sum=_state.idx_squared_sum.at[
                _my_color(_state), small_id
            ]
            .add(_other_idx_squared_sum)
            .at[_my_color(_state), large_id]
            .set(0),
        ),
    )


def _set_stone_next_to_oppo_ren(_state: GoState, _xy, _adj_xy):
    oppo_color = _opponent_color(_state)
    oppo_ren_id = _state.ren_id_board[oppo_color, _adj_xy]

    # x = x.at[idx].add(y)      x[idx] += y みたいな感じで
    # x = x.at[idx].minus(y)    x[idx] -= y は無いのだろうか
    return jax.lax.cond(
        _is_atari(_state, oppo_color, oppo_ren_id)
        & (_single_liberty(_state, oppo_color, oppo_ren_id) == _xy),
        lambda: _remove_stones(_state, oppo_ren_id, _adj_xy),
        lambda: _state.replace(  # type:ignore
            num_pseudo=_state.num_pseudo.at[oppo_color, oppo_ren_id].set(
                _state.num_pseudo[oppo_color, oppo_ren_id] - 1
            ),
            idx_sum=_state.idx_sum.at[oppo_color, oppo_ren_id].set(
                _state.idx_sum[oppo_color, oppo_ren_id] - (_xy + 1)
            ),
            idx_squared_sum=_state.idx_squared_sum.at[
                oppo_color, oppo_ren_id
            ].set(
                _state.idx_squared_sum[oppo_color, oppo_ren_id]
                - (_xy + 1) ** 2
            ),
        ),
    )


def _remove_stones(_state: GoState, _rm_ren_id, _rm_stone_xy) -> GoState:
    oppo_color = _opponent_color(_state)
    surrounded_stones = _state.ren_id_board[oppo_color] == _rm_ren_id
    agehama = jnp.count_nonzero(surrounded_stones)
    # 一時的に無効化（後で取り除く）
    oppo_ren_id_board = jnp.where(
        surrounded_stones, INVALID_POINT, _state.ren_id_board[oppo_color]
    )

    # 取り除かれた連に隣接する連の呼吸点を増やす
    # TODO vmap化できそう
    _state = jax.lax.fori_loop(
        0,
        _state.size * _state.size,
        lambda i, s: _add_adj_liberty(s, i, surrounded_stones),
        _state,
    )

    return _state.replace(  # type:ignore
        ren_id_board=_state.ren_id_board.at[oppo_color].set(oppo_ren_id_board),
        num_pseudo=_state.num_pseudo.at[oppo_color, _rm_ren_id].set(0),
        idx_sum=_state.idx_sum.at[oppo_color, _rm_ren_id].set(0),
        idx_squared_sum=_state.idx_squared_sum.at[oppo_color, _rm_ren_id].set(
            0
        ),
        agehama=_state.agehama.at[_my_color(_state)].add(agehama),
        kou=jnp.int32(_rm_stone_xy),  # type:ignore
    )


def _add_adj_liberty(state: GoState, xy, s_stones):
    # 取り除かれた連の場所を呼吸点に追加する
    my_color = _my_color(state)
    ren_id = state.ren_id_board[my_color, xy]
    replaced_state = jax.lax.fori_loop(
        0,
        4,
        lambda i, s: _check_if_adj_is_removed(s, i, xy, s_stones, ren_id),
        state,
    )
    return jax.lax.cond(ren_id != -1, lambda: replaced_state, lambda: state)


def _check_if_adj_is_removed(state, i, xy, s_stones, ren_id):
    my_color = _my_color(state)
    x = xy // state.size + dx[i]
    y = xy % state.size + dy[i]
    adj_xy = x * state.size + y
    is_off = _is_off_board(x, y, state.size)
    is_adj_removed = s_stones[adj_xy]
    return jax.lax.cond(
        ~is_off & is_adj_removed,
        lambda: state.replace(  # type:ignore
            num_pseudo=state.num_pseudo.at[my_color, ren_id].add(1),
            idx_sum=state.idx_sum.at[my_color, ren_id].add(adj_xy + 1),
            idx_squared_sum=state.idx_squared_sum.at[my_color, ren_id].add(
                (adj_xy + 1) ** 2
            ),
        ),
        lambda: state,
    )


def legal_actions(state: GoState, size: int) -> jnp.ndarray:
    def is_exception(xy, board):
        ren_id = state.ren_id_board[_opponent_color(state), xy]
        exception_xy = _single_liberty(state, _opponent_color(state), ren_id)
        return jax.lax.cond(
            (ren_id >= 0) & _is_atari(state, _opponent_color(state), ren_id),
            lambda: board.at[exception_xy].set(TRUE),
            lambda: board,
        )

    exception = jnp.zeros(size**2, dtype=jnp.bool_)
    exception = jax.lax.fori_loop(0, size**2, is_exception, exception)

    _legal_action_mask = jax.lax.map(
        lambda xy: _is_legal_action(state, xy),
        jnp.arange(0, size**2),
    )
    _legal_action_mask = _legal_action_mask | exception
    return jax.lax.cond(
        (state.kou == -1),
        lambda: _legal_action_mask,
        lambda: _legal_action_mask.at[state.kou].set(FALSE),
    )


def _is_legal_action(state: GoState, xy):
    point = (state.ren_id_board[BLACK, xy] == -1) & (
        state.ren_id_board[WHITE, xy] == -1
    )

    def check_around(i, xy, state):
        # 呼吸点か、呼吸点2つ以上の味方連があればセーフ
        my_color = _my_color(state)
        x = xy // state.size + dx[i]
        y = xy % state.size + dy[i]
        adj_xy = x * state.size + y
        is_off = _is_off_board(x, y, state.size)
        my_ren_id = state.ren_id_board[my_color, adj_xy]
        return ~is_off & (
            _is_point(state, x, y)
            | ((my_ren_id >= 0) & ~_is_atari(state, my_color, my_ren_id))
        )

    is_legal = point & jax.lax.fori_loop(
        0, 4, lambda i, legal: legal | check_around(i, xy, state), FALSE
    )

    return is_legal


def _is_point(_state, x, y):
    return (_state.ren_id_board[0, x * _state.size + y] == -1) & (
        _state.ren_id_board[1, x * _state.size + y] == -1
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
    print(state.ren_id_board[BLACK].reshape((5, 5)))
    print(state.ren_id_board[WHITE].reshape((5, 5)))
    print(state.kou)


def _my_color(_state: GoState):
    return jnp.int32(_state.turn % 2)


def _opponent_color(_state: GoState):
    return jnp.int32((_state.turn + 1) % 2)


def _is_off_board(_x, _y, _size) -> bool:
    return (_x < 0) | (_size <= _x) | (_y < 0) | (_size <= _y)


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
    score = count_ji(jnp.array([BLACK, WHITE]))
    r = jax.lax.cond(
        score[BLACK] - _state.komi > score[WHITE],
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
