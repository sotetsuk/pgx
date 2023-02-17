from functools import partial
from typing import Tuple

import jax
from jax import numpy as jnp

from pgx.flax.struct import dataclass

BLACK = 0
WHITE = 1
POINT = 2
BLACK_CHAR = "@"
WHITE_CHAR = "O"
POINT_CHAR = "+"

dx = jnp.int32([-1, +1, 0, 0])
dy = jnp.int32([0, 0, -1, +1])

FALSE = jnp.bool_(False)
TRUE = jnp.bool_(True)


@dataclass
class GoState:
    # 横幅, マスの数ではない
    size: jnp.ndarray = jnp.int32(19)  # type:ignore

    # 連の代表点（一番小さいマス目）のマス目の座標
    ren_id_board: jnp.ndarray = jnp.full(
        (2, 19 * 19), -1, dtype=jnp.int32
    )  # type:ignore

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
    passed: jnp.ndarray = FALSE  # type:ignore

    # コウによる着手禁止点(xy), 無ければ(-1)
    kou: jnp.ndarray = jnp.int32(-1)  # type:ignore

    # コミ
    komi: jnp.ndarray = jnp.float32(6.5)  # type:ignore

    # 終局判定
    terminated: jnp.ndarray = FALSE  # type:ignore


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
    num_player_log = 8
    my_color = jax.lax.cond(
        player_id == state.curr_player,
        lambda: state.turn % 2,
        lambda: (state.turn + 1) % 2,
    )

    @jax.vmap
    def _make_log(i):
        return state.game_log[i % num_player_log] == (
            (my_color + i // num_player_log) % 2
        )

    log = _make_log(jnp.arange(num_player_log * 2))
    color = jnp.full_like(
        state.game_log[0], (my_color + 1) % 2
    )  # AlphaZeroでは黒だと1

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
        legal_action_mask=jnp.ones(size**2 + 1, dtype=jnp.bool_),
        game_log=jnp.full((8, size**2), 2, dtype=jnp.int32),
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
        lambda: _not_pass_move(_state, _action, _size),
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
        lambda: (_state.replace(passed=True), jnp.zeros(2, dtype=jnp.int8)),  # type: ignore
    )


def _not_pass_move(
    _state: GoState, _action: int, size
) -> Tuple[GoState, jnp.ndarray]:
    state = _state.replace(passed=FALSE)  # type: ignore
    xy = _action
    my_color = _my_color(state)
    agehama_before = state.agehama[my_color]
    is_illegal = ~state.legal_action_mask[xy]

    kou_occurred = _kou_occurred(state, xy)

    # 周囲の連から敵石を除く
    state = jax.lax.fori_loop(
        0, 4, lambda i, s: _remove_around_xy(i, s, xy, size), state
    )
    # 石を置く
    state = _set_stone(state, xy)
    # 周囲をマージ
    state = jax.lax.fori_loop(
        0, 4, lambda i, s: _merge_around_xy(i, s, xy, size), state
    )

    # コウの確認
    state = jax.lax.cond(
        kou_occurred & state.agehama[my_color] - agehama_before == 1,
        lambda: state,
        lambda: state.replace(kou=jnp.int32(-1)),  # type:ignore
    )

    return jax.lax.cond(
        is_illegal,
        lambda: _illegal_move(_set_stone(_state, xy)),  # 石くらいは置いておく
        lambda: (state, jnp.zeros(2, dtype=jnp.int8)),
    )


def _remove_around_xy(i, state: GoState, xy, size):
    adj_xy = _neighbour(xy, size)[i]
    is_off = adj_xy == -1
    is_opp_ren = state.ren_id_board[_opponent_color(state), adj_xy] != -1
    state = jax.lax.cond(
        ((~is_off) & is_opp_ren),
        lambda: _set_stone_next_to_oppo_ren(state, xy, adj_xy, size),
        lambda: state,
    )
    return state


def _merge_around_xy(i, state: GoState, xy, size):
    my_color = _my_color(state)
    adj_xy = _neighbour(xy, size)[i]
    is_off = adj_xy == -1
    is_my_ren = state.ren_id_board[my_color, adj_xy] != -1
    state = jax.lax.cond(
        ((~is_off) & is_my_ren),
        lambda: _merge_ren(state, xy, adj_xy),
        lambda: state,
    )
    return state


def _illegal_move(
    _state: GoState,
) -> Tuple[GoState, jnp.ndarray]:
    r: jnp.ndarray = jnp.ones(2, dtype=jnp.int8)  # type:ignore
    return _state.replace(terminated=TRUE), r.at[_state.turn % 2].set(-1)  # type: ignore


def _set_stone(_state: GoState, _xy: int) -> GoState:
    my_color = _my_color(_state)
    return _state.replace(  # type:ignore
        ren_id_board=_state.ren_id_board.at[my_color, _xy].set(_xy),
    )


def _merge_ren(_state: GoState, _xy: int, _adj_xy: int):
    my_color = _my_color(_state)
    my_ren_id_board = _state.ren_id_board[my_color]

    new_id = my_ren_id_board[_xy]
    adj_ren_id = my_ren_id_board[_adj_xy]
    small_id, large_id = jnp.minimum(new_id, adj_ren_id), jnp.maximum(
        new_id, adj_ren_id
    )

    # 大きいidの連を消し、小さいidの連と繋げる
    ren_id_board = jnp.where(
        my_ren_id_board == large_id, small_id, my_ren_id_board
    )

    return _state.replace(  # type:ignore
        ren_id_board=_state.ren_id_board.at[my_color].set(ren_id_board),
    )


def _set_stone_next_to_oppo_ren(_state: GoState, _xy, _adj_xy, size):
    oppo_color = _opponent_color(_state)
    oppo_ren_id = _state.ren_id_board[oppo_color, _adj_xy]

    num_pseudo, idx_sum, idx_squared_sum = _count(_state, oppo_color, size)

    # fmt: off
    is_atari = ((idx_sum[oppo_ren_id] ** 2) == idx_squared_sum[oppo_ren_id] * num_pseudo[oppo_ren_id])
    single_liberty = (idx_squared_sum[oppo_ren_id] // idx_sum[oppo_ren_id]) - 1
    # fmt: on

    return jax.lax.cond(
        is_atari & (single_liberty == _xy),
        lambda: _remove_stones(_state, oppo_ren_id, _adj_xy),
        lambda: _state,
    )


def _remove_stones(_state: GoState, _rm_ren_id, _rm_stone_xy) -> GoState:
    oppo_color = _opponent_color(_state)
    surrounded_stones = _state.ren_id_board[oppo_color] == _rm_ren_id
    agehama = jnp.count_nonzero(surrounded_stones)
    oppo_ren_id_board = jnp.where(
        surrounded_stones, -1, _state.ren_id_board[oppo_color]
    )

    return _state.replace(  # type:ignore
        ren_id_board=_state.ren_id_board.at[oppo_color].set(oppo_ren_id_board),
        agehama=_state.agehama.at[_my_color(_state)].add(agehama),
        kou=jnp.int32(_rm_stone_xy),  # type:ignore
    )


def legal_actions(state: GoState, size: int) -> jnp.ndarray:
    is_empty = (state.ren_id_board[BLACK] == -1) & (
        state.ren_id_board[WHITE] == -1
    )

    my_color = _my_color(state)
    opp_color = _opponent_color(state)
    my_ren = state.ren_id_board[my_color]
    opp_ren = state.ren_id_board[opp_color]
    my_num_pseudo, my_idx_sum, my_idx_squared_sum = _count(
        state, my_color, size
    )
    opp_num_pseudo, opp_idx_sum, opp_idx_squared_sum = _count(
        state, opp_color, size
    )

    # fmt: off
    my_in_atari = (my_idx_sum[my_ren] ** 2) == my_idx_squared_sum[my_ren] * my_num_pseudo[my_ren]
    opp_in_atari = (opp_idx_sum[opp_ren] ** 2) == opp_idx_squared_sum[opp_ren] * opp_num_pseudo[opp_ren]
    # fmt: on
    is_empty = (state.ren_id_board[0] == -1) & (state.ren_id_board[1] == -1)
    has_liberty = (my_ren >= 0) & ~my_in_atari
    kills_opp = (opp_ren >= 0) & opp_in_atari

    @jax.vmap
    def is_neighbor_ok(xy):
        neighbors = _neighbour(xy, size)
        on_board = neighbors != -1
        _has_empty = is_empty[neighbors]
        _has_liberty = has_liberty[neighbors]
        _kills_opp = kills_opp[neighbors]
        return (
            (on_board & _has_empty).any()
            | (on_board & _kills_opp).any()
            | (on_board & _has_liberty).any()
        )

    neighbor_ok = is_neighbor_ok(jnp.arange(size**2))
    legal_action_mask = is_empty & neighbor_ok

    return jax.lax.cond(
        (state.kou == -1),
        lambda: legal_action_mask,
        lambda: legal_action_mask.at[state.kou].set(FALSE),
    )


def _count(state: GoState, color, size):
    ZERO = jnp.int32(0)
    is_empty = (state.ren_id_board[BLACK] == -1) & (
        state.ren_id_board[WHITE] == -1
    )
    idx_sum = jnp.where(is_empty, jnp.arange(1, size**2 + 1), ZERO)
    idx_squared_sum = jnp.where(
        is_empty, jnp.arange(1, size**2 + 1) ** 2, ZERO
    )

    @jax.vmap
    def _count_neighbor(xy):
        neighbors = _neighbour(xy, size)
        on_board = neighbors != -1
        # fmt: off
        return (jnp.where(on_board, is_empty[neighbors], ZERO).sum(),
                jnp.where(on_board, idx_sum[neighbors], ZERO).sum(),
                jnp.where(on_board, idx_squared_sum[neighbors], ZERO).sum())
        # fmt: on

    idx = jnp.arange(size**2)
    my_ren = state.ren_id_board[color]

    num_pseudo, idx_sum, idx_squared_sum = _count_neighbor(idx)
    # num_pseudo = jnp.where(my_ren >= 0, num_pseudo, ZERO)
    # idx_sum = jnp.where(my_ren >= 0, idx_sum, ZERO)
    # idx_squared_sum = jnp.where(my_ren >= 0, idx_squared_sum, ZERO)

    @jax.vmap
    def _num_pseudo(x):
        return jnp.where(my_ren == x, num_pseudo, ZERO).sum()

    @jax.vmap
    def _idx_sum(x):
        return jnp.where(my_ren == x, idx_sum, ZERO).sum()

    @jax.vmap
    def _idx_squared_sum(x):
        return jnp.where(my_ren == x, idx_squared_sum, ZERO).sum()

    return _num_pseudo(idx), _idx_sum(idx), _idx_squared_sum(idx)


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


def _kou_occurred(_state: GoState, xy: int) -> jnp.ndarray:
    size = _state.size
    x = xy // size
    y = xy % size
    oob = jnp.bool_([x - 1 < 0, x + 1 >= size, y - 1 < 0, y + 1 >= size])
    oppo_color = _opponent_color(_state)
    is_occupied = _state.ren_id_board[oppo_color][_neighbour(xy, size)] != -1
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
        lambda: jnp.array([1, -1], dtype=jnp.int8),
        lambda: jnp.array([-1, 1], dtype=jnp.int8),
    )

    return r


def _neighbour(xy, size):
        xs = xy // size + dx
        ys = xy % size + dy
        on_board = (0 <= xs) & (xs < size) & (0 <= ys) & (ys < size)
        return jnp.where(on_board, xs * size + ys, -1)


def _neighbours(size):
    return jax.vmap(partial(_neighbour, size=size))(jnp.arange(size**2))


def _count_ji(state: GoState, color: int, size: int):
    board = jnp.zeros_like(state.ren_id_board[0])
    board = jnp.where(state.ren_id_board[color] >= 0, 1, board)
    board = jnp.where(state.ren_id_board[1 - color] >= 0, -1, board)
    # 0 = empty, 1 = mine, -1 = opponent's

    neighbours = _neighbours(size)

    def is_opp_neighbours(b):
        # 空点かつ、隣接する4箇所のいずれかが敵石の場合True
        return (b == 0) & (
            (b[neighbours.flatten()] == -1).reshape(size**2, 4)
            & (neighbours != -1)
        ).any(axis=1)

    def fill_opp(x):
        b, _ = x
        mask = is_opp_neighbours(b)
        return jnp.where(mask, -1, b), mask.any()

    # fmt off
    b, _ = jax.lax.while_loop(lambda x: x[1], fill_opp, (board, TRUE))
    # fmt on

    return (b == 0).sum()
