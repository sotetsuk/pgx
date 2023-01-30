from typing import Tuple

import jax
from flax import struct
from jax import numpy as jnp

import pgx.core as core

BLACK = 0
WHITE = 1
POINT = 2
BLACK_CHAR = "@"
WHITE_CHAR = "O"
POINT_CHAR = "+"

NSEW = jnp.array([[-1, 0], [1, 0], [0, 1], [0, -1]], dtype=jnp.int32)

FALSE = jnp.bool_(False)
TRUE = jnp.bool_(True)


@struct.dataclass
class State(core.State):
    rng: jax.random.KeyArray = jax.random.PRNGKey(0)
    curr_player: jnp.ndarray = jnp.int8(0)
    reward: jnp.ndarray = jnp.float32([0.0, 0.0])
    terminated: jnp.ndarray = FALSE
    legal_action_mask: jnp.ndarray = jnp.zeros(19 * 19, dtype=jnp.bool_)

    # 横幅, マスの数ではない
    size: jnp.ndarray = jnp.int32(19)  # type:ignore

    # 連
    ren_id_board: jnp.ndarray = jnp.full(
        (2, 19 * 19), -1, dtype=jnp.int32
    )  # type:ignore

    # 連idが使えるか
    available_ren_id: jnp.ndarray = jnp.ones((2, 19 * 19), dtype=jnp.bool_)

    # 連周りの情報 0:None 1:呼吸点 2:石
    liberty: jnp.ndarray = jnp.zeros((2, 19 * 19, 19 * 19), dtype=jnp.int32)

    # 隣接している敵の連id
    adj_ren_id: jnp.ndarray = jnp.zeros((2, 19 * 19, 19 * 19), dtype=jnp.bool_)

    # 直近8回のログ
    game_log: jnp.ndarray = jnp.full(
        (8, 19 * 19), 2, dtype=jnp.int32
    )  # type:ignore

    # 経過ターン, 0始まり
    turn: jnp.ndarray = jnp.int32(0)  # type:ignore

    # [0]: 黒の得たアゲハマ, [1]: 白の方
    agehama: jnp.ndarray = jnp.zeros(2, dtype=jnp.int32)

    # 直前のactionがパスだとTrue
    passed: jnp.ndarray = jnp.bool_(False)  # type:ignore

    # コウによる着手禁止点(xy), 無ければ(-1)
    kou: jnp.ndarray = jnp.int32(-1)  # type:ignore

    # コミ
    komi: jnp.ndarray = jnp.float32(6.5)  # type:ignore


class Go(core.Env):

    def __init__(self, size: int = 5):
        super().__init__()
        self.size: int = size

    def init(self, rng: jnp.ndarray) -> core.State:
        return init(rng, self.size)[1]

    def step(self, state: core.State, action: jnp.ndarray) -> core.State:
        return step(state, action, self.size)[1]

    def observe(self, state: core.State, player_id: jnp.ndarray) -> jnp.ndarray:
        return observe(state, player_id)


def observe(state: State, player_id, observe_all=False):
    return _get_alphazero_features(state, player_id, observe_all)


def _get_alphazero_features(state: State, player_id, observe_all):
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
    rng: jnp.ndarray, size: int = 5
) -> Tuple[jnp.ndarray, State]:
    curr_player = jnp.int8(jax.random.bernoulli(rng))
    return curr_player, State(  # type:ignore
        size=jnp.int32(size),  # type:ignore
        ren_id_board=jnp.full(
            (2, size * size), -1, dtype=jnp.int32
        ),  # type:ignore
        available_ren_id=jnp.ones((2, size * size), dtype=jnp.bool_),
        liberty=jnp.zeros((2, size * size, size * size), dtype=jnp.int32),
        adj_ren_id=jnp.zeros((2, size * size, size * size), dtype=jnp.bool_),
        legal_action_mask=jnp.ones(size * size, dtype=jnp.bool_),
        game_log=jnp.full((8, size * size), 2, dtype=jnp.int32),  # type:ignore
        curr_player=curr_player,  # type:ignore
    )


def step(
    state: State, action: jnp.ndarray, size: int
) -> Tuple[jnp.ndarray, State, jnp.ndarray]:
    # update state
    _state, reward = _update_state_wo_legal_action(state, action, size)

    # add legal actions
    _state = State(  # type:ignore
        size=_state.size,
        ren_id_board=_state.ren_id_board,
        available_ren_id=_state.available_ren_id,
        liberty=_state.liberty,
        adj_ren_id=_state.adj_ren_id,
        legal_action_mask=legal_actions(_state, size),
        game_log=_state.game_log,
        turn=_state.turn,
        curr_player=_state.curr_player,
        agehama=_state.agehama,
        passed=_state.passed,
        kou=_state.kou,
        komi=_state.komi,
        terminated=_state.terminated,
    )

    # update log
    new_log = jnp.roll(_state.game_log, size * size)
    new_log = new_log.at[0].set(get_board(_state))
    _state = State(  # type:ignore
        size=_state.size,
        ren_id_board=_state.ren_id_board,
        available_ren_id=_state.available_ren_id,
        liberty=_state.liberty,
        adj_ren_id=_state.adj_ren_id,
        legal_action_mask=_state.legal_action_mask,
        game_log=new_log,
        turn=_state.turn,
        curr_player=_state.curr_player,
        agehama=_state.agehama,
        passed=_state.passed,
        kou=_state.kou,
        komi=_state.komi,
        terminated=_state.terminated,
    )

    return (_state.curr_player, _state, reward)


def _update_state_wo_legal_action(
    _state: State, _action: int, _size: int
) -> Tuple[State, jnp.ndarray]:
    _state, _reward = jax.lax.cond(
        _action < 0,
        lambda: _pass_move(_state, _size),
        lambda: _not_pass_move(_state, _action),
    )

    # increase turn
    _state = _increase_turn(_state)

    # change player
    _state = _change_player(_state)

    return _state, _reward


def _pass_move(_state: State, _size: int) -> Tuple[State, jnp.ndarray]:
    return jax.lax.cond(
        _state.passed,
        # 連続でパスならば終局
        lambda: (
            _update_terminated(_state),
            _get_reward(_state, _size),
        ),
        # 1回目のパスならばStateにパスを追加してそのまま続行
        lambda: (_set_pass(_state, True), jnp.array([0, 0])),
    )


def _increase_turn(_state: State) -> State:
    return State(  # type:ignore
        size=_state.size,
        ren_id_board=_state.ren_id_board,
        available_ren_id=_state.available_ren_id,
        liberty=_state.liberty,
        adj_ren_id=_state.adj_ren_id,
        legal_action_mask=_state.legal_action_mask,
        game_log=_state.game_log,
        turn=_state.turn + 1,
        curr_player=_state.curr_player,
        agehama=_state.agehama,
        passed=_state.passed,
        kou=_state.kou,
        komi=_state.komi,
        terminated=_state.terminated,
    )


def _change_player(_state: State) -> State:
    return State(  # type:ignore
        size=_state.size,
        ren_id_board=_state.ren_id_board,
        available_ren_id=_state.available_ren_id,
        liberty=_state.liberty,
        adj_ren_id=_state.adj_ren_id,
        legal_action_mask=_state.legal_action_mask,
        game_log=_state.game_log,
        turn=_state.turn,
        curr_player=(_state.curr_player + 1) % 2,
        agehama=_state.agehama,
        passed=_state.passed,
        kou=_state.kou,
        komi=_state.komi,
        terminated=_state.terminated,
    )


def _set_pass(_state: State, _pass: bool) -> State:
    return State(  # type:ignore
        size=_state.size,
        ren_id_board=_state.ren_id_board,
        available_ren_id=_state.available_ren_id,
        liberty=_state.liberty,
        adj_ren_id=_state.adj_ren_id,
        legal_action_mask=_state.legal_action_mask,
        game_log=_state.game_log,
        turn=_state.turn,
        curr_player=_state.curr_player,
        agehama=_state.agehama,
        passed=_pass,  # type:ignore
        kou=_state.kou,
        komi=_state.komi,
        terminated=_state.terminated,
    )


def _update_terminated(_state: State) -> State:
    return State(  # type:ignore
        size=_state.size,
        ren_id_board=_state.ren_id_board,
        available_ren_id=_state.available_ren_id,
        liberty=_state.liberty,
        adj_ren_id=_state.adj_ren_id,
        legal_action_mask=_state.legal_action_mask,
        game_log=_state.game_log,
        turn=_state.turn,
        curr_player=jnp.int8(-1),  # type:ignore
        agehama=_state.agehama,
        passed=_state.passed,
        kou=_state.kou,
        komi=_state.komi,
        terminated=jnp.bool_(True),  # type:ignore
    )


def _not_pass_move(
    _state: State, _action: int
) -> Tuple[State, jnp.ndarray]:
    state = _set_pass(_state, False)
    xy = _action
    agehama_before = state.agehama[_my_color(state)]
    is_illegal = _is_illegal_move(state, xy)  # 既に他の石が置かれている or コウ

    # 石を置く
    kou_occurred = _kou_occurred(state, xy)
    state = _set_stone(state, xy)

    # 周囲の連を調べる
    state_and_xy = (state, xy)
    state_and_xy = jax.lax.fori_loop(0, 4, _check_around_xy, state_and_xy)
    state = state_and_xy[0]

    # 自殺手
    is_illegal = (
        jnp.count_nonzero(
            state.liberty[
                _my_color(state), state.ren_id_board[_my_color(state), xy]
            ]
            == 1
        )
        == 0
    ) | is_illegal

    # コウの確認
    state = State(  # type:ignore
        size=state.size,
        ren_id_board=state.ren_id_board,
        available_ren_id=state.available_ren_id,
        liberty=state.liberty,
        adj_ren_id=state.adj_ren_id,
        legal_action_mask=state.legal_action_mask,
        game_log=state.game_log,
        turn=state.turn,
        curr_player=state.curr_player,
        agehama=state.agehama,
        passed=state.passed,
        kou=jax.lax.cond(
            kou_occurred & state.agehama[_my_color(state)] - agehama_before
            == 1,
            lambda: state.kou,
            lambda: jnp.int32(-1),
        ),
        komi=state.komi,
        terminated=state.terminated,
    )

    return jax.lax.cond(
        is_illegal,
        lambda: _illegal_move(state),
        lambda: (state, jnp.array([0, 0])),
    )


def _check_around_xy(i, state_and_xy):
    state = state_and_xy[0]
    xy = state_and_xy[1]
    adj_pos = (
        jnp.array([xy // state.size, xy % state.size], dtype=jnp.int32)
        + NSEW[i]  # type:ignore
    )
    adj_xy = adj_pos[0] * state.size + adj_pos[1]
    state = jax.lax.cond(
        _is_off_board(adj_pos, state.size),
        lambda: state,  # 盤外
        lambda: jax.lax.cond(
            state.ren_id_board[_my_color(state), adj_xy] != -1,
            lambda: _merge_ren(state, xy, adj_xy),
            lambda: jax.lax.cond(
                state.ren_id_board[_opponent_color(state), adj_xy] != -1,
                lambda: _set_stone_next_to_oppo_ren(state, xy, adj_xy),
                lambda: State(  # type:ignore
                    size=state.size,
                    ren_id_board=state.ren_id_board,
                    available_ren_id=state.available_ren_id,
                    liberty=state.liberty.at[
                        _my_color(state),
                        state.ren_id_board[_my_color(state), xy],
                        adj_xy,
                    ].set(1),
                    adj_ren_id=state.adj_ren_id,
                    legal_action_mask=state.legal_action_mask,
                    game_log=state.game_log,
                    turn=state.turn,
                    curr_player=state.curr_player,
                    agehama=state.agehama,
                    passed=state.passed,
                    kou=state.kou,
                    komi=state.komi,
                    terminated=state.terminated,
                ),
            ),
        ),
    )
    return (state, xy)


def _is_illegal_move(_state: State, _xy):
    """
    既に石があるorコウ
    """
    my_color = _my_color(_state)
    oppo_color = _opponent_color(_state)
    return jnp.logical_or(
        jnp.logical_or(
            _state.ren_id_board[my_color, _xy] != -1,
            _state.ren_id_board[oppo_color, _xy] != -1,
        ),
        _xy == _state.kou,
    )


def _illegal_move(
    _state: State,
) -> Tuple[State, jnp.ndarray]:
    r: jnp.ndarray = jnp.array([1, 1])  # type:ignore
    return _update_terminated(_state), r.at[_state.turn % 2].set(-1)


def _set_stone(_state: State, _xy: int) -> State:
    available_ren_id = _state.available_ren_id[_my_color(_state)]
    next_ren_id = jnp.argmax(available_ren_id)
    available_ren_id = available_ren_id.at[next_ren_id].set(False)
    return State(  # type:ignore
        size=_state.size,
        ren_id_board=_state.ren_id_board.at[_my_color(_state), _xy].set(
            next_ren_id
        ),
        available_ren_id=_state.available_ren_id.at[_my_color(_state)].set(
            available_ren_id
        ),
        liberty=_state.liberty,
        adj_ren_id=_state.adj_ren_id,
        legal_action_mask=_state.legal_action_mask,
        game_log=_state.game_log,
        turn=_state.turn,
        curr_player=_state.curr_player,
        agehama=_state.agehama,
        passed=_state.passed,
        kou=_state.kou,
        komi=_state.komi,
        terminated=_state.terminated,
    )


def _merge_ren(_state: State, _xy: int, _adj_xy: int):
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

    _adj_ren_id = _state.adj_ren_id.at[_my_color(_state)].get()

    _oppo_adj_ren_id = jax.lax.map(
        lambda _a: jnp.where(
            _a[large_id], _a.at[large_id].set(False).at[small_id].set(True), _a
        ),
        _state.adj_ren_id.at[_opponent_color(_state)].get(),
    )
    _adj_ren_id = _adj_ren_id.at[small_id].set(
        jnp.logical_or(_adj_ren_id[small_id], _adj_ren_id[large_id])
    )
    _adj_ren_id = _adj_ren_id.at[large_id, :].set(False)

    return jax.lax.cond(
        new_id == adj_ren_id,
        lambda: _state,
        lambda: State(  # type:ignore
            size=_state.size,
            ren_id_board=_state.ren_id_board.at[_my_color(_state)].set(
                ren_id_board
            ),
            available_ren_id=_state.available_ren_id.at[
                _my_color(_state), large_id
            ].set(True),
            liberty=_state.liberty.at[_my_color(_state)].set(liberty),
            adj_ren_id=_state.adj_ren_id.at[_my_color(_state)]
            .set(_adj_ren_id)
            .at[_opponent_color(_state)]
            .set(_oppo_adj_ren_id),
            legal_action_mask=_state.legal_action_mask,
            game_log=_state.game_log,
            turn=_state.turn,
            curr_player=_state.curr_player,
            agehama=_state.agehama,
            passed=_state.passed,
            kou=_state.kou,
            komi=_state.komi,
            terminated=_state.terminated,
        ),
    )


def _set_stone_next_to_oppo_ren(_state: State, _xy, _adj_xy):
    oppo_ren_id = _state.ren_id_board.at[
        _opponent_color(_state), _adj_xy
    ].get()

    liberty = (
        _state.liberty.at[_opponent_color(_state), oppo_ren_id, _xy]
        .set(2)
        .at[
            _my_color(_state),
            _state.ren_id_board[_my_color(_state), _xy],
            _adj_xy,
        ]
        .set(2)
    )
    adj_ren_id = (
        _state.adj_ren_id.at[
            _my_color(_state),
            _state.ren_id_board[_my_color(_state), _xy],
            oppo_ren_id,
        ]
        .set(True)
        .at[
            _opponent_color(_state),
            oppo_ren_id,
            _state.ren_id_board[_my_color(_state), _xy],
        ]
        .set(True)
    )

    state = State(  # type:ignore
        size=_state.size,
        ren_id_board=_state.ren_id_board,
        available_ren_id=_state.available_ren_id,
        liberty=liberty,
        adj_ren_id=adj_ren_id,
        legal_action_mask=_state.legal_action_mask,
        game_log=_state.game_log,
        turn=_state.turn,
        curr_player=_state.curr_player,
        agehama=_state.agehama,
        passed=_state.passed,
        kou=_state.kou,
        komi=_state.komi,
        terminated=_state.terminated,
    )

    return jax.lax.cond(
        jnp.count_nonzero(
            state.liberty[_opponent_color(state), oppo_ren_id] == 1
        )
        == 0,
        lambda: _remove_stones(state, oppo_ren_id, _adj_xy),
        lambda: state,
    )


def _remove_stones(_state: State, _rm_ren_id, _rm_stone_xy) -> State:
    surrounded_stones = (
        _state.ren_id_board[_opponent_color(_state)] == _rm_ren_id
    )
    agehama = jnp.count_nonzero(surrounded_stones)
    oppo_ren_id_board = jnp.where(
        surrounded_stones, -1, _state.ren_id_board[_opponent_color(_state)]
    )
    liberty = jax.lax.map(
        lambda l: jnp.where((l > 0) & surrounded_stones, 1, l),
        _state.liberty[_my_color(_state)],
    )
    available_ren_id = _state.available_ren_id.at[
        _opponent_color(_state), _rm_ren_id
    ].set(True)

    return State(  # type:ignore
        size=_state.size,
        ren_id_board=_state.ren_id_board.at[_opponent_color(_state)].set(
            oppo_ren_id_board
        ),
        available_ren_id=available_ren_id,
        liberty=_state.liberty.at[_my_color(_state)]
        .set(liberty)
        .at[_opponent_color(_state), _rm_ren_id, :]
        .set(0),
        adj_ren_id=_state.adj_ren_id.at[_opponent_color(_state), _rm_ren_id, :]
        .set(False)
        .at[_my_color(_state), :, _rm_ren_id]
        .set(False),
        legal_action_mask=_state.legal_action_mask,
        game_log=_state.game_log,
        turn=_state.turn,
        curr_player=_state.curr_player,
        agehama=_state.agehama.at[_my_color(_state)].add(agehama),
        passed=_state.passed,
        kou=jnp.int32(_rm_stone_xy),  # type:ignore
        terminated=_state.terminated,
    )


def legal_actions(state: State, size: int) -> jnp.ndarray:
    return jnp.logical_not(
        jax.lax.map(
            lambda xy: _update_state_wo_legal_action(state, xy, size)[
                0
            ].terminated,
            jnp.arange(0, size * size),
        )
    )


def get_board(state: State) -> jnp.ndarray:
    board = jnp.full_like(state.ren_id_board[BLACK], 2)
    board = jnp.where(state.ren_id_board[BLACK] != -1, 0, board)
    board = jnp.where(state.ren_id_board[WHITE] != -1, 1, board)
    return board  # type:ignore


def show(state: State) -> None:
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


def _show_details(state: State) -> None:
    show(state)
    print(state.ren_id_board[BLACK].reshape((5, 5)))
    print(state.ren_id_board[WHITE].reshape((5, 5)))
    print(state.kou)


def _my_color(_state: State):
    return jnp.int32(_state.turn % 2)


def _opponent_color(_state: State):
    return jnp.int32((_state.turn + 1) % 2)


def _is_off_board(_pos: jnp.ndarray, size) -> bool:
    x = _pos[0]
    y = _pos[1]
    return jnp.logical_or(
        jnp.logical_or(x < 0, size <= x),
        jnp.logical_or(y < 0, size <= y),
    )


def _kou_occurred(_state: State, xy: int) -> bool:
    size = _state.size
    x = xy // size
    y = xy % size

    oppo_color = _opponent_color(_state)

    return jnp.logical_and(
        jnp.logical_and(
            jnp.logical_and(
                jnp.logical_or(
                    x - 1 < 0,
                    _state.ren_id_board[oppo_color][_to_xy(x - 1, y, size)]
                    != -1,
                ),
                jnp.logical_or(
                    x + 1 >= size,
                    _state.ren_id_board[oppo_color][_to_xy(x + 1, y, size)]
                    != -1,
                ),
            ),
            jnp.logical_or(
                y - 1 < 0,
                _state.ren_id_board[oppo_color][_to_xy(x, y - 1, size)] != -1,
            ),
        ),
        jnp.logical_or(
            y + 1 >= size,
            _state.ren_id_board[oppo_color][_to_xy(x, y + 1, size)] != -1,
        ),
    )


def _to_xy(x, y, size) -> int:
    return x * size + y


def _get_reward(_state: State, _size: int) -> jnp.ndarray:
    b = _count_ji(_state, BLACK, _size) - _state.agehama[WHITE] - _state.komi
    w = _count_ji(_state, WHITE, _size) - _state.agehama[BLACK]
    r = jax.lax.cond(
        b > w, lambda: jnp.array([1, -1]), lambda: jnp.array([-1, 1])
    )

    return r


def _count_ji(_state: State, _color, _size):
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
