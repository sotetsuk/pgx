from functools import partial

import jax
from jax import numpy as jnp

import pgx
from pgx.flax.struct import dataclass

BLACK = 1
WHITE = -1
POINT = 2
BLACK_CHAR = "@"
WHITE_CHAR = "O"
POINT_CHAR = "+"

dx = jnp.int32([-1, +1, 0, 0])
dy = jnp.int32([0, 0, -1, +1])

FALSE = jnp.bool_(False)
TRUE = jnp.bool_(True)


@dataclass
class State(pgx.State):
    steps: jnp.ndarray = jnp.int32(0)
    current_player: jnp.ndarray = jnp.int8(0)
    reward: jnp.ndarray = jnp.float32([0.0, 0.0])
    terminated: jnp.ndarray = FALSE
    truncated: jnp.ndarray = FALSE
    legal_action_mask: jnp.ndarray = jnp.zeros(19 * 19 + 1, dtype=jnp.bool_)
    observation: jnp.ndarray = jnp.zeros((17, 19, 19), dtype=jnp.bool_)
    _rng_key: jax.random.KeyArray = jax.random.PRNGKey(0)
    # ---
    size: jnp.ndarray = jnp.int32(19)  # NOTE: require 19 * 19 > int8
    # ids of representative stone id (smallest) in the connected stones
    # positive for black, negative for white, and zero for empty.
    # require at least 19 * 19 > int8, idx_squared_sum can be 361^2 > int16
    chain_id_board: jnp.ndarray = jnp.zeros(19 * 19, dtype=jnp.int32)
    board_history: jnp.ndarray = jnp.full((8, 19 * 19), 2, dtype=jnp.int8)
    turn: jnp.ndarray = jnp.int32(0)
    num_captured_stones: jnp.ndarray = jnp.zeros(2, dtype=jnp.int32)
    passed: jnp.ndarray = FALSE  # TRUE if last action is pass
    ko: jnp.ndarray = jnp.int32(-1)  # by SSK
    komi: jnp.ndarray = jnp.float32(7.5)
    black_player: jnp.ndarray = jnp.int8(0)


class Go(pgx.Env):
    def __init__(
        self,
        *,
        auto_reset: bool = False,
        max_truncation_steps: int = -1,
        size: int = 19,
        komi: float = 7.5,
        history_length: int = 8
    ):
        super().__init__(
            auto_reset=auto_reset, max_truncation_steps=max_truncation_steps
        )
        self.size = size
        self.komi = komi
        self.history_length = history_length
        self.max_termination_steps = self.size * self.size * 2

    def _init(self, key: jax.random.KeyArray) -> State:
        return partial(init, size=self.size, komi=self.komi)(key=key)

    def _step(self, state: pgx.State, action: jnp.ndarray) -> State:
        assert isinstance(state, State)
        state = partial(step, size=self.size)(state, action)
        # terminates if size * size * 2 (722 if size=19) steps are elapsed
        state = jax.lax.cond(
            (0 <= self.max_termination_steps)
            & (self.max_termination_steps <= state.steps),
            lambda: state.replace(  # type: ignore
                terminated=TRUE,
                reward=partial(_get_reward, size=self.size)(state),
            ),
            lambda: state,
        )
        return state  # type: ignore

    def _observe(
        self, state: pgx.State, player_id: jnp.ndarray
    ) -> jnp.ndarray:
        assert isinstance(state, State)
        return partial(
            observe, size=self.size, history_length=self.history_length
        )(state=state, player_id=player_id)

    @property
    def version(self) -> str:
        return "beta"

    @property
    def num_players(self) -> int:
        return 2


def observe(state: State, player_id, size, history_length):
    """Return AlphaGoZero [Silver+17] feature

        obs = (size, size, history_length * 2 + 1)
        e.g., (19, 19, 17) if size=19 and history_length=8 (used in AlphaZero)

        obs[:, :, 0]: stones of `player_id`          @ current board
        obs[:, :, 1]: stones of `player_id` opponent @ current board
        obs[:, :, 2]: stones of `player_id`          @ 1-step before
        obs[:, :, 3]: stones of `player_id` opponent @ 1-step before
        ...
        obs[:, :, -1]: color of `player_id`

        NOTE: For the final dimension, there are two possible options:

          - Use the color of current player to play
          - Use the color of `player_id`

        This ambiguity happens because `observe` function is available even if state.current_player != player_id.
        In the AlphaGoZero paper, the final dimension C is explained as:

          > The final feature plane, C, represents the colour to play, and has a constant value of either 1 if black
    is to play or 0 if white is to play.

        however, it also describes as

          > the colour feature C is necessary because the komi is not observable.

        So, we use player_id's color to let the agent komi information.
        As long as it's called when state.current_player == player_id, this doesn't matter.
    """
    current_player_color = _my_color(state)  # -1 or 1
    my_color, opp_color = jax.lax.cond(
        player_id == state.current_player,
        lambda: (current_player_color, -1 * current_player_color),
        lambda: (-1 * current_player_color, current_player_color),
    )

    @jax.vmap
    def _make(i):
        color = jnp.int8([1, -1])[i % 2] * my_color
        return state.board_history[i // 2] == color

    log = _make(jnp.arange(history_length * 2))
    color = jnp.full_like(log[0], my_color == 1)  # black=1, white=0

    return jnp.vstack([log, color]).transpose().reshape((size, size, -1))


def init(key: jax.random.KeyArray, size: int, komi: float = 7.5) -> State:
    black_player = jnp.int8(jax.random.bernoulli(key))
    current_player = black_player
    return State(  # type:ignore
        size=jnp.int32(size),
        chain_id_board=jnp.zeros(size**2, dtype=jnp.int32),
        legal_action_mask=jnp.ones(size**2 + 1, dtype=jnp.bool_),
        board_history=jnp.full((8, size**2), 2, dtype=jnp.int8),
        current_player=current_player,
        komi=jnp.float32(komi),
        black_player=black_player,
    )


def step(state: State, action: int, size: int) -> State:
    state = state.replace(ko=jnp.int32(-1))  # type: ignore
    # update state
    _state = _update_state_wo_legal_action(state, action, size)

    # add legal actions
    _state = _state.replace(  # type:ignore
        legal_action_mask=_state.legal_action_mask.at[:-1]
        .set(legal_actions(_state, size))
        .at[-1]
        .set(TRUE)
    )

    # update log
    new_log = jnp.roll(_state.board_history, size**2)
    new_log = new_log.at[0].set(
        jnp.clip(_state.chain_id_board, -1, 1).astype(jnp.int8)
    )
    return _state.replace(board_history=new_log)  # type:ignore


def _update_state_wo_legal_action(
    _state: State, _action: int, _size: int
) -> State:
    _state = jax.lax.cond(
        (_action < _size * _size),
        lambda: _not_pass_move(_state, _action, _size),
        lambda: _pass_move(_state, _size),
    )

    # increase turn
    _state = _state.replace(turn=_state.turn + 1)  # type: ignore

    # change player
    _state = _state.replace(current_player=(_state.current_player + 1) % 2)  # type: ignore

    return _state


def _pass_move(_state: State, _size: int) -> State:
    return jax.lax.cond(
        _state.passed,
        # 連続でパスならば終局
        lambda: _state.replace(terminated=TRUE, reward=_get_reward(_state, _size)),  # type: ignore
        # 1回目のパスならばStateにパスを追加してそのまま続行
        lambda: _state.replace(passed=True, reward=jnp.zeros(2, dtype=jnp.float32)),  # type: ignore
    )


def _not_pass_move(_state: State, _action: int, size) -> State:
    state = _state.replace(passed=FALSE)  # type: ignore
    xy = _action
    my_color_ix = _my_color_ix(state)
    num_captured_stones_before = state.num_captured_stones[my_color_ix]

    ko_may_occur = _ko_may_occur(state, xy)

    # 周囲の連から敵石を除く
    adj_xy = _neighbour(xy, size)
    oppo_color = _opponent_color(state)
    ren_id = state.chain_id_board[adj_xy]
    num_pseudo, idx_sum, idx_squared_sum = _count(state, size)
    ren_ix = jnp.abs(ren_id) - 1
    # fmt: off
    is_atari = ((idx_sum[ren_ix] ** 2) == idx_squared_sum[ren_ix] * num_pseudo[ren_ix])
    single_liberty = (idx_squared_sum[ren_ix] // idx_sum[ren_ix]) - 1
    is_killed = (adj_xy != -1) & (ren_id * oppo_color > 0) & is_atari & (single_liberty == xy)
    state = jax.lax.fori_loop(
        0, 4,
        lambda i, s: jax.lax.cond(
            is_killed[i],
            lambda: _remove_stones(s, ren_id[i], adj_xy[i], ko_may_occur),
            lambda: s,
        ),
        state,
    )
    # fmt: on
    state = _set_stone(state, xy)

    # 周囲をマージ
    state = jax.lax.fori_loop(
        0, 4, lambda i, s: _merge_around_xy(i, s, xy, size), state
    )

    # コウの確認
    state = jax.lax.cond(
        state.num_captured_stones[my_color_ix] - num_captured_stones_before
        == 1,
        lambda: state,
        lambda: state.replace(ko=jnp.int32(-1)),  # type:ignore
    )

    return state.replace(reward=jnp.zeros(2, dtype=jnp.float32))  # type: ignore


def _merge_around_xy(i, state: State, xy, size):
    my_color = _my_color(state)
    adj_xy = _neighbour(xy, size)[i]
    is_off = adj_xy == -1
    is_my_ren = state.chain_id_board[adj_xy] * my_color > 0
    state = jax.lax.cond(
        ((~is_off) & is_my_ren),
        lambda: _merge_ren(state, xy, adj_xy),
        lambda: state,
    )
    return state


def _set_stone(_state: State, _xy: int) -> State:
    my_color = _my_color(_state)
    return _state.replace(  # type:ignore
        chain_id_board=_state.chain_id_board.at[_xy].set((_xy + 1) * my_color),
    )


def _merge_ren(_state: State, _xy: int, _adj_xy: int):
    my_color = _my_color(_state)
    new_id = jnp.abs(_state.chain_id_board[_xy])
    adj_ren_id = jnp.abs(_state.chain_id_board[_adj_xy])
    # fmt: off
    small_id = jnp.minimum(new_id, adj_ren_id) * my_color
    large_id = jnp.maximum(new_id, adj_ren_id) * my_color
    # fmt: on

    # 大きいidの連を消し、小さいidの連と繋げる
    chain_id_board = jnp.where(
        _state.chain_id_board == large_id, small_id, _state.chain_id_board
    )

    return _state.replace(  # type:ignore
        chain_id_board=chain_id_board,
    )


def _remove_stones(
    _state: State, _rm_ren_id, _rm_stone_xy, ko_may_occur
) -> State:
    surrounded_stones = _state.chain_id_board == _rm_ren_id
    num_captured_stones = jnp.count_nonzero(surrounded_stones)
    chain_id_board = jnp.where(surrounded_stones, 0, _state.chain_id_board)
    ko = jax.lax.cond(
        ko_may_occur & (num_captured_stones == 1),
        lambda: jnp.int32(_rm_stone_xy),
        lambda: _state.ko,
    )
    return _state.replace(  # type:ignore
        chain_id_board=chain_id_board,
        num_captured_stones=_state.num_captured_stones.at[
            _my_color_ix(_state)
        ].add(num_captured_stones),
        ko=ko,
    )


def legal_actions(state: State, size: int) -> jnp.ndarray:
    is_empty = state.chain_id_board == 0

    my_color = _my_color(state)
    opp_color = _opponent_color(state)
    num_pseudo, idx_sum, idx_squared_sum = _count(state, size)

    ren_ix = jnp.abs(state.chain_id_board) - 1
    # fmt: off
    in_atari = (idx_sum[ren_ix] ** 2) == idx_squared_sum[ren_ix] * num_pseudo[ren_ix]
    # fmt: on
    has_liberty = (state.chain_id_board * my_color > 0) & ~in_atari
    kills_opp = (state.chain_id_board * opp_color > 0) & in_atari

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
        (state.ko == -1),
        lambda: legal_action_mask,
        lambda: legal_action_mask.at[state.ko].set(FALSE),
    )


def _count(state: State, size):
    ZERO = jnp.int32(0)
    chain_id_board = jnp.abs(state.chain_id_board)
    is_empty = chain_id_board == 0
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
    num_pseudo, idx_sum, idx_squared_sum = _count_neighbor(idx)

    @jax.vmap
    def _num_pseudo(x):
        return jnp.where(chain_id_board == (x + 1), num_pseudo, ZERO).sum()

    @jax.vmap
    def _idx_sum(x):
        return jnp.where(chain_id_board == (x + 1), idx_sum, ZERO).sum()

    @jax.vmap
    def _idx_squared_sum(x):
        return jnp.where(
            chain_id_board == (x + 1), idx_squared_sum, ZERO
        ).sum()

    return _num_pseudo(idx), _idx_sum(idx), _idx_squared_sum(idx)


def show(state: State) -> None:
    print("===========")
    for xy in range(state.size * state.size):
        if state.chain_id_board[xy] > 0:
            print(" " + BLACK_CHAR, end="")
        elif state.chain_id_board[xy] < 0:
            print(" " + WHITE_CHAR, end="")
        else:
            print(" " + POINT_CHAR, end="")

        if xy % state.size == state.size - 1:
            print()


def _show_details(state: State) -> None:
    show(state)
    print(state.chain_id_board.reshape((5, 5)))
    print(state.ko)


def _my_color(_state: State):
    return jnp.int32([1, -1])[_state.turn % 2]


def _my_color_ix(_state: State):
    return _state.turn % 2


def _opponent_color(_state: State):
    return jnp.int32([-1, 1])[_state.turn % 2]


def _opponent_color_ix(_state: State):
    return (_state.turn + 1) % 2


def _ko_may_occur(_state: State, xy: int) -> jnp.ndarray:
    size = _state.size
    x = xy // size
    y = xy % size
    oob = jnp.bool_([x - 1 < 0, x + 1 >= size, y - 1 < 0, y + 1 >= size])
    oppo_color = _opponent_color(_state)
    is_occupied_by_opp = (
        _state.chain_id_board[_neighbour(xy, size)] * oppo_color > 0
    )
    return (oob | is_occupied_by_opp).all()


def _count_point(state, size):
    # NEED FIX: Japanese rule → Tromp-Taylor rule
    return jnp.array(
        [
            _count_ji(state, 1, size)
            + jnp.count_nonzero(state.chain_id_board > 0),
            _count_ji(state, -1, size)
            + jnp.count_nonzero(state.chain_id_board < 0),
        ],
        dtype=jnp.float32,
    )


def _get_reward(_state: State, size: int) -> jnp.ndarray:
    score = _count_point(_state, size)
    reward_bw = jax.lax.cond(
        score[0] - _state.komi > score[1],
        lambda: jnp.array([1, -1], dtype=jnp.float32),
        lambda: jnp.array([-1, 1], dtype=jnp.float32),
    )
    black_player = _state.black_player
    reward = jax.lax.cond(
        black_player == 0,
        lambda: reward_bw,
        lambda: reward_bw[jnp.int8([1, 0])],
    )

    return reward


def _neighbour(xy, size):
    xs = xy // size + dx
    ys = xy % size + dy
    on_board = (0 <= xs) & (xs < size) & (0 <= ys) & (ys < size)
    return jnp.where(on_board, xs * size + ys, -1)


def _neighbours(size):
    return jax.vmap(partial(_neighbour, size=size))(jnp.arange(size**2))


def _count_ji(state: State, color: int, size: int):
    board = jnp.zeros_like(state.chain_id_board)
    board = jnp.where(state.chain_id_board * color > 0, 1, board)
    board = jnp.where(state.chain_id_board * color < 0, -1, board)
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
