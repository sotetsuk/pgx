import jax
import jax.numpy as jnp

import pgx
from pgx.flax.struct import dataclass

FALSE = jnp.bool_(False)
TRUE = jnp.bool_(True)

# fmt: off
IDX = jnp.int8([[0, 7, 14, 21], [1, 8, 15, 22], [2, 9, 16, 23], [3, 10, 17, 24], [4, 11, 18, 25], [5, 12, 19, 26], [6, 13, 20, 27], [7, 14, 21, 28], [8, 15, 22, 29], [9, 16, 23, 30], [10, 17, 24, 31], [11, 18, 25, 32], [12, 19, 26, 33], [13, 20, 27, 34], [14, 21, 28, 35], [15, 22, 29, 36], [16, 23, 30, 37], [17, 24, 31, 38], [18, 25, 32, 39], [19, 26, 33, 40], [20, 27, 34, 41], [0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [7, 8, 9, 10], [8, 9, 10, 11], [9, 10, 11, 12], [10, 11, 12, 13], [14, 15, 16, 17], [15, 16, 17, 18], [16, 17, 18, 19], [17, 18, 19, 20], [21, 22, 23, 24], [22, 23, 24, 25], [23, 24, 25, 26], [24, 25, 26, 27], [28, 29, 30, 31], [29, 30, 31, 32], [30, 31, 32, 33], [31, 32, 33, 34], [35, 36, 37, 38], [36, 37, 38, 39], [37, 38, 39, 40], [38, 39, 40, 41], [0, 8, 16, 24], [1, 9, 17, 25], [2, 10, 18, 26], [3, 11, 19, 27], [7, 15, 23, 31], [8, 16, 24, 32], [9, 17, 25, 33], [10, 18, 26, 34], [14, 22, 30, 38], [15, 23, 31, 39], [16, 24, 32, 40], [17, 25, 33, 41], [3, 9, 15, 21], [4, 10, 16, 22], [5, 11, 17, 23], [6, 12, 18, 24], [10, 16, 22, 28], [11, 17, 23, 29], [12, 18, 24, 30], [13, 19, 25, 31], [17, 23, 29, 35], [18, 24, 30, 36], [19, 25, 31, 37], [20, 26, 32, 38]])
# fmt: on


def make_cache():
    IDX = []
    # 縦
    for i in range(3):
        for j in range(7):
            a = i * 7 + j
            IDX.append([a, a + 7, a + 14, a + 21])
    # 横
    for i in range(6):
        for j in range(4):
            a = i * 7 + j
            IDX.append([a, a + 1, a + 2, a + 3])

    # 斜め
    for i in range(3):
        for j in range(4):
            a = i * 7 + j
            IDX.append([a, a + 8, a + 16, a + 24])
    for i in range(3):
        for j in range(3, 7):
            a = i * 7 + j
            IDX.append([a, a + 6, a + 12, a + 18])
    print(IDX)


@dataclass
class State(pgx.State):
    steps: jnp.ndarray = jnp.int32(0)
    current_player: jnp.ndarray = jnp.int8(0)
    observation: jnp.ndarray = jnp.zeros(27, dtype=jnp.bool_)
    reward: jnp.ndarray = jnp.float32([0.0, 0.0])
    terminated: jnp.ndarray = FALSE
    truncated: jnp.ndarray = FALSE
    legal_action_mask: jnp.ndarray = jnp.ones(7, dtype=jnp.bool_)
    _rng_key: jax.random.KeyArray = jax.random.PRNGKey(0)
    # ---
    turn: jnp.ndarray = jnp.int8(0)
    # 6x7 board
    # [[ 0,  1,  2,  3,  4,  5,  6],
    #  [ 7,  8,  9, 10, 11, 12, 13],
    #  [14, 15, 16, 17, 18, 19, 20],
    #  [21, 22, 23, 24, 25, 26, 27],
    #  [28, 29, 30, 31, 32, 33, 34],
    #  [35, 36, 37, 38, 39, 40, 41]]
    board: jnp.ndarray = -jnp.ones(42, jnp.int8)  # -1 (empty), 0, 1
    blank_row: jnp.ndarray = jnp.full(7, 5)


class ConnectFour(pgx.Env):
    def __init__(
        self,
        *,
        auto_reset: bool = False,
        max_truncation_steps: int = -1,
    ):
        super().__init__(
            auto_reset=auto_reset, max_truncation_steps=max_truncation_steps
        )

    def _init(self, key: jax.random.KeyArray) -> State:
        return init(key)

    def _step(self, state: pgx.State, action: jnp.ndarray) -> State:
        assert isinstance(state, State)
        return step(state, action)

    def _observe(
        self, state: pgx.State, player_id: jnp.ndarray
    ) -> jnp.ndarray:
        assert isinstance(state, State)
        return observe(state, player_id)

    @property
    def version(self) -> str:
        return "alpha"

    @property
    def num_players(self) -> int:
        return 2


def init(rng: jax.random.KeyArray) -> State:
    rng, subkey = jax.random.split(rng)
    current_player = jnp.int8(jax.random.bernoulli(subkey))
    return State(current_player=current_player)  # type:ignore


def step(state: State, action: jnp.ndarray) -> State:
    board = state.board
    row = state.blank_row[action]
    blank_row = state.blank_row.at[action].set(row - 1)
    board = board.at[_to_idx(row, action)].set(state.turn)
    won = _win_check(board, state.turn)
    reward = jax.lax.cond(
        won,
        lambda: jnp.float32([-1, -1]).at[state.current_player].set(1),
        lambda: jnp.zeros(2, jnp.float32),
    )
    return state.replace(  # type: ignore
        current_player=1 - state.current_player,
        legal_action_mask=blank_row >= 0,
        turn=1 - state.turn,
        board=board,
        blank_row=blank_row,
        terminated=won | jnp.all(blank_row == -1),
        reward=reward,
    )


def _to_idx(row, col):
    return row * 7 + col


def _win_check(board, turn) -> jnp.ndarray:
    return ((board[IDX] == turn).all(axis=1)).any()


def observe(state: State, player_id: jnp.ndarray) -> jnp.ndarray:
    ...
