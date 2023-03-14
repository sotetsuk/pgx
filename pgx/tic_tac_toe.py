from typing import Tuple

import jax
import jax.numpy as jnp

import pgx.core as core
from pgx.flax.struct import dataclass

FALSE = jnp.bool_(False)
TRUE = jnp.bool_(True)

# fmt off
IDX = jnp.int8(
    [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [0, 3, 6],
        [1, 4, 7],
        [2, 5, 8],
        [0, 4, 8],
        [2, 4, 6],
    ]
)
# fmt on


@dataclass
class State(core.State):
    steps: jnp.ndarray = jnp.int32(0)
    curr_player: jnp.ndarray = jnp.int8(0)
    observation: jnp.ndarray = jnp.zeros(27, dtype=jnp.bool_)
    reward: jnp.ndarray = jnp.float32([0.0, 0.0])
    terminated: jnp.ndarray = FALSE
    truncated: jnp.ndarray = FALSE
    legal_action_mask: jnp.ndarray = jnp.ones(9, dtype=jnp.bool_)
    _rng_key: jax.random.KeyArray = jax.random.PRNGKey(0)
    # ---
    turn: jnp.ndarray = jnp.int8(0)
    # 0 1 2
    # 3 4 5
    # 6 7 8
    board: jnp.ndarray = -jnp.ones(9, jnp.int8)  # -1 (empty), 0, 1


class TicTacToe(core.Env):
    def __init__(self, *, auto_reset: bool = False,
                 max_truncation_steps: int = -1,
                 ):
        super().__init__(auto_reset=auto_reset, max_truncation_steps=max_truncation_steps)

    def _init(self, key: jax.random.KeyArray) -> State:
        return init(key)

    def _step(self, state: core.State, action: jnp.ndarray) -> State:
        assert isinstance(state, State)
        return step(state, action)

    def observe(
        self, state: core.State, player_id: jnp.ndarray
    ) -> jnp.ndarray:
        assert isinstance(state, State)
        return observe(state, player_id)

    @property
    def num_players(self) -> int:
        return 2

    @property
    def reward_range(self) -> Tuple[float, float]:
        return -1.0, 1.0


def init(rng: jax.random.KeyArray) -> State:
    rng, subkey = jax.random.split(rng)
    curr_player = jnp.int8(jax.random.bernoulli(subkey))
    return State(curr_player=curr_player)  # type:ignore


def step(state: State, action: jnp.ndarray) -> State:
    board = state.board.at[action].set(state.turn)
    won = _win_check(board, state.turn)
    reward = jax.lax.cond(
        won,
        lambda: jnp.float32([-1, -1]).at[state.curr_player].set(1),
        lambda: jnp.zeros(2, jnp.float32),
    )
    terminated = won | jnp.all(board != -1)
    curr_player = (state.curr_player + 1) % 2
    legal_action_mask = board < 0
    return State(
        curr_player=curr_player,
        legal_action_mask=legal_action_mask,
        reward=reward,
        terminated=terminated,
        turn=(state.turn + 1) % 2,
        board=board,
    )  # type: ignore


def _win_check(board, turn) -> jnp.ndarray:
    return ((board[IDX] == turn).all(axis=1)).any()


def observe(state: State, player_id: jnp.ndarray) -> jnp.ndarray:
    empty_board = state.board == -1
    my_board, opp_obard = jax.lax.cond(
        state.curr_player == player_id,  # flip board if player_id is opposite
        lambda: (state.turn == state.board, (1 - state.turn) == state.board),
        lambda: ((1 - state.turn) == state.board, state.turn == state.board),
    )
    return jnp.concatenate(
        [empty_board, my_board, opp_obard],
        dtype=jnp.bool_,
    )
