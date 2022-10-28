from typing import Tuple

import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class State:
    curr_player: jnp.ndarray = jnp.zeros(1, jnp.int8)
    legal_action_mask: jnp.ndarray = jnp.ones(9, jnp.bool_)
    terminated: jnp.ndarray = jnp.zeros(0, jnp.bool_)
    # 0: 先手, 1: 後手
    turn: jnp.ndarray = jnp.zeros(1, jnp.int8)
    # 0 1 2
    # 3 4 5
    # 6 7 8
    # -1: empty, 0: 先手, 1: 後手
    board: jnp.ndarray = -jnp.ones(9, jnp.int8)


@jax.jit
def init(rng: jax.random.KeyArray) -> Tuple[jnp.ndarray, State]:
    curr_player = jnp.int8(jax.random.bernoulli(rng))
    return curr_player, State(curr_player=curr_player)  # type:ignore


def step(
    state: State, action: jnp.ndarray
) -> Tuple[jnp.ndarray, State, jnp.ndarray]:
    # TODO(sotetsuk): illegal action check
    # if state.legal_action_mask.at[action]:
    #     ...
    board = state.board.at[action].set(state.turn[0])
    won = _win_check(board, state.turn)

    rewards = jax.lax.cond(
        won,
        lambda: jnp.int16([-1, -1]).at[state.curr_player].set(1),
        lambda: jnp.zeros(2, jnp.int16),
    )

    curr_player = (state.curr_player + 1) % 2
    state = State(
        curr_player=curr_player,
        legal_action_mask=board < 0,
        terminated=jnp.bool_(won),
        turn=(state.turn + 1) % 2,
        board=board,
    )  # type: ignore
    return curr_player, state, rewards


def _win_check(board, turn) -> bool:
    # board:
    #   0 1 2
    #   3 4 5
    #   6 7 8
    won = False
    for i in range(0, 9, 3):
        # e.g., [0, 1, 2]
        won = jax.lax.cond(
            jnp.all(board[i : i + 3] == turn), lambda: True, lambda: won
        )
    for i in range(3):
        # e.g., [0, 3, 6]
        won = jax.lax.cond(
            jnp.all(board[i:9:3] == turn), lambda: True, lambda: won
        )
    won = jax.lax.cond(
        jnp.all((board[0] == turn) & (board[4] == turn) & (board[8] == turn)),
        lambda: True,
        lambda: won,
    )
    won = jax.lax.cond(
        jnp.all((board[2] == turn) & (board[4] == turn) & (board[6] == turn)),
        lambda: True,
        lambda: won,
    )
    return won


def observe(state: State) -> jnp.ndarray:
    ...
