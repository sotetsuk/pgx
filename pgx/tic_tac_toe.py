from typing import Tuple

import jax
import jax.numpy as jnp
from flax import struct

FALSE = jnp.bool_(False)
TRUE = jnp.bool_(True)


@struct.dataclass
class State:
    curr_player: jnp.ndarray = jnp.int8(0)
    legal_action_mask: jnp.ndarray = jnp.ones(9, jnp.bool_)
    terminated: jnp.ndarray = jnp.bool_(False)
    # 0: 先手, 1: 後手
    turn: jnp.ndarray = jnp.int8(0)
    # 0 1 2
    # 3 4 5
    # 6 7 8
    # -1: empty, 0: 先手, 1: 後手
    board: jnp.ndarray = -jnp.ones(9, jnp.int8)


def init(rng: jax.random.KeyArray) -> Tuple[jnp.ndarray, State]:
    curr_player = jnp.int8(jax.random.bernoulli(rng))
    return curr_player, State(curr_player=curr_player)  # type:ignore


def step(
    state: State, action: jnp.ndarray
) -> Tuple[jnp.ndarray, State, jnp.ndarray]:
    # TODO(sotetsuk): illegal action check
    # if state.legal_action_mask.at[action]:
    #     ...
    board = state.board.at[action].set(state.turn)
    won = _win_check(board, state.turn)
    rewards = jax.lax.cond(
        won,
        lambda: jnp.int16([-1, -1]).at[state.curr_player].set(1),
        lambda: jnp.zeros(2, jnp.int16),
    )
    terminated = won | jnp.all(board != -1)
    curr_player = (state.curr_player + 1) % 2
    state = State(
        curr_player=curr_player,
        legal_action_mask=board < 0,
        terminated=terminated,
        turn=(state.turn + 1) % 2,
        board=board,
    )  # type: ignore
    return curr_player, state, rewards


def _win_check(board, turn) -> jnp.ndarray:
    # board:
    #   0 1 2
    #   3 4 5
    #   6 7 8
    won = FALSE
    for i in range(0, 9, 3):
        # e.g., [0, 1, 2]
        won = jax.lax.cond(
            jnp.all(board[i : i + 3] == turn), lambda: TRUE, lambda: won
        )
    for i in range(3):
        # e.g., [0, 3, 6]
        won = jax.lax.cond(
            jnp.all(board[i:9:3] == turn), lambda: TRUE, lambda: won
        )
    won = jax.lax.cond(
        jnp.all(board[jnp.int8([0, 4, 8])] == turn),
        lambda: TRUE,
        lambda: won,
    )
    won = jax.lax.cond(
        jnp.all(board[jnp.int8([2, 4, 6])] == turn),
        lambda: TRUE,
        lambda: won,
    )
    return won


def observe(state: State) -> jnp.ndarray:
    ...
