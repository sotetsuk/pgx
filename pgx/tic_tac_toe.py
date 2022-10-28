from typing import Tuple

import jax
import jax.numpy as jnp
from flax import struct

FALSE = jnp.zeros(1, jnp.bool_)
TRUE = jnp.ones(1, jnp.bool_)


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


def init(rng: jax.random.KeyArray) -> Tuple[jnp.ndarray, State]:
    curr_player = jax.random.bernoulli(rng)
    return curr_player, State(curr_player)


def step(
    state: State, action: jnp.ndarray
) -> Tuple[jnp.ndarray, State, jnp.ndarray]:
    # TODO(sotetsuk): illegal action check
    # if state.legal_action_mask.at[action]:
    #     ...
    board = state.board.at[action].set(state.turn[0])
    won = _win_check(board, state.turn)
    next_player = jnp.int8((state.curr_player + 1) % 2)

    rewards = jnp.zeros(2, jnp.int16)
    rewards = jax.lax.cond(
        jnp.all(won),
        lambda: rewards.at[jnp.int8(state.curr_player)].set(1),
        lambda: rewards,
    )
    rewards = jax.lax.cond(
        jnp.all(won),
        lambda: rewards.at[jnp.int8(next_player)].set(-1),
        lambda: rewards,
    )
    # if won:
    #     rewards = rewards.at[state.curr_player].set(1)
    #     rewards =

    state = State(
        curr_player=next_player,
        legal_action_mask=board < 0,
        terminated=won,
        turn=(state.turn + 1) % 2,
        board=board,
    )

    return next_player, state, rewards


def _win_check(board, turn):
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
        jnp.all((board[0] == turn) & (board[4] == turn) & (board[8] == turn)),
        lambda: TRUE,
        lambda: won,
    )
    won = jax.lax.cond(
        jnp.all((board[2] == turn) & (board[4] == turn) & (board[6] == turn)),
        lambda: TRUE,
        lambda: won,
    )
    return won


def observe(state: State) -> jnp.ndarray:
    ...
