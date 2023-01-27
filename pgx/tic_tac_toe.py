import jax
import jax.numpy as jnp
from flax.struct import dataclass

import pgx.core

FALSE = jnp.bool_(False)
TRUE = jnp.bool_(True)


@dataclass
class State(pgx.core.State):
    rng: jax.random.KeyArray = jax.random.PRNGKey(0)
    curr_player: jnp.ndarray = jnp.int8(0)
    reward: jnp.ndarray = jnp.float32([0.0, 0.0])
    terminated: jnp.ndarray = jnp.bool_(False)
    legal_action_mask: jnp.ndarray = jnp.ones(9, dtype=jnp.bool_)
    # 0: 先手, 1: 後手
    turn: jnp.ndarray = jnp.int8(0)
    # 0 1 2
    # 3 4 5
    # 6 7 8
    # -1: empty, 0: 先手, 1: 後手
    board: jnp.ndarray = -jnp.ones(9, jnp.int8)


def init(rng: jax.random.KeyArray) -> State:
    rng, subkey = jax.random.split(rng)
    curr_player = jnp.int8(jax.random.bernoulli(subkey))
    return State(rng=rng, curr_player=curr_player)  # type:ignore


def step(state: State, action: jnp.ndarray) -> State:
    # TODO(sotetsuk): illegal action check
    # if state.legal_action_mask.at[action]:
    #     ...
    board = state.board.at[action].set(state.turn)
    won = _win_check(board, state.turn)
    reward = jax.lax.cond(
        won,
        lambda: jnp.int16([-1, -1]).at[state.curr_player].set(1),
        lambda: jnp.zeros(2, jnp.int16),
    )
    terminated = won | jnp.all(board != -1)
    curr_player = jax.lax.cond(
        terminated, lambda: jnp.int8(-1), lambda: (state.curr_player + 1) % 2
    )
    state = State(
        curr_player=curr_player,
        legal_action_mask=board < 0,
        reward=reward,
        terminated=terminated,
        turn=(state.turn + 1) % 2,
        board=board,
    )  # type: ignore
    return state


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


def observe(state: State, player_id: jnp.ndarray) -> jnp.ndarray:
    empty_board = state.board == -1
    my_board, opp_obard = jax.lax.cond(
        state.curr_player == player_id,  # flip board if player_id is opposite
        lambda: (state.turn == state.board, (1 - state.turn) == state.board),
        lambda: ((1 - state.turn) == state.board, state.turn == state.board),
    )
    obs = jnp.concatenate(
        [empty_board, my_board, opp_obard],
        dtype=jnp.float16,
    )
    # Zero if player_id is -1
    obs = jax.lax.cond(
        player_id == -1, lambda: jnp.zeros_like(obs), lambda: obs
    )
    return obs
