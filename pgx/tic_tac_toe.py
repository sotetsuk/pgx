import jax
import jax.numpy as jnp
from flax.struct import dataclass

import pgx.core as core

FALSE = jnp.bool_(False)
TRUE = jnp.bool_(True)


@dataclass
class State(core.State):
    curr_player: jnp.ndarray = jnp.int8(0)
    reward: jnp.ndarray = jnp.float32([0.0, 0.0])
    terminated: jnp.ndarray = FALSE
    legal_action_mask: jnp.ndarray = jnp.ones(9, dtype=jnp.bool_)
    # 0: 先手, 1: 後手
    turn: jnp.ndarray = jnp.int8(0)
    # 0 1 2
    # 3 4 5
    # 6 7 8
    # -1: empty, 0: 先手, 1: 後手
    board: jnp.ndarray = -jnp.ones(9, jnp.int8)


class TicTacToe(core.Env):
    def __init__(self):
        super().__init__()

    def init(self, rng: jax.random.KeyArray) -> State:
        return init(rng)

    def _step(self, state: core.State, action: jnp.ndarray) -> State:
        assert isinstance(state, State)
        return step(state, action)

    def observe(
        self, state: core.State, player_id: jnp.ndarray
    ) -> jnp.ndarray:
        assert isinstance(state, State)
        return observe(state, player_id)

    def num_players(self) -> int:
        return 2


def init(rng: jax.random.KeyArray) -> State:
    rng, subkey = jax.random.split(rng)
    curr_player = jnp.int8(jax.random.bernoulli(subkey))
    return State(curr_player=curr_player)  # type:ignore


def step(state: State, action: jnp.ndarray) -> State:
    # TODO(sotetsuk): illegal action check
    # if state.legal_action_mask.at[action]:
    #     ...
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
    legal_action_mask = jax.lax.cond(
        terminated,
        lambda: jnp.zeros_like(legal_action_mask),
        lambda: legal_action_mask,
    )
    return State(
        curr_player=curr_player,
        legal_action_mask=legal_action_mask,
        reward=reward,
        terminated=terminated,
        turn=(state.turn + 1) % 2,
        board=board,
    )  # type: ignore


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
    return jnp.concatenate(
        [empty_board, my_board, opp_obard],
        dtype=jnp.float16,
    )
