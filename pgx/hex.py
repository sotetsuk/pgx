import jax.numpy as jnp

import pgx.core as core
from pgx.flax.struct import dataclass

FALSE = jnp.bool_(False)
TRUE = jnp.bool_(True)


@dataclass
class State(core.State):
    size: jnp.ndarray = jnp.int8(11)
    curr_player: jnp.ndarray = jnp.int8(0)
    observation: jnp.ndarray = jnp.zeros(11 * 11, dtype=jnp.bool_)
    reward: jnp.ndarray = jnp.float32([0.0, 0.0])
    terminated: jnp.ndarray = FALSE
    legal_action_mask: jnp.ndarray = jnp.ones(11 * 11, dtype=jnp.bool_)
    # ---
    turn: jnp.ndarray = jnp.int8(0)
    # 11x11 board
    # [[  0,  1,  2,  ...,  8,  9, 10],
    #  [ 11,  12, 13, ..., 19, 20, 21],
    #  .
    #  .
    #  .
    #  [110, 111, 112, ...,  119, 120]]
    board: jnp.ndarray = -jnp.ones(
        11 * 11, jnp.int8
    )  # -1(oppo), 0(empty), 1(self)
