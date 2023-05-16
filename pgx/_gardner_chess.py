import jax
import jax.numpy as jnp

import pgx.v1 as v1
from pgx._src.struct import dataclass

FALSE = jnp.bool_(False)


INIT_BOARD = jnp.int8(
    [
        4,
        1,
        0,
        -1,
        -4,
        2,
        1,
        0,
        -1,
        -2,
        3,
        1,
        0,
        -1,
        -3,
        5,
        1,
        0,
        -1,
        -5,
        6,
        1,
        0,
        -1,
        -6,
    ]
)


@dataclass
class State(v1.State):
    current_player: jnp.ndarray = jnp.int8(0)
    rewards: jnp.ndarray = jnp.float32([0.0, 0.0])
    terminated: jnp.ndarray = FALSE
    truncated: jnp.ndarray = FALSE
    legal_action_mask: jnp.ndarray = jnp.ones(1)  # TODO: fix me
    observation: jnp.ndarray = jnp.zeros((8, 8, 19), dtype=jnp.float32)
    _rng_key: jax.random.KeyArray = jax.random.PRNGKey(0)
    _step_count: jnp.ndarray = jnp.int32(0)
    # --- Chess specific ---
    _turn: jnp.ndarray = jnp.int8(0)
    _board: jnp.ndarray = INIT_BOARD  # 左上からFENと同じ形式で埋めていく
