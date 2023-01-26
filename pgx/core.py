from typing import Literal

import jax
import jax.numpy as jnp
from flax.struct import dataclass

EnvId = Literal[
    "tic_tac_toe/v0",
    "minatar/breakout/v0",
    "suzume_jong/v0",
    "go/v0",
]


@dataclass
class State:
    rng: jax.random.KeyArray
    curr_player: jnp.int8
    reward: jnp.float32
    terminated: jnp.bool_
    legal_action_mask: jnp.bool_
