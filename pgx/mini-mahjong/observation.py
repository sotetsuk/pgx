from dataclasses import dataclass

import jax.numpy as jnp


@dataclass
class Observation:
    hand: jnp.ndarray
    target: int
