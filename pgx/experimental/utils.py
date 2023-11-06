import jax
import jax.numpy as jnp

from pgx._src.types import Array


def act_randomly(rng: PRNGKey, legal_action_mask: Array) -> Array:
    logits = jnp.log(legal_action_mask.astype(jnp.float32))
    return jax.random.categorical(rng, logits=logits, axis=1)
