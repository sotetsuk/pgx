import jax
import jax.numpy as jnp

import pgx


def act_randomly(rng: jax.random.KeyArray, state: pgx.State) -> jnp.ndarray:
    logits = jnp.log(state.legal_action_mask.astype(jnp.float16))
    return jax.random.categorical(rng, logits=logits, axis=1)
