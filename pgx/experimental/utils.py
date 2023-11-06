import jax
import jax.numpy as jnp


def act_randomly(rng: Array, legal_action_mask: Array) -> jax.Array:
    logits = jnp.log(legal_action_mask.astype(jnp.float32))
    return jax.random.categorical(rng, logits=logits, axis=1)
