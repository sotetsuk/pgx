import jax
import jax.numpy as jnp


def act_randomly(
    rng: jax.random.KeyArray, legal_action_mask: jax.Array
) -> jax.Array:
    logits = jnp.log(legal_action_mask.astype(jnp.float32))
    return jax.random.categorical(rng, logits=logits, axis=1)
