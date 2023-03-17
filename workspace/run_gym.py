import jax
import jax.numpy as jnp

import pgx.gym as gym

rng = jax.random.PRNGKey(0)
envs = gym.RandomOpponentEnv("tic_tac_toe/v0", 10, False, store_states=True)
obs, info = envs.reset(0)
rewards = []
observations = [obs]
for _ in range(10):
    legal_action_mask = info["legal_action_mask"]
    logits = jnp.log(legal_action_mask.astype(jnp.float16))
    action = jax.random.categorical(rng, logits=logits, axis=1)
    state, reward, terminated, truncated, info = envs.step(action)
    rewards.append(reward)
    observations.append(obs)
    if (terminated | truncated).all():
        break
