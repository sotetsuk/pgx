# %%

import jax
import jax.numpy as jnp

import pgx.gym as gym
from pgx.visualizer import Visualizer

viz = Visualizer(color_mode="dark", scale=0.5)

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
# %%
viz.show_svg(envs.states[0])
print(observations[0])

# %%
viz.show_svg(envs.states[1])
print(rewards[0])
print(observations[1])

# %%
viz.show_svg(envs.states[2])
print(rewards[1])
print(observations[2])

# %%
viz.show_svg(envs.states[3])
print(rewards[2])
print(observations[3])

# %%
viz.show_svg(envs.states[4])
print(rewards[3])
print(observations[4])

# %%
viz.show_svg(envs.states[5])
print(rewards[4])
print(observations[5])
