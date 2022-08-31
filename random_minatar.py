import jax.random

from pgx.envs import MinAtar

batch_size = 8
episode_length = 20
env = MinAtar(game="breakout", batch_size=batch_size, auto_reset=True)
obs, info = env.reset(seed=0)
# print(obs.shape)
for _ in range(episode_length):
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    action = jax.random.choice(subkey, 5, shape=(batch_size,))
    obs, r, done, _ = env.step(action)
    # print(obs.shape)
    print(done)
