import jax.random

from pgx.envs import MinAtar

batch_size = 128
episode_length = 100
env = MinAtar(game="asterix", batch_size=batch_size, auto_reset=True)
obs = env.reset(seed=0)
# print(obs.shape)
for i in range(episode_length):
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    action = jax.random.choice(subkey, 5, shape=(batch_size,))
    obs, r, done, _ = env.step(action)
    # print(i, obs.shape)
    print(done[:8])
