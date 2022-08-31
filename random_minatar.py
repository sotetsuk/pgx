import jax.random
from tqdm import tqdm

from pgx.envs import MinAtar

batch_size = 10_000
N = 100
env = MinAtar(game="breakout", batch_size=batch_size, auto_reset=False)
obs = env.reset(seed=0)
# print(obs.shape)
for i in tqdm(range(N)):
    while True:
        key = jax.random.PRNGKey(0)
        key, subkey = jax.random.split(key)
        action = jax.random.choice(subkey, 5, shape=(batch_size,))
        obs, r, done, _ = env.step(action)
        # print(i, obs.shape)
        if done.all():
            break
