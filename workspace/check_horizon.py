import sys
import jax
import pgx
from pgx.experimental.utils import act_randomly

env_id = sys.argv[1]
env = pgx.make(env_id)
init = jax.jit(jax.vmap(env.init))  # vectorize and JIT-compile
step = jax.jit(jax.vmap(env.step))

batch_size = 32
num_reps = 10
key = jax.random.PRNGKey(42)


def random_play_length(key):
    def step_fn(x):
        k, s = x
        k, subkey = jax.random.split(k)
        action = act_randomly(subkey, s)
        s = step(s, action)
        return k, s

    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, batch_size)
    state = init(keys)  # vectorized states
    _, state = jax.lax.while_loop(
        lambda x: ~(x[1].terminated | x[1].terminated).all(),
        step_fn,
        (key, state)
    )

    return state._step_count


lengths = jax.lax.map(
        random_play_length, jax.random.split(key, num_reps)
)
print(lengths.shape)
print(lengths.mean())
print(lengths.min())
print(lengths.max())

import matplotlib.pyplot as plt

plt.figure()
plt.hist(lengths.flatten(), bins=25)
plt.show()
