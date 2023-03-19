import sys
import json
import time
import jax
import pgx
from pgx.experimental.utils import act_randomly


env_id: pgx.EnvId = sys.argv[1]
env = pgx.make(env_id)

# run api test
pgx.api_test(env, 100)

# jit
init = jax.jit(jax.vmap(env.init))
step = jax.jit(jax.vmap(env.step))
act_randomly = jax.jit(act_randomly)

# show multi visualizations
N = 4
key = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key)
keys = jax.random.split(subkey, N)
state: pgx.State = init(keys)
i = 0
while not state.terminated.all():
    key, subkey = jax.random.split(key)
    action = act_randomly(subkey, state)
    print("================")
    print(f"{i:04d}")
    print("================")
    print(f"curr_player: {state.current_player}\naction: {action}")
    state.save_svg(f"{i:04d}.svg")
    state = step(state, action)
    print(f"reward:\n{state.reward}")
    i += 1
state.save_svg(f"{i:04d}.svg")
