import os
import sys
import jax
import pgx
from pgx.experimental.utils import act_randomly

os.makedirs("tmp", exist_ok=True)

env_id: pgx.EnvId = sys.argv[1]
color_theme = sys.argv[2]
env = pgx.make(env_id, auto_reset=True)
init = jax.jit(env.init)
step = jax.jit(env.step)

rng = jax.random.PRNGKey(9999)

states = []
rng, subkey = jax.random.split(rng)
state = init(subkey)
# while not state.terminated.all():
for i in range(50):
    state.save_svg(f"tmp/{env_id}_{i:03d}.svg", color_theme=color_theme)
    rng, subkey = jax.random.split(rng)
    action = act_randomly(subkey, state.legal_action_mask)
    state = step(state, action)
