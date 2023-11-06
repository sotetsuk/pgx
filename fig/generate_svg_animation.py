import sys
import jax
import pgx
from pgx.experimental.utils import act_randomly

env_id: pgx.EnvId = sys.argv[1]
env = pgx.make(env_id, auto_reset=True)
init = jax.jit(env.init)
step = jax.jit(env.step)

rng = jax.random.PRNGKey(9999)

states = []
rng, subkey = jax.random.split(rng)
state = init(subkey)
states.append(state)
# while not state.terminated.all():
for _ in range(18):
    rng, subkey = jax.random.split(rng)
    action = act_randomly(subkey, state.legal_action_mask)
    state = step(state, action)
    states.append(state)

pgx.save_svg_animation(states, f"{env_id}_light.svg", frame_duration_seconds=0.3)
pgx.save_svg_animation(states, f"{env_id}_dark.svg", frame_duration_seconds=0.3, color_theme="dark")
