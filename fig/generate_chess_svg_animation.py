import jax

import pgx
from pgx.experimental.utils import act_randomly

env = pgx.make("chess")
init = jax.jit(jax.vmap(env.init))
step = jax.jit(jax.vmap(env.step))

batch_size = 4
rng = jax.random.PRNGKey(9999)

states = []
rng, subkey = jax.random.split(rng)
keys = jax.random.split(subkey, batch_size)
state = init(keys)
states.append(state)
while not state.terminated.all():
    rng, subkey = jax.random.split(rng)
    action = act_randomly(subkey, state)
    state = step(state, action)
    states.append(state)

pgx.save_svg_animation(states, "chess.svg", frame_duration_seconds=0.1)
pgx.save_svg(states[0], "chess_initial.svg")
pgx.save_svg(states[-1], "chess_final.svg")
