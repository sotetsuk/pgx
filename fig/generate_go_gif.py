import pgx
from pgx.go import Go
from pgx.experimental.utils import act_randomly
from pgx.experimental.wrappers import auto_reset
import jax
import time

N = 9

env = Go(size=9)
init = jax.jit(jax.vmap(env.init))
step = jax.jit(jax.vmap(auto_reset(env.step, env.init)))

rng = jax.random.PRNGKey(0)
rng, subkey = jax.random.split(rng)
subkeys = jax.random.split(subkey, N)
# warmup
print("warmup starts ...")
s = init(subkeys)
rng, subkey = jax.random.split(rng)
a = act_randomly(subkey, s)
s = step(s, a)
print("warmup ends")

st = time.time()
s = init(subkeys)
to_display_states = []
for i in range(1000 + 200):
    if i >= 1000 and i % 5 == 0:
        to_display_states.append(s)
    rng, subkey = jax.random.split(rng)
    a = act_randomly(subkey, s)
    s = step(s, a)
et = time.time()


pgx.save_svg_animation(to_display_states, "go_dark.svg", color_theme="dark", frame_duration_seconds=0.1)
pgx.save_svg_animation(to_display_states, "go_light.svg", color_theme="light", frame_duration_seconds=0.1)

print(et - st)

"""
$ inkscape --export-type=png *.svg
$ convert *.png go.gif
"""
