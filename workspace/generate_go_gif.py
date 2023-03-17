from pgx.go import Go
from functools import partial
from pgx.experimental.utils import act_randomly
from pgx.experimental.wrappers import auto_reset
from pgx.visualizer import global_config
import jax
import time

global_config.color_theme = "dark"
N = 20

env = Go(size=5)
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
for i in range(1000 + 100):
    if i >= 1000 and i % 3 == 0:
        s.save_svg(f"{i % 1000:03d}.svg")
    rng, subkey = jax.random.split(rng)
    a = act_randomly(subkey, s)
    s = step(s, a)
et = time.time()
print(et - st)

"""
$ inkscape --export-type=png *.svg
$ convert *.png go.gif
"""
