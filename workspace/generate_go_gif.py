from pgx.go import init, step
from functools import partial
from pgx.utils import act_randomly
from pgx.visualizer import Visualizer
import jax
import time
v = Visualizer(color_mode="dark")

N = 20

init = jax.jit(jax.vmap(partial(init, size=5)))
step = jax.jit(jax.vmap(partial(step, size=5)))

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
        v.save_svg(s, f"{i % 1000:03d}.svg")
    rng, subkey = jax.random.split(rng)
    a = act_randomly(subkey, s)
    # print(a)
    s = step(s, a)
et = time.time()
print(et - st)

"""
$ inkscape --export-type=png *.svg
$ convert *.png go.gif
"""
