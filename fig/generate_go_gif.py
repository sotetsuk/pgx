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
a = act_randomly(subkey, s.legal_action_mask)
s = step(s, a)
print("warmup ends")

st = time.time()
s = init(subkeys)
for i in range(1000 + 200):
    if i >= 1000 and i % 5 == 0:
        s.save_svg(f"{i % 1000:03d}_dark.svg", color_theme="dark")
        s.save_svg(f"{i % 1000:03d}_light.svg", color_theme="light")
    rng, subkey = jax.random.split(rng)
    a = act_randomly(subkey, s.legal_action_mask)
    s = step(s, a)
et = time.time()
print(et - st)

"""
$ inkscape --export-type=png *.svg
$ convert *.png go.gif
"""
