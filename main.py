import jax
import jax.numpy as jnp


def f(x):
    return - x ** 2


def g(x):
    return x ** 2


def cond_fn(x):
    return x < 10

def body_fn(x):
    return x + 1

@jax.vmap
def h(x):
    x = jax.lax.while_loop(cond_fn, body_fn, x) 
    # x = jax.lax.cond(x < 0, f, g, x)
    return x


print(jax.make_jaxpr(h)(jnp.array([-1, 2])))
print(h(jnp.array([-1, 2])))
