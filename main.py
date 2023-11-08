import jax
import pgx
from pgx.experimental import act_randomly, auto_reset

print(f"{pgx.__version__=}")

env = pgx.make("tic_tac_toe")

init = jax.jit(jax.vmap(env.init))  # vectorize and JIT-compile
step = jax.jit(jax.vmap(env.step))
act_randomly = jax.jit(act_randomly)

batch_size = 9

# prepare PRNGKeys
key = jax.random.PRNGKey(42)
key, subkey = jax.random.split(key)
keys = jax.random.split(subkey, batch_size)

state = init(keys)  # vectorized states
for _ in range(100):
    key, subkey = jax.random.split(key)
    # action = act_randomly(subkey, state.legal_action_mask)
    action = act_randomly(subkey, state)
    keys = jax.random.split(key, batch_size)
    state = step(state, action)
