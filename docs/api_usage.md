# Pgx API Usage

## Example.1: Random play

```py
import jax
import jax.numpy as jnp
import pgx

seed = 42
batch_size = 10
key = jax.random.PRNGKey(seed)


def act_randomly(rng_key, obs, mask):
    """Ignore observation and choose randomly from legal actions"""
    del obs
    probs = mask / mask.sum()
    logits = jnp.log(probs)
    logits = jnp.maximum(logits, jnp.finfo(logits.dtype).min)
    return jax.random.categorical(rng_key, logits=logits, axis=-1)


# Load the environment
env = pgx.make("go_9x9")
init_fn = jax.jit(jax.vmap(env.init))
step_fn = jax.jit(jax.vmap(env.step))

# Initialize the states
key, subkey = jax.random.split(key)
keys = jax.random.split(subkey, batch_size)
state = init_fn(keys)

# Run random simulation
while not (state.terminated | state.truncated).all():
    key, subkey = jax.random.split(key)
    action = act_randomly(subkey, state.observation, state.legal_action_mask)
    state = step_fn(state, action)  # state.reward (2,)
```

## Example.2: Random agent vs Baseline model

This illustrative example helps to understand

- How `state.current_player` is defined
- How to access the reward of each player
- How `Env.step` behaves against already terminated states
- How to use baseline models probided by Pgx

```py
import jax
import jax.numpy as jnp
import pgx

seed = 42
batch_size = 10
key = jax.random.PRNGKey(seed)

# Prepare agent A and B
#   Agent A: random player
#   Agent B: baseline player provided by Pgx
A = 0
B = 1

# Load the environment
env = pgx.make("go_9x9")
init_fn = jax.jit(jax.vmap(env.init))
step_fn = jax.jit(jax.vmap(env.step))

# Prepare random player
from pgx.experimental.utils import act_randomly
act_randomly = jax.jit(act_randomly)
# Prepare baseline model
# Note that it additionaly requires Haiku library ($ pip install dm-haiku)
model_id = "go_9x9_v0"
model = pgx.make_baseline_model(model_id)

# Initialize the states
key, subkey = jax.random.split(key)
keys = jax.random.split(subkey, batch_size)
state = init_fn(keys)
print(f"Game index: {jnp.arange(batch_size)}")  #  [0 1 2 3 4 5 6 7 8 9]
print(f"Black player: {state.current_player}")  #  [1 1 0 1 0 0 1 1 1 1]
# In other words
print(f"A is black: {state.current_player == A}")  # [False False  True False  True  True False False False False]
print(f"B is black: {state.current_player == B}")  # [ True  True False  True False False  True  True  True  True]

# Run
R = state.rewards
while not (state.terminated | state.truncated).all():
    # Action of random player A
    key, subkey = jax.random.split(key)
    action_A = act_randomly(subkey, state)
    # Greedy action of baseline model B
    logits, value = model(state.observation)
    action_B = logits.argmax(axis=-1)
    
    action = jnp.where(state.current_player == A, action_A, action_B)
    state = step_fn(state, action)
    R += state.rewards

print(f"Return of agent A = {R[:, A]}")  # [-1. -1. -1. -1. -1. -1. -1. -1. -1. -1.]
print(f"Return of agent B = {R[:, B]}")  # [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
```
