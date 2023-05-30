import sys
import json
import time
import jax
import pgx
import numpy as np
import jax.numpy as jnp
from pgx.experimental.utils import act_randomly
from pgx.bridge_bidding import BridgeBidding
import pickle
import haiku as hk


NUM_ACTIONS = 38
MIN_ACTION = 52
FALSE = jnp.bool_(False)
TRUE = jnp.bool_(True)


def net_fn(x):
    """Haiku module for our network."""
    net = hk.Sequential(
        [
            hk.Linear(1024),
            jax.nn.relu,
            hk.Linear(1024),
            jax.nn.relu,
            hk.Linear(1024),
            jax.nn.relu,
            hk.Linear(1024),
            jax.nn.relu,
            hk.Linear(NUM_ACTIONS),
            jax.nn.log_softmax,
        ]
    )
    return net(x)


net = hk.without_apply_rng(hk.transform(net_fn))

params = pickle.load(
    open("bridge_bidding_sl_networks/params-240000.pkl", "rb")
)


def act_sl_model(params, observation) -> int:
    print(observation)
    policy = jnp.exp(net.apply(params, observation))
    print(policy)
    return jnp.argmax(policy, axis=1)


env = BridgeBidding()
# run api test
pgx.v1_api_test(env, 100)

# jit
init = jax.jit(jax.vmap(env.init))
step = jax.jit(jax.vmap(env.step))
act_randomly = jax.jit(act_randomly)


N = 4
key = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key)
keys = jax.random.split(subkey, N)
state: pgx.State = init(keys)
state = state.replace(
    _vul_NS=jnp.zeros(N, dtype=jnp.bool_),
    _vul_EW=jnp.zeros(N, dtype=jnp.bool_),
)  # wbridge5のデータセットはノンバルのみ
print(state)
i = 0
while not state.terminated.all():
    key, subkey = jax.random.split(key)
    action = act_sl_model(params, state.observation)
    print(action)
    print("================")
    print(f"{i:04d}")
    print("================")
    print(f"curr_player: {state.current_player}\naction: {action}")
    state.save_svg(f"test/{i:04d}.svg")
    state = step(state, action)
    print(f"reward:\n{state.rewards}")
    i += 1
state.save_svg(f"{i:04d}.svg")
