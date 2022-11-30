import jax
import jax.numpy as jnp
from pgx.suzume_jong import _is_completed, init, _to_str, _is_valid, step


def test_is_completed():
    hand = jnp.int8([0,0,1,1,1,0,0,0,0,3,0])
    assert _is_completed(hand)
    hand = jnp.int8([0,0,1,1,1,0,0,0,0,2,1])
    assert not _is_completed(hand)
    hand = jnp.int8([0,0,1,0,1,0,0,0,0,3,0])
    assert not _is_completed(hand)
    hand = jnp.int8([0,0,1,2,1,0,0,0,0,3,0])
    assert not _is_completed(hand)


def test_init():
    curr_player, state = init(jax.random.PRNGKey(1))
    assert _is_valid(state)
    assert _to_str(state) == """ dora: r
*[2] 23358g, xxxxxxxxxx 
 [1] 34459 , xxxxxxxxxx 
 [0] 5789r , xxxxxxxxxx 
"""


def test_step():
    curr_player, state = init(jax.random.PRNGKey(1))
    assert _is_valid(state)
    assert _to_str(state) == """ dora: r
*[2] 23358g, xxxxxxxxxx 
 [1] 34459 , xxxxxxxxxx 
 [0] 5789r , xxxxxxxxxx 
"""
    curr_player, state, r = step(state, jnp.int8(1))
    print(state.legal_action_mask)
    assert not state.terminated
    assert _is_valid(state)
    assert _to_str(state) == """ dora: r
 [2] 3358g , 2xxxxxxxxx 
*[1] 344599, xxxxxxxxxx 
 [0] 5789r , xxxxxxxxxx 
"""


def test_random_play():
    for seed in range(10):
        # print("=================================")
        key = jax.random.PRNGKey(seed)
        key, subkey = jax.random.split(key)
        curr_player, state = init(subkey)
        # print(_to_str(state))
        while not state.terminated:
            legal_actions = jnp.where(state.legal_action_mask)[0]
            key, subkey = jax.random.split(key)
            action = jax.random.choice(subkey, legal_actions)
            curr_player, state, r = step(state, action)
        print(_to_str(state))
        print(r)
