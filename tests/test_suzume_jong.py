import jax
import jax.numpy as jnp
from pgx.suzume_jong import _is_completed, init, _to_str, _validate, step, _to_base5, _hand_to_score


def test_to_base5():
    hand = jnp.int8([0,0,0,0,0,0,0,0,0,3,3])
    assert _to_base5(hand) == 18
    hand = jnp.int8([4,1,1,0,0,0,0,0,0,0,0])
    assert _to_base5(hand) == 41406250


def test_to_hand_to_score():
    hand = jnp.int8([0,0,0,0,0,0,0,0,0,3,3])
    assert _hand_to_score(hand) == (4, 15)
    hand = jnp.int8([4,1,1,0,0,0,0,0,0,0,0])
    assert _hand_to_score(hand) == (3, 0)


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
    _validate(state)
    print(_to_str(state))
    assert _to_str(state) == """ dora: r
*[2] 2*3 3 5 8*g : _ _ _ _ _ _ _ _ _ _  
 [1] 3 4 4 5*9   : _ _ _ _ _ _ _ _ _ _  
 [0] 5 7 8 9 r*  : _ _ _ _ _ _ _ _ _ _  
"""


def test_step():
    curr_player, state = init(jax.random.PRNGKey(1))
    _validate(state)
    print(_to_str(state))
    assert _to_str(state) == """ dora: r
*[2] 2*3 3 5 8*g : _ _ _ _ _ _ _ _ _ _  
 [1] 3 4 4 5*9   : _ _ _ _ _ _ _ _ _ _  
 [0] 5 7 8 9 r*  : _ _ _ _ _ _ _ _ _ _  
"""

    curr_player, state, r = step(state, jnp.int8(1))
    assert not state.terminated
    _validate(state)
    print(_to_str(state))
    assert _to_str(state) == """ dora: r
 [2] 3 3 5 8*g   : 2*_ _ _ _ _ _ _ _ _  
*[1] 3 4 4 5*9*9 : _ _ _ _ _ _ _ _ _ _  
 [0] 5 7 8 9 r*  : _ _ _ _ _ _ _ _ _ _  
"""


def test_random_play():
    results = ""
    print()
    for seed in range(10):
        print("=================================")
        print(seed)
        print("=================================")
        key = jax.random.PRNGKey(seed)
        key, subkey = jax.random.split(key)
        curr_player, state = init(subkey)
        _validate(state)
        # print(_to_str(state))
        while not state.terminated:
            legal_actions = jnp.where(state.legal_action_mask)[0]
            key, subkey = jax.random.split(key)
            action = jax.random.choice(subkey, legal_actions)
            curr_player, state, r = step(state, action)
            _validate(state)
        print(_to_str(state))
        print(r)
        results += _to_str(state)

    expected = """[terminated] dora: 5
 [1] 1 4 6 8*g   : 7*_ _ _ _ _ _ _ _ _  
 [0] 5 8 9*9 g   : 5 _ _ _ _ _ _ _ _ _  
 [2] 2*3 4 6 7   : _ _ _ _ _ _ _ _ _ _  
[terminated] dora: 4
 [2] 3 8*9*g*r*  : 1 5 5 1 8 1*2*4*9 4  
 [0] 4 g r*r*r*  : 8 6 2 g 7 5 9 2 7*_  
 [1] 3*6*6 6 8   : 7 1 5*7 2 3 g 3 9 _  
[terminated] dora: 3
 [2] 1 2 4 g r*  : 7 r*r*8*_ _ _ _ _ _  
 [1] 5 6*7 8 9   : 1 7*3 2 _ _ _ _ _ _  
 [0] 1*6 8 g*g   : 5 9*5 4 _ _ _ _ _ _  
[terminated] dora: 3
 [1] 4 7 9 r*r*  : 3*8 5 7 _ _ _ _ _ _  
 [2] 1 2 3 5 7*  : 8*2 g 7 _ _ _ _ _ _  
 [0] 2 5 9*r*r*  : 4 9 8 6*_ _ _ _ _ _  
[terminated] dora: 3
 [1] 3*7 9 g*r*  : g 5*9 7 9 r*3 6 g 3  
 [0] 1*1 5 6 8*  : 1 8 4 g 2 8 2*4*5 _  
 [2] 2 4 7 8 9*  : r*5 2 r*6 7*6*4 1 _  
[terminated] dora: 3
 [0] 1 4 5 6 8   : 7 r*_ _ _ _ _ _ _ _  
 [2] 4 6*8 9*9   : 3 3 _ _ _ _ _ _ _ _  
 [1] 1 2*3 4 5   : r*_ _ _ _ _ _ _ _ _  
[terminated] dora: 6
 [1] 4*5*5 8*8   : 2 6 8 4 9 7*g r*1*2  
 [2] 1 3 5 6 r*  : 2 4 2*9 1 5 3*9*r*_  
 [0] 1 7 9 g*g   : g 3 7 3 4 6 r*8 7 _  
[terminated] dora: 5
 [2] 3 5*5 7*g   : 9*2*6 8 2 g 1*8 9 4  
 [1] 6*7 7 8*9   : 5 8 4*4 9 7 r*6 2 _  
 [0] 1 2 3*3 r*  : 4 r*1 3 6 r*g 1 g _  
[terminated] dora: 3
 [0] 2*2 7 9 g   : 1 r r*r*2 7 _ _ _ _  
 [2] 4 4 8 g*g   : 7 9 1 5 8 _ _ _ _ _  
 [1] 2 3 4*8 9*  : 9 6 6*r*1 _ _ _ _ _  
[terminated] dora: r
 [2] 1 2 6 g r*  : 4*_ _ _ _ _ _ _ _ _  
 [0] 2 3*4 6*6 6 : _ _ _ _ _ _ _ _ _ _  
 [1] 1*1 4 7 9   : _ _ _ _ _ _ _ _ _ _  
"""
    assert results == expected
