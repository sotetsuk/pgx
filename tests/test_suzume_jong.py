import jax
import jax.numpy as jnp
from pgx.suzume_jong import _is_completed, init, _to_str, _validate, step, _to_base5, _hand_to_score, observe, _init


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
    curr_player, state = _init(jax.random.PRNGKey(1))
    _validate(state)
    print(_to_str(state))
    assert _to_str(state) == """ dora: r
*[2] 2*3 3 5 8*g : _ _ _ _ _ _ _ _ _ _  
 [1] 3 4 4 5*9   : _ _ _ _ _ _ _ _ _ _  
 [0] 5 7 8 9 r*  : _ _ _ _ _ _ _ _ _ _  
"""

    for seed in range(1000):
        curr_player, state = init(jax.random.PRNGKey(seed))
        assert jnp.logical_not(state.terminated)


def test_step():
    curr_player, state = _init(jax.random.PRNGKey(1))
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
    for seed in range(100):
        print("=================================")
        print(seed)
        print("=================================")
        key = jax.random.PRNGKey(seed)
        key, subkey = jax.random.split(key)
        curr_player, state = _init(subkey)
        # _validate(state)
        # print(_to_str(state))
        while not state.terminated:
            legal_actions = jnp.where(state.legal_action_mask)[0]
            key, subkey = jax.random.split(key)
            action = jax.random.choice(subkey, legal_actions)
            curr_player, state, r = step(state, action)
            # _validate(state)
        print(_to_str(state))
        print(r)
        results += _to_str(state)

    expected = """[terminated] dora: 5
 [1] 1 4 6 8*g   : 7*_ _ _ _ _ _ _ _ _  
 [0] 5 8 9*9 g   : 5 _ _ _ _ _ _ _ _ _  
 [2] 2*3 4 6 7   : _ _ _ _ _ _ _ _ _ _  
[terminated] dora: 4
 [2] 3 8*9*g r*  : 1 5 5 1 8 1*2*4*9 4  
 [0] 4 g r*r*r*  : 8 6 2 g 7 5 9 2 7*_  
 [1] 3*6*6 6 8   : 7 1 5*7 2 3 g 3 9 _  
[terminated] dora: 3
 [2] 2 3 6 r*r*  : 7 r*r*8*2 g 1 4 4*9  
 [1] 2*5 6 9 g   : 1 7*3 2 7 6*3*8 8 _  
 [0] 1*1 4 6 8   : 5 9*5 4 g 7 9 5*g _  
[terminated] dora: 3
 [1] 4*6 g g r*  : 3*8 5 7 9 r*4 5*6 7  
 [2] 1 3 3 9 g   : 8*2 g 7 2 5 1 8 7*_  
 [0] 1*1 2*5 r*  : 4 9 8 6*9*6 4 r*2 _  
[terminated] dora: 3
 [1] 3*7 9 g r*  : g 5*9 7 9 r*3 6 g 3  
 [0] 1*1 5 6 8*  : 1 8 4 g 2 8 2*4*5 _  
 [2] 2 4 7 8 9*  : r*5 2 r*6 7*6*4 1 _  
[terminated] dora: 3
 [0] 1 4 5 6 8   : 7 r*_ _ _ _ _ _ _ _  
 [2] 4 6*8 9*9   : 3 3 _ _ _ _ _ _ _ _  
 [1] 1 2*3 4 5   : r*_ _ _ _ _ _ _ _ _  
[terminated] dora: 6
 [1] 4*5*5 8*8   : 2 6 8 4 9 7*g r*1*2  
 [2] 1 3 5 6 r*  : 2 4 2*9 1 5 3*9*r*_  
 [0] 1 7 9 g g   : g 3 7 3 4 6 r*8 7 _  
[terminated] dora: 5
 [2] 3 5*5 7*g   : 9*2*6 8 2 g 1*8 9 4  
 [1] 6*7 7 8*9   : 5 8 4*4 9 7 r*6 2 _  
 [0] 1 2 3*3 r*  : 4 r*1 3 6 r*g 1 g _  
[terminated] dora: 3
 [0] 2*2 7 9 g   : 1 r r*r*2 7 _ _ _ _  
 [2] 4 4 8 g g   : 7 9 1 5 8 _ _ _ _ _  
 [1] 2 3 4*8 9*  : 9 6 6*r*1 _ _ _ _ _  
[terminated] dora: r
 [2] 1 2 6 g r*  : 4*_ _ _ _ _ _ _ _ _  
 [0] 2 3*4 6*6 6 : _ _ _ _ _ _ _ _ _ _  
 [1] 1*1 4 7 9   : _ _ _ _ _ _ _ _ _ _  
[terminated] dora: 6
 [0] 2*2 3*4 8   : 5 5*8*9 1 6*7 g 4 9* 
 [2] 1*4*5 8 r*  : 4 3 r*g 2 3 3 1 6 _  
 [1] 2 6 7*8 r*  : r*g 9 7 g 9 1 5 7 _  
[terminated] dora: 7
 [2] 2*2 7 9 g   : 1 8 6 g 1 3*6 7 r*g  
 [1] 1 5*8 r*r*  : 9 5 7*9*8 3 3 4 5 _  
 [0] 4*4 6*6 8*  : 2 9 4 g 2 5 r*1*3 _  
[terminated] dora: 5
 [2] 2*2 3 4 8*  : 4 g 4*7 2 r*8 7*2 r* 
 [1] 1*5 6*9*9   : 8 6 9 r*7 3 3*5 1 _  
 [0] 1 1 9 g r*  : 3 6 g g 6 4 7 8 5*_  
[terminated] dora: r
 [1] 3 5 5 g r*  : 7 9*7 8 1*3 6 8*1 2  
 [0] 3*3 5 6*7   : 9 2 4 8 1 4 6 7 g _  
 [2] 1 2*2 g r*  : 9 4 4*8 r*9 g 5*6 _  
[terminated] dora: 1
 [1] 2*2 4 9 g   : 9 8*1 2 5*1 g 2 5 8  
 [2] 7 8 g r*r*  : 1*6 4 9*8 5 4 6 9 _  
 [0] 3*3 6 r*r*  : 4*7 7*6*3 5 g 7 3 _  
[terminated] dora: 6
 [1] 2*3 4 5*7*  : 9 1 g r*2 _ _ _ _ _  
 [0] 1 2 9*9 r*  : 8 6*4 5 r*_ _ _ _ _  
 [2] 1*2 3*3 5   : 7 1 4 8 6 _ _ _ _ _  
[terminated] dora: 2
 [1] 5*5 9 g r*  : 3 3*1 2 9 g 8*7 _ _  
 [2] 2*4*7 8 9   : g 2 4 1*r*4 r*6 _ _  
 [0] 1 3 7*g r*  : 8 4 1 9 7 6 6*3 _ _  
[terminated] dora: 5
 [1] 1 5 5 8 9   : 8 7 9 9 3*_ _ _ _ _  
 [2] 2 3 4*4 5*  : r*6 9*1 _ _ _ _ _ _  
 [0] 2*g g g g   : 8*4 4 6 _ _ _ _ _ _  
[terminated] dora: 5
 [1] 1*4 7 9 9   : r g 1 5 _ _ _ _ _ _  
 [0] 1 6*7 7 r*  : g 3*4 8*_ _ _ _ _ _  
 [2] 2 3 3 4*4 5*: r*8 2 _ _ _ _ _ _ _  
[terminated] dora: 6
 [0] 1 1 7*r*r*  : 3 g 3 2 9*4 r*1 1 6* 
 [1] 2*3 4 8 g   : 5*2 8 8 2 4 8*7 9 _  
 [2] 5 5 6 9 r*  : 9 g 7 5 7 3*6 g 4*_  
[terminated] dora: 1
 [0] 3*3 3 7 8*  : 2 1 8 1 7*g 8 9 9 g  
 [1] 1*3 5 6 r*  : g 8 6 5 4 7 6 2*4 _  
 [2] 4*4 5*9*g   : r*r*5 6*9 r*2 2 7 _  
[terminated] dora: g
 [1] 1 1 3 3 6*  : 9 9 r*8 7 6 8 4*1 2  
 [2] 5*5 6 8 r*  : 4 3*4 g 5 4 7 5 9 _  
 [0] 2*7*g g r*  : 2 3 9 1*8*r*7 2 6 _  
[terminated] dora: 4
 [0] 3*4 5 7*7   : 2 1 9 1 3 2*_ _ _ _  
 [2] 1*4*6*6 6   : 8*3 2 g g 5 _ _ _ _  
 [1] 1 7 8 9*9   : 5*g r*2 8 7 _ _ _ _  
[terminated] dora: 7
 [2] 5*5 6 7 g   : r g g r*9 3 7 1 4*1* 
 [1] 1 4 4 7*8   : 8*5 9 r*9*g 5 2*3 _  
 [0] 1 3*6*6 r*  : 3 6 8 2 8 4 9 2 2 _  
[terminated] dora: 6
 [0] 1*1 1 4 7   : r*8 5 g 3 g 3 r*8 4  
 [1] 8*9 9 r*r*  : 2 4 g 6 7 g 5 8 9 _  
 [2] 2 2 5*7*7   : 4*3 9*6 2*3*6*1 5 _  
[terminated] dora: g
 [2] 1*3 5 7*7   : 3*r r*g r*8 4 9 g 9* 
 [0] 2*2 3 8 9   : 2 r*6 8*2 5 4*g 9 _  
 [1] 1 1 3 4 6   : 7 6*4 6 7 5 5*8 1 _  
[terminated] dora: 3
 [0] 6 6 9*9 g   : g 4 3 2 7 5 8 8 8 7  
 [2] 2*2 5*8*g   : r*9 7*5 2 g 1 3 4*_  
 [1] 3 4 7 9 r*  : 6 r*r*1 6*1 1*4 5 _  
[terminated] dora: 5
 [0] 1*5*7*7 7   : 8 g 6 g 2*r*r*4*1 r* 
 [2] 1 1 3*4 9   : 9 7 2 8 2 4 3 4 9 _  
 [1] 2 5 6*8*r*  : 3 9*3 g 5 6 g 6 8 _  
[terminated] dora: 3
 [0] 1 3 3 7*g   : 2 9 9 9 3 2 4*r*8 2  
 [2] 1*5*7 7 r*  : 5 r*g 8*6 5 8 6*4 _  
 [1] 1 5 7 g g   : 1 9*2*4 4 6 8 6 r*_  
[terminated] dora: g
 [2] 1*2*3*3 5   : 7 9 1 2 8 _ _ _ _ _  
 [0] 1 4 7 9 r*  : g 6*5 3 4 _ _ _ _ _  
 [1] 1 2 6 8*8   : 2 r*5*r*_ _ _ _ _ _  
[terminated] dora: 5
 [0] 1*2*2 4 7   : 3 g 8 r*r*g 2 5*g 1  
 [2] 1 4*6 7 9   : 8 2 5 6*r*9 4 9 3 _  
 [1] 4 7 8 g r*  : 7*5 8*3 6 3*1 6 9*_  
[terminated] dora: 5
 [2] 1*3*3 6 r*  : r*9 3 2 1 4 5 9 2*g  
 [1] 1 7 7 8*g   : 5 6 r*2 5 8 2 6 8 _  
 [0] 4*4 4 8 r*  : g 1 7 6*3 9 9*7*g _  
[terminated] dora: 9
 [0] 3 3 9 g g   : 4*1 7 8 4 r*_ _ _ _  
 [1] 2 4 8*9 9   : 7 r*8 2 g 5*_ _ _ _  
 [2] 4 5 5 6 6 7 : r*3*8 5 1*_ _ _ _ _  
[terminated] dora: r
 [1] 2 3 3 7 7   : 8 4 g 1 2 2 9 r*7 9  
 [2] 1 4 7 8 9   : 1*8*3 6 2*9*g 5 5 _  
 [0] 4*5 6*6 g   : 5*r*4 g 8 1 3*r*6 _  
[terminated] dora: 2
 [1] 1 3*3 9*g   : 6 7 r*4 5 3 7 8*3 6* 
 [2] 6 8 8 9 r*  : 1 1*4*2 5 1 5*r*9 _  
 [0] 4 4 7 9 g   : 8 6 2 5 2*g 7 g r*_  
[terminated] dora: 9
 [0] 1*3 4 8 g   : 4 2 1 3 6 1 g _ _ _  
 [2] 4*5 6*7*9*  : 5 9 6 4 2 7 7 _ _ _  
 [1] 5*5 7 r*r*  : 3 1 g r*9 2*8 _ _ _  
[terminated] dora: 9
 [0] 2*5 5 6 r*  : g 2 8 5 1 7 g 4 7 2  
 [1] 1 4 8*9 g   : g r*7 r*3 8 8 7*6*_  
 [2] 3*3 3 4*9*  : 6 4 1 5*r*6 9 2 1*_  
[terminated] dora: 6
 [2] 1 1 8*9 9   : 2 5 7 3 5*9 6 g 6*_  
 [1] 1 4*4 g g   : 2 7 5 3 8 3 3*7 4 _  
 [0] 4 5 6 r*r*r*: 1 7*8 8 r*2 9*g _ _  
[terminated] dora: 5
 [1] 2 5*5 7 r*  : 1 6 7 4 2*3 8 6 g r* 
 [0] 1 4 5 9*g   : g r*6 2 r*4 9 8 6*_  
 [2] 1 2 3*8*8   : 1*3 g 7 4*7*9 3 9 _  
[terminated] dora: 7
 [1] 1 3 7*8 r*  : 6 4 _ _ _ _ _ _ _ _  
 [2] 3 4*6*8*8   : 2 _ _ _ _ _ _ _ _ _  
 [0] 3*5*7 8 9*  : 5 _ _ _ _ _ _ _ _ _  
[terminated] dora: 9
 [1] 2*4 5 6*g   : 6 9*r*7*2 5*2 6 1 2  
 [2] 4 4 7 7 g   : r*8 r*1 8 g 1*r*5 _  
 [0] 3*3 3 7 9   : 8 4*5 6 3 1 g 8*9 _  
[terminated] dora: 6
 [2] 2*2 7 g g   : 3 2 4 4*7 6 3*r*7 9  
 [1] 1 3 5*8 9*  : 5 9 6 2 g 1*5 r*7*_  
 [0] 1 1 3 4 g   : 5 8 r*r*6 8 9 8*4 _  
[terminated] dora: 6
 [0] 5 7 7 9 9   : r 8 2 8*_ _ _ _ _ _  
 [1] 3 4 5*6*8   : 2 4 g 5 _ _ _ _ _ _  
 [2] 1*1 2*2 g   : 7 8 5 7*_ _ _ _ _ _  
[terminated] dora: 5
 [1] 2 5 6 8 r*  : 2 3 8 4 1 5 4 3 8 9  
 [0] 1*5*7 8*9   : r*3 6 6 9 1 7 4 r*_  
 [2] 2 4*g g r*  : 3*9*1 2 g 7 g 7*6*_  
[terminated] dora: r
 [1] 5*8*9*9 g   : r*g 3 6*8 9 4*5 7 2  
 [0] 1 1 8 9 r*  : r*4 2 6 4 5 4 7 7*_  
 [2] 2 3*3 6 g   : 3 1 8 1*2*6 5 7 g _  
[terminated] dora: 4
 [1] 3 5*5 8*8   : 1 9 4 4*8 9 5 2*6 g  
 [0] 1*7*7 7 7   : 1 9 2 4 r*8 6*9*6 _  
 [2] 3*3 6 r*r*  : 1 g g 5 2 3 r*2 g _  
[terminated] dora: g
 [1] 1*3*4 6*9*  : 5 g 2 4 r*6 4 5 9 1  
 [2] 1 5 8 8 g   : 1 9 r*3 2 4*6 9 2 _  
 [0] 5*7 8*g r*  : r*6 7 2*7*3 7 8 3 _  
[terminated] dora: 1
 [1] 1 6 9*g g   : 2 2 9 7 5 4 9 4*r*_  
 [2] 3*3 5 6*6   : 5 3 4 7 g 7 8 3 _ _  
 [0] 4 5*6 r*r*  : 2*8 1 9 g 2 8 8*_ _  
[terminated] dora: 5
 [0] 1 2 4*4 r*  : 7 8 4 1 9 8*3 3*2 6  
 [2] 3 5*6 7 9*  : g r*8 5 g r*g 8 1*_  
 [1] 2*4 7 9 9   : 1 2 6 3 g 7*5 6*r*_  
[terminated] dora: 9
 [0] 1 5 5 g g   : 1 7 1*6 3 2 6*7 9 9* 
 [1] 5*8 8 r*r*  : g 8 2*4 7 3 3*6 r*_  
 [2] 1 2 6 9 r*  : 4 4 4*7*8*g 3 5 2 _  
[terminated] dora: 6
 [0] 1 2 7 8 9   : r*3 9 g 3*g 7 7 6 5  
 [1] 1*1 2 8*9   : 9 8 2 5 g 4 3 5 5*_  
 [2] 2*4*4 6 r*  : 7*1 8 3 r*6*g 4 r*_  
[terminated] dora: 5
 [1] 4 9 9 g r*  : 5 3*g 9 2 2*r*1 1*7  
 [0] 1 3 4*g g   : 6 8 8 2 r*8*3 8 7 _  
 [2] 5*6*6 7*r*  : 7 9*4 6 3 2 4 5 1 _  
[terminated] dora: 2
 [2] 2 7*8 9*9   : g 1 6 6 1 r*g r*2 r* 
 [1] 1 5*5 9 r*  : 4 3*6*8 8 4 8*3 g _  
 [0] 4*5 7 7 g   : 6 9 5 4 3 3 7 1*2 _  
[terminated] dora: g
 [0] 1*1 6 8 9   : 2 8 9 9 4 9 r*r*8*r* 
 [2] 1 3 6 g r*  : 1 5 5 4 g 7 5*3 g _  
 [1] 2*3*5 7*7   : 7 4 6 4*3 8 2 6*2 _  
[terminated] dora: 5
 [1] 3 4*8 g g   : 9 1 5 4 7 g r*7 7*6  
 [2] 1*4 8*8 r*  : 9 5 r*6 3 1 6 3*2*_  
 [0] 6*7 8 9 r*  : 9 3 5 1 g 2 4 2 2 _  
[terminated] dora: 7
 [2] 4 9*g g r*  : 6 6*r*7 4 9 3 3 4*3  
 [1] 2 7 7 8*g   : 1 4 9 2 2 9 1*8 5 _  
 [0] 1 6 6 r*r*  : 8 3*1 5*2*5 g 8 5 _  
[terminated] dora: 2
 [0] 1*1 3 4 r*  : 8 r*7 8 5 3*9 4 5 6* 
 [1] 3 6 8 g r*  : 1 2 6 9*5 4 5*3 2*_  
 [2] 1 7*7 8*g   : 4 r*g 7 2 g 9 6 9 _  
[terminated] dora: 9
 [0] 4 6 7*8 g   : 7 _ _ _ _ _ _ _ _ _  
 [1] 1 1 5 g g   : 5 _ _ _ _ _ _ _ _ _  
 [2] 3 4 r*r*r*  : _ _ _ _ _ _ _ _ _ _  
[terminated] dora: 2
 [2] 3 8 8 9 r*  : 3 r*1 1*5 4 4*6 2 3  
 [0] 1 5 6*6 r*  : 7 7 4 g 5 2 4 9 7*_  
 [1] 3*5*6 8*g   : r*9 7 9*1 g 2 g 8 _  
[terminated] dora: 2
 [1] 2*9*r*r*r   : 4 _ _ _ _ _ _ _ _ _  
 [2] 7 7 8*8 8   : 5 _ _ _ _ _ _ _ _ _  
 [0] 4*4 5 7*9   : 7 _ _ _ _ _ _ _ _ _  
[terminated] dora: 3
 [0] 1 5 7 9*r*  : 6 _ _ _ _ _ _ _ _ _  
 [1] 1*2 3 5 7*  : _ _ _ _ _ _ _ _ _ _  
 [2] 2 4 6*8*r*  : _ _ _ _ _ _ _ _ _ _  
[terminated] dora: 7
 [1] 1 5 8 8 g   : 5 9 6 2 6*3 5*2*g 9  
 [2] 1*5 7 7 7   : 4*2 3 3*1 g 9 1 r*_  
 [0] 2 3 4 9*r*  : 8 8*g 6 4 r*4 r*6 _  
[terminated] dora: 3
 [0] 3*5*6 7 r*  : 4 g 6 9 4 1 4 5 9 r  
 [1] 2 2 5 8*8   : 6 1 4*3 7*1*9*9 6*_  
 [2] 2*3 5 7 g   : 1 2 g r*7 g 8 r*8 _  
[terminated] dora: 7
 [0] 3 4*4 6*8   : 2 5 r 6 9 r*7 r*7 8  
 [1] 1 2*2 8*8   : 5 4 5 6 2 6 1 g r*_  
 [2] 1*9*9 9 g   : g 7 1 4 g 5*3 3 3*_  
[terminated] dora: 1
 [2] 4*7 9 9 r*  : 5*8 9 r*7 g 6 g 1 4  
 [1] 3*5 6*6 g   : 8 6 2 4 3 3 7 2 3 _  
 [0] 5 8*8 r*r*  : 2 2*7*4 9*5 1 g 1*_  
[terminated] dora: 2
 [2] 3*4 7 8*g   : 6*5 2 8 g 1*r*2*6 7  
 [0] 1 5 6 8 r*  : r*1 r*6 4 3 7 g 2 _  
 [1] 4*8 9*9 g   : 7*5*9 3 3 1 4 5 9 _  
[terminated] dora: 8
 [2] 5 5 6*6 7   : 7 g 6 4 1 9 g r*4 3  
 [1] 2*5 7*7 r*  : 4 2 6 1*9 4*8 3*2 _  
 [0] 3 5*8 g r*  : 8 9*r*g 2 1 9 1 3 _  
[terminated] dora: 9
 [2] 2 5 6*7 7   : 2 r*g 9 4 8 6 _ _ _  
 [1] 1*2*3 5 7*  : 4 3 3 1 1 8 _ _ _ _  
 [0] 3*4*5*6 8*  : 7 g 4 5 r*g _ _ _ _  
[terminated] dora: r
 [1] 1*4 8 8 9   : 8 6 9*3 3*2 5*1 g 4  
 [2] 3 3 5 7*7   : 5 r*9 6 8 r*g 1 6 _  
 [0] 2*6*9 g g   : r*2 1 2 7 5 7 4 4*_  
[terminated] dora: r
 [2] 2*3 7 8*8   : 9 5*r*4 9 g g 1 4 7  
 [0] 1*5 6*6 r*  : 7 r*g 2 6 4 4*3 8 _  
 [1] 2 5 9*9 g   : 2 1 7*1 3 8 3*6 5 _  
[terminated] dora: 2
 [0] 1 5 7 8 r*  : 4 4*6 4 9*3 3 r*8 2* 
 [2] 1*5*9 g g   : 9 8 5 3 7 4 6*5 g _  
 [1] 1 6 6 7*7   : 8*r*3*2 g 1 9 r*2 _  
[terminated] dora: 5
 [1] 4*8 8 8 g   : 8 6 1 5 4 r*9*6 6 9  
 [0] 2*3 7*7 9   : 2 5 r*3 7 9 2 5*2 _  
 [2] 1*4 g g r*  : 7 1 r*3 1 4 6*3*g _  
[terminated] dora: g
 [1] 1 5 6*6 9   : 9 4*2 r 3 3 2 4 1*5  
 [0] 4 6 8 r*r*  : r*8 9 9*g 2 5*7 4 _  
 [2] 1 1 6 7*g   : 5 g 7 8 2*8*3 7 3*_  
[terminated] dora: 1
 [2] 5 5 8 9 9   : 4 _ _ _ _ _ _ _ _ _  
 [1] 1*2 3 4 4   : _ _ _ _ _ _ _ _ _ _  
 [0] 2*5 6*8*9*  : _ _ _ _ _ _ _ _ _ _  
[terminated] dora: 6
 [1] 5 6 7*8 8   : 4 _ _ _ _ _ _ _ _ _  
 [2] 1 2 5 6 7   : 8*_ _ _ _ _ _ _ _ _  
 [0] 2 3 8 g g   : _ _ _ _ _ _ _ _ _ _  
[terminated] dora: 7
 [0] 3*6 6 8*r*  : 5*1 9 5 9 g 2*g 3 6  
 [1] 1*2 5 9*g   : 3 4 1 5 8 r*3 8 4 _  
 [2] 2 4*4 6 g   : 7 8 r*7 1 2 r*7*9 _  
[terminated] dora: 3
 [2] 4*5 5 7 g   : 6 2 g r*3 7 2 8 4 3  
 [0] 1*1 4 6 9*  : r*6*8 6 g 8 5 2 1 _  
 [1] 1 3 4 r*r*  : 2*g 7 7*5*9 9 8*9 _  
[terminated] dora: 7
 [2] 2 6 8*8 8   : 7*g 3 2 8 5 5 r*1 4  
 [1] 2*3*9*9 r*  : 1 6 4 1 7 3 g 6*2 _  
 [0] 1*5 6 9 r*  : 5*3 g 4 g 9 7 4*r*_  
[terminated] dora: 7
 [1] 5 5 7 9 r*  : 5 _ _ _ _ _ _ _ _ _  
 [0] 3 4 5*6*r*  : 8*_ _ _ _ _ _ _ _ _  
 [2] 2*3 4 7*8 9*: _ _ _ _ _ _ _ _ _ _  
[terminated] dora: 8
 [2] 4 4 9*g r*  : 6 1 3 6 7 9 9 8 7*9  
 [0] 1 3*3 7 8   : g r*5 2*r*4*5 g 6*_  
 [1] 1*3 5*5 6   : 2 7 2 g 4 8 1 r*2 _  
[terminated] dora: 4
 [1] 1 3 6 g r*  : g 6*8 2 r*7 1 7 g r* 
 [2] 2 3 5 6 g   : 5 9 5*r*3 9*4*7 9 _  
 [0] 1*2*2 4 8*  : 5 4 6 8 8 9 1 7*3*_  
[terminated] dora: r
 [1] 1*2 2 7 g   : 6 2*9 7 1 5 g 4 8 4  
 [2] 3*4*5*r*r*  : 3 4 3 1 7 2 1 g 9*_  
 [0] 3 5 6*6 8   : 8*9 9 5 8 r*g 6 7*_  
[terminated] dora: 3
 [1] 1 2 6*r*r*  : r*1 5 3 4 9 r*7 3*6  
 [0] 4 6 8*8 g   : 9 1*2 8 5 9*6 7 7 _  
 [2] 4*4 5*9 g   : 5 1 2*3 7*g 8 g 2 _  
[terminated] dora: 7
 [1] 5*5 9 g r*  : 6*3 6 1 g 1 4 8 4 2  
 [0] 2 3 4 7*9   : r*r*8*6 1*8 7 g 2 _  
 [2] 4*6 7 9*r*  : 1 2*g 3 3*9 5 8 5 _  
[terminated] dora: 8
 [1] 2 7 8 g r*  : r*2 5 g r*3 3 7 7 1* 
 [2] 1 4*4 6 7*  : 9 r*g 4 5*3 1 3*8*_  
 [0] 4 5 5 6*9*  : 8 2*1 6 g 9 9 2 6 _  
[terminated] dora: 7
 [1] 1*1 5*5 6*  : 4 4*9 8 3*9 8*6 1 5  
 [2] 2 2 2 3 6   : r*2*1 g 8 7 g 4 3 _  
 [0] 3 7*g g r*  : 9*7 4 9 8 5 6 r*r*_  
[terminated] dora: 8
 [1] 3*3 6 9 r*  : 4 2*g g 2 6 r*_ _ _  
 [0] 2 2 4*5 7   : 8 5*7 4 6*r*3 _ _ _  
 [2] 1*1 1 4 5 6 : 7*3 8 g 5 9 _ _ _ _  
[terminated] dora: 9
 [0] 1 4 6 g r*  : 8 9 3 5 g 9 1 8 2 3  
 [1] 1*1 4*8 g   : 9*8*g 5 r*6 4 6*7 _  
 [2] 2*3 4 5*r*  : 3 7 2 7*2 7 5 6 r*_  
[terminated] dora: 1
 [0] 9*9 9 g r*  : 5*1 1*8 4 6 7*6 8 r* 
 [2] 4*6*6 7 9   : 2 5 2 g 7 2*8 8*3*_  
 [1] 2 3 5 7 g   : 3 r*g r*4 3 1 5 4 _  
[terminated] dora: 2
 [2] 1 1 5*6*g   : 2 9*3 6 6 r 8 3*3 5  
 [1] 4*4 9 9 r*  : 9 8 1 7 7*r*g 4 2 _  
 [0] 4 5 5 7 g   : 3 1*8 g r*7 6 2 8*_  
[terminated] dora: 1
 [1] 2*2 6 8 g   : 9 1 7 7 _ _ _ _ _ _  
 [0] 1*3 4 5*6   : r*9 1 5 _ _ _ _ _ _  
 [2] 2 3 4*4 5 6 : 7*6*5 _ _ _ _ _ _ _  
[terminated] dora: 4
 [2] 3 5 8 g g   : 6*8 1 7*_ _ _ _ _ _  
 [1] 1*1 1 7 g   : 9 r*6 8 _ _ _ _ _ _  
 [0] 3 4*5 7 8*9 : 2 6 2*_ _ _ _ _ _ _  
[terminated] dora: 3
 [1] 6 6 7*8*9   : 2*1 r*5 5*r*r*g 7 1  
 [2] 2 4*4 7 g   : 7 1 8 6 5 4 3 2 g _  
 [0] 3 6*9*9 9   : 2 1 4 g 8 8 r*5 3 _  
[terminated] dora: r
 [0] 1 1 1 2*5   : 7 5 4*9 6 2 8 3 2 g  
 [2] 4 7 9*r*r*  : 1*3 g 8 8*4 5 3*5*_  
 [1] 6*8 9 g r*  : 3 2 6 4 9 7 7*g 6 _  
[terminated] dora: 4
 [1] 4 8 9 g r*  : 5*9 8 2 r*3 6 g 2*8  
 [0] 1 3*6*7*7   : 9 2 7 g 1*5 3 9 5 _  
 [2] 1 2 4*4 6   : 5 r*7 1 g r*8*3 6 _  
[terminated] dora: g
 [0] 3*5 6 6 8   : r*g 2*8 3 r*1 2 9*3  
 [2] 1*1 4*6 9   : 8 7 4 7 7*2 1 8*6 _  
 [1] 5*7 9 9 g   : g r*3 2 r*5 4 5 4 _  
[terminated] dora: 7
 [2] 3 4 5 7*8   : 3 7 2 4 r*1 2*g 6*5  
 [0] 9*9 9 g g   : 6 1*6 r*g 9 8 r*3 _  
 [1] 2 2 3 5*5   : 6 8*1 r*4 7 4*1 8 _  
[terminated] dora: 4
 [0] 4 4 6 8 8   : 7 5*2 5 1 1 5 1*9 2  
 [1] 6*7*9*9 9   : 6 8 g 1 3 r*6 7 g _  
 [2] 2 3 g r*r*  : 3*7 2*4*g 8*r*5 3 _  
[terminated] dora: 3
 [0] 1 4 6*7 9   : 5 8*7 g 2 g 9 6 5*1  
 [1] 1 4*8 8 r*  : 7*2 6 8 r*g 6 3 r*_  
 [2] 3 4 4 9*r*  : 5 7 g 9 3*1*2*5 2 _  
[terminated] dora: 4
 [2] 6 7*7 g g   : 8 r 4*2 6 3 1*3 8 3  
 [0] 2*4 5*5 7   : 9 8*5 6*9*8 g r*3*_  
 [1] 1 1 2 5 9   : 6 r*g 9 7 1 4 r*2 _  
"""
    assert results == expected


def test_observe():
    curr_player, state = _init(jax.random.PRNGKey(1))
    curr_player, state, r = step(state, jnp.int8(1))
    print(_to_str(state))
    obs = observe(state, player_id=jnp.int8(2))
    assert obs.shape[0] == 15
    assert obs.shape[1] == 11
    assert jnp.all(obs[0] == jnp.bool_([0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0]))
    assert jnp.all(obs[1] == jnp.bool_([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert jnp.all(obs[2] == jnp.bool_([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert jnp.all(obs[3] == jnp.bool_([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert jnp.all(obs[4] == jnp.bool_([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]))
    assert jnp.all(obs[5] == jnp.bool_([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]))
    assert jnp.all(obs[6] == jnp.bool_([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert jnp.all(obs[7] == jnp.bool_([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert jnp.all(obs[8] == jnp.bool_([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))

    seed = 5
    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)
    curr_player, state = _init(subkey)
    while not state.terminated:
        legal_actions = jnp.where(state.legal_action_mask)[0]
        key, subkey = jax.random.split(key)
        action = jax.random.choice(subkey, legal_actions)
        curr_player, state, r = step(state, action)
    print(_to_str(state))
    """
    [terminated] dora: 3
     [0] 1 4 5 6 8   : 7 r*_ _ _ _ _ _ _ _  
     [2] 4 6*8 9*9   : 3 3 _ _ _ _ _ _ _ _  
     [1] 1 2*3 4 5   : r*_ _ _ _ _ _ _ _ _  
    """
    obs = observe(state, player_id=jnp.int8(0))
    assert obs.shape[0] == 15
    assert obs.shape[1] == 11
    assert jnp.all(obs[0] == jnp.bool_([1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0]))
    assert jnp.all(obs[1] == jnp.bool_([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert jnp.all(obs[2] == jnp.bool_([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert jnp.all(obs[3] == jnp.bool_([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert jnp.all(obs[4] == jnp.bool_([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert jnp.all(obs[5] == jnp.bool_([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert jnp.all(obs[6] == jnp.bool_([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1]))
    assert jnp.all(obs[7] == jnp.bool_([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert jnp.all(obs[8] == jnp.bool_([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]))
    assert jnp.all(obs[9] == jnp.bool_([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert jnp.all(obs[10] == jnp.bool_([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert jnp.all(obs[11] == jnp.bool_([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert jnp.all(obs[12] == jnp.bool_([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]))
    assert jnp.all(obs[13] == jnp.bool_([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert jnp.all(obs[14] == jnp.bool_([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    obs = observe(state, player_id=jnp.int8(1))
    assert obs.shape[0] == 15
    assert obs.shape[1] == 11
    assert jnp.all(obs[0] == jnp.bool_([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]))
    assert jnp.all(obs[1] == jnp.bool_([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert jnp.all(obs[2] == jnp.bool_([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert jnp.all(obs[3] == jnp.bool_([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert jnp.all(obs[4] == jnp.bool_([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert jnp.all(obs[5] == jnp.bool_([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert jnp.all(obs[6] == jnp.bool_([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]))
    assert jnp.all(obs[7] == jnp.bool_([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1]))
    assert jnp.all(obs[8] == jnp.bool_([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert jnp.all(obs[9] == jnp.bool_([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]))
    assert jnp.all(obs[10] == jnp.bool_([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]))
    assert jnp.all(obs[11] == jnp.bool_([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert jnp.all(obs[12] == jnp.bool_([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert jnp.all(obs[13] == jnp.bool_([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert jnp.all(obs[14] == jnp.bool_([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    obs = observe(state, player_id=jnp.int8(2))
    assert obs.shape[0] == 15
    assert obs.shape[1] == 11
    assert jnp.all(obs[0] == jnp.bool_([0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0]))
    assert jnp.all(obs[1] == jnp.bool_([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]))
    assert jnp.all(obs[2] == jnp.bool_([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert jnp.all(obs[3] == jnp.bool_([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert jnp.all(obs[4] == jnp.bool_([0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0]))
    assert jnp.all(obs[5] == jnp.bool_([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert jnp.all(obs[6] == jnp.bool_([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert jnp.all(obs[7] == jnp.bool_([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]))
    assert jnp.all(obs[8] == jnp.bool_([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1]))
    assert jnp.all(obs[9] == jnp.bool_([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]))
    assert jnp.all(obs[10] == jnp.bool_([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert jnp.all(obs[11] == jnp.bool_([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert jnp.all(obs[12] == jnp.bool_([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]))
    assert jnp.all(obs[13] == jnp.bool_([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]))
    assert jnp.all(obs[14] == jnp.bool_([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
