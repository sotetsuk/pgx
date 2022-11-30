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
    assert _to_str(state) == """ dora: r
*[2] 23358g, xxxxxxxxxx 
 [1] 34459 , xxxxxxxxxx 
 [0] 5789r , xxxxxxxxxx 
"""


def test_step():
    curr_player, state = init(jax.random.PRNGKey(1))
    _validate(state)
    assert _to_str(state) == """ dora: r
*[2] 23358g, xxxxxxxxxx 
 [1] 34459 , xxxxxxxxxx 
 [0] 5789r , xxxxxxxxxx 
"""
    curr_player, state, r = step(state, jnp.int8(1))
    print(state.legal_action_mask)
    assert not state.terminated
    _validate(state)
    assert _to_str(state) == """ dora: r
 [2] 3358g , 2xxxxxxxxx 
*[1] 344599, xxxxxxxxxx 
 [0] 5789r , xxxxxxxxxx 
"""


def test_random_play():
    results = ""
    for seed in range(10):
        # print("=================================")
        # print(seed)
        # print("=================================")
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
        # print(r)
        results += _to_str(state)

    expected = """[terminated] dora: 5
 [1] 1468g , 7xxxxxxxxx 
 [0] 5899g , 5xxxxxxxxx 
 [2] 23467 , xxxxxxxxxx 
[terminated] dora: 4
 [2] 389gr , 1551812494 
 [0] 4grrr , 862g75927x 
 [1] 36668 , 715723g39x 
[terminated] dora: 3
 [2] 124gr , 7rr8xxxxxx 
 [1] 56789 , 1732xxxxxx 
 [0] 168gg , 5954xxxxxx 
[terminated] dora: 3
 [1] 479rr , 3857xxxxxx 
 [2] 12357 , 82g7xxxxxx 
 [0] 259rr , 4986xxxxxx 
[terminated] dora: 3
 [1] 379gr , g5979r36g3 
 [0] 11568 , 184g28245x 
 [2] 24789 , r52r67641x 
[terminated] dora: 3
 [0] 14568 , 7rxxxxxxxx 
 [2] 46899 , 33xxxxxxxx 
 [1] 12345 , rxxxxxxxxx 
[terminated] dora: 6
 [1] 45588 , 268497gr12 
 [2] 1356r , 24291539rx 
 [0] 179gg , g37346r87x 
[terminated] dora: 5
 [2] 3557g , 92682g1894 
 [1] 67789 , 584497r62x 
 [0] 1233r , 4r136rg1gx 
[terminated] dora: 3
 [0] 2279g , 1rrr27xxxx 
 [2] 448gg , 79158xxxxx 
 [1] 23489 , 966r1xxxxx 
[terminated] dora: r
 [2] 126gr , 4xxxxxxxxx 
 [0] 234666, xxxxxxxxxx 
 [1] 11479 , xxxxxxxxxx 
"""
    # assert results == expected