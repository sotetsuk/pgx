import jax
import jax.numpy as jnp
from pgx.sparrow_mahjong import _is_completed, _to_str, _validate, _to_base5, _hand_to_score, _init, SparrowMahjong

env = SparrowMahjong()
init = jax.jit(env.init)
step = jax.jit(env.step)
observe = jax.jit(env.observe)


def test_to_base5():
    hand = jnp.int32([0,0,0,0,0,0,0,0,0,3,3])
    assert _to_base5(hand) == 18
    hand = jnp.int32([4,1,1,0,0,0,0,0,0,0,0])
    assert _to_base5(hand) == 41406250


def test_to_hand_to_score():
    hand = jnp.int32([0,0,0,0,0,0,0,0,0,3,3])
    assert _hand_to_score(hand) == (4, 15)
    hand = jnp.int32([4,1,1,0,0,0,0,0,0,0,0])
    assert _hand_to_score(hand) == (3, 0)


def test_is_completed():
    hand = jnp.int32([0,0,1,1,1,0,0,0,0,3,0])
    assert _is_completed(hand)
    hand = jnp.int32([0,0,1,1,1,0,0,0,0,2,1])
    assert not _is_completed(hand)
    hand = jnp.int32([0,0,1,0,1,0,0,0,0,3,0])
    assert not _is_completed(hand)
    hand = jnp.int32([0,0,1,2,1,0,0,0,0,3,0])
    assert not _is_completed(hand)


def test_init():
    state = _init(jax.random.PRNGKey(1))
    _validate(state)
    print(_to_str(state))
    assert _to_str(state) == """ dora: r
*[2] 2*3 3 5 8*g : _ _ _ _ _ _ _ _ _ _  
 [1] 3 4 4 5*9   : _ _ _ _ _ _ _ _ _ _  
 [0] 5 7 8 9 r*  : _ _ _ _ _ _ _ _ _ _  
"""

    for seed in range(1000):
        state = init(jax.random.PRNGKey(seed))
        assert jnp.logical_not(state.terminated)


def test_step():
    state = _init(jax.random.PRNGKey(1))
    _validate(state)
    print(_to_str(state))
    assert _to_str(state) == """ dora: r
*[2] 2*3 3 5 8*g : _ _ _ _ _ _ _ _ _ _  
 [1] 3 4 4 5*9   : _ _ _ _ _ _ _ _ _ _  
 [0] 5 7 8 9 r*  : _ _ _ _ _ _ _ _ _ _  
"""

    state = step(state, jnp.int32(1))
    assert not state.terminated
    _validate(state)
    print(_to_str(state))
    assert _to_str(state) == """ dora: r
 [2] 3 3 5 8*g   : 2*_ _ _ _ _ _ _ _ _  
*[1] 3 4 4 5*9*9 : _ _ _ _ _ _ _ _ _ _  
 [0] 5 7 8 9 r*  : _ _ _ _ _ _ _ _ _ _  
"""


def test_observe():
    state = _init(jax.random.PRNGKey(1))
    state = step(state, jnp.int32(1))
    print(_to_str(state))
    obs = observe(state, player_id=jnp.int32(2))
    assert obs.shape[0] == 11
    assert obs.shape[1] == 15
    assert jnp.all(obs[:, 0] == jnp.bool_([0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0]))
    assert jnp.all(obs[:, 1] == jnp.bool_([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert jnp.all(obs[:, 2] == jnp.bool_([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert jnp.all(obs[:, 3] == jnp.bool_([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert jnp.all(obs[:, 4] == jnp.bool_([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]))
    assert jnp.all(obs[:, 5] == jnp.bool_([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]))
    assert jnp.all(obs[:, 6] == jnp.bool_([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert jnp.all(obs[:, 7] == jnp.bool_([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert jnp.all(obs[:, 8] == jnp.bool_([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))

    seed = 5
    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)
    state = _init(subkey)
    while not state.terminated:
        legal_actions = jnp.where(state.legal_action_mask)[0]
        key, subkey = jax.random.split(key)
        action = jax.random.choice(subkey, legal_actions)
        state = step(state, action)
    print(_to_str(state))
    """
    [terminated] dora: 3
     [0] 1 4 5 6 8   : 7 r*_ _ _ _ _ _ _ _  
     [2] 4 6*8 9*9   : 3 3 _ _ _ _ _ _ _ _  
     [1] 1 2*3 4 5   : r*_ _ _ _ _ _ _ _ _  
    """
    obs = observe(state, player_id=jnp.int32(0))
    assert obs.shape[0] == 11
    assert obs.shape[1] == 15
    assert jnp.all(obs[:, 0] == jnp.bool_([1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0]))
    assert jnp.all(obs[:, 1] == jnp.bool_([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert jnp.all(obs[:, 2] == jnp.bool_([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert jnp.all(obs[:, 3] == jnp.bool_([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert jnp.all(obs[:, 4] == jnp.bool_([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert jnp.all(obs[:, 5] == jnp.bool_([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert jnp.all(obs[:, 6] == jnp.bool_([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1]))
    assert jnp.all(obs[:, 7] == jnp.bool_([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert jnp.all(obs[:, 8] == jnp.bool_([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]))
    assert jnp.all(obs[:, 9] == jnp.bool_([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert jnp.all(obs[:, 10] == jnp.bool_([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert jnp.all(obs[:, 11] == jnp.bool_([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert jnp.all(obs[:, 12] == jnp.bool_([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]))
    assert jnp.all(obs[:, 13] == jnp.bool_([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert jnp.all(obs[:, 14] == jnp.bool_([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    obs = observe(state, player_id=jnp.int32(1))
    assert obs.shape[0] == 11
    assert obs.shape[1] == 15
    assert jnp.all(obs[:, 0] == jnp.bool_([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]))
    assert jnp.all(obs[:, 1] == jnp.bool_([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert jnp.all(obs[:, 2] == jnp.bool_([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert jnp.all(obs[:, 3] == jnp.bool_([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert jnp.all(obs[:, 4] == jnp.bool_([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert jnp.all(obs[:, 5] == jnp.bool_([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert jnp.all(obs[:, 6] == jnp.bool_([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]))
    assert jnp.all(obs[:, 7] == jnp.bool_([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1]))
    assert jnp.all(obs[:, 8] == jnp.bool_([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert jnp.all(obs[:, 9] == jnp.bool_([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]))
    assert jnp.all(obs[:, 10] == jnp.bool_([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]))
    assert jnp.all(obs[:, 11] == jnp.bool_([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert jnp.all(obs[:, 12] == jnp.bool_([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert jnp.all(obs[:, 13] == jnp.bool_([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert jnp.all(obs[:, 14] == jnp.bool_([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    obs = observe(state, player_id=jnp.int32(2))
    assert obs.shape[0] == 11
    assert obs.shape[1] == 15
    assert jnp.all(obs[:, 0] == jnp.bool_([0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0]))
    assert jnp.all(obs[:, 1] == jnp.bool_([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]))
    assert jnp.all(obs[:, 2] == jnp.bool_([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert jnp.all(obs[:, 3] == jnp.bool_([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert jnp.all(obs[:, 4] == jnp.bool_([0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0]))
    assert jnp.all(obs[:, 5] == jnp.bool_([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert jnp.all(obs[:, 6] == jnp.bool_([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert jnp.all(obs[:, 7] == jnp.bool_([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]))
    assert jnp.all(obs[:, 8] == jnp.bool_([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1]))
    assert jnp.all(obs[:, 9] == jnp.bool_([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]))
    assert jnp.all(obs[:, 10] == jnp.bool_([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert jnp.all(obs[:, 11] == jnp.bool_([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert jnp.all(obs[:, 12] == jnp.bool_([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]))
    assert jnp.all(obs[:, 13] == jnp.bool_([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]))
    assert jnp.all(obs[:, 14] == jnp.bool_([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))


def test_api():
    import pgx
    env = pgx.make("sparrow_mahjong")
    pgx.api_test(env, 3, use_key=False)
    pgx.api_test(env, 3, use_key=True)