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
    print(state.hands)
    print(state.walls)
    print(_to_str(state))
    assert _is_valid(state)
    assert _to_str(state) == """dora: r
*[2] 23358g, xxxxxxxxx
 [1] 34459 , xxxxxxxxx
 [0] 5789r , xxxxxxxxx
"""


def test_step():
    curr_player, state = init(jax.random.PRNGKey(1))
    assert _is_valid(state)
    assert _to_str(state) == """dora: r
*[2] 23358g, xxxxxxxxx
 [1] 34459 , xxxxxxxxx
 [0] 5789r , xxxxxxxxx
"""
    curr_player, state, r = step(state, jnp.int8(1))
    assert _is_valid(state)
    assert _to_str(state) == """dora: r
 [2] 3358g , 2xxxxxxxx
*[1] 344599, xxxxxxxxx
 [0] 5789r , xxxxxxxxx
"""
