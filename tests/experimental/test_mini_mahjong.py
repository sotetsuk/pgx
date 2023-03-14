import jax
import jax.numpy as jnp

from pgx.mahjong.mini_mahjong import Deck, Hand, Action, init, step


def test_deck():
    deck = Deck.init(jax.random.PRNGKey(seed=0))
    assert deck.idx == 0
    deck, tile1 = Deck.draw(deck)
    assert deck.idx == 1
    deck = Deck.init(jax.random.PRNGKey(seed=1))
    assert deck.idx == 0
    deck, tile2 = Deck.draw(deck)
    assert tile1 != tile2


def test_hand():
    hand = jnp.zeros(34, dtype=jnp.uint8)
    hand = Hand.add(hand, 0)
    assert Hand.can_ron(hand, 0)
    assert not Hand.can_ron(hand, 1)
    hand = Hand.add(hand, 0)
    assert Hand.can_tsumo(hand)
    assert Hand.can_pon(hand, 0)

    R, M, L = 0, 1, 2
    assert not Hand.can_chi(hand, 2, R)
    hand = Hand.add(hand, 1)
    assert Hand.can_chi(hand, 2, R)
    assert not Hand.can_chi(hand, 1, M)
    hand = Hand.add(hand, 2)
    assert Hand.can_chi(hand, 1, M)
    assert Hand.can_chi(hand, 0, L)
    assert not Hand.can_chi(hand, 1, L)

    hand = Hand.chi(hand, 0, L)
    assert hand[0] == 2
    assert hand[1] == 0
    assert hand[2] == 0
    hand = Hand.pon(hand, 0)
    assert hand[0] == 0

    hand = jnp.array([
        3,1,1,1,1,1,0,0,0,
        0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0
        ])
    assert Hand.can_tsumo(hand)

    hand = jnp.array([
        2,0,0,0,0,0,0,1,0,
        1,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0
        ])
    assert not Hand.can_chi(hand, 8, M)

    hand = jnp.array([
        2,0,0,0,0,0,0,0,1,
        1,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0
        ])
    assert not Hand.can_chi(hand, 7, L)


def test_state():
    state = init(jax.random.PRNGKey(seed=0))

    MANZU_6 = 5
    state, _, _ = step(
        state, jnp.array([MANZU_6, Action.NONE, Action.NONE, Action.NONE])
    )

    state, _, _ = step(
        state, jnp.array([Action.NONE, Action.CHI_R, Action.NONE, Action.NONE])
    )

    SOUZU_7 = 24
    state, _, _ = step(
        state, jnp.array([Action.NONE, SOUZU_7, Action.NONE, Action.NONE])
    )

    state, _, _ = step(
        state, jnp.array([Action.NONE, Action.NONE, Action.NONE, Action.PON])
    )

    DRAGON_R = 33
    state, _, _ = step(
        state, jnp.array([Action.NONE, Action.NONE, Action.NONE, DRAGON_R])
    )

    MANZU_1 = 0
    state, _, _ = step(
        state, jnp.array([MANZU_1, Action.NONE, Action.NONE, Action.NONE])
    )

    state, _, _ = step(
        state, jnp.array([Action.NONE, Action.PASS, Action.NONE, Action.NONE])
    )
