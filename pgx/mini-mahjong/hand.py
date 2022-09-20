import jax
import jax.numpy as jnp
from jax import jit


@jit
def can_ron(hand: jnp.ndarray, tile: int) -> bool:
    return jnp.sum(hand.at[tile].set(hand[tile] + 1) > 0) == 1


@jit
def can_tsumo(hand: jnp.ndarray) -> bool:
    return jnp.sum(hand > 0) == 1


@jit
def can_pon(hand: jnp.ndarray, tile: int) -> bool:
    return hand[tile] >= 2


@jit
def can_chi(hand: jnp.ndarray, tile: int, pos: int) -> bool:
    # pos:
    #    0: 45[6]
    #    1: 4[5]6
    #    2: [4]56
    return jax.lax.switch(
        pos,
        [
            lambda: (
                (tile < 27)
                & (tile % 9 > 1)
                & (hand[tile - 2] > 0)
                & (hand[tile - 1] > 0)
            ),
            lambda: (
                (tile < 27)
                & (tile % 9 > 0)
                & (tile % 9 < 9)
                & (hand[tile - 1] > 0)
                & (hand[tile + 1] > 0)
            ),
            lambda: (
                (tile < 27)
                & (tile % 9 < 8)
                & (hand[tile + 1] > 0)
                & (hand[tile + 2] > 0)
            ),
        ],
    )


@jit
def add(hand: jnp.ndarray, tile: int, x: int = 1) -> jnp.ndarray:
    return hand.at[tile].set(hand[tile] + x)


@jit
def sub(hand: jnp.ndarray, tile: int, x: int = 1) -> jnp.ndarray:
    return add(hand, tile, -x)


@jit
def pon(hand: jnp.ndarray, tile: int) -> jnp.ndarray:
    return sub(hand, tile, 2)


@jit
def chi(hand: jnp.ndarray, tile: int, pos: int) -> jnp.ndarray:
    return jax.lax.switch(
        pos,
        [
            lambda: sub(sub(hand, tile - 2), tile - 1),
            lambda: sub(sub(hand, tile - 1), tile + 1),
            lambda: sub(sub(hand, tile + 1), tile + 2),
        ],
    )


if __name__ == "__main__":
    hand = jnp.zeros(34, dtype=jnp.uint8)
    hand = add(hand, 0)
    assert can_ron(hand, 0)
    assert not can_ron(hand, 1)
    hand = add(hand, 0)
    assert can_tsumo(hand)
    assert can_pon(hand, 0)

    R, M, L = 0, 1, 2
    assert not can_chi(hand, 2, R)
    hand = add(hand, 1)
    assert can_chi(hand, 2, R)
    assert not can_chi(hand, 1, M)
    hand = add(hand, 2)
    assert can_chi(hand, 1, M)
    assert can_chi(hand, 0, L)
    assert not can_chi(hand, 1, L)

    hand = chi(hand, 0, L)
    assert hand[0] == 2
    assert hand[1] == 0
    assert hand[2] == 0
    hand = pon(hand, 0)
    assert hand[0] == 0
