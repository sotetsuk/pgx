from dataclasses import dataclass
from typing import List, Tuple

import jax
import jax.numpy as jnp
from jax import jit, tree_util
from shanten_tools import shanten


@dataclass
class Hand:
    arr: jnp.ndarray = jnp.zeros((4, 34), dtype=jnp.uint8)

    @jit
    def size(self, player: int) -> int:
        return jnp.sum(self.arr[player])

    @jit
    def can_ron(self, player: int, tile: int) -> bool:
        # TODO
        return (
            jnp.sum(
                self.arr.at[(player, tile)].set(self.arr[player][tile] + 1) > 0
            )
            == 1
        )

    # def can_ron(self, player: int, tile: int) -> bool:
    #    return (
    #        shanten(
    #            self.arr.at[(player, tile)].set(self.arr[player][tile] + 1)[
    #                player
    #            ]
    #        )
    #        == -1
    #    )

    @jit
    def can_tsumo(self, player: int) -> bool:
        # TODO
        return jnp.sum(self.arr[player] > 0) == 1

    # def can_tsumo(self, player: int) -> bool:
    #    return shanten(self.arr[player]) == -1

    @jit
    def can_pon(self, player: int, tile: int) -> bool:
        return self.arr[player][tile] >= 2

    @jit
    def can_chi(self, player: int, tile: int, pos: int) -> bool:
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
                    & (self.arr[player][tile - 2] > 0)
                    & (self.arr[player][tile - 1] > 0)
                ),
                lambda: (
                    (tile < 27)
                    & (tile % 9 > 0)
                    & (tile % 9 < 9)
                    & (self.arr[player][tile - 1] > 0)
                    & (self.arr[player][tile + 1] > 0)
                ),
                lambda: (
                    (tile < 27)
                    & (tile % 9 < 8)
                    & (self.arr[player][tile + 1] > 0)
                    & (self.arr[player][tile + 2] > 0)
                ),
            ],
        )

    def _tree_flatten(self):
        children = (self.arr,)
        aux_data = {}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)


tree_util.register_pytree_node(Hand, Hand._tree_flatten, Hand._tree_unflatten)


@jit
def add(hand: Hand, player: int, tile: int, x: int = 1) -> Hand:
    hand.arr = hand.arr.at[(player, tile)].set(hand.arr[player][tile] + x)
    return hand


@jit
def sub(hand: Hand, player: int, tile: int, x: int = 1) -> Hand:
    return add(hand, player, tile, -x)


@jit
def pon(hand: Hand, player: int, tile: int) -> Hand:
    return sub(hand, player, tile, 2)


@jit
def chi(hand: Hand, player: int, tile: int, pos: int) -> Hand:
    return jax.lax.switch(
        pos,
        [
            lambda: sub(sub(hand, player, tile - 2), player, tile - 1),
            lambda: sub(sub(hand, player, tile - 1), player, tile + 1),
            lambda: sub(sub(hand, player, tile + 1), player, tile + 2),
        ],
    )


if __name__ == "__main__":
    hand = Hand()
    player = 0
    assert hand.size(player) == 0
    hand = add(hand, player, 0)
    assert hand.size(player) == 1
    assert hand.can_ron(player, 0)
    assert not hand.can_ron(player, 1)
    hand = add(hand, player, 0)
    assert hand.can_tsumo(player)
    assert hand.can_pon(player, 0)

    R, M, L = 0, 1, 2
    assert not hand.can_chi(player, 2, R)
    hand = add(hand, player, 1)
    assert hand.can_chi(player, 2, R)
    assert not hand.can_chi(player, 1, M)
    hand = add(hand, player, 2)
    assert hand.can_chi(player, 1, M)
    assert hand.can_chi(player, 0, L)
    assert not hand.can_chi(player, 1, L)

    hand = chi(hand, player, 0, L)
    assert hand.arr[player][0] == 2
    assert hand.arr[player][1] == 0
    assert hand.arr[player][2] == 0
    hand = pon(hand, player, 0)
    assert hand.arr[player][0] == 0
