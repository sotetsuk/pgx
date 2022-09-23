from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import jit, tree_util


@dataclass
class Deck:
    idx: int
    arr: jnp.ndarray

    @jit
    def is_empty(self) -> bool:
        return self.idx == 122

    def _tree_flatten(self):
        children = (self.idx, self.arr)
        aux_data = {}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)


tree_util.register_pytree_node(Deck, Deck._tree_flatten, Deck._tree_unflatten)


@jit
def init(key) -> Deck:
    arr = jax.random.permutation(key, jnp.array([i // 4 for i in range(136)]))
    return Deck(0, arr)


@jit
def draw(deck: Deck) -> Tuple[Deck, int]:
    tile = deck.arr[deck.idx]
    deck.idx += 1
    return deck, tile


if __name__ == "__main__":
    deck = init(jax.random.PRNGKey(seed=0))
    assert deck.idx == 0
    deck, tile1 = draw(deck)
    assert deck.idx == 1
    deck = init(jax.random.PRNGKey(seed=1))
    assert deck.idx == 0
    deck, tile2 = draw(deck)
    assert tile1 != tile2
