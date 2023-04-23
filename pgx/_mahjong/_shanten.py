import json
import os

import jax
import jax.numpy as jnp

DIR = os.path.join(os.path.dirname(__file__), "cache")


def load_shanten_cache():
    with open(os.path.join(DIR, "shanten_cache.json")) as f:
        return jnp.array(json.load(f), dtype=jnp.uint32)


class Shanten:
    # See the link below for the algorithm details.
    # https://github.com/sotetsuk/pgx/pull/123

    CACHE = load_shanten_cache()

    @staticmethod
    def discard(hand: jnp.ndarray) -> jnp.ndarray:
        return jax.lax.map(
            lambda i: jax.lax.cond(
                hand[i] == 0,
                lambda: 0,
                lambda: Shanten.number(hand.at[i].set(hand[i] - 1)),
            ),
            jnp.arange(34),
        )

    @staticmethod
    def number(hand: jnp.ndarray):
        return jnp.min(
            jnp.array(
                [
                    Shanten.normal(hand),
                    Shanten.seven_pairs(hand),
                    Shanten.thirteen_orphan(hand),
                ]
            )
        )

    @staticmethod
    def seven_pairs(hand: jnp.ndarray):
        n_pair = jnp.sum(hand >= 2)
        n_kind = jnp.sum(hand > 0)
        return 7 - n_pair + jax.lax.max(7 - n_kind, 0)

    @staticmethod
    def thirteen_orphan(hand: jnp.ndarray):
        n_pair = (
            (hand[0] >= 2).astype(int)
            + (hand[8] >= 2).astype(int)
            + (hand[9] >= 2).astype(int)
            + (hand[17] >= 2).astype(int)
            + (hand[18] >= 2).astype(int)
            + jnp.sum(hand[26:34] >= 2)
        )
        n_kind = (
            (hand[0] > 0).astype(int)
            + (hand[8] > 0).astype(int)
            + (hand[9] > 0).astype(int)
            + (hand[17] > 0).astype(int)
            + (hand[18] > 0).astype(int)
            + jnp.sum(hand[26:34] > 0)
        )
        return 14 - n_kind - (n_pair > 0)

    @staticmethod
    def normal(hand: jnp.ndarray):
        code = jax.lax.map(
            lambda suit: jax.lax.cond(
                suit == 3,
                lambda: jax.lax.fori_loop(
                    27,
                    34,
                    lambda i, code: code * 5 + hand[i].astype(int),
                    0,
                )
                + 1953125,
                lambda: jax.lax.fori_loop(
                    9 * suit,
                    9 * (suit + 1),
                    lambda i, code: code * 5 + hand[i].astype(int),
                    0,
                ),
            ),
            jnp.arange(4),
        )

        n_set = jnp.sum(hand).astype(int) // 3

        return jnp.min(
            jax.lax.map(
                lambda suit: Shanten._normal(code, n_set, suit), jnp.arange(4)
            )
        )

    @staticmethod
    def _normal(code: jnp.ndarray, n_set, head_suit) -> int:
        cost = Shanten.CACHE[code[head_suit]][4]
        idx = jnp.full(4, 0).at[head_suit].set(5)
        cost, idx = jax.lax.fori_loop(
            0,
            n_set,
            lambda _, tpl: Shanten._update(code, *tpl),
            (cost, idx),
        )
        return cost

    @staticmethod
    def _update(code: jnp.ndarray, cost: int, idx: jnp.ndarray):
        i = jnp.argmin(Shanten.CACHE[code][[0, 1, 2, 3], idx])
        cost += Shanten.CACHE[code][i][idx[i]]
        idx = idx.at[i].set(idx[i] + 1)
        return (cost, idx)
