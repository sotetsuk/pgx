import json
import os

import jax
import jax.numpy as jnp

from pgx._mahjong._action import Action  # type: ignore

DIR = os.path.join(os.path.dirname(__file__), "cache")


def load_hand_cache():
    with open(os.path.join(DIR, "hand_cache.json")) as f:
        return jnp.array(json.load(f), dtype=jnp.uint32)


class Hand:
    CACHE = load_hand_cache()

    @staticmethod
    def make_init_hand(deck: jnp.ndarray):
        hand = jnp.zeros((4, 34), dtype=jnp.uint8)
        for i in range(3):
            for j in range(4):
                hand = hand.at[j].set(
                    jax.lax.fori_loop(
                        0,
                        4,
                        lambda k, h: Hand.add(
                            h, deck[-(16 * i + 4 * j + k + 1)]  # type: ignore
                        ),
                        hand[j],
                    )
                )
        for j in range(4):
            hand = hand.at[j].set(Hand.add(hand[j], deck[-(16 * 3 + j + 1)]))  # type: ignore

        last_draw = deck[-(16 * 3 + 4 + 1)].astype(int)
        hand = hand.at[0].set(Hand.add(hand[0], last_draw))  # type: ignore

        return hand

    @staticmethod
    def cache(code: int):
        return (Hand.CACHE[code >> 5] >> (code & 0b11111)) & 1

    @staticmethod
    def can_ron(hand: jnp.ndarray, tile: int):
        return Hand.can_tsumo(Hand.add(hand, tile))

    @staticmethod
    def can_riichi(hand: jnp.ndarray):
        """手牌は14枚"""
        return jax.vmap(
            lambda i: (hand[i] != 0) & Hand.is_tenpai(Hand.sub(hand, i))
        )(jnp.arange(34)).any()

    @staticmethod
    def is_tenpai(hand: jnp.ndarray):
        """手牌は13枚"""
        return jax.vmap(
            lambda tile: (hand[tile] != 4) & Hand.can_ron(hand, tile)
        )(jnp.arange(34)).any()

    @staticmethod
    def can_tsumo(hand: jnp.ndarray):
        thirteen_orphan = (
            (hand[0] > 0)
            & (hand[8] > 0)
            & (hand[9] > 0)
            & (hand[17] > 0)
            & (hand[18] > 0)
            & jnp.all(hand[26:] > 0)
            & (
                (
                    hand[0]
                    + hand[8]
                    + hand[9]
                    + hand[17]
                    + hand[18]
                    + jnp.sum(hand[26:])
                )
                == 14
            )
        )
        seven_pairs = jnp.sum(hand == 2) == 7

        def _is_valid(suit):
            return Hand.cache(
                jax.lax.fori_loop(
                    9 * suit,
                    9 * (suit + 1),
                    lambda i, code: code * 5 + hand[i].astype(int),
                    0,
                )
            )

        valid = jax.vmap(_is_valid)(jnp.arange(3)).all()

        # これはうまく行かない
        # heads = (
        #     (jnp.sum(hand[0:9]) % 3 == 2)
        #     + (jnp.sum(hand[9:18]) % 3 == 2)
        #     + (jnp.sum(hand[18:27]) % 3 == 2)
        # )

        heads = jnp.int32(0)
        for suit in range(3):
            heads += jnp.sum(hand[9 * suit : 9 * (suit + 1)]) % 3 == 2

        heads, valid = jax.lax.fori_loop(
            27,
            34,
            lambda i, tpl: (
                tpl[0] + (hand[i] == 2),
                tpl[1] & (hand[i] != 1) & (hand[i] != 4),
            ),
            (heads, valid),
        )

        return ((valid & (heads == 1)) | thirteen_orphan | seven_pairs) == 1

    @staticmethod
    def can_pon(hand: jnp.ndarray, tile: int) -> bool:
        return hand[tile] >= 2  # type: ignore

    @staticmethod
    def can_minkan(hand: jnp.ndarray, tile: int) -> bool:
        return hand[tile] == 3  # type: ignore

    @staticmethod
    def can_kakan(hand: jnp.ndarray, tile: int) -> bool:
        return hand[tile] == 1  # type: ignore

    @staticmethod
    def can_ankan(hand: jnp.ndarray, tile: int) -> bool:
        return hand[tile] == 4  # type: ignore

    @staticmethod
    def can_chi(hand: jnp.ndarray, tile: int, action: int) -> bool:
        return jax.lax.cond(
            (tile >= 27) | (action < Action.CHI_L) | (Action.CHI_R < action),
            lambda: False,
            lambda: jax.lax.switch(
                action - Action.CHI_L,
                [
                    lambda: (tile % 9 < 7)
                    & (hand[tile + 1] > 0)
                    & (hand[tile + 2] > 0),
                    lambda: (
                        (tile % 9 < 8)
                        & (tile % 9 > 0)
                        & (hand[tile - 1] > 0)
                        & (hand[tile + 1] > 0)
                    ),
                    lambda: (tile % 9 > 1)
                    & (hand[tile - 2] > 0)
                    & (hand[tile - 1] > 0),
                ],
            ),
        )

    @staticmethod
    def add(hand: jnp.ndarray, tile: int, x: int = 1) -> jnp.ndarray:
        return hand.at[tile].set(hand[tile] + x)

    @staticmethod
    def sub(hand: jnp.ndarray, tile: int, x: int = 1) -> jnp.ndarray:
        return Hand.add(hand, tile, -x)

    @staticmethod
    def pon(hand: jnp.ndarray, tile: int) -> jnp.ndarray:
        return Hand.sub(hand, tile, 2)

    @staticmethod
    def minkan(hand: jnp.ndarray, tile: int) -> jnp.ndarray:
        return Hand.sub(hand, tile, 3)

    @staticmethod
    def kakan(hand: jnp.ndarray, tile: int) -> jnp.ndarray:
        return Hand.sub(hand, tile)

    @staticmethod
    def ankan(hand: jnp.ndarray, tile: int) -> jnp.ndarray:
        return Hand.sub(hand, tile, 4)

    @staticmethod
    def chi(hand: jnp.ndarray, tile: int, action: int) -> jnp.ndarray:
        return jax.lax.switch(
            action - Action.CHI_L,
            [
                lambda: Hand.sub(Hand.sub(hand, tile + 1), tile + 2),
                lambda: Hand.sub(Hand.sub(hand, tile - 1), tile + 1),
                lambda: Hand.sub(Hand.sub(hand, tile - 2), tile - 1),
            ],
        )

    @staticmethod
    def to_str(hand: jnp.ndarray) -> str:
        s = ""
        for i in range(4):
            t = ""
            for j in range(9 if i < 3 else 7):
                t += str(j + 1) * hand[9 * i + j]
            if t:
                t += ["m", "p", "s", "z"][i]
            s += t
        return s

    @staticmethod
    def from_str(s: str) -> jnp.ndarray:
        base = 0
        hand = jnp.zeros(34, dtype=jnp.uint8)
        for c in reversed(s):
            if c == "m":
                base = 0
            elif c == "p":
                base = 9
            elif c == "s":
                base = 18
            elif c == "z":
                base = 27
            else:
                hand = Hand.add(hand, ord(c) - ord("1") + base)
        return hand
