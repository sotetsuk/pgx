from __future__ import annotations

import json
import os
from dataclasses import dataclass

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

    @staticmethod
    @jit
    def init(key) -> Deck:
        arr = jax.random.permutation(
            key, jnp.array([i // 4 for i in range(136)])
        )
        return Deck(0, arr)

    @staticmethod
    @jit
    def draw(deck: Deck) -> tuple[Deck, jnp.ndarray]:
        # -> tuple[Deck, int]
        tile = deck.arr[deck.idx]
        deck.idx += 1
        return deck, tile

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)


tree_util.register_pytree_node(Deck, Deck._tree_flatten, Deck._tree_unflatten)


class CacheLoader:
    DIR = os.path.join(os.path.dirname(__file__), "cache")

    @staticmethod
    @jit
    def load_hand_cache():
        with open(os.path.join(CacheLoader.DIR, "hand_cache.json")) as f:
            return jnp.array(json.load(f), dtype=jnp.uint32)


class Hand:
    CACHE = CacheLoader.load_hand_cache()

    @staticmethod
    @jit
    def cache(code: int) -> int:
        return (Hand.CACHE[code >> 5] >> (code & 0b11111)) & 1

    @staticmethod
    @jit
    def can_ron(hand: jnp.ndarray, tile: int) -> bool:
        return Hand.can_tsumo(hand.at[tile].set(hand[tile] + 1))

    @staticmethod
    @jit
    def can_tsumo(hand: jnp.ndarray):
        heads, valid = jnp.int32(0), jnp.int32(1)
        for suit in range(3):
            valid &= Hand.cache(
                jax.lax.fori_loop(
                    9 * suit,
                    9 * (suit + 1),
                    lambda i, code: code * 5 + hand[i].astype(int),
                    0,
                )
            )
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

        return (valid & (heads == 1)) == 1

    @staticmethod
    @jit
    def can_pon(hand: jnp.ndarray, tile: int) -> jnp.ndarray:
        # -> bool
        return hand[tile] >= 2

    @staticmethod
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
                    & (tile % 9 < 8)
                    & (hand[tile - 1] > 0)
                    & (hand[tile + 1] > 0)
                ),
                lambda: (
                    (tile < 27)
                    & (tile % 9 < 7)
                    & (hand[tile + 1] > 0)
                    & (hand[tile + 2] > 0)
                ),
            ],
        )

    @staticmethod
    @jit
    def add(hand: jnp.ndarray, tile: int, x: int = 1) -> jnp.ndarray:
        return hand.at[tile].set(hand[tile] + x)

    @staticmethod
    @jit
    def sub(hand: jnp.ndarray, tile: int, x: int = 1) -> jnp.ndarray:
        return Hand.add(hand, tile, -x)

    @staticmethod
    @jit
    def pon(hand: jnp.ndarray, tile: int) -> jnp.ndarray:
        return Hand.sub(hand, tile, 2)

    @staticmethod
    @jit
    def chi(hand: jnp.ndarray, tile: int, pos: int) -> jnp.ndarray:
        return jax.lax.switch(
            pos,
            [
                lambda: Hand.sub(Hand.sub(hand, tile - 2), tile - 1),
                lambda: Hand.sub(Hand.sub(hand, tile - 1), tile + 1),
                lambda: Hand.sub(Hand.sub(hand, tile + 1), tile + 2),
            ],
        )


class Action:
    # discard: 0~33
    RON = 34
    PON = 35
    CHI_R = 36  # 45[6]
    CHI_M = 37  # 4[5]6
    CHI_L = 38  # [4]56
    PASS = 39
    TSUMO = 40
    NONE = 41


@dataclass
class Observation:
    hand: jnp.ndarray
    target: int


@dataclass
class State:
    deck: Deck
    hand: jnp.ndarray
    turn: int
    target: int

    @jit
    def legal_actions(self) -> jnp.ndarray:
        legal_actions = jnp.full((4, 41), False)

        # discard, tsumo
        legal_actions = legal_actions.at[self.turn].set(
            jax.lax.cond(
                self.target != -1,
                lambda arr: arr,
                lambda arr: jax.lax.fori_loop(
                    0,
                    34,
                    lambda i, arr: arr.at[i].set(self.hand[self.turn][i] > 0),
                    arr,
                )
                .at[Action.TSUMO]
                .set(Hand.can_tsumo(self.hand[self.turn])),
                legal_actions[self.turn],
            )
        )

        # ron, pon, chi
        for player in range(4):
            legal_actions = legal_actions.at[player].set(
                jax.lax.cond(
                    (player == self.turn) | (self.target == -1),
                    lambda: legal_actions[player],
                    lambda: legal_actions[player]
                    .at[Action.RON]
                    .set(Hand.can_ron(self.hand[player], self.target)),
                )
            )
            legal_actions = legal_actions.at[player].set(
                jax.lax.cond(
                    (player == self.turn)
                    | (self.target == -1)
                    | self.deck.is_empty(),
                    lambda: legal_actions[player],
                    lambda: legal_actions[player]
                    .at[Action.PON]
                    .set(Hand.can_pon(self.hand[player], self.target)),
                )
            )
            legal_actions = legal_actions.at[player].set(
                jax.lax.cond(
                    (player != (self.turn + 1) % 4)
                    | (self.target == -1)
                    | self.deck.is_empty(),
                    lambda: legal_actions[player],
                    lambda: legal_actions[player]
                    .at[Action.CHI_R]
                    .set(Hand.can_chi(self.hand[player], self.target, 0))
                    .at[Action.CHI_M]
                    .set(Hand.can_chi(self.hand[player], self.target, 1))
                    .at[Action.CHI_L]
                    .set(Hand.can_chi(self.hand[player], self.target, 2)),
                )
            )
            legal_actions = legal_actions.at[(player, Action.PASS)].set(
                (player != self.turn) & (jnp.sum(legal_actions[player]) > 0)
            )

        return legal_actions

    def observe(self, player: int) -> Observation:
        return Observation(self.hand[player], self.target)

    @staticmethod
    @jit
    def init(key) -> State:
        deck = Deck.init(key)
        hand = jnp.zeros((4, 34), dtype=jnp.uint32)
        for i in range(4):
            for _ in range(13):
                deck, tile = Deck.draw(deck)
                hand = hand.at[i].set(Hand.add(hand[i], tile))
        deck, tile = Deck.draw(deck)
        hand = hand.at[0].set(Hand.add(hand[0], tile))
        return State(deck, hand, 0, -1)

    @staticmethod
    @jit
    def step(
        state: State, actions: jnp.ndarray
    ) -> tuple[State, jnp.ndarray, bool]:
        player = jnp.argmin(actions)
        return State._step(state, player, actions[player])

    @staticmethod
    @jit
    def _step(
        state: State, player: int, action: int
    ) -> tuple[State, jnp.ndarray, bool]:
        return jax.lax.cond(
            action < 34,
            lambda: State._discard(state, action),
            lambda: jax.lax.switch(
                action - 34,
                [
                    lambda: State._ron(state, player),
                    lambda: State._pon(state, player),
                    lambda: State._chi(state, player, 0),
                    lambda: State._chi(state, player, 1),
                    lambda: State._chi(state, player, 2),
                    lambda: State._try_draw(state),
                    lambda: State._tsumo(state),
                ],
            ),
        )

    @staticmethod
    @jit
    def _discard(state: State, tile: int) -> tuple[State, jnp.ndarray, bool]:
        state.hand = state.hand.at[state.turn].set(
            Hand.sub(state.hand[state.turn], tile)
        )
        state.target = tile
        return jax.lax.cond(
            jnp.any(state.legal_actions()),
            lambda: (state, jnp.full(4, 0), False),
            lambda: State._try_draw(state),
        )

    @staticmethod
    @jit
    def _try_draw(state: State) -> tuple[State, jnp.ndarray, bool]:
        state.target = -1
        return jax.lax.cond(
            state.deck.is_empty(),
            lambda: State._ryukyoku(state),
            lambda: State._draw(state),
        )

    @staticmethod
    @jit
    def _draw(state: State) -> tuple[State, jnp.ndarray, bool]:
        state.turn += 1
        state.turn %= 4
        state.deck, tile = Deck.draw(state.deck)
        state.hand = state.hand.at[state.turn].set(
            Hand.add(state.hand[state.turn], tile)
        )
        return state, jnp.full(4, 0), False

    @staticmethod
    @jit
    def _ryukyoku(state: State) -> tuple[State, jnp.ndarray, bool]:
        return state, jnp.full(4, 0), True

    @staticmethod
    @jit
    def _ron(state: State, player: int) -> tuple[State, jnp.ndarray, bool]:
        return (
            state,
            jnp.full(4, 0).at[state.turn].set(-1).at[player].set(1),
            True,
        )

    @staticmethod
    @jit
    def _pon(state: State, player: int) -> tuple[State, jnp.ndarray, bool]:
        state.hand = state.hand.at[player].set(
            Hand.pon(state.hand[player], state.target)
        )
        state.target = -1
        state.turn = player
        return state, jnp.full(4, 0), False

    @staticmethod
    @jit
    def _chi(
        state: State, player: int, pos: int
    ) -> tuple[State, jnp.ndarray, bool]:
        state.hand = state.hand.at[player].set(
            Hand.chi(state.hand[player], state.target, pos)
        )
        state.target = -1
        state.turn = player
        return state, jnp.full(4, 0), False

    @staticmethod
    @jit
    def _tsumo(state: State) -> tuple[State, jnp.ndarray, bool]:
        return state, jnp.full(4, -1).at[state.turn].set(1), True

    def _tree_flatten(self):
        children = (self.deck, self.hand, self.turn, self.target)
        aux_data = {}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)


tree_util.register_pytree_node(
    State, State._tree_flatten, State._tree_unflatten
)


@jit
def init(key) -> State:
    return State.init(key)


@jit
def step(
    state: State, actions: jnp.ndarray
) -> tuple[State, jnp.ndarray, bool]:
    return State.step(state, actions)
