from __future__ import annotations

import json
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

    @staticmethod
    @jit
    def init(key) -> Deck:
        arr = jax.random.permutation(
            key, jnp.array([i // 4 for i in range(136)])
        )
        return Deck(0, arr)

    @staticmethod
    @jit
    def draw(deck: Deck) -> Tuple[Deck, int]:
        tile = deck.arr[deck.idx]
        deck.idx += 1
        return deck, tile

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)


tree_util.register_pytree_node(Deck, Deck._tree_flatten, Deck._tree_unflatten)


class Hand:
    # fmt: off
    CACHE_SUITED = jnp.array(
        [16673, 570983072, 1065280, 1094779908, 0, 0, 0, 0, 573046784, 2726988456, 2261664, 2189559816, 2189568512, 8712, 570983072, 570983072, 0, 1094779908, 1065280, 1094779908, 262144, 16704, 0, 0, 1094778880, 16, 1094779904, 0, 1065216, 1094779908, 1065280, 1094779908, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2190092800, 41608, 570983072, 570983072, 2729050112, 2726988456, 2191690400, 2189559816, 2191698432, 2189568520, 573080224, 2726988456, 0, 2189593224, 2138752, 2189559816, 570982912, 570983072, 0, 0, 2189557760, 32, 2189559808, 0, 2130432, 2189559816, 2130560, 42076168, 0, 0, 2726988448, 570983072, 2261504, 2189559816, 2130560, 2189559816, 2189557760, 8712, 570983072, 570983072, 0, 0, 0, 0, 0, 2726988456, 2261664, 2189559816, 2189568512, 8712, 570983072, 570983072, 573046784, 2726988456, 2261664, 42076168, 2189568512, 8712, 570983072, 34079392, 0, 0, 0, 0, 0, 0, 0, 0, 1094778880, 16, 1094779904, 0, 1065216, 1094779908, 1065280, 1094779908, 0, 1094779908, 1065280, 1094779908, 262144, 16704, 0, 0, 1094778880, 16, 1094779904, 0, 1065216, 1094779908, 1065280, 21038084, 0, 0, 16, 0, 1094779904, 0, 0, 0, 1048576, 1094779908, 1065280, 1094779908, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1065216, 1094779908, 1065280, 1094779908, 262144, 16704, 0, 0, 0, 0, 0, 0, 0, 16, 1094779904, 0, 1065216, 1094779908, 1065280, 21038084, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1065280, 1094779908, 262144, 16704, 0, 0, 1094778880, 16, 1094779904, 0, 1065216, 21038084, 1065280, 16843780, 0, 1094779908, 1065280, 21038084, 262144, 16704, 0, 0, 1094778880, 16, 21038080, 0, 1065216, 16843780, 1048896, 65540, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2726988448, 570983072, 2191690240, 2189559816, 2130560, 2189559816, 2191654912, 2189568520, 573080224, 2726988456, 0, 0, 0, 0, 0, 2726988456, 2261664, 2189559816, 2189568512, 8712, 570983072, 570983072, 573046784, 2726988456, 2261664, 42076168, 2189568512, 8712, 570983072, 34079392, 0, 0, 0, 0, 2192222720, 2189601416, 573080224, 2726988456, 2729050112, 2726988456, 2191690400, 2189559816, 2191698432, 2189568520, 573080224, 579504808, 0, 2189593256, 2191698560, 2189559816, 573080064, 2726988456, 2130560, 42076168, 2189557760, 32, 2189559808, 0, 2130432, 42076168, 2130560, 33687560, 0, 0, 2729085600, 2726988456, 2785792, 2189593224, 2130560, 42076168, 2189557760, 8744, 2726988448, 570983072, 2130432, 42076168, 2130560, 33687560, 0, 2726988456, 2261664, 42076168, 2190092800, 41608, 570983072, 34079392, 2729050112, 579504808, 44206752, 33687560, 44214784, 33696264, 36176544, 131624, 0, 0, 0, 0, 0, 0, 0, 0, 2191654912, 2189559848, 2191690368, 2189559816, 2130432, 2189559816, 2130560, 42076168, 0, 2189568520, 573080224, 2726988456, 524288, 33408, 0, 0, 2189557760, 32, 2189559808, 0, 2130432, 42076168, 2130560, 33687560, 0, 0, 2261664, 2189559816, 2189568512, 8712, 570983072, 570983072, 573046784, 2726988456, 2261664, 42076168, 2189568512, 8712, 570983072, 34079392, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2130432, 2189559816, 2130560, 42076168, 524288, 33408, 0, 0, 0, 0, 0, 0, 0, 32, 2189559808, 0, 2130432, 42076168, 2130560, 33687560, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2130560, 42076168, 524288, 33408, 0, 0, 2189557760, 32, 42076160, 0, 2130432, 33687560, 2097792, 131080, 0, 42076168, 2130560, 33687560, 524288, 640, 0, 0, 42074112, 32, 33687552, 0, 2097664, 131080, 512, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2726988456, 2191690400, 2189559816, 2191698432, 2189568520, 573080224, 579504808, 573046784, 2726988456, 2261664, 42076168, 2189568512, 8712, 570983072, 34079392, 0, 0, 2138752, 2189559816, 570982912, 570983072, 0, 0, 2189557760, 32, 2189559808, 0, 2130432, 42076168, 2130560, 33687560, 0, 2189559816, 2130560, 42076168, 524288, 33408, 0, 0, 2189557760, 32, 42076160, 0, 2130432, 33687560, 2097792, 131080, 0, 0, 0, 0, 2261504, 2189559816, 2130560, 42076168, 2189557760, 8712, 570983072, 570983072, 0, 0, 0, 0, 0, 2726988456, 2261664, 42076168, 2189568512, 8712, 570983072, 34079392, 573046784, 579504808, 2261664, 33687560, 42084864, 8712, 34079392, 544, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2729050112, 2726988456, 2191690400, 42076168, 2191698432, 42084872, 573080224, 34212520, 0, 2189593224, 2138752, 42076168, 570982912, 570983072, 0, 0, 2189557760, 32, 42076160, 0, 2130432, 33687560, 2097792, 131080, 0, 0, 2726988448, 570983072, 2261504, 42076168, 2130560, 33687560, 2189557760, 8712, 570983072, 34079392, 0, 0, 0, 0, 0, 579504808, 2261664, 33687560, 42084864, 8712, 34079392, 544, 573046784, 34212520, 2228896, 131080, 33696256, 520, 544, 0, 0, 0, 0, 0, 2190092800, 41608, 570983072, 34079392, 2729050112, 579504808, 44206752, 33687560, 44214784, 33696264, 36176544, 131624, 0, 42109576, 2138752, 33687560, 570982912, 34079392, 0, 0, 42074112, 32, 33687552, 0, 2097664, 131080, 512, 0, 0, 0, 579504800, 34079392, 2261504, 33687560, 2097792, 131080, 42074112, 8712, 34079392, 544, 0, 0, 0, 0, 0, 34212520, 2228896, 131080, 33696256, 520, 544, 0, 36175872, 131624, 131616, 0, 131584]
        , dtype=jnp.uint32,
    )
    # fmt: on
    CACHE_HONOR = jnp.array([1, 0, 2, 1, 0])

    @staticmethod
    @jit
    def cache_suited(code: int) -> int:
        return (Hand.CACHE_SUITED[code >> 4] >> (2 * (code & 0b1111))) & 0b11

    @staticmethod
    @jit
    def can_ron(hand: jnp.ndarray, tile: int) -> bool:
        return Hand.can_tsumo(hand.at[tile].set(hand[tile] + 1))

    @staticmethod
    @jit
    def can_tsumo(hand: jnp.ndarray) -> bool:
        heads = 0
        valid = True
        for i in range(3):
            code = 0
            for j in range(9):
                heads, valid, code = jax.lax.cond(
                    hand[9 * i + j] == 0,
                    lambda: (
                        heads + (Hand.cache_suited(code) == 2),
                        valid & (Hand.cache_suited(code) != 0),
                        0,
                    ),
                    lambda: (
                        heads,
                        valid,
                        ((code << 1) + 1) << (hand[9 * i + j].astype(int) - 1),
                    ),
                )
            heads += Hand.cache_suited(code) == 2
            valid &= Hand.cache_suited(code) != 0

        for i in range(27, 34):
            heads += Hand.CACHE_HONOR[hand[i]] == 2
            valid &= Hand.CACHE_HONOR[hand[i]] != 0

        return valid & (heads == 1)

    @staticmethod
    @jit
    def can_pon(hand: jnp.ndarray, tile: int) -> bool:
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
                    .set(Hand.can_ron(self.hand[player], self.target))
                    .at[Action.PON]
                    .set(Hand.can_pon(self.hand[player], self.target)),
                )
            )
            legal_actions = legal_actions.at[player].set(
                jax.lax.cond(
                    # (player != (self.turn + 1) % 4) | (self.target == -1),
                    # NOTE: どこからでもチーできるようにしている
                    (player == self.turn) | (self.target == -1),
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
        hand = jnp.zeros((4, 34), dtype=jnp.uint8)
        for i in range(4):
            for _ in range(14 if i == 0 else 13):
                deck, tile = Deck.draw(deck)
                hand = hand.at[i].set(Hand.add(hand[i], tile))
        return State(deck, hand, 0, -1)

    @staticmethod
    @jit
    def step(
        state: State, actions: jnp.ndarray
    ) -> Tuple[State, jnp.ndarray, bool]:
        player = jnp.argmin(actions)
        return State._step(state, player, actions[player])

    @staticmethod
    @jit
    def _step(
        state: State, player: int, action: int
    ) -> Tuple[State, jnp.ndarray, bool]:
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
    def _discard(state: State, tile: int) -> Tuple[State, jnp.ndarray, bool]:
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
    def _try_draw(state: State) -> Tuple[State, jnp.ndarray, bool]:
        state.target = -1
        return jax.lax.cond(
            state.deck.is_empty(),
            lambda: State._ryukyoku(state),
            lambda: State._draw(state),
        )

    @staticmethod
    @jit
    def _draw(state: State) -> Tuple[State, jnp.ndarray, bool]:
        state.turn += 1
        state.turn %= 4
        state.deck, tile = Deck.draw(state.deck)
        state.hand = state.hand.at[state.turn].set(
            Hand.add(state.hand[state.turn], tile)
        )
        return state, jnp.full(4, 0), False

    @staticmethod
    @jit
    def _ryukyoku(state: State) -> Tuple[State, jnp.ndarray, bool]:
        return state, jnp.full(4, 0), True

    @staticmethod
    @jit
    def _ron(state: State, player: int) -> Tuple[State, jnp.ndarray, bool]:
        return (
            state,
            jnp.full(4, 0).at[state.turn].set(-1).at[player].set(1),
            True,
        )

    @staticmethod
    @jit
    def _pon(state: State, player: int) -> Tuple[State, jnp.ndarray, bool]:
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
    ) -> Tuple[State, jnp.ndarray, bool]:
        state.hand = state.hand.at[player].set(
            Hand.chi(state.hand[player], state.target, pos)
        )
        state.target = -1
        state.turn = player
        return state, jnp.full(4, 0), False

    @staticmethod
    @jit
    def _tsumo(state: State) -> Tuple[State, jnp.ndarray, bool]:
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
) -> Tuple[State, jnp.ndarray, bool]:
    return State.step(state, actions)

