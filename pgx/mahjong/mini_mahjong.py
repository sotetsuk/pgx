from __future__ import annotations

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
        [1385955477, 2569143448, 0, 0, 3686684160, 2569143708, 5413232, 1385976476, 2569142272, 2569143448, 9961984, 0, 301312, 39200, 2569143440, 2569143448, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14326640, 1385976476, 3686719232, 2569182652, 2574425584, 3686684316, 2579103744, 2569143512, 1385976464, 0, 301312, 39200, 2569143440, 421659800, 0, 1386011580, 2569143696, 2569143448, 5413120, 1385976476, 0, 0, 3686662144, 2569143708, 5413232, 1385976476, 3686684160, 421660060, 5413232, 303846044, 0, 0, 0, 0, 301312, 39200, 2569143440, 2569143448, 2569142272, 2569143448, 9961984, 0, 301312, 39200, 2569143440, 421659800, 0, 4, 39200, 0, 2569143296, 2569143448, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2569143440, 2569143448, 9961984, 0, 0, 0, 262144, 39200, 2569143440, 421659800, 0, 0, 0, 0, 0, 2569143448, 9961984, 0, 301312, 39200, 421659792, 287442072, 2569142272, 421659800, 9961984, 0, 301312, 6432, 287442064, 16909336, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1386011580, 2569182640, 2569143448, 2574425344, 3686684316, 0, 0, 3686662144, 2569143708, 5413232, 1385976476, 3686684160, 421660060, 5413232, 303846044, 0, 0, 2583338992, 3686684316, 3686719232, 2569182652, 2574425584, 1539200668, 2579365888, 2569182712, 3686684304, 421659800, 301312, 39200, 421659792, 287442072, 0, 3686719420, 2579105680, 421659800, 5675264, 1386011580, 421659792, 287442072, 3686662144, 421660060, 14326640, 303846044, 1539235584, 287448508, 292691440, 18224668, 0, 0, 0, 0, 2569444608, 2569182648, 2569143440, 421659800, 2574385152, 3686684316, 9961984, 0, 301312, 39200, 421659792, 287442072, 0, 2569143708, 5413232, 1385976476, 3686684160, 421660060, 5413232, 303846044, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2569143440, 421659800, 9961984, 0, 0, 0, 262144, 39200, 421659792, 287442072, 0, 0, 0, 0, 0, 421659800, 9961984, 0, 301312, 6432, 287442064, 16909336, 421658624, 287442072, 1573376, 0, 268544, 4384, 16909328, 16, 0, 0, 0, 0, 0, 0, 0, 0, 3686662144, 2569182652, 2574425584, 1539200668, 3686684160, 421660060, 5413232, 303846044, 0, 2569143512, 1385976464, 0, 301312, 39200, 421659792, 287442072, 2569142272, 421659800, 9961984, 0, 301312, 6432, 287442064, 16909336, 0, 0, 2569143696, 421659800, 5413120, 1385976476, 0, 0, 3686662144, 421660060, 5413232, 303846044, 1539200512, 287442332, 5380464, 1315356, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3686719232, 421699004, 426941936, 322852508, 2579103744, 421659864, 1385976464, 0, 301312, 6432, 287442064, 16909336, 0, 1386011580, 421660048, 287442072, 5413120, 303846044, 0, 0, 1539178496, 287442332, 5380464, 1315356, 322852352, 16909596, 1184112, 20, 0, 0, 14326640, 303846044, 1539235584, 287448508, 292691440, 18224668, 431620096, 287442136, 303846032, 0, 268544, 4384, 16909328, 16, 0, 303848380, 287442320, 16909336, 5380352, 1315356, 0, 0, 322830336, 16909596, 1184112, 20, 18224640, 276, 272]
        , dtype=jnp.uint32,
    )
    # fmt: on
    CACHE_HONOR = jnp.array([1, 0, 1, 1, 0])

    @staticmethod
    @jit
    def cache_suited(code: int) -> int:
        return (Hand.CACHE_SUITED[code >> 5] >> (code & 0b11111)) & 1

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
            size = 0
            for j in range(9):
                heads, valid, code, size = jax.lax.cond(
                    hand[9 * i + j] == 0,
                    lambda: (
                        heads + (size % 3 == 2),
                        valid & (Hand.cache_suited(code) != 0),
                        0,
                        0,
                    ),
                    lambda: (
                        heads,
                        valid,
                        ((code << 1) + 1) << (hand[9 * i + j].astype(int) - 1),
                        size + hand[9 * i + j].astype(int),
                    ),
                )
            heads += size % 3 == 2
            valid &= Hand.cache_suited(code) != 0

        for i in range(27, 34):
            heads += hand[i] % 3 == 2
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
