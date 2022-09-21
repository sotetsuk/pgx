from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import jit, tree_util

import agent
import deck
import hand
from actions import RON, PON, CHI_R, CHI_M, CHI_L, PASS, TSUMO, NONE


@dataclass
class Observation:
    hand: jnp.ndarray
    target: int


@dataclass
class State:
    deck: deck.Deck
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
                .at[TSUMO]
                .set(hand.can_tsumo(self.hand[self.turn])),
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
                    .at[RON]
                    .set(hand.can_ron(self.hand[player], self.target))
                    .at[PON]
                    .set(hand.can_pon(self.hand[player], self.target)),
                )
            )
            legal_actions = legal_actions.at[player].set(
                jax.lax.cond(
                    # (player != (self.turn + 1) % 4) | (self.target == -1),
                    # NOTE: どこからでもチーできるようにしている
                    (player == self.turn) | (self.target == -1),
                    lambda: legal_actions[player],
                    lambda: legal_actions[player]
                    .at[CHI_R]
                    .set(hand.can_chi(self.hand[player], self.target, 0))
                    .at[CHI_M]
                    .set(hand.can_chi(self.hand[player], self.target, 1))
                    .at[CHI_L]
                    .set(hand.can_chi(self.hand[player], self.target, 2)),
                )
            )
            legal_actions = legal_actions.at[(player, PASS)].set(
                (player != self.turn) & (jnp.sum(legal_actions[player]) > 0)
            )

        return legal_actions

    def observe(self, player: int) -> Observation:
        return Observation(self.hand[player], self.target)

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
def init(key: jax.random.PRNGKey) -> State:
    _deck = deck.init(key)
    _hand = jnp.zeros((4, 34), dtype=jnp.uint8)
    for i in range(4):
        for _ in range(14 if i == 0 else 13):
            _deck, tile = deck.draw(_deck)
            _hand = _hand.at[i].set(hand.add(_hand[i], tile))
    return State(_deck, _hand, 0, -1)


@jit
def step(
    state: State, actions: jnp.ndarray
) -> Tuple[State, jnp.ndarray, bool]:
    player = jnp.argmin(actions)
    return _step(state, player, actions[player])


@jit
def _step(
    state: State, player: int, action: int
) -> Tuple[State, jnp.ndarray, bool]:
    return jax.lax.cond(
        action < 34,
        lambda: _discard(state, action),
        lambda: jax.lax.switch(
            action - 34,
            [
                lambda: _ron(state, player),
                lambda: _pon(state, player),
                lambda: _chi(state, player, 0),
                lambda: _chi(state, player, 1),
                lambda: _chi(state, player, 2),
                lambda: _try_draw(state),
                lambda: _tsumo(state),
            ],
        ),
    )


@jit
def _discard(state: State, tile: int) -> Tuple[State, jnp.ndarray, bool]:
    state.hand = state.hand.at[state.turn].set(
        hand.sub(state.hand[state.turn], tile)
    )
    state.target = tile
    return jax.lax.cond(
        jnp.any(state.legal_actions()),
        lambda: (state, jnp.full(4, 0), False),
        lambda: _try_draw(state),
    )


@jit
def _try_draw(state: State) -> Tuple[State, jnp.ndarray, bool]:
    state.target = -1
    return jax.lax.cond(
        state.deck.is_empty(), lambda: _ryukyoku(state), lambda: _draw(state)
    )


@jit
def _draw(state: State) -> Tuple[State, jnp.ndarray, bool]:
    state.turn += 1
    state.turn %= 4
    state.deck, tile = deck.draw(state.deck)
    state.hand = state.hand.at[state.turn].set(
        hand.add(state.hand[state.turn], tile)
    )
    return state, jnp.full(4, 0), False


@jit
def _ryukyoku(state: State) -> Tuple[State, jnp.ndarray, bool]:
    return state, jnp.full(4, 0), True


@jit
def _ron(state: State, player: int) -> Tuple[State, jnp.ndarray, bool]:
    return state, jnp.full(4, 0).at[state.turn].set(-1).at[player].set(1), True


@jit
def _pon(state: State, player: int) -> Tuple[State, jnp.ndarray, bool]:
    state.hand = state.hand.at[player].set(
        hand.pon(state.hand[player], state.target)
    )
    state.target = -1
    state.turn = player
    return state, jnp.full(4, 0), False


@jit
def _chi(
    state: State, player: int, pos: int
) -> Tuple[State, jnp.ndarray, bool]:
    state.hand = state.hand.at[player].set(
        hand.chi(state.hand[player], state.target, pos)
    )
    state.target = -1
    state.turn = player
    return state, jnp.full(4, 0), False


@jit
def _tsumo(state: State) -> Tuple[State, jnp.ndarray, bool]:
    return state, jnp.full(4, -1).at[state.turn].set(1), True


def test():
    state = init(jax.random.PRNGKey(seed=0))
    print(state.hand)
    print(state.turn)
    print(state.target)
    print(state.legal_actions())
    print("-" * 30)

    MANZU_6 = 5
    state, _, _ = step(state, jnp.array([MANZU_6, NONE, NONE, NONE]))

    print(state.hand)
    print(state.turn)
    print(state.target)
    print(state.legal_actions())
    print("-" * 30)

    state, _, _ = step(state, jnp.array([NONE, CHI_R, NONE, NONE]))

    print(state.hand)
    print(state.turn)
    print(state.target)
    print(state.legal_actions())
    print("-" * 30)

    SOUZU_7 = 24
    state, _, _ = step(state, jnp.array([NONE, SOUZU_7, NONE, NONE]))

    print(state.hand)
    print(state.turn)
    print(state.target)
    print(state.legal_actions())
    print("-" * 30)

    state, _, _ = step(state, jnp.array([NONE, NONE, NONE, PON]))

    print(state.hand)
    print(state.turn)
    print(state.target)
    print(state.legal_actions())
    print("-" * 30)

    DRAGON_R = 33
    state, _, _ = step(state, jnp.array([NONE, NONE, NONE, DRAGON_R]))

    print(state.hand)
    print(state.turn)
    print(state.target)
    print(state.legal_actions())
    print("-" * 30)

    MANZU_1 = 0
    state, _, _ = step(state, jnp.array([MANZU_1, NONE, NONE, NONE]))

    print(state.hand)
    print(state.turn)
    print(state.target)
    print(state.legal_actions())
    print("-" * 30)

    state, _, _ = step(state, jnp.array([NONE, PASS, NONE, NONE]))

    print(state.hand)
    print(state.turn)
    print(state.target)
    print(state.legal_actions())
    print("-" * 30)


if __name__ == "__main__":
    # test()
    for i in range(5):
        state = init(jax.random.PRNGKey(seed=i))
        done = False
        while not done:
            legal_actions = state.legal_actions()
            selected = jnp.array(
                [agent.act(legal_actions[i], state.observe(i)) for i in range(4)]
            )
            state, reward, done = step(state, selected)

        print("hand:", state.hand)
        print("reward:", reward)
        print("-" * 30)
