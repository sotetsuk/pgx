from dataclasses import dataclass
from typing import Tuple

import deck
import hand
import jax
import jax.numpy as jnp
from deck import Deck
from hand import Hand
from jax import jit, tree_util

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
    hand: Hand
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
                    lambda i, arr: arr.at[i].set(
                        self.hand.arr[self.turn][i] > 0
                    ),
                    arr,
                )
                .at[TSUMO]
                .set(self.hand.can_tsumo(self.turn)),
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
                    .set(self.hand.can_ron(player, self.target))
                    .at[PON]
                    .set(self.hand.can_pon(player, self.target)),
                )
            )
            legal_actions = legal_actions.at[player].set(
                jax.lax.cond(
                    #(player != (self.turn + 1) % 4) | (self.target == -1),
                    # NOTE: どこからでもチーできるようにしている
                    (player == self.turn) | (self.target == -1),
                    lambda: legal_actions[player],
                    lambda: legal_actions[player]
                    .at[CHI_R]
                    .set(self.hand.can_chi(player, self.target, 0))
                    .at[CHI_M]
                    .set(self.hand.can_chi(player, self.target, 1))
                    .at[CHI_L]
                    .set(self.hand.can_chi(player, self.target, 2)),
                )
            )
            legal_actions = legal_actions.at[(player, PASS)].set(
                (player != self.turn) & (jnp.sum(legal_actions[player]) > 0)
            )

        return legal_actions

    def observe(self, player: int) -> Observation:
        return Observation(self.hand.arr[player], self.target)

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
    _hand = Hand()
    for i in range(4):
        for _ in range(14 if i == 0 else 13):
            _deck, tile = deck.draw(_deck)
            _hand = hand.add(_hand, i, tile)
    return State(_deck, _hand, 0, -1)


@jit
def step(state: State, actions: jnp.ndarray) -> Tuple[State, jnp.ndarray, bool]:
    player = jnp.argmin(actions)
    return _step(state, player, actions[player])


@jit
def _step(state: State, player: int, action: int) -> Tuple[State, jnp.ndarray, bool]:
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
    state.hand = hand.sub(state.hand, state.turn, tile)
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
    state.hand = hand.add(state.hand, state.turn, tile)
    return state, jnp.full(4, 0), False


@jit
def _ryukyoku(state: State) -> Tuple[State, jnp.ndarray, bool]:
    return state, jnp.full(4, 0), True


@jit
def _ron(state: State, player: int) -> Tuple[State, jnp.ndarray, bool]:
    return state, jnp.full(4, 0).at[state.turn].set(-1).at[player].set(1), True


@jit
def _pon(state: State, player: int) -> Tuple[State, jnp.ndarray, bool]:
    state.hand = hand.pon(state.hand, player, state.target)
    state.target = -1
    state.turn = player
    return state, jnp.full(4, 0), False


@jit
def _chi(state: State, player: int, pos: int) -> Tuple[State, jnp.ndarray, bool]:
    state.hand = hand.chi(state.hand, player, state.target, pos)
    state.target = -1
    state.turn = player
    return state, jnp.full(4, 0), False


@jit
def _tsumo(state: State) -> Tuple[State, jnp.ndarray, bool]:
    return state, jnp.full(4, -1).at[state.turn].set(1), True


def test():
    state = init(jax.random.PRNGKey(seed=0))
    print(state.hand.arr)
    print(state.turn)
    print(state.target)
    print(state.legal_actions())
    print("-" * 30)

    MANZU_6 = 5
    state, done = step(state, jnp.array([MANZU_6, NONE, NONE, NONE]))

    print(state.hand.arr)
    print(state.turn)
    print(state.target)
    print(state.legal_actions())
    print("-" * 30)

    state, done = step(state, jnp.array([NONE, CHI_R, NONE, NONE]))

    print(state.hand.arr)
    print(state.turn)
    print(state.target)
    print(state.legal_actions())
    print("-" * 30)

    SOUZU_7 = 24
    state, done = step(state, jnp.array([NONE, SOUZU_7, NONE, NONE]))

    print(state.hand.arr)
    print(state.turn)
    print(state.target)
    print(state.legal_actions())
    print("-" * 30)

    state, done = step(state, jnp.array([NONE, NONE, NONE, PON]))

    print(state.hand.arr)
    print(state.turn)
    print(state.target)
    print(state.legal_actions())
    print("-" * 30)

    state, done = step(state, jnp.array([NONE, NONE, NONE, DRAGON_R]))

    print(state.hand.arr)
    print(state.turn)
    print(state.target)
    print(state.legal_actions())
    print("-" * 30)

    MANZU_1 = 0
    state, done = step(state, jnp.array([MANZU_1, NONE, NONE, NONE]))

    print(state.hand.arr)
    print(state.turn)
    print(state.target)
    print(state.legal_actions())
    print("-" * 30)

    state, done = step(state, jnp.array([NONE, PASS, NONE, NONE]))

    print(state.hand.arr)
    print(state.turn)
    print(state.target)
    print(state.legal_actions())
    print("-" * 30)


from shanten_tools import shanten


def act(legal_actions: jnp.ndarray, obs: Observation) -> int:
    if not jnp.any(legal_actions):
        return NONE

    if legal_actions[TSUMO]:
        return TSUMO
    if legal_actions[RON]:
        return RON

    if jnp.sum(obs.hand) % 3 == 2:
        min_shanten = 999
        discard = -1
        for tile in range(34):
            if not legal_actions[tile]:
                continue
            s = shanten(obs.hand.at[tile].set(obs.hand[tile] - 1))
            if s < min_shanten:
                s = min_shanten
                discard = tile
        return discard

    if legal_actions[PON]:
        s = shanten(obs.hand.at[obs.target].set(obs.hand[obs.target] - 2))
        if s <= shanten(obs.hand):
            return PON

    if legal_actions[CHI_R]:
        s = shanten(
            obs.hand.at[obs.target - 2]
            .set(obs.hand[obs.target - 2] - 1)
            .at[obs.target - 1]
            .set(obs.hand[obs.target - 1] - 1)
        )
        if s <= shanten(obs.hand):
            return CHI_R

    if legal_actions[CHI_M]:
        s = shanten(
            obs.hand.at[obs.target - 1]
            .set(obs.hand[obs.target - 1] - 1)
            .at[obs.target + 1]
            .set(obs.hand[obs.target + 1] - 1)
        )
        if s <= shanten(obs.hand):
            return CHI_M

    if legal_actions[CHI_L]:
        s = shanten(
            obs.hand.at[obs.target + 1]
            .set(obs.hand[obs.target + 1] - 1)
            .at[obs.target + 2]
            .set(obs.hand[obs.target + 2] - 1)
        )
        if s <= shanten(obs.hand):
            return CHI_L

    return PASS


if __name__ == "__main__":
    for i in range(5):
        state = init(jax.random.PRNGKey(seed=i))
        done = False
        while not done:
            legal_actions = state.legal_actions()
            selected = jnp.array(
                [act(legal_actions[i], state.observe(i)) for i in range(4)]
            )
            state, reward, done = step(state, selected)

        print('hand:', state.hand.arr)
        print('reward:', reward)
        print("-" * 30)
