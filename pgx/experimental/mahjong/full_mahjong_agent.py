# type: ignore
# flake8: noqa

import time

import _full_mahjong
import numpy as np
from _full_mahjong import (
    Action,
    Deck,
    Hand,
    Meld,
    Observation,
    Shanten,
    State,
    Tile,
    step,
)

np.random.seed(0)


def to_np(obs: Observation) -> _full_mahjong.Observation:
    return _full_mahjong.Observation(
        np.array(obs.hand),
        np.array(obs.red),
        obs.target,
        obs.last_draw,
        np.array(obs.riichi),
    )


def act(
    player: int, legal_actions: np.ndarray, obs: _full_mahjong.Observation
) -> int:
    if not np.any(legal_actions):
        return Action.NONE

    if legal_actions[Action.TSUMO]:
        return Action.TSUMO
    if legal_actions[Action.RON]:
        return Action.RON
    if legal_actions[Action.RIICHI]:
        return Action.RIICHI
    if np.any(legal_actions[37:108]):  # 槓
        return np.argmax(legal_actions[37:108]) + 37

    if np.sum(obs.hand) % 3 == 2:
        if obs.riichi[player]:
            return Action.TSUMOGIRI
        else:
            discard = np.argmin(
                (obs.hand == 0) * 99 + Shanten.discard(obs.hand)
            )
            if legal_actions[discard]:
                return discard
            if (
                Tile.unred(obs.last_draw) == discard
                and legal_actions[Action.TSUMOGIRI]
            ):
                return Action.TSUMOGIRI
            assert Tile.num(discard) == 4 and Tile.suit(discard) < 3
            return Tile.suit(discard) + 34

    if obs.riichi[player]:
        return Action.PASS

    if legal_actions[Action.MINKAN]:
        return Action.MINKAN

    s = Shanten.number(obs.hand)
    target = Tile.unred(obs.target)
    if legal_actions[Action.PON] | legal_actions[Action.PON]:
        obs.hand[target] -= 2
        if np.random.random() < 0.5 and s < Shanten.number(obs.hand):
            return (
                Action.PON
                if legal_actions[Action.PON]
                else Action.PON_EXPOSE_RED
            )
        obs.hand[target] += 2

    if legal_actions[Action.CHI_L] | legal_actions[Action.CHI_L_EXPOSE_RED]:
        obs.hand[target + 1] -= 1
        obs.hand[target + 2] -= 1
        if np.random.random() < 0.5 and s < Shanten.number(obs.hand):
            return (
                Action.CHI_L
                if legal_actions[Action.CHI_L]
                else legal_actions[Action.CHI_R_EXPOSE_RED]
            )
        obs.hand[target + 1] += 1
        obs.hand[target + 2] += 1

    if legal_actions[Action.CHI_M] | legal_actions[Action.CHI_M_EXPOSE_RED]:
        obs.hand[target - 1] -= 1
        obs.hand[target + 1] -= 1
        if np.random.random() < 0.5 and s < Shanten.number(obs.hand):
            return (
                Action.CHI_M
                if legal_actions[Action.CHI_M]
                else Action.CHI_M_EXPOSE_RED
            )
        obs.hand[target - 1] += 1
        obs.hand[target + 1] += 1

    if legal_actions[Action.CHI_R] | legal_actions[Action.CHI_R_EXPOSE_RED]:
        obs.hand[target - 2] -= 1
        obs.hand[target - 1] -= 1
        if np.random.random() < 0.5 and s < Shanten.number(obs.hand):
            return (
                Action.CHI_R
                if legal_actions[Action.CHI_R]
                else Action.CHI_R_EXPOSE_RED
            )
        obs.hand[target - 2] += 1
        obs.hand[target - 1] += 1

    return Action.PASS


if __name__ == "__main__":
    legal_action_time = 0.0
    obs_time = 0.0
    select_time = 0.0
    step_time = 0.0

    for i in range(20):
        state = State.init_with_deck_arr(np.random.permutation(Deck.DeckList))
        done = False
        while not done:
            tmp = time.time()
            legal_actions = np.array(state.legal_actions())
            if i != 0:
                legal_action_time += time.time() - tmp

            tmp = time.time()
            obs = [to_np(state.observe(i)) for i in range(4)]
            if i != 0:
                obs_time += time.time() - tmp

            tmp = time.time()
            selected = np.array(
                [act(i, legal_actions[i], obs[i]) for i in range(4)]
            )
            if i != 0:
                select_time += time.time() - tmp

            tmp = time.time()
            state, reward, done = step(state, selected)
            if i != 0:
                step_time += time.time() - tmp

        print("hand:", Hand.to_str(state.hand[0], state.red[0]))
        for i in range(1, 4):
            print("     ", Hand.to_str(state.hand[i], state.red[i]))
        print(
            "melds:",
            list(map(Meld.to_str, state.melds[0][: state.n_meld[0]])),
        )
        for i in range(1, 4):
            print(
                "      ",
                list(map(Meld.to_str, state.melds[i][: state.n_meld[i]])),
            )
        print(f"{state.riichi=}")
        print(f"{state.is_menzen=}")
        print(f"{state.deck.n_dora=}")
        print(f"{state.deck.end=}")
        if state.target != -1:
            print("target:", Tile.to_str(state.target))
        if state.last_draw != -1:
            print("last_draw:", Tile.to_str(state.last_draw))
        print(f"{reward=}")
        print("-" * 30)

    print(f"{legal_action_time=}")
    print(f"{obs_time=}")
    print(f"{select_time=}")
    print(f"{step_time=}")
