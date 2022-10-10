import random

import numpy as np
from _full_mahjong import Action, Hand, Meld, Observation, Tile, init, step
from shanten_tools import shanten  # type: ignore


def act(legal_actions: np.ndarray, obs: Observation) -> int:
    if not np.any(legal_actions):
        return Action.NONE

    for action in list(range(34, 68)) + [
        Action.TSUMO,
        Action.RON,
        Action.RIICHI,
        Action.MINKAN,
    ]:
        if legal_actions[action]:
            return action

    if np.sum(obs.hand) % 3 == 2:
        min_shanten = 999
        discard = -1
        for tile in range(34):
            if legal_actions[tile] or (
                tile == obs.last_draw and legal_actions[Action.TSUMOGIRI]
            ):
                s = shanten(Hand.sub(obs.hand, tile))
                if s < min_shanten:
                    s = min_shanten
                    discard = tile
        return discard if obs.last_draw != discard else Action.TSUMOGIRI

    if legal_actions[Action.PON]:
        s = shanten(Hand.pon(obs.hand, obs.target))
        if s < shanten(obs.hand) and random.random() < 0.5:
            return Action.PON

    if legal_actions[Action.CHI_R]:
        s = shanten(Hand.chi(obs.hand, obs.target, Action.CHI_R))
        if s < shanten(obs.hand) and random.random() < 0.5:
            return Action.CHI_R

    if legal_actions[Action.CHI_M]:
        s = shanten(Hand.chi(obs.hand, obs.target, Action.CHI_M))
        if s < shanten(obs.hand) and random.random() < 0.5:
            return Action.CHI_M

    if legal_actions[Action.CHI_L]:
        s = shanten(Hand.chi(obs.hand, obs.target, Action.CHI_L))
        if s < shanten(obs.hand) and random.random() < 0.5:
            return Action.CHI_L

    return Action.PASS


if __name__ == "__main__":
    for i in range(50):
        state = init()
        reward = np.full(4, 0)
        done = False
        while not done:
            legal_actions = state.legal_actions()
            selected = np.array(
                [act(legal_actions[i], state.observe(i)) for i in range(4)]
            )
            state, reward, done = step(state, selected)

        print("hand:", Hand.to_str(state.hand[0]))
        for i in range(1, 4):
            print("     ", Hand.to_str(state.hand[i]))
        print(
            "melds:",
            list(map(Meld.to_str, state.melds[0][: state.meld_num[0]])),
        )
        for i in range(1, 4):
            print(
                "      ",
                list(map(Meld.to_str, state.melds[i][: state.meld_num[i]])),
            )
        print("riichi:", state.riichi)
        print("is_menzen:", state.is_menzen)
        print("doras:", state.deck.doras)
        print("end:", state.deck.end)
        if state.target != -1:
            print("target:", Tile.to_str(state.target))
        if state.last_draw != -1:
            print("last_draw:", Tile.to_str(state.last_draw))
        print("reward:", reward)
        print("-" * 30)
