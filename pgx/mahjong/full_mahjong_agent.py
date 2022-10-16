import time

import _full_mahjong
import numpy as np
from _full_mahjong import Action, Hand, Meld, Observation, State, Tile, step
from shanten_tools import shanten, shanten_discard  # type: ignore

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
    if np.any(legal_actions[34:68]):
        return np.argmax(legal_actions[34:68]) + 34  # type: ignore

    if np.sum(obs.hand) % 3 == 2:
        if obs.riichi[player]:
            return Action.TSUMOGIRI
        else:
            discard = np.argmin((obs.hand == 0) * 99 + shanten_discard(obs.hand))  # type: ignore
            return discard if obs.last_draw != discard else Action.TSUMOGIRI  # type: ignore

    if obs.riichi[player]:
        return Action.PASS

    if legal_actions[Action.MINKAN]:
        return Action.MINKAN

    s = shanten(obs.hand)
    if legal_actions[Action.PON]:
        obs.hand[obs.target] -= 2
        if np.random.random() < 0.5 and s < shanten(obs.hand):  # type: ignore
            return Action.PON
        obs.hand[obs.target] += 2

    if legal_actions[Action.CHI_R]:
        obs.hand[obs.target - 2] -= 1
        obs.hand[obs.target - 1] -= 1
        if np.random.random() < 0.5 and s < shanten(obs.hand):  # type: ignore
            return Action.CHI_R
        obs.hand[obs.target - 2] += 1
        obs.hand[obs.target - 1] += 1

    if legal_actions[Action.CHI_M]:
        obs.hand[obs.target - 1] -= 1
        obs.hand[obs.target + 1] -= 1
        if np.random.random() < 0.5 and s < shanten(obs.hand):  # type: ignore
            return Action.CHI_M
        obs.hand[obs.target - 1] += 1
        obs.hand[obs.target + 1] += 1

    if legal_actions[Action.CHI_L]:
        obs.hand[obs.target + 1] -= 1
        obs.hand[obs.target + 2] -= 1
        if np.random.random() < 0.5 and s < shanten(obs.hand):  # type: ignore
            return Action.CHI_L
        obs.hand[obs.target + 1] += 1
        obs.hand[obs.target + 2] += 1

    return Action.PASS


if __name__ == "__main__":
    legal_action_time = 0.0
    obs_time = 0.0
    select_time = 0.0
    step_time = 0.0

    for i in range(20):
        state = State.init_with_deck_arr(
            np.random.permutation(np.arange(136) // 4)
        )
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
            # state, reward, done = State.step(state, selected)  # type: ignore
            state, reward, done = step(state, selected)  # type: ignore
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
