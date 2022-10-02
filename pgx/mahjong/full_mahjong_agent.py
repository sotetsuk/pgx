import random

import jax
import jax.numpy as jnp
from full_mahjong import Action, Observation, init, step
from shanten_tools import shanten  # type: ignore

random.seed(0)


def act(legal_actions: jnp.ndarray, obs: Observation) -> int:
    if not jnp.any(legal_actions):
        return Action.NONE

    if legal_actions[Action.TSUMO]:
        return Action.TSUMO
    if legal_actions[Action.RON]:
        return Action.RON
    if legal_actions[Action.RIICHI]:
        return Action.RIICHI

    if jnp.sum(obs.hand) % 3 == 2:
        min_shanten = 999
        discard = -1
        for tile in range(34):
            if legal_actions[tile] or (
                tile == obs.last_draw and legal_actions[Action.TSUMOGIRI]
            ):
                s = shanten(obs.hand.at[tile].set(obs.hand[tile] - 1))
                if s < min_shanten:
                    s = min_shanten
                    discard = tile
        return discard if obs.last_draw != discard else Action.TSUMOGIRI

    if legal_actions[Action.PON]:
        s = shanten(obs.hand.at[obs.target].set(obs.hand[obs.target] - 2))
        if s < shanten(obs.hand) and random.random() < 0.5:
            return Action.PON

    if legal_actions[Action.CHI_R]:
        s = shanten(
            obs.hand.at[obs.target - 2]
            .set(obs.hand[obs.target - 2] - 1)
            .at[obs.target - 1]
            .set(obs.hand[obs.target - 1] - 1)
        )
        if s < shanten(obs.hand) and random.random() < 0.5:
            return Action.CHI_R

    if legal_actions[Action.CHI_M]:
        s = shanten(
            obs.hand.at[obs.target - 1]
            .set(obs.hand[obs.target - 1] - 1)
            .at[obs.target + 1]
            .set(obs.hand[obs.target + 1] - 1)
        )
        if s < shanten(obs.hand) and random.random() < 0.5:
            return Action.CHI_M

    if legal_actions[Action.CHI_L]:
        s = shanten(
            obs.hand.at[obs.target + 1]
            .set(obs.hand[obs.target + 1] - 1)
            .at[obs.target + 2]
            .set(obs.hand[obs.target + 2] - 1)
        )
        if s < shanten(obs.hand) and random.random() < 0.5:
            return Action.CHI_L

    return Action.PASS


if __name__ == "__main__":
    for i in range(10):
        state = init(jax.random.PRNGKey(seed=i))
        reward = jnp.full(4, 0)
        done = False
        while not done:
            legal_actions = state.legal_actions()
            selected = jnp.array(
                [act(legal_actions[i], state.observe(i)) for i in range(4)]
            )
            state, reward, done = step(state, selected)

        print("hand:", state.hand)
        print("riichi:", state.riichi)
        print("reward:", reward)
        print("-" * 30)
