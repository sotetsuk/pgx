import jax.numpy as jnp
from actions import CHI_L, CHI_M, CHI_R, NONE, PASS, PON, RON, TSUMO


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
        if s < shanten(obs.hand):
            return PON

    if legal_actions[CHI_R]:
        s = shanten(
            obs.hand.at[obs.target - 2]
            .set(obs.hand[obs.target - 2] - 1)
            .at[obs.target - 1]
            .set(obs.hand[obs.target - 1] - 1)
        )
        if s < shanten(obs.hand):
            return CHI_R

    if legal_actions[CHI_M]:
        s = shanten(
            obs.hand.at[obs.target - 1]
            .set(obs.hand[obs.target - 1] - 1)
            .at[obs.target + 1]
            .set(obs.hand[obs.target + 1] - 1)
        )
        if s < shanten(obs.hand):
            return CHI_M

    if legal_actions[CHI_L]:
        s = shanten(
            obs.hand.at[obs.target + 1]
            .set(obs.hand[obs.target + 1] - 1)
            .at[obs.target + 2]
            .set(obs.hand[obs.target + 2] - 1)
        )
        if s < shanten(obs.hand):
            return CHI_L

    return PASS
