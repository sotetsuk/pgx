# Copyright 2023 The Pgx Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
import jax.numpy as jnp

import pgx.v1 as v1
from pgx._src.struct import dataclass

FALSE = jnp.bool_(False)
TRUE = jnp.bool_(True)

INVALID_ACTION = jnp.int8(-1)
CALL = jnp.int8(0)
RAISE = jnp.int8(1)
FOLD = jnp.int8(2)

MAX_RAISE = jnp.int8(2)


@dataclass
class State(v1.State):
    current_player: jnp.ndarray = jnp.int8(0)
    observation: jnp.ndarray = jnp.zeros((8, 8, 2), dtype=jnp.bool_)
    rewards: jnp.ndarray = jnp.float32([0.0, 0.0])
    terminated: jnp.ndarray = FALSE
    truncated: jnp.ndarray = FALSE
    legal_action_mask: jnp.ndarray = jnp.ones(3, dtype=jnp.bool_)
    _rng_key: jax.random.KeyArray = jax.random.PRNGKey(0)
    _step_count: jnp.ndarray = jnp.int32(0)
    # --- Leduc Hold'Em specific ---
    _first_player: jnp.ndarray = jnp.int8(0)
    # [(player 0), (player 1), (public)]
    _cards: jnp.ndarray = jnp.int8([-1, -1, -1])
    # 0(Call)  1(Bet)  2(Fold)  3(Check)
    _last_action: jnp.ndarray = INVALID_ACTION
    _chips: jnp.ndarray = jnp.ones(2, dtype=jnp.int8)
    _round: jnp.ndarray = jnp.int8(0)
    _raise_count: jnp.ndarray = jnp.int8(0)

    @property
    def env_id(self) -> v1.EnvId:
        return "leduc_holdem"


class LeducHoldem(v1.Env):
    def __init__(self):
        super().__init__()

    def _init(self, key: jax.random.KeyArray) -> State:
        return _init(key)

    def _step(self, state: v1.State, action: jnp.ndarray) -> State:
        assert isinstance(state, State)
        return _step(state, action)

    def _observe(self, state: v1.State, player_id: jnp.ndarray) -> jnp.ndarray:
        assert isinstance(state, State)
        return _observe(state, player_id)

    @property
    def id(self) -> v1.EnvId:
        return "leduc_holdem"

    @property
    def version(self) -> str:
        return "beta"

    @property
    def num_players(self) -> int:
        return 2


def _init(rng: jax.random.KeyArray) -> State:
    rng1, rng2, rng3 = jax.random.split(rng, 3)
    current_player = jnp.int8(jax.random.bernoulli(rng1))
    init_card = jax.random.permutation(
        rng2, jnp.int8([0, 0, 1, 1, 2, 2]), independent=True
    )
    return State(  # type:ignore
        _rng_key=rng3,
        _first_player=current_player,
        current_player=current_player,
        _cards=init_card[:3],
        legal_action_mask=jnp.bool_([1, 1, 0]),
        _chips=jnp.ones(2, dtype=jnp.int8),
    )


def _step(state: State, action):
    action = jnp.int8(action)
    chips = jax.lax.switch(
        action,
        [
            lambda: state._chips.at[state.current_player].set(
                state._chips[1 - state.current_player]
            ),  # CALL
            lambda: state._chips.at[state.current_player].set(
                jnp.max(state._chips) + _raise_chips(state)
            ),  # RAISE
            lambda: state._chips,  # FOLD
        ],
    )

    round_over, terminated, reward = _check_round_over(state, action)
    last_action = jax.lax.select(round_over, INVALID_ACTION, action)
    current_player = jax.lax.select(
        round_over, state._first_player, 1 - state.current_player
    )
    raise_count = jax.lax.select(
        round_over, jnp.int8(0), state._raise_count + jnp.int8(action == RAISE)
    )

    reward *= jnp.min(chips)

    legal_action = jax.lax.switch(
        action,
        [
            lambda: jnp.bool_([1, 1, 0]),  # CALL
            lambda: jnp.bool_([1, 1, 1]),  # RAISE
            lambda: jnp.bool_([0, 0, 0]),  # FOLD
        ],
    )
    legal_action = legal_action.at[RAISE].set(raise_count < MAX_RAISE)

    return state.replace(  # type:ignore
        current_player=current_player,
        _last_action=last_action,
        legal_action_mask=legal_action,
        terminated=terminated,
        rewards=reward,
        _round=state._round + jnp.int8(round_over),
        _chips=chips,
        _raise_count=raise_count,
    )


def _check_round_over(state, action):
    round_over = (action == FOLD) | (
        (state._last_action != INVALID_ACTION) & (action == CALL)
    )
    terminated = round_over & (state._round == 1)

    reward = jax.lax.select(
        terminated & (action == FOLD),
        jnp.float32([-1, -1]).at[1 - state.current_player].set(1),
        jnp.float32([0, 0]),
    )
    reward = jax.lax.select(
        terminated & (action != FOLD),
        _get_unit_reward(state),
        reward,
    )
    return round_over, terminated, reward


def _get_unit_reward(state: State):
    win_by_one_pair = state._cards[state.current_player] == state._cards[2]
    lose_by_one_pair = (
        state._cards[1 - state.current_player] == state._cards[2]
    )
    win = win_by_one_pair | (
        ~lose_by_one_pair
        & (
            state._cards[state.current_player]
            > state._cards[1 - state.current_player]
        )
    )
    reward = jax.lax.select(
        win,
        jnp.float32([-1, -1]).at[state.current_player].set(1),
        jnp.float32([-1, -1]).at[1 - state.current_player].set(1),
    )
    return jax.lax.select(
        state._cards[state.current_player]
        == state._cards[1 - state.current_player],  # Draw
        jnp.float32([0, 0]),
        reward,
    )


def _raise_chips(state):
    """raise amounts is 2 in the first round and 4 in the second round."""
    return (state._round + 1) * 2


def _observe(state: State, player_id) -> jnp.ndarray:
    """
    Index   Meaning
    0~2     J ~ K in hand
    3~5     J ~ K as public card
    6~19    0 ~ 13 chips for the current player
    20~33   0 ~ 13 chips for the opponent
    """
    obs = jnp.zeros(34, dtype=jnp.bool_)
    obs = obs.at[state._cards[player_id]].set(TRUE)
    obs = jax.lax.select(
        state._round == 1, obs.at[3 + state._cards[2]].set(TRUE), obs
    )
    obs = obs.at[6 + state._chips[player_id]].set(TRUE)
    obs = obs.at[20 + state._chips[1 - player_id]].set(TRUE)

    return obs
