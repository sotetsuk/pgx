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

from functools import partial

import jax
from jax import numpy as jnp

import pgx.core as core
from pgx._src.games.go import (
    GameState,
    _get_reward_bw,
    _init_game_state,
    _observe_game_state,
    _step_game_state,
    legal_actions,
)
from pgx._src.struct import dataclass
from pgx._src.types import Array, PRNGKey

FALSE = jnp.bool_(False)
TRUE = jnp.bool_(True)


@dataclass
class State(core.State):
    current_player: Array = jnp.int32(0)
    rewards: Array = jnp.float32([0.0, 0.0])
    terminated: Array = FALSE
    truncated: Array = FALSE
    legal_action_mask: Array = jnp.zeros(19 * 19 + 1, dtype=jnp.bool_)
    observation: Array = jnp.zeros((19, 19, 17), dtype=jnp.bool_)
    _step_count: Array = jnp.int32(0)
    _x: GameState = GameState()

    @property
    def env_id(self) -> core.EnvId:
        try:
            size = int(self._x._size.item())
        except TypeError:
            size = int(self._x._size[0].item())
        return f"go_{size}x{size}"  # type: ignore

    @staticmethod
    def _from_sgf(sgf: str):
        return _from_sgf(sgf)


class Go(core.Env):
    def __init__(
        self,
        *,
        size: int = 19,
        komi: float = 7.5,
        history_length: int = 8,
    ):
        super().__init__()
        assert isinstance(size, int)
        self.size = size
        self.komi = komi
        self.history_length = history_length
        self.max_termination_steps = self.size * self.size * 2

    def _init(self, key: PRNGKey) -> State:
        return partial(_init, size=self.size, komi=self.komi)(key=key)

    def _step(self, state: core.State, action: Array, key) -> State:
        del key
        assert isinstance(state, State)
        state = partial(_step, size=self.size)(state, action)
        # terminates if size * size * 2 (722 if size=19) steps are elapsed
        state = jax.lax.cond(
            (0 <= self.max_termination_steps)
            & (self.max_termination_steps <= state._step_count),
            lambda: state.replace(  # type: ignore
                terminated=TRUE,
                rewards=partial(terminal_values, size=self.size)(state),
            ),
            lambda: state,
        )
        return state  # type: ignore

    def _observe(self, state: core.State, player_id: Array) -> Array:
        assert isinstance(state, State)
        return partial(
            _observe, size=self.size, history_length=self.history_length
        )(state=state, player_id=player_id)

    @property
    def id(self) -> core.EnvId:
        return f"go_{int(self.size)}x{int(self.size)}"  # type: ignore

    @property
    def version(self) -> str:
        return "v0"

    @property
    def num_players(self) -> int:
        return 2


def _observe(state: State, player_id, size, history_length):
    """Return AlphaGo Zero [Silver+17] feature

        obs = (size, size, history_length * 2 + 1)
        e.g., (19, 19, 17) if size=19 and history_length=8 (used in AlphaZero)

        obs[:, :, 0]: stones of `player_id`          @ current board
        obs[:, :, 1]: stones of `player_id` opponent @ current board
        obs[:, :, 2]: stones of `player_id`          @ 1-step before
        obs[:, :, 3]: stones of `player_id` opponent @ 1-step before
        ...
        obs[:, :, -1]: color of `player_id`

        NOTE: For the final dimension, there are two possible options:

          - Use the color of current player to play
          - Use the color of `player_id`

        This ambiguity happens because `observe` function is available even if state.current_player != player_id.
        In the AlphaGo Zero paper, the final dimension C is explained as:

          > The final feature plane, C, represents the colour to play, and has a constant value of either 1 if black
    is to play or 0 if white is to play.

        however, it also describes as

          > the colour feature C is necessary because the komi is not observable.

        So, we use player_id's color to let the agent komi information.
        As long as it's called when state.current_player == player_id, this doesn't matter.
    """
    my_turn = jax.lax.select(
        player_id == state.current_player, state._x._turn, 1 - state._x._turn
    )
    return _observe_game_state(state._x, my_turn, size, history_length)


def _init(key: PRNGKey, size: int, komi: float = 7.5) -> State:
    current_player = jnp.int32(jax.random.bernoulli(key))
    return State(  # type:ignore
        legal_action_mask=jnp.ones(size**2 + 1, dtype=jnp.bool_),
        current_player=current_player,
        _x=_init_game_state(size, komi),
    )


def _step(state: State, action: int, size: int) -> State:
    x = _step_game_state(state._x, action, size)

    current_player = (state.current_player + 1) % 2  # player to act
    state = state.replace(  # type:ignore
        current_player=current_player,
        terminated=x.is_terminal,
        legal_action_mask=state.legal_action_mask.at[:-1]
        .set(legal_actions(x, size))
        .at[-1]
        .set(TRUE),
        _x=x,
    )

    rewards = terminal_values(state, size)
    rewards = jax.lax.select(state._x.is_terminal, rewards, jnp.zeros_like(rewards))
    return state.replace(rewards=rewards)  # type:ignore


def terminal_values(state: State, size) -> Array:
    reward_bw = _get_reward_bw(state._x, size)
    reward = jax.lax.select(
        state.current_player == state._x._turn,
        reward_bw,
        reward_bw[jnp.int32([1, 0])],
    )
    return reward


# only for debug
def _show(state: State) -> None:
    BLACK_CHAR = "@"
    WHITE_CHAR = "O"
    POINT_CHAR = "+"
    print("===========")
    for xy in range(state._x._size * state._x._size):
        if state._x._chain_id_board[xy] > 0:
            print(" " + BLACK_CHAR, end="")
        elif state._x._chain_id_board[xy] < 0:
            print(" " + WHITE_CHAR, end="")
        else:
            print(" " + POINT_CHAR, end="")

        if xy % state._x._size == state._x._size - 1:
            print()


# load sgf
def _from_sgf(sgf: str):
    indexes = "abcdefghijklmnopqrs"
    infos = sgf.split(";")
    game_info = infos[1]
    game_record = infos[2:]
    # assert game_info[game_info.find('GM') + 3] == "1"
    # set default to 19
    size = 19
    if game_info.find("SZ") != -1:
        sz = game_info[game_info.find("SZ") + 3 : game_info.find("SZ") + 5]
        if sz[1] == "]":
            sz = sz[0]
        size = int(sz)
    env = Go(size=size)
    init = jax.jit(env.init)
    step = jax.jit(env.step)
    key = jax.random.PRNGKey(0)
    state = init(key)
    has_branch = False
    for reco in game_record:
        if reco[-2] == ")":
            # The end of main branch
            print("this sgf has some branches")
            print("loaded main branch")
            has_branch = True
        if reco[2] == "]":
            # pass
            state = step(state, size * size)
            # check branches
            if has_branch:
                return state
            continue
        pos = reco[2:4]
        yoko = indexes.index(pos[0])
        tate = indexes.index(pos[1])
        action = yoko + size * tate
        state = step(state, action)
        # We only follow the main branch
        # Return when the main branch ends
        if has_branch:
            return state
    return state
