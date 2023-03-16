# Modified from https://github.com/Farama-Foundation/PettingZoo

import sys
from typing import Literal, Optional

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from IPython.display import display_svg  # type:ignore
from pettingzoo import AECEnv  # type: ignore
from pettingzoo.utils import wrappers  # type: ignore

import pgx


def pettingzoo_env(
    env_id: pgx.EnvId, render_mode=Optional[Literal["svg"]]
) -> AECEnv:
    env = PettingZooEnv(env_id, render_mode=render_mode)
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


def get_agents_spaces(env_id: pgx.EnvId):
    if env_id == "tic_tac_toe":
        agents = ["player_0", "player_1"]
        action_spaces = {i: spaces.Discrete(9) for i in agents}
        observation_spaces = {
            i: spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=0, high=1, shape=(27,), dtype=np.int8
                    ),
                    "action_mask": spaces.Box(
                        low=0, high=1, shape=(9,), dtype=np.int8
                    ),
                }
            )
            for i in agents
        }
        return agents, action_spaces, observation_spaces
    elif env_id == "go-19x19":
        agents = ["player_0", "player_1"]
        size = 19
        action_spaces = {i: spaces.Discrete(size * size + 1) for i in agents}
        observation_spaces = {
            i: spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=0, high=1, shape=(size, size, 17), dtype=np.int8
                    ),
                    "action_mask": spaces.Box(
                        low=0, high=1, shape=(size * size + 1,), dtype=np.int8
                    ),
                }
            )
            for i in agents
        }
        return agents, action_spaces, observation_spaces
    else:
        assert False


class PettingZooEnv(AECEnv):
    metadata = {
        "render_modes": ["svg"],
    }

    def __init__(self, env_id: pgx.EnvId, render_mode=None):
        super().__init__()
        self.pgx_env: pgx.Env = pgx.make(env_id)
        self.pgx_env.init = jax.jit(self.pgx_env.init)  # type: ignore
        self.pgx_env.step = jax.jit(self.pgx_env.step)  # type: ignore
        self._state: pgx.State = self.pgx_env.init(jax.random.PRNGKey(0))  # type: ignore

        (
            self.agents,
            self.action_spaces,
            self.observation_spaces,
        ) = get_agents_spaces(env_id)
        self.possible_agents = self.agents[:]

        self.rewards = {i: 0 for i in self.agents}
        self.terminations = {i: False for i in self.agents}
        self.truncations = {i: False for i in self.agents}
        self.infos = {i: {} for i in self.agents}  # type: ignore

        self.agent_selection = f"player_{self._state.current_player}"
        self.render_mode = render_mode

    def observe(self, agent):
        # TODO: use agent
        return {
            "observation": np.array(self._state.observation),
            "action_mask": np.array(self._state.legal_action_mask),
        }

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            return self._was_dead_step(action)  # type: ignore

        self._state = self.pgx_env.step(self._state, jnp.int32(action))

        next_agent = f"player_{self._state.current_player}"

        if self._state.terminated:
            for i in range(self.pgx_env.num_players):
                self.rewards[f"player_{i}"] = float(self._state.reward[i])
            self.terminations = {i: True for i in self.agents}

        self._cumulative_rewards[self.agent_selection] = 0
        self.agent_selection = next_agent

        self._accumulate_rewards()
        if self.render_mode == "svg":
            self.render()

    def reset(self, seed=None, options=None):
        if seed is None:
            seed = np.random.randint(9999)  # TODO: fix me
        key = jax.random.PRNGKey(seed)
        self._state = self.pgx_env.init(key)

        self.agents = self.possible_agents[:]
        self.rewards = {i: 0 for i in self.agents}
        self._cumulative_rewards = {i: 0 for i in self.agents}
        self.terminations = {i: False for i in self.agents}
        self.truncations = {i: False for i in self.agents}
        self.infos = {i: {} for i in self.agents}
        self.agent_selection = f"player_{self._state.current_player}"

    def render(self):
        if "ipykernel" not in sys.modules:
            raise RuntimeError("Svg rendering only supports jupyter notebook")
        display_svg(self._state._repr_html_(), raw=True)

    def close(self):
        ...
