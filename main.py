# %%

import sys
import time
import jax
import jax.numpy as jnp
import pgx
import gymnasium
import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from IPython.display import display_svg  # type:ignore
from pettingzoo.utils.agent_selector import agent_selector

env = pgx.make("tic_tac_toe")
env.init = jax.jit(env.init)
env.step = jax.jit(env.step)

# %%

s = env.init(jax.random.PRNGKey(0))
for i in range(5):
    s = env.step(s, i)
    display_svg(s._repr_html_(), raw=True)
# %%


class ToPettingZoo(AECEnv):
    metadata = {
        "render_modes": ["svg"],
    }

    def __init__(self, env: pgx.Env, render_mode=None):
        super().__init__()
        self.pgx_env: pgx.Env = env
        env.init = jax.jit(env.init)
        env.step = jax.jit(env.step)
        self._state = env.init(jax.random.PRNGKey(0))

        # TODO: fix me
        self.agents = ["player_0", "player_1"]  # TODO: Tic-tac-toe dependent
        self.possible_agents = self.agents[:]

        self.action_spaces = {i: spaces.Discrete(9) for i in self.agents}  # TODO: Tic-tac-toe dependent
        self.observation_spaces = {  # TODO: Tic-tac-toe dependent
            i: spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=0, high=1, shape=(27,), dtype=np.int8
                    ),
                    "action_mask": spaces.Box(low=0, high=1, shape=(9,), dtype=np.int8),
                }
            )
            for i in self.agents
        }

        self.rewards = {i: 0 for i in self.agents}
        self.terminations = {i: False for i in self.agents}
        self.truncations = {i: False for i in self.agents}
        self.infos = {i: {"legal_moves": list(range(0, 9))} for i in self.agents}

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        self.render_mode = render_mode

    def observe(self, agent):
        # TODO: use agent
        return {"observation": self._state.observation, "action_mask": self._state.legal_action_mask}

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

        next_agent = self._agent_selector.next()  # TODO: fix me

        if self._state.terminated:
            for i in range(self.pgx_env.num_players):
                self.rewards[f"player_{i}"] = float(self._state.reward[i])
            self.terminations =  {a: True for a in self.agents}

        # Switch selection to next agents
        self._cumulative_rewards[self.agent_selection] = 0
        self.agent_selection = next_agent

        self._accumulate_rewards()
        if self.render_mode == "svg":
            self.render()

    def reset(self, seed=None, options=None):
        if seed is not None:
            key = jax.random.PRNGKey(seed)
        else:
            key = jax.random.PRNGKey(0)  # TODO: fix me
        self._state = self.pgx_env.init(key)

        self.agents = self.possible_agents[:]
        self.rewards = {i: 0 for i in self.agents}
        self._cumulative_rewards = {i: 0 for i in self.agents}
        self.terminations = {i: False for i in self.agents}
        self.truncations = {i: False for i in self.agents}
        self.infos = {i: {} for i in self.agents}
        # selects the first agent
        self._agent_selector.reinit(self.agents)
        self._agent_selector.reset()
        self.agent_selection = self._agent_selector.reset()

    def render(self):
        if "ipykernel" not in sys.modules:
            raise RuntimeError("Svg rendering only supports jupyter notebook")
        display_svg(self._state._repr_html_(), raw=True)

    def close(self):
        pass

# %%

env = pgx.make("tic_tac_toe")
pz = ToPettingZoo(env, render_mode="svg")

pz.reset()
obs, *_ = pz.last()
# %%
obs
# %%
pz.step(0)
obs, *_ = pz.last()
obs
# %%

pz.step()
obs, *_ = pz.last()
obs
# %%
