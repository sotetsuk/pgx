import warnings
from abc import ABC
from typing import Any, Dict, List, Tuple

import pettingzoo
from gymnasium import spaces
from packaging import version
from pettingzoo.utils.env import AECEnv
from pettingzoo.utils.wrappers import BaseWrapper
from open_spiel.python.rl_environment import Environment, ChanceEventSampler


if version.parse(pettingzoo.__version__) < version.parse("1.21.0"):
    warnings.warn(
        f"You are using PettingZoo {pettingzoo.__version__}. "
        f"Future tianshou versions may not support PettingZoo<1.21.0. "
        f"Consider upgrading your PettingZoo version.", DeprecationWarning
    )


class OpenSpielEnv(AECEnv, ABC):  # これをopen_spielように変える.
    """The interface for petting zoo environments.

    Multi-agent environments must be wrapped as
    :class:`~tianshou.env.PettingZooEnv`. Here is the usage:
    ::

        env = PettingZooEnv(...)
        # obs is a dict containing obs, agent_id, and mask
        obs = env.reset()
        action = policy(obs)
        obs, rew, trunc, term, info = env.step(action)
        env.close()

    The available action's mask is set to True, otherwise it is set to False.
    Further usage can be found at :ref:`marl_example`.
    """

    def __init__(self, env: Environment):
        super().__init__()
        self.env = env
        # agent idx list
        #self.agents = self.env.possible_agents
        #self.agent_idx = {}
        #for i, agent_id in enumerate(self.agents):
        #    self.agent_idx[agent_id] = i

        #self.rewards = [0] * len(self.agents)

        # Get first observation space, assuming all agents have equal space
        #self.observation_space: Any = self.env.observation_space(self.agents[0])

        # Get first action space, assuming all agents have equal space
        #self.action_space: Any = self.env.action_space(self.agents[0])

        #assert all(self.env.observation_space(agent) == self.observation_space
        #           for agent in self.agents), \
        "Observation spaces for all agents must be identical. Perhaps " \
        "SuperSuit's pad_observations wrapper can help (useage: " \
        "`supersuit.aec_wrappers.pad_observations(env)`"

        #assert all(self.env.action_space(agent) == self.action_space
        #           for agent in self.agents), \
        "Action spaces for all agents must be identical. Perhaps " \
        "SuperSuit's pad_action_space wrapper can help (useage: " \
        "`supersuit.aec_wrappers.pad_action_space(env)`"

        self.reset()


    def reset(self, *args: Any, **kwargs: Any) -> Tuple[dict, dict]:
        time_step = self.env.reset()

        obs = time_step.observations  # open_spielのEnvironmentの出力
        observation_dict = {
            "agent_id": obs["current_player"],
            "obs": obs["serialized_state"],
            "mask": obs["legal_actions"][obs["current_player"]]
        }  # tinshouのPettingZooEnvの形式に直す.

        return observation_dict, {"info": obs["info_state"]}


    def step(self, action: Any, reset_if_done=True) -> Tuple[Dict, List[int], bool, bool, Dict]:

        time_step = self.env.step([action])
        term = time_step.last()
        if term:  # AutoReset
            time_step = self.env.reset()
        
        reward = time_step.rewards if not term else [0., 0.]  # open_spielのEnvironmentの出力
        spiel_observation = time_step.observations

        if isinstance(spiel_observation, dict) and 'legal_actions' in spiel_observation:
            obs = {
                'agent_id': spiel_observation['current_player'],
                'obs': spiel_observation['serialized_state'],
                'mask': spiel_observation["legal_actions"][spiel_observation["current_player"]]
            }
        else:
            if isinstance(self.action_space, spaces.Discrete):
                obs = {
                    'agent_id': spiel_observation['current_player'],
                    'obs': spiel_observation['serialized_state'],
                    'mask': [True] * self.env.action_space(spiel_observation['current_player']).n
                }
            else:
                obs = {'agent_id': spiel_observation['current_player'], 'obs': spiel_observation}
        return obs, reward, term, False, {"info": spiel_observation["info_state"]}


    def close(self) -> None:
        self.env.close()

    def seed(self, seed: Any = None) -> None:
        try:
            self.env.seed(seed)
        except (NotImplementedError, AttributeError):
            self.env.reset(seed=seed)

    def render(self) -> Any:
        return self.env.render()
