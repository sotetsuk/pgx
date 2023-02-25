import json
from tianshou.env import SubprocVectorEnv
import numpy as np
import time
import argparse
from abc import ABC
from typing import Any, Dict, List, Tuple

import pyspiel
from pettingzoo.utils.env import AECEnv
from open_spiel.python.rl_environment import Environment, ChanceEventSampler


# OpenSpielEnv is modified from TianShou PettingZooEnv (see #384 for changes):
# This wrapper enables to use TianShou's SubprocVectorEnv for OpenSpiel
#
# https://github.com/thu-ml/tianshou/blob/master/tianshou/env/pettingzoo_env.py
#
# Distributed under MIT LICENSE:
#
# https://github.com/thu-ml/tianshou/blob/master/LICENSE
class OpenSpielEnv(AECEnv, ABC):
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
        self.reset()

    def reset(self, *args: Any, **kwargs: Any) -> Tuple[dict, dict]:
        time_step = self.env.reset()

        obs = time_step.observations
        observation_dict = {
            "agent_id": obs["current_player"],
            "obs": np.array(obs["info_state"][obs["current_player"]]),
            "mask": np.array(obs["legal_actions"][obs["current_player"]])
        }  # align to tianshou petting zoo format

        return observation_dict, {}

    def step(self, action: Any, reset_if_done=True) -> Tuple[Dict, List[int], bool, bool, Dict]:

        time_step = self.env.step([action])
        term = time_step.last()
        if term:  # AutoReset
            time_step = self.env.reset()

        reward = time_step.rewards if not term else [0., 0.]
        obs = time_step.observations
        observation_dict = {
            "agent_id": obs["current_player"],
            "obs": np.array(obs["info_state"][obs["current_player"]]),
            "mask": np.array(obs["legal_actions"][obs["current_player"]])
        }  # align to tianshou petting zoo format

        return observation_dict, reward, term, False, {}

    def close(self) -> None:
        self.env.close()

    def seed(self, seed: Any = None) -> None:
        try:
            self.env.seed(seed)
        except (NotImplementedError, AttributeError):
            self.env.reset(seed=seed)

    def render(self) -> Any:
        return self.env.render()


def make_single_env(env_name: str):
    game = pyspiel.load_game(env_name)
    return Environment(game,  chance_event_sampler=ChanceEventSampler())


def make_env(env_name: str, n_envs: int):
    return SubprocVectorEnv([lambda: OpenSpielEnv(make_single_env(env_name)) for _ in range(n_envs)])


def random_play(env: SubprocVectorEnv, n_steps_lim: int, batch_size: int):
    assert n_steps_lim % batch_size == 0
    step_num = 0
    observation, info = env.reset()
    while step_num < n_steps_lim:
        legal_actions = [observation[i]["mask"] for i in range(batch_size)]
        # observation = np.stack([observation[i]["obs"] for i in range(batch_size)])
        # print(observation.shape)
        action = [np.random.choice(legal_actions[i]) for i in range(batch_size)]
        observation, _, _, _, _ = env.step(action)
        step_num += batch_size
    return step_num


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name")
    parser.add_argument("batch_size", type=int)
    parser.add_argument("n_steps_lim", type=int)
    args = parser.parse_args()
    env = make_env(args.env_name, args.batch_size)
    time_sta = time.time()
    steps_num = random_play(env, args.n_steps_lim, args.batch_size)
    time_end = time.time()
    sec = time_end - time_sta
    print(json.dumps({"game": args.env_name, "library": "open_spiel", "venv": "subproc", "total_steps": steps_num, "total_sec": sec, "steps/sec": steps_num/sec, "batch_size": args.batch_size}))
