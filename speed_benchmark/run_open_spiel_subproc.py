import json
from tianshou.env import SubprocVectorEnv
import numpy as np
import time
import argparse
from abc import ABC
from typing import Any, Dict, List, Tuple

import pyspiel
from pettingzoo.utils.env import AECEnv


class OpenSpiel:

    def __init__(self, game_name):
        self.game = pyspiel.load_game(game_name)
        self.state = self.game.new_initial_state()

    def skip_chance(self):
        if self.state.is_chance_node():
            # print("came to chance node")
            outcomes_with_probs = self.state.chance_outcomes()
            action_list, prob_list = zip(*outcomes_with_probs)
            action = np.random.choice(action_list, p=prob_list)
            self.state.apply_action(action)

    def reset(self):
        self.state = self.game.new_initial_state()
        self.skip_chance()
        obs = self.state.observation_tensor(self.state.current_player())
        legal_actions = self.state.legal_actions()
        return obs, legal_actions

    def step(self, action):
        self.state.apply_action(action)
        self.skip_chance()
        done = self.state.is_terminal()

        if done:  # auto reset
            self.state = self.game.new_initial_state()

        obs = self.state.observation_tensor(self.state.current_player())
        legal_actions = self.state.legal_actions()
        return obs, legal_actions


# OpenSpielEnv is modified from TianShou repository (see #384 for changes):
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

    def __init__(self, env):
        super().__init__()
        self.env = env
        self.reset()

    def reset(self, *args: Any, **kwargs: Any) -> Tuple[dict, dict]:
        obs, legal_actions = self.env.reset()

        observation_dict = {
            "agent_id": 0,
            "obs": np.array(obs),  # wrap by np.array is several times faster
            "mask": np.array(legal_actions)
        }  # align to tianshou petting zoo format

        # return empty dict for significant speed up
        return observation_dict, {}

    def step(self, action: Any, reset_if_done=True) -> Tuple[Dict, List[int], bool, bool, Dict]:
        obs, legal_actions = self.env.step(action)
        obs = {
            'agent_id': 0,
            'obs': np.array(obs),
            'mask': np.array(legal_actions)
        }

        return obs, 0, False, False, {}

    def close(self) -> None:
        self.env.close()

    def seed(self, seed: Any = None) -> None:
        try:
            self.env.seed(seed)
        except (NotImplementedError, AttributeError):
            self.env.reset(seed=seed)

    def render(self) -> Any:
        return self.env.render()


def make_env(env_name: str, n_envs: int,  seed: int):
    return SubprocVectorEnv([lambda: OpenSpielEnv(OpenSpiel(env_name)) for _ in range(n_envs)])


def random_play(env: SubprocVectorEnv, n_steps_lim: int, batch_size: int):
    assert n_steps_lim % batch_size == 0
    step_num = 0
    observation, legal_actions = env.reset()
    for _ in range(n_steps_lim // batch_size):
        legal_actions = [observation[i]["mask"] for i in range(batch_size)]
        observation = np.array([observation[i]["obs"] for i in range(batch_size)])
        action = [np.random.choice(legal_actions[i]) for i in range(batch_size)]
        observation, _, _, _, _ = env.step(action)
        step_num += batch_size
    return step_num


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name")
    parser.add_argument("batch_size", type=int)
    parser.add_argument("n_steps_lim", default=2 ** 10 * 10, type=int)
    parser.add_argument("--seed", default=100, type=bool)
    args = parser.parse_args()
    assert args.n_steps_lim % args.batch_size == 0
    env = make_env(args.env_name, args.batch_size, args.seed)
    time_sta = time.time()
    steps_num = random_play(env, args.n_steps_lim, args.batch_size)
    time_end = time.time()
    sec = time_end - time_sta
    print(json.dumps({"game": args.env_name, "library": "open_spiel", "venv": "subproc", "total_steps": steps_num, "total_sec": sec, "steps/sec": steps_num/sec, "batch_size": args.batch_size}))
