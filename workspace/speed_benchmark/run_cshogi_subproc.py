import json
from tianshou.env import SubprocVectorEnv
import numpy as np
import time
import argparse
import warnings
from abc import ABC
from typing import Any, Dict, List, Tuple
from pettingzoo.utils.env import AECEnv
import cshogi
from cshogi.gym_shogi.envs import ShogiEnv
from cshogi._cshogi import _dlshogi_FEATURES1_NUM as FEATURES1_NUM
from cshogi._cshogi import _dlshogi_FEATURES2_NUM as FEATURES2_NUM


def make_input_features(board):
    features1 = np.zeros((1, FEATURES1_NUM, 9, 9), dtype=np.float32)
    features2 = np.zeros((1, FEATURES2_NUM, 9, 9), dtype=np.float32)
    board._dlshogi_make_input_features(features1, features2)
    return np.vstack([features1.reshape(FEATURES1_NUM, 81), features2.reshape(FEATURES2_NUM, 81)])


def make_legal_actions(board):
    moves = [m for m in board.legal_moves]  # note: not int
    return moves


# Shogi is modified from TianShou repository's PettingZooEnv
# This wrapper enables to use TianShou's SubprocVectorEnv for cshogi's environment
#
# https://github.com/thu-ml/tianshou/blob/master/tianshou/env/pettingzoo_env.py
#
# Distributed under MIT LICENSE:
#
# https://github.com/thu-ml/tianshou/blob/master/LICENSE
class Shogi(AECEnv, ABC):
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

    def __init__(self):
        super().__init__()

    def reset(self, *args: Any, **kwargs: Any) -> Tuple[dict, dict]:
        self.current_player = np.random.randint(2)
        self.board = cshogi.Board()

        obs = make_input_features(self.board)
        legal_actions = make_legal_actions(self.board)
        observation_dict = {
            "agent_id": self.current_player,
            "obs": obs,
            "mask": legal_actions
        }

        return observation_dict, {}

    def step(self, action: Any, reset_if_done=True) -> Tuple[Dict, List[int], bool, bool, Dict]:
        self.current_player = (self.current_player + 1) % 2
        self.board.push(action)
        done = self.board.is_game_over()

        if done and reset_if_done:
            self.current_player = np.random.randint(2)
            self.board = cshogi.Board()
            done = False

        obs = make_input_features(self.board)
        legal_actions = make_legal_actions(self.board)
        obs = {
            'agent_id': self.current_player,
            'obs': obs,
            'mask': legal_actions
        }

        return obs, 0, done, False, {}


def make_env(n_envs: int):
    return SubprocVectorEnv([lambda: Shogi() for _ in range(n_envs)])


def random_play(env: SubprocVectorEnv, n_steps_lim: int, batch_size: int):
    assert n_steps_lim % batch_size == 0
    step_num = 0
    observation, info = env.reset()
    for _ in range(n_steps_lim // batch_size):
        legal_actions = [observation[i]["mask"] for i in range(batch_size)]
        observation = np.stack([observation[i]["obs"] for i in range(batch_size)])
        # print(observation.shape)
        action = [np.random.choice(legal_actions[i]) for i in range(batch_size)]
        observation, reward, terminated, _, info = env.step(action)
        step_num += batch_size
    return step_num


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("batch_size", type=int)
    parser.add_argument("n_steps_lim", default=2 ** 10 * 10, type=int)
    parser.add_argument("--seed", default=100, type=bool)
    args = parser.parse_args()
    assert args.n_steps_lim % args.batch_size == 0
    env = make_env(args.batch_size)
    time_sta = time.time()
    steps_num = random_play(env, args.n_steps_lim, args.batch_size)
    time_end = time.time()
    sec = time_end - time_sta
    print(json.dumps({"game": "shogi", "library": "cshogi/subproc", "total_steps": steps_num, "total_sec": sec, "steps/sec": steps_num/sec, "batch_size": args.batch_size}))
