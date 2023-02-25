import argparse
import json
import collections
import time
import numpy as np
import pyspiel
from open_spiel.python.rl_environment import Environment, ChanceEventSampler


# SyncVectorEnv is copied from
# https://github.com/deepmind/open_spiel/blob/master/open_spiel/python/vector_env.py
#
# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
class SyncVectorEnv(object):
  """A vectorized RL Environment.
  This environment is synchronized - games do not execute in parallel. Speedups
  are realized by calling models on many game states simultaneously.
  """

  def __init__(self, envs):
    if not isinstance(envs, list):
      raise ValueError(
          "Need to call this with a list of rl_environment.Environment objects")
    self.envs = envs

  def __len__(self):
    return len(self.envs)

  def observation_spec(self):
    return self.envs[0].observation_spec()

  @property
  def num_players(self):
    return self.envs[0].num_players

  def step(self, step_outputs, reset_if_done=False):
    """Apply one step.
    Args:
      step_outputs: the step outputs
      reset_if_done: if True, automatically reset the environment
          when the epsiode ends
    Returns:
      time_steps: the time steps,
      reward: the reward
      done: done flag
      unreset_time_steps: unreset time steps
    """
    time_steps = [
        self.envs[i].step([step_outputs[i].action])
        for i in range(len(self.envs))
    ]
    reward = [step.rewards for step in time_steps]
    done = [step.last() for step in time_steps]
    unreset_time_steps = time_steps  # Copy these because you may want to look
                                     # at the unreset versions to extract
                                     # information from them

    if reset_if_done:
      time_steps = self.reset(envs_to_reset=done)

    return time_steps, reward, done, unreset_time_steps

  def reset(self, envs_to_reset=None):
    if envs_to_reset is None:
      envs_to_reset = [True for _ in range(len(self.envs))]

    time_steps = [
        self.envs[i].reset()
        if envs_to_reset[i] else self.envs[i].get_time_step()
        for i in range(len(self.envs))
    ]
    return time_steps


def make_single_env(env_name: str):
    game = pyspiel.load_game(env_name)
    return Environment(game,  chance_event_sampler=ChanceEventSampler())


def make_env(env_name: str, n_envs: int) -> SyncVectorEnv:
    return SyncVectorEnv([make_single_env(env_name) for i in range(n_envs)])


def random_play(env: SyncVectorEnv, n_steps_lim: int, batch_size: int):
    assert n_steps_lim % batch_size == 0
    StepOutput = collections.namedtuple("step_output", ["action"])
    time_step = env.reset()
    assert len(env.envs) == len(time_step)  # ensure parallerization
    step_num = 0
    while step_num < n_steps_lim:
        # obs = np.stack([ts.observations["info_state"][ts.observations["current_player"]] for ts in time_step])
        # print(obs.shape)
        actions = [np.random.choice(ts.observations["legal_actions"][ts.observations["current_player"]]) for ts in time_step]
        step_outputs = [StepOutput(action=action) for action in actions]
        time_step, reward, done, unreset_time_steps = env.step(step_outputs, reset_if_done=True)
        step_num += batch_size
    return step_num


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name")  # go, chess backgammon tic_tac_toe
    parser.add_argument("batch_size", type=int)
    parser.add_argument("n_steps_lim", type=int)
    args = parser.parse_args()
    env = make_env(args.env_name, args.batch_size)
    time_sta = time.time()
    steps_num = random_play(env, args.n_steps_lim, args.batch_size)
    time_end = time.time()
    sec = time_end-time_sta
    print(json.dumps({"game": args.env_name, "library": "open_spiel", "venv": "for-loop", "total_steps": steps_num, "total_sec": sec, "steps/sec": steps_num/sec, "batch_size": args.batch_size}))
