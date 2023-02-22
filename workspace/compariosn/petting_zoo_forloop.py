from vector_env import SyncVectorEnv
import argparse
import time
import numpy as np
import collections
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from pettingzoo.classic.tictactoe import tictactoe
from pettingzoo.classic.go import go

class AutoResetPettingZooEnv(PettingZooEnv):
    def __init__(self, env):
        super().__init__(env)
    
    def step(self, action):
        obs, reward, term, trunc, info = super().step(action)
        if term:
            obs = super().reset()
        return obs, reward, term, trunc, info


def make_env(env_name, n_envs):
    #from pettingzoo.classic import chess_v5
    def get_go_env():
        return AutoResetPettingZooEnv(go.env())
    def get_tictactoe_env():
        return AutoResetPettingZooEnv(tictactoe.env())
    if env_name == "go":
        return DummyVectorEnv([get_go_env for _ in range(n_envs)])
    elif env_name == "tictactoe":
        return DummyVectorEnv([get_tictactoe_env() for _ in range(n_envs)])
    elif env_name == "chess":
        #return chess_v5.env()
        raise ValueError("Chess will be added later")
    else:
        raise ValueError("no such environment in petting zoo")


def random_play(env: DummyVectorEnv, n_steps_lim: int) -> int:
    # petting zooのgo環境でrandom gaentを終局まで動かす.
    step_num = 0
    rng = np.random.default_rng()
    observation = env.reset()
    terminated = np.zeros(len(env._env_fns))
    while step_num < n_steps_lim:
        assert len(env._env_fns) == len(observation)  # ensure parallerization
        legal_action_mask = np.array([observation[i]["mask"] for i in range(len(observation))])
        action = [rng.choice(np.where(legal_action_mask[i]==1)[0]) for i in range(len(legal_action_mask))]  # chose action randomly
        observation, reward, terminated, _, _ = env.step(action)
        step_num += 1
    return step_num


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name")
    parser.add_argument("batch_size", type=int)
    parser.add_argument("n_steps_lim", type=int)
    args = parser.parse_args()
    env = make_env(args.env_name, args.batch_size)
    time_sta = time.time()
    step_num = random_play(env, args.n_steps_lim)
    time_end = time.time()
    print((args.batch_size*step_num)/(time_end-time_sta))