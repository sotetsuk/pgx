import argparse
import time
import numpy as np
import collections
import supersuit as ss
from pettingzoo.utils.env import AECEnv
from pettingzoo.utils.conversions import aec_to_parallel


def transpose(ll):
    return [[ll[i][j] for i in range(len(ll))] for j in range(len(ll[0]))]

class AutoResetPettingZooEnv(ss.ConcatVecEnv):
    """
    Autoresetを実装
    """
    def __init__(self, env):
        super().__init__(env)
    

    def reset(self, seed=None, options=None):
        obs = super().reset()
        return obs

    def step(self, actions):
        data = []
        idx = 0
        for venv in self.vec_envs:
            data.append(
                venv.step(
                    self.concatenate_actions(
                        actions[idx : idx + venv.num_envs], venv.num_envs
                    )
                )
            )
            idx += venv.num_envs
        observations, rewards, terminations, truncations, infos = transpose(data)
        observations = self.concat_obs(observations)
        rewards = np.concatenate(rewards, axis=0)
        terminations = np.concatenate(terminations, axis=0)
        truncations = np.concatenate(truncations, axis=0)
        infos = sum(infos, [])
        if terminations:
            obs, _ = self.reset
        return observations, rewards, terminations, truncations, infos


def make_env(env_name, n_envs):
    from pettingzoo.classic.go import go
    from pettingzoo.classic.tictactoe import tictactoe

    #from pettingzoo.classic import chess_v5
    def get_go_env():
        env = go.env()
        env.metadata["is_parallelizable"] = True
        return aec_to_parallel(env)
    
    def get_tictactoe_env():
        return tictactoe.env()
    if env_name == "go":
        env = ss.pettingzoo_env_to_vec_env_v1(get_go_env())
        envs = ss.concat_vec_envs_v1(env, n_envs, num_cpus=0)
        envs.single_observation_space = envs.observation_space
        envs.single_action_space = envs.action_space
        envs.is_vector_env = True
        return envs
    elif env_name == "tictactoe":
        env = ss.pettingzoo_env_to_vec_env_v1(env)
        envs = ss.concat_vec_envs_v1(env, n_envs, num_cpus=0)
        return envs
    elif env_name == "chess":
        #return chess_v5.env()
        raise ValueError("Chess will be added later")
    else:
        raise ValueError("no such environment in petting zoo")


def random_play(env, n_steps_lim: int) -> int:
    # petting zooのgo環境でrandom gaentを終局まで動かす.
    step_num = 0
    rng = np.random.default_rng()
    observation = env.reset()
    player = 0
    while step_num < n_steps_lim:
        legal_action_mask = np.array([mask for mask in observation["action_mask"]])
        print(legal_action_mask.sum(axis=1))
        action = [rng.choice(np.where(legal_action_mask[2*i]==1)[0]) if player == 0 else rng.choice(np.where(legal_action_mask[2*i+1]==1)[0]) for i in range(len(legal_action_mask)//2)]  # chose action randomly
        observation, reward, done, info = env.step(action)
        player = (player + 1) % 2
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