from tianshou_env.pettingzoo_env import OpenSpielEnv
from tianshou_env.venvs import SubprocVectorEnv
import numpy as np
import time
import argparse


def make_single_env(env_name: str, seed: int):
    import pyspiel
    from open_spiel.python.rl_environment import Environment, ChanceEventSampler
    def gen_env():
        game = pyspiel.load_game(env_name)
        return Environment(game,  chance_event_sampler=ChanceEventSampler(seed=seed))
    return gen_env()

def make_env(env_name: str, n_envs: int,  seed: int):
    return SubprocVectorEnv([lambda: OpenSpielEnv(make_single_env(env_name, seed)) for _ in range(n_envs)])


def random_play(env: SubprocVectorEnv, n_steps_lim: int):
    step_num = 0
    rng = np.random.default_rng()
    observation, info = env.reset()
    terminated = np.zeros(len(env._env_fns))
    while step_num < n_steps_lim:
        legal_action_mask = [observation[i]["mask"] for i in range(len(observation))]
        action = [rng.choice(legal_action_mask[i]) for i in range(len(legal_action_mask))]  # chose action randomly
        observation, reward, terminated, _, info = env.step(action)
        step_num += 1
    return step_num


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name")
    parser.add_argument("batch_size", type=int)
    parser.add_argument("n_steps_lim", type=int)
    parser.add_argument("--seed", default=100, type=bool)
    args = parser.parse_args()
    env = make_env(args.env_name, args.batch_size, args.seed)
    time_sta = time.time()
    step_num = random_play(env, args.n_steps_lim)
    time_end = time.time()
    print((args.batch_size*step_num)/(time_end-time_sta))
