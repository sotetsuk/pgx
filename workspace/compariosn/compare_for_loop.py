import supersuit as ss
from vector_env import SyncVectorEnv
import argparse
import time
import numpy as np
import collections

def petting_zoo_make_env(env_name):
    from pettingzoo.classic.go import go
    from pettingzoo.classic import chess_v5
    if env_name == "go":
        return go.env()
    elif env_name == "chess":
        #return chess_v5.env()
        raise ValueError("Chess will be added later")
    else:
        raise ValueError("no such environment in petting zoo")

def petting_zoo_random_play(tup):
    # petting zooのgo環境でrandom gaentを終局まで動かす.
    id, do_print, env_name = tup
    env = petting_zoo_make_env(env_name)
    step_nums = 0
    env.reset()
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        action = None if termination or truncation else np.random.choice(np.where(observation["action_mask"]==1)[0])  # this is where you would insert your policy
        env.step(action)
        step_nums += 1
    if do_print:
        print(id, step_nums)
    return id, step_nums


def open_spile_make_single_env(env_name: str, seed: int):
    import pyspiel
    from open_spiel.python.rl_environment import Environment, ChanceEventSampler
    def gen_env():
        game = pyspiel.load_game(env_name)
        return Environment(game, chance_event_sampler=ChanceEventSampler(seed=seed))
    return gen_env()


def open_spiel_make_env(env_name: str, n_envs: int, seed: int) -> SyncVectorEnv:
    return SyncVectorEnv([open_spile_make_single_env(env_name, seed) for i in range(n_envs)])


def open_spile_random_play(env: SyncVectorEnv, n_steps_lim: int):
    # random play for  open spiel
    StepOutput = collections.namedtuple("step_output", ["action"])
    time_step = env.reset()
    rng = np.random.default_rng()
    step_num = 0
    player_id = 0
    while step_num < n_steps_lim:
        legal_actions = np.array([ts.observations["legal_actions"][player_id] for ts in time_step])
        assert len(env.envs) == len(legal_actions)  # ensure parallerization
        action = rng.choice(legal_actions, axis=1)  # same actions so far
        step_outputs = [StepOutput(action=a) for a in action]
        time_step, reward, done, unreset_time_steps = env.step(step_outputs, reset_if_done=True)
        player_id = (player_id + 1) % 2
        step_num += 1
    return step_num


def measure_time(args):
    if args.library == "open_spiel":
        env = open_spiel_make_env(args.env_name, args.n_envs, args.seed)
        random_play_fn = open_spile_random_play
    elif args.library == "petting_zoo":
        random_play_fn = petting_zoo_random_play
    else:
        raise ValueError("Incorrect library name")
    time_sta = time.time()
    random_play_fn(env, args.n_steps_lim)
    time_end = time.time()
    tim = time_end- time_sta
    tim_per_step = tim / args.n_steps_lim * args.n_envs
    print("library: {} env: {} n_envs: {} n_steps_lim: {} execution time is {} time_per_step is {}".format(args.library, args.env_name, args.n_envs, args.n_steps_lim, tim, tim_per_step))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("library")
    parser.add_argument("env_name")
    parser.add_argument("n_envs", type=int)
    parser.add_argument("n_steps_lim", type=int)
    parser.add_argument("--seed", default=100, type=bool)
    args = parser.parse_args()
    measure_time(args)