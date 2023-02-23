from vector_env import SyncVectorEnv
import argparse
import time
import numpy as np
import collections


def make_single_env(env_name: str, seed: int):
    import pyspiel
    from open_spiel.python.rl_environment import Environment, ChanceEventSampler
    def gen_env():
        game = pyspiel.load_game(env_name)
        return Environment(game,  chance_event_sampler=ChanceEventSampler(seed=seed))
    return gen_env()


def make_env(env_name: str, n_envs: int, seed: int) -> SyncVectorEnv:
    return SyncVectorEnv([make_single_env(env_name, seed) for i in range(n_envs)])


def random_play(env: SyncVectorEnv, n_steps_lim: int, batch_size: int):
    # random play for  open spiel
    StepOutput = collections.namedtuple("step_output", ["action"])
    time_step = env.reset()
    rng = np.random.default_rng()
    step_num = 0
    while step_num < n_steps_lim:
        legal_actions = np.array([ts.observations["legal_actions"][ts.observations["current_player"]] for ts in time_step])
        assert len(env.envs) == len(legal_actions)  # ensure parallerization
        action = [rng.choice(legal_actions[i]) for i in range(len(legal_actions))]
        step_outputs = [StepOutput(action=a) for a in action]
        time_step, reward, done, unreset_time_steps = env.step(step_outputs, reset_if_done=True)
        step_num += batch_size
    return step_num


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name")
    parser.add_argument("batch_size", type=int)
    parser.add_argument("n_steps_lim", type=int)
    parser.add_argument("--seed", default=100, type=bool)
    args = parser.parse_args()
    assert args.n_steps_lim % args.batch_size
    env = make_env(args.env_name, args.batch_size, args.seed)
    time_sta = time.time()
    step_num = random_play(env, args.n_steps_lim, args.batch_size)
    time_end = time.time()
    print((step_num)/(time_end-time_sta), time_end-time_sta)
