import argparse
import time
import json
import numpy as np
from tianshou.env.pettingzoo_env import PettingZooEnv


class AutoResetPettingZooEnv(PettingZooEnv):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, term, trunc, info = super().step(action)
        if term:
            obs = super().reset()
        return obs, reward, term, trunc, info


def make_env(env_name, n_envs, vec_env):
    if env_name == "go":
        def env_fn():
            from pettingzoo.classic.go import go
            return AutoResetPettingZooEnv(go.env())
    elif env_name == "tic_tac_toe":
        def env_fn():
            from pettingzoo.classic.tictactoe import tictactoe
            return AutoResetPettingZooEnv(tictactoe.env())
    elif env_name == "chess":
        def env_fn():
            from pettingzoo.classic.chess import chess
            return AutoResetPettingZooEnv(chess.env())

    if vec_env == "for-loop":
        from tianshou.env import DummyVectorEnv as VecEnv
    elif vec_env == "subproc":
        from tianshou.env import SubprocVectorEnv as VecEnv

    return VecEnv([env_fn for _ in range(n_envs)])


def random_play(env, n_steps_lim: int, batch_size: int) -> int:
    step_num = 0
    observation = env.reset()
    assert len(env._env_fns) == len(observation)  # ensure parallerization
    while step_num < n_steps_lim:
        legal_action_mask = [observation[i]["mask"] for i in range(batch_size)]
        # observation = np.stack([observation[i]["obs"] for i in range(batch_size)])
        # print(observation.shape)
        action = [np.random.choice(np.where(legal_action_mask[i])[0]) for i in range(batch_size)]
        observation, reward, terminated, _, _ = env.step(action)
        step_num += batch_size
    return step_num


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name")  # go, chess, tic_tac_toe
    parser.add_argument("venv")  # for-loop, subproc
    parser.add_argument("batch_size", type=int)
    parser.add_argument("num_batch_steps", type=int)
    args = parser.parse_args()
    env = make_env(args.env_name, args.batch_size, args.venv)
    time_sta = time.time()
    steps_num = random_play(env, args.num_batch_steps * args.batch_size, args.batch_size)
    time_end = time.time()
    sec = time_end - time_sta
    print(json.dumps({"game": args.env_name, "library": f"petting_zoo/{args.venv}", "total_steps": steps_num, "total_sec": sec, "steps/sec": steps_num/sec, "batch_size": args.batch_size}))
