import argparse
import time
import json
import numpy as np
import collections
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv


class AutoResetPettingZooEnv(PettingZooEnv):
    def __init__(self, env):
        super().__init__(env)
    
    def step(self, action):
        obs, reward, term, trunc, info = super().step(action)
        if term:
            obs = super().reset()
        return obs, reward, term, trunc, info


def make_env(env_name, n_envs):
    
    def get_go_env():
        from pettingzoo.classic.go import go
        return AutoResetPettingZooEnv(go.env())

    def get_tictactoe_env():
        from pettingzoo.classic.tictactoe import tictactoe
        return AutoResetPettingZooEnv(tictactoe.env())
    
    def get_chess_env():
        from pettingzoo.classic.chess import chess
        return AutoResetPettingZooEnv(chess.env())
    
    if env_name == "go":
        return DummyVectorEnv([get_go_env for _ in range(n_envs)])
    elif env_name == "tic_tac_toe":
        return DummyVectorEnv([get_tictactoe_env for _ in range(n_envs)])
    elif env_name == "chess":
        return DummyVectorEnv([get_chess_env for _ in range(n_envs)])
    else:
        raise ValueError(f"wrong argument: {env_name}")


def random_play(env: DummyVectorEnv, n_steps_lim: int, batch_size: int) -> int:
    # petting zooのgo環境でrandom gaentを終局まで動かす.
    step_num = 0
    rng = np.random.default_rng()
    observation = env.reset()
    while step_num < n_steps_lim:
        assert len(env._env_fns) == len(observation)  # ensure parallerization
        legal_action_mask = np.array([observation[i]["mask"] for i in range(batch_size)])
        action = [rng.choice(np.where(legal_action_mask[i])[0]) for i in range(batch_size)]  # chose action randomly
        observation, reward, terminated, _, _ = env.step(action)
        step_num += batch_size
    return step_num


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name")  # go, chess, tic_tac_toe
    parser.add_argument("batch_size", type=int)
    parser.add_argument("n_steps_lim", type=int)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()
    assert args.n_steps_lim % args.batch_size == 0
    env = make_env(args.env_name, args.batch_size)
    time_sta = time.time()
    steps_num = random_play(env, args.n_steps_lim, args.batch_size)
    time_end = time.time()
    sec = time_end - time_sta
    print(json.dumps({"game": args.env_name, "library": "open_spiel", "total_steps": steps_num, "total_sec": sec, "steps/sec": steps_num/sec}))