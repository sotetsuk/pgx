import time
import json
import argparse
import numpy as np
from cshogi.gym_shogi.envs import ShogiVecEnv
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


class Shogi:
    """Vector shogi env w/ for-loop with auto-reset"""

    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.env = ShogiVecEnv(self.batch_size)

    def reset(self):
        self.env.reset()
        obs = np.stack([make_input_features(self.env.envs[i].board) for i in range(self.batch_size)])
        legal_actions = [make_legal_actions (self.env.envs[i].board) for i in range(self.batch_size)]
        return obs, {"legal_actions": legal_actions}

    def step(self, action):
        # auto reset is implemented by ShogiVecEnv
        reward, done, _ = self.env.step(action)
        obs = np.stack([make_input_features(self.env.envs[i].board) for i in range(self.batch_size)])
        legal_actions = [make_legal_actions (self.env.envs[i].board) for i in range(self.batch_size)]
        return obs, reward, done, {"legal_actions": legal_actions}


def random_play(env, n_steps_lim, batch_size):
    assert n_steps_lim % batch_size == 0
    step_num = 0
    observation, info = env.reset()
    for _ in range(n_steps_lim // batch_size):
        action = [np.random.choice(info["legal_actions"][i]) for i in range(batch_size)]
        obs, reward, terminated, info = env.step(action)
        step_num += batch_size
    return step_num


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("batch_size", type=int)
    parser.add_argument("n_steps_lim", type=int)
    args = parser.parse_args()
    env = Shogi(args.batch_size)
    time_sta = time.time()
    steps_num = random_play(env, args.n_steps_lim, args.batch_size)
    time_end = time.time()
    sec = time_end - time_sta
    print(json.dumps({"game": "shogi", "library": "cshogi/for-loop", "total_steps": steps_num, "total_sec": sec, "steps/sec": steps_num/sec, "batch_size": args.batch_size}))
