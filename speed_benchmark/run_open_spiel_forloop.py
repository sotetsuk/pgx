import argparse
import json
import collections
import time
import numpy as np
import pyspiel


class OpenSpielEnv:

    def __init__(self, game_name):
        self.game = pyspiel.load_game(game_name)
        self.state = self.game.new_initial_state()

    def skip_chance(self):
        if self.state.is_chance_node():
            # print("came to chance node")
            outcomes_with_probs = self.state.chance_outcomes() 
            action_list, prob_list = zip(*outcomes_with_probs)
            action = np.random.choice(action_list, p=prob_list)
            self.state.apply_action(action)

    def reset(self):
        self.state = self.game.new_initial_state()
        self.skip_chance()
        obs = self.state.observation_tensor(self.state.current_player())
        legal_actions = self.state.legal_actions()
        return obs, legal_actions

    def step(self, action):
        self.state.apply_action(action)
        self.skip_chance()
        done = self.state.is_terminal()

        if done:  # auto reset
            self.state = self.game.new_initial_state()

        obs = self.state.observation_tensor(self.state.current_player())
        legal_actions = self.state.legal_actions()
        return obs, legal_actions


class DummyVecOpenSpielEnv:

    def __init__(self, game, batch_size):
        self.batch_size = batch_size
        self.envs = [OpenSpielEnv(game) for _ in range(self.batch_size)]

    def reset(self):
        observations, legal_actions = zip(*[env.reset() for env in self.envs])
        return np.array(observations), legal_actions

    def step(self, action):
        observations, legal_actions = zip(*[env.step(a) for env, a in zip(self.envs, action)])
        return np.array(observations), legal_actions


def random_play(env, n_steps_lim, batch_size):
    assert n_steps_lim % batch_size == 0
    obs, legal_actions = env.reset()
    step_num = 0
    for _ in range(n_steps_lim // batch_size):
        # print(obs.shape)
        action = [np.random.choice(a) for a in legal_actions]
        obs, legal_actions = env.step(action)
        step_num += batch_size
    return step_num


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name")  # go, chess backgammon tic_tac_toe
    parser.add_argument("batch_size", type=int)
    parser.add_argument("n_steps_lim", default=2 ** 10 * 10, type=int)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()
    assert args.n_steps_lim % args.batch_size == 0
    env = DummyVecOpenSpielEnv(args.env_name, args.batch_size)
    time_sta = time.time()
    steps_num = random_play(env, args.n_steps_lim, args.batch_size)
    time_end = time.time()
    sec = time_end-time_sta
    print(json.dumps({"game": args.env_name, "library": "open_spiel", "venv": "for-loop", "total_steps": steps_num, "total_sec": sec, "steps/sec": steps_num/sec, "batch_size": args.batch_size}))
