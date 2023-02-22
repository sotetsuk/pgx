from pettingzoo_env import OpenSpielEnv
from venvs import SubprocVectorEnv
import numpy as np
import time
import pyspiel


if __name__ == "__main__":
    env_name = "go"
    n_envs = 2
    seed = 100
    n_steps_lim = 1000

    def open_spile_make_single_env(env_name: str, seed: int):
        import pyspiel
        from open_spiel.python.rl_environment import Environment, ChanceEventSampler
        def gen_env():
            game = pyspiel.load_game(env_name)
            return Environment(game,  chance_event_sampler=ChanceEventSampler(seed=seed))
        return gen_env()
    def get_go_env():
        return OpenSpielEnv(open_spile_make_single_env(env_name, seed))
    
    env = SubprocVectorEnv([lambda: OpenSpielEnv(open_spile_make_single_env(env_name, seed)) for _ in range(n_envs)])

    # petting zooのgo環境でrandom gaentを終局まで動かす.
    step_num = 0
    rng = np.random.default_rng()
    observation, info = env.reset()
    terminated = np.zeros(len(env._env_fns))
    time_sta = time.time()
    while step_num < n_steps_lim:
        legal_action_mask = [observation[i]["mask"] for i in range(len(observation))]
        #print(len(info[0]["info"]), len(info[0]["info"][0]), len(info[0]["info"][1]))
        action = [rng.choice(legal_action_mask[i]) for i in range(len(legal_action_mask))]  # chose action randomly
        observation, reward, terminated, _, info = env.step(action)
        new_legal_action_mask = [observation[i]["mask"] for i in range(len(observation))]
        assert sum([((not action[i] in new_legal_action_mask[i]) & (not terminated[i])) | (action[i]==361)| terminated[i] for i in range(n_envs)]) == n_envs  # 実行済みのactionが消えていることを確認. 361はパス.
        step_num += 1
    time_end = time.time()
    tim = time_end - time_sta
    print(tim)


