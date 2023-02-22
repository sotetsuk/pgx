from pettingzoo_env import OpenSpielEnv
from venvs import SubprocVectorEnv
import numpy as np
import time
import pyspiel


if __name__ == "__main__":
    env_name = "go"
    n_envs = 10
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
        #old_board_black = [info[i]["info"][0][:361] if observation[i]["agent_id"] == 0 else info[i]["info"][0][361: 722] for i in range(n_envs)]  # 手番のplayerから見たboard
        action = [rng.choice(legal_action_mask[i]) for i in range(len(legal_action_mask))]  # chose action randomly
        observation, reward, terminated, _, info = env.step(action)
        #new_legal_action_mask = [observation[i]["mask"] for i in range(len(observation))]
        #assert sum([((not action[i] in new_legal_action_mask[i]) & (not terminated[i])) | (action[i]==361)| terminated[i] for i in range(n_envs)]) == n_envs  # 実行済みのactionが消えていることを確認. 361はパス.
        #new_board_black = [info[i]["info"][0][361: 722] if observation[i]["agent_id"] == 0 else info[i]["info"][0][:361] for i in range(n_envs)]  # 直前の手番のplayerから見たboard
        #assert sum([(sum(new_board_black[i]) > sum(old_board_black[i])) | (action[i]==361) | terminated[i] for i in range(n_envs)])  # 石を置いた場合は増えているかどうか  361はパス.
        #step_num += 1
    time_end = time.time()
    tim = time_end - time_sta
    print(tim)


