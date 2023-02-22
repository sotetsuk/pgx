from tianshou.env import SubprocVectorEnv
from pettingzoo.classic.go import go
from tianshou.env.pettingzoo_env import PettingZooEnv
import numpy as np

class AutoResetPettingZooEnv(PettingZooEnv):  # 全体でpetting_zooの関数, classをimportするとopen_spielの速度が落ちる.
    def __init__(self, env):
        super().__init__(env)
    
    def step(self, action):
        obs, reward, term, trunc, info = super().step(action)
        if term:
            obs = super().reset()
        return obs, reward, term, trunc, info


def petting_zoo_make_env(env_name, n_envs):
            
    #from pettingzoo.classic import chess_v5
    def get_go_env():
        return AutoResetPettingZooEnv(go.env())
    if env_name == "go":
        return SubprocVectorEnv([get_go_env for _ in range(n_envs)])
    elif env_name == "chess":
        #return chess_v5.env()
        raise ValueError("Chess will be added later")
    else:
        raise ValueError("no such environment in petting zoo")


def petting_zoo_random_play(env, n_steps_lim: int) -> int:
    # petting zooのgo環境でrandom gaentを終局まで動かす.
    step_num = 0
    rng = np.random.default_rng()
    observation = env.reset()
    terminated = np.zeros(len(env._env_fns))
    while step_num < n_steps_lim:
        assert len(env._env_fns) == len(observation)  # ensure parallerization
        legal_action_mask = np.array([observation[i]["mask"] for i in range(len(observation))])
        action = [rng.choice(np.where(legal_action_mask[i]==1)[0]) for i in range(len(legal_action_mask))]  # chose action randomly
        observation, reward, terminated, _, _ = env.step(action)
        step_num += 1
    return step_num