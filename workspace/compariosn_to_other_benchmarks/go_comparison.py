"""
Petting zooとopen spileの囲碁環境とpgxの囲碁環境の速度比較を行う
"""
from pettingzoo.classic.go import go
import random
import pyspiel
import numpy as np
import time



def main():
    pass



if __name__ == "__main__":
    # petting_zoo
    petting_zoo_go_env = go.env()
    petting_zoo_go_env.reset()
    time_sta = time.time()
    for agent in petting_zoo_go_env.agent_iter():
        observation, reward, termination, truncation, info = petting_zoo_go_env.last()
        action = None if termination or truncation else petting_zoo_go_env.action_space(agent).sample()  # this is where you would insert your policy
        petting_zoo_go_env.step(action)
    time_end = time.time()
    tim = time_end- time_sta
    print("execution time in petting zoo is {}".format(tim))

    # open spile
    time_sta = time.time()
    game = pyspiel.load_game("go")
    state = game.new_initial_state()
    while not state.is_terminal():
        legal_actions = state.legal_actions()
        if state.is_chance_node():
            # Sample a chance event outcome.
            outcomes_with_probs = state.chance_outcomes()
            action_list, prob_list = zip(*outcomes_with_probs)
            action = np.random.choice(action_list, p=prob_list)
            state.apply_action(action)
        else:
            # The algorithm can pick an action based on an observation (fully observable
            # games) or an information state (information available for that player)
            # We arbitrarily select the first available action as an example.
            action = legal_actions[0]
            state.apply_action(action)
    tim = time_end- time_sta
    print("execution time in open spile is {}".format(tim))