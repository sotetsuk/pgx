"""
Petting zooとopen spileの囲碁環境とpgxの囲碁環境の速度比較を行う
"""
from pettingzoo.classic.go import go
import random
import pyspiel
import numpy as np
import time
import argparse
from multiprocessing import Pool
from multiprocessing import Queue
import multiprocessing
from pgx.go import init, step, observe
import jax
from jax import vmap, jit
import jax.numpy as jnp
import numpy as np



def petting_zoo_random_go(tup):
    # petting zooのgo環境でrandom gaentを終局まで動かす.
    id, do_print = tup
    step_nums = 0
    petting_zoo_go_env = go.env()
    petting_zoo_go_env.reset()
    for agent in petting_zoo_go_env.agent_iter():
        observation, reward, termination, truncation, info = petting_zoo_go_env.last()
        action = None if termination or truncation else np.random.choice(np.where(observation["action_mask"]==1)[0])  # this is where you would insert your policy
        petting_zoo_go_env.step(action)
        step_nums += 1
    if do_print:
        print(id, step_nums)
    return id, step_nums


def open_spile_random_go(tup):
    # open spileのgo環境でrandom gaentを終局まで動かす.
    id, do_print = tup
    game = pyspiel.load_game("go")
    state = game.new_initial_state()
    step_nums = 0
    while not state.is_terminal():
        legal_actions = state.legal_actions()
        # Sample a chance event outcome.
        action = np.random.choice(legal_actions)
        state.apply_action(action)
        step_nums += 1
    if do_print:
        print(id, step_nums)
    return id, step_nums



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("n_processes", type=int)
    parser.add_argument("n_games", type=int)
    parser.add_argument("--print_per_game", default=False, type=bool)
    args = parser.parse_args()
    n_processes = args.n_processes
    n_games = args.n_games

    # petting_zoo
    # open spile
    p = Pool(n_processes)
    process_list = []
    time_sta = time.time()
    ex = p.map_async(petting_zoo_random_go, iterable=[(i, args.print_per_game) for i in range(n_games)])
    result = ex.get()
    time_end = time.time()
    tim = time_end- time_sta
    p.close()
    avarage_steps = sum(list(map(lambda x: x[1], ex.get())))//n_games
    print("n_games: {} n_processes: {} execution time in petting zoo is {}, avarage number of steps is {}".format(n_games, n_processes, tim, avarage_steps))


    # open spile
    p = Pool(n_processes)
    process_list = []
    time_sta = time.time()
    ex = p.map_async(open_spile_random_go, iterable=[(i, args.print_per_game) for i in range(n_games)])
    result = ex.get()
    time_end = time.time()
    tim = time_end- time_sta
    p.close()
    avarage_steps = sum(list(map(lambda x: x[1], ex.get())))//n_games
    print("n_games: {} n_processes: {} execution time in open_spile is {}, avarage number of steps is {}".format(n_games, n_processes, tim, avarage_steps))