"""
Petting zooとopen spileの囲碁環境とpgxの囲碁環境の速度比較を行う
"""
import random
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

def petting_zoo_make_env(env_name):
    from pettingzoo.classic.go import go
    from pettingzoo.classic import chess_v5
    if env_name == "go":
        return go.env()
    elif env_name == "chess":
        #return chess_v5.env()
        raise ValueError("Chess will be added later")
    else:
        raise ValueError("no such environment in petting zoo")

def petting_zoo_random_play(tup):
    # petting zooのgo環境でrandom gaentを終局まで動かす.
    id, do_print, env_name = tup
    env = petting_zoo_make_env(env_name)
    step_nums = 0
    env.reset()
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()
        action = None if termination or truncation else np.random.choice(np.where(observation["action_mask"]==1)[0])  # this is where you would insert your policy
        env.step(action)
        step_nums += 1
    if do_print:
        print(id, step_nums)
    return id, step_nums


def open_spile_make_env(env_name):
    import pyspiel
    if env_name == "go":
        return pyspiel.load_game("go")
    elif env_name == "backgammon":
        return pyspiel.load_game("backgammon")
    elif env_name == "bridge":
        return pyspiel.load_game("bridge")
    else:
        raise ValueError("no such environment in open spile")



def open_spile_random_play(tup):
    import pyspiel
    # open spileのgo環境でrandom gaentを終局まで動かす.
    id, do_print, env_name = tup
    game = open_spile_make_env(env_name)
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


def measure_time(args):
    if args.library == "open_spiel":
        random_play_fn = open_spile_random_play
    elif args.library == "petting_zoo":
        random_play_fn = petting_zoo_random_play
    else:
        raise ValueError("Incorrect library name")
    
    p = Pool(args.n_processes)
    process_list = []
    time_sta = time.time()
    ex = p.map_async(random_play_fn, iterable=[(i, args.print_per_game, args.env_name) for i in range(args.n_games)])
    result = ex.get()
    time_end = time.time()
    tim = time_end- time_sta
    p.close()
    avarage_steps = sum(list(map(lambda x: x[1], ex.get())))//args.n_games
    print("library: {} env: {} n_games: {} n_processes: {} execution time is {}, avarage number of steps is {}".format(args.library, args.env_name, args.n_games, args.n_processes, tim, avarage_steps))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("library")
    parser.add_argument("env_name")
    parser.add_argument("n_processes", type=int)
    parser.add_argument("n_games", type=int)
    parser.add_argument("--print_per_game", default=False, type=bool)
    args = parser.parse_args()
    measure_time(args)

    