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
from pgx.go import init, step, observe
import jax
from jax import vmap, jit
import jax.numpy as jnp
import numpy as np



def petting_zoo_random_go():
    # petting zooのgo環境でrandom gaentを終局まで動かす.
    petting_zoo_go_env = go.env()
    petting_zoo_go_env.reset()
    for agent in petting_zoo_go_env.agent_iter():
        observation, reward, termination, truncation, info = petting_zoo_go_env.last()
        action = None if termination or truncation else petting_zoo_go_env.action_space(agent).sample()  # this is where you would insert your policy
        petting_zoo_go_env.step(action)


def open_spile_random_go():
    # open spileのgo環境でrandom gaentを終局まで動かす.
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
@jit
def _init(key):
    return init(key, size=19)


@jit
def _step(state, action):
    return jit(step, static_argnums=(2,))(state, action, 19)


init_vmap = vmap(_init)
step_vmap = vmap(_step)


def pgx_random_go(n_envs, key):
    # pgxのgo環境でrandom agentを終局まで動かす.
    for_init, rng = jax.random.split(key)
    keys = jax.random.split(for_init, n_envs)
    curr_player, state = init_vmap(keys)
    print(curr_player.shape)
    while state.terminated.sum() != n_envs:
        rng = jax.random.split(rng)[0]
        logits = jnp.log(state.legal_action_mask.astype(jnp.float16))
        action = jax.random.categorical(rng, logits=logits)
        curr_player, state, reward = step_vmap(state, action)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("n_processes", type=int)
    args = parser.parse_args()
    n_processes = args.n_processes

    # petting_zoo
    p = Pool(n_processes)
    time_sta = time.time()
    for i in range(n_processes):
        p.apply(petting_zoo_random_go)
    p.close()
    time_end = time.time()
    tim = time_end- time_sta
    print("execution time in petting zoo is {}".format(tim))

    # open spile
    p = Pool(n_processes)
    time_sta = time.time()
    for i in range(n_processes):
        p.apply(open_spile_random_go)
    p.close()
    time_end = time.time()
    tim = time_end- time_sta
    print("execution time in open spile is {}".format(tim))

    # pgx
    seed = np.random.randint(0, 1000)
    key = jax.random.PRNGKey(seed)
    time_sta = time.time()
    pgx_random_go(n_processes, key)
    time_end = time.time()
    tim = time_end- time_sta
    print("execution time in pgx is {}".format(tim))
