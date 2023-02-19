"""
Code references: https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/vec_env/base_vec_env.py, https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/vec_env/subproc_vec_env.py
Copyright (c) 2019 Antonin Raffin

"""

import multiprocessing as mp
from comparison import open_spile_make_env
from collections import OrderedDict
from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, Union, Dict, Iterable
import numpy as np
import time
import cloudpickle
import argparse


class CloudpickleWrapper:
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)

    :param var: the variable you wish to wrap for pickling with cloudpickle
    """

    def __init__(self, var: Any):
        self.var = var

    def __getstate__(self) -> Any:
        return cloudpickle.dumps(self.var)

    def __setstate__(self, var: Any) -> None:
        self.var = cloudpickle.loads(var)


def _worker(
    remote, parent_remote, state_wrapper: CloudpickleWrapper, env_name: str
) -> None:
    parent_remote.close()
    state = state_wrapper.var()
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                action = data
                state.apply_action(action)
                legal_actions = state.legal_actions()
                terminated = state.is_terminal()
                if terminated:  # auto_reset
                    state = open_spile_make_env(env_name).new_initial_state()
                    legal_actions = state.legal_actions()
                remote.send((legal_actions, terminated))

            elif cmd == "reset":
                legal_actions = state.legal_actions()
                terminated = state.is_terminal()
                remote.send((legal_actions, terminated))
            elif cmd == "render":
                pass
            elif cmd == "close":
                remote.close()
                break
            elif cmd == "get_spaces":
                pass
            elif cmd == "env_method":
                pass
            elif cmd == "get_attr":
                pass
            elif cmd == "set_attr":
                pass
            elif cmd == "is_wrapped":
                pass
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except EOFError:
            break



class SubprocVecEnv(object):
    """
    Creates a multiprocess vectorized wrapper for multiple environments, distributing each environment to its own
    process, allowing significant speed up when the environment is computationally complex.

    For performance reasons, if your environment is not IO bound, the number of environments should not exceed the
    number of logical cores on your CPU.

    .. warning::

        Only 'forkserver' and 'spawn' start methods are thread-safe,
        which is important when TensorFlow sessions or other non thread-safe
        libraries are used in the parent (see issue #217). However, compared to
        'fork' they incur a small start-up cost and have restrictions on
        global variables. With those methods, users must wrap the code in an
        ``if __name__ == "__main__":`` block.
        For more information, see the multiprocessing documentation.

    :param env_fns: Environments to run in subprocesses
    :param start_method: method used to start the subprocesses.
           Must be one of the methods returned by multiprocessing.get_all_start_methods().
           Defaults to 'forkserver' on available platforms, and 'spawn' otherwise.
    """

    def __init__(self, states, env_name, start_method: Optional[str] = None):
        self.waiting = False
        self.closed = False
        self.n_envs = len(states)
        env_names = [env_name] * self.n_envs

        if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context(start_method)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.n_envs)])
        self.processes = []
        for work_remote, remote, state, env_name in zip(self.work_remotes, self.remotes, states, env_names):
            args = (work_remote, remote, CloudpickleWrapper(state), env_name)
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(target=_worker, args=args, daemon=True)  # pytype:disable=attribute-error
            process.start()
            self.processes.append(process)
            work_remote.close()


    def step_async(self, actions: np.ndarray) -> None:
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", (action)))
        self.waiting = True


    def step_wait(self) -> Tuple:
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        legal_actions, terminated = zip(*results)
        return np.stack(legal_actions), np.stack(terminated)


    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        if seed is None:
            seed = np.random.randint(0, 2**32 - 1)
        for idx, remote in enumerate(self.remotes):
            remote.send(("seed", seed + idx))
        return [remote.recv() for remote in self.remotes]


    def reset(self) -> Tuple:
        for remote in self.remotes:
            remote.send(("reset", None))
        results = [remote.recv() for remote in self.remotes]
        legal_actions, terminated = zip(*results)
        return np.stack(legal_actions), np.stack(terminated)


    def step(self, actions) -> Tuple:
        self.step_async(actions)
        return self.step_wait()


    def close(self) -> None:
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()
        self.closed = True


def open_spiel_random_play(env, n_steps_lim):
    legal_actions, terminated = env.reset()
    n_steps = 0
    rng = np.random.default_rng()
    while n_steps < n_steps_lim:
        action = rng.choice(legal_actions, axis=1) # n_envs
        legal_actions, terminated = env.step(action)
        assert env.n_envs == legal_actions.shape[0]  # 並列化されていることを確認.
        n_steps += 1
    return n_steps





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name")
    parser.add_argument("n_envs", type=int)
    parser.add_argument("n_steps_lim", type=int)
    args = parser.parse_args()
    states = [lambda: open_spile_make_env(args.env_name).new_initial_state() for i in range(args.n_envs)]
    env = SubprocVecEnv(states, args.env_name)
    process_list = []
    time_sta = time.time()
    n_steps = open_spiel_random_play(env, args.n_steps_lim)
    time_end = time.time()
    tim = time_end- time_sta
    env.close()
    print(n_steps)
    print(tim)