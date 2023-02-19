"""
Stable baselineのSubprocEnvと同様の実装方針でopen_spielのgoを並列化する.
"""

import multiprocessing as mp
from comparison import open_spile_make_env
from collections import OrderedDict
from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, Union, Dict, Iterable
import numpy as np
import time


def _worker(
    remote, parent_remote, env_name: str
) -> None:
    parent_remote.close()
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                state, action = data
                state.apply_action(action)
                legal_actions = state.legal_actions()
                terminated = state.is_terminal()
                remote.send((state, legal_actions, terminated))
            elif cmd == "reset":
                game = open_spile_make_env(env_name)
                state = game.new_initial_state()
                legal_actions = state.legal_actions()
                terminated = state.is_terminal()
                remote.send((state, legal_actions, terminated))
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

    def __init__(self, env_name, n_envs, start_method: Optional[str] = None):
        self.waiting = False
        self.closed = False
        env_names = [env_name] * n_envs

        if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context(start_method)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)])
        self.processes = []
        for work_remote, remote, env_name in zip(self.work_remotes, self.remotes, env_names):
            args = (work_remote, remote, env_name)
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(target=_worker, args=args, daemon=True)  # pytype:disable=attribute-error
            process.start()
            self.processes.append(process)
            work_remote.close()


    def step_async(self, states, actions: np.ndarray) -> None:
        for remote, state, action in zip(self.remotes, states, actions):
            remote.send(("step", (state, action)))
        self.waiting = True


    def step_wait(self) -> Tuple:
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        states, legal_actions, terminated = zip(*results)
        return states, np.stack(legal_actions), np.stack(terminated)


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
        states, legal_actions, terminated = zip(*results)
        return states, np.stack(legal_actions), np.stack(terminated)


    def step(self, states, actions) -> Tuple:
        self.step_async(states, actions)
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


if __name__ == "__main__":
    n_envs = 10
    env_name = "go"
    env = SubprocVecEnv(env_name, n_envs)
    
    process_list = []
    time_sta = time.time()
    state, legal_actions, terminated = env.reset()
    n_steps = 0
    rng = np.random.default_rng()
    while not terminated.all():
        action = rng.choice(legal_actions, axis=1) # n_envs
        state, legal_actions, terminated = env.step(state, action)
        n_steps += 1
    time_end = time.time()
    tim = time_end- time_sta
    env.close()
    print(n_steps)
    print(tim)