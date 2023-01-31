import sys
import time

import jax

from pgx.animal_shogi import (
    _move,
    _update_legal_move_actions,
    _effected_positions,
    _is_check,
    _is_try,
    init,
    _legal_actions,
    step,
    _init_legal_actions,
    _board_status,
    _dlaction_to_action
)


def test(func):
    rng = jax.random.PRNGKey(0)
    state = init(rng)
    if func.__name__ == "_legal_actions" or func.__name__ == "_is_check"  or func.__name__ == "_init_legal_actions" or func.__name__ == "_board_status":
        time_sta = time.perf_counter()
        jax.jit(func)(state)
        time_end = time.perf_counter()
        delta = (time_end - time_sta) * 1000
    elif func.__name__ == "init":
        time_sta = time.perf_counter()
        jax.jit(func)(rng)
        time_end = time.perf_counter()
        delta = (time_end - time_sta) * 1000
    elif func.__name__ == "_effected_positions":
        time_sta = time.perf_counter()
        jax.jit(func)(state, 0)
        time_end = time.perf_counter()
        delta = (time_end - time_sta) * 1000
    elif func.__name__ == "step":
        time_sta = time.perf_counter()
        jax.jit(func)(state, 0)
        time_end = time.perf_counter()
        delta = (time_end - time_sta) * 1000
    elif func.__name__ == "_move" or func.__name__ == "_update_legal_move_actions":
        a = _dlaction_to_action(0, state)
        time_sta = time.perf_counter()
        jax.jit(func)(state, a)
        time_end = time.perf_counter()
        delta = (time_end - time_sta) * 1000
    elif func.__name__ == "_is_try":
        a = _dlaction_to_action(0, state)
        time_sta = time.perf_counter()
        jax.jit(func)(a)
        time_end = time.perf_counter()
        delta = (time_end - time_sta) * 1000
    elif func.__name__ == "_dlaction_to_action":
        time_sta = time.perf_counter()
        jax.jit(func)(5, state)
        time_end = time.perf_counter()
        delta = (time_end - time_sta) * 1000
    print(f"| `{func.__name__}` | {delta:.1f}ms |")
    return


func_name = sys.argv[1]
if func_name == "_legal_actions":
    func = _legal_actions
elif func_name == "_move":
    func = _move
elif func_name == "_update_legal_move_actions":
    func = _update_legal_move_actions
elif func_name == "_effected_positions":
    func = _effected_positions
elif func_name == "_is_check":
    func = _is_check
elif func_name == "_is_try":
    func = _is_try
elif func_name == "_init_legal_actions":
    func = _init_legal_actions
elif func_name == "_board_status":
    func = _board_status
elif func_name == "_dlaction_to_action":
    func = _dlaction_to_action
elif func_name == "step":
    func = step
elif func_name == "init":
    func = init
else:
    print(func_name)
    assert False

test(func=func)
