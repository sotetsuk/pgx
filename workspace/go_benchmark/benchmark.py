import sys
import time

import jax

from pgx.go import (
    _count_ji,
    _get_alphazero_features,
    _get_reward,
    _legal_actions,
    _merge_ren,
    _not_pass_move,
    _pass_move,
    _remove_stones,
    _set_stone_next_to_oppo_ren,
    _update_state_wo_legal_action,
    init,
    step,
)

rng = jax.random.PRNGKey(0)
size = 19
_, state = init(rng, size)


def test(func):
    rng = jax.random.PRNGKey(0)
    size = 19
    _, state = init(rng, size)
    if func.__name__ == "_legal_actions":
        time_sta = time.perf_counter()
        jax.jit(func, static_argnums=(1,))(state, 19)
        time_end = time.perf_counter()
        delta = (time_end - time_sta) * 1000
        exp = jax.make_jaxpr(func, static_argnums=(1,))(state, 19)
        n_line = len(str(exp).split("\n"))
        print(f"| `{func.__name__}` | {n_line} | {delta:.1f}ms |")
        return

    try:
        time_sta = time.perf_counter()
        jax.jit(func)(state)
        time_end = time.perf_counter()
        delta = (time_end - time_sta) * 1000
    except TypeError:
        try:
            time_sta = time.perf_counter()
            jax.jit(func, static_argnums=(1,))(state, 0)
            time_end = time.perf_counter()
            delta = (time_end - time_sta) * 1000
            exp = jax.make_jaxpr(func, static_argnums=(1,))(state, 0)
            n_line = len(str(exp).split("\n"))
        except ZeroDivisionError:
            time_sta = time.perf_counter()
            jax.jit(func, static_argnums=(1,))(state, 19)
            time_end = time.perf_counter()
            delta = (time_end - time_sta) * 1000
            exp = jax.make_jaxpr(func, static_argnums=(1,))(state, 19)
            n_line = len(str(exp).split("\n"))
        except TypeError:
            time_sta = time.perf_counter()
            jax.jit(func, static_argnums=(2,))(state, 0, 19)
            time_end = time.perf_counter()
            delta = (time_end - time_sta) * 1000
            exp = jax.make_jaxpr(func, static_argnums=(2,))(state, 0, 19)
            n_line = len(str(exp).split("\n"))
    print(f"| `{func.__name__}` | {n_line} | {delta:.1f}ms |")


func_name = sys.argv[1]
if func_name == "_count_ji":
    func = _count_ji
elif func_name == "_get_alphazero_features":
    func = _get_alphazero_features
elif func_name == "_get_reward":
    func = _get_reward
elif func_name == "_merge_ren":
    func = _merge_ren
elif func_name == "_not_pass_move":
    func = _not_pass_move
elif func_name == "_pass_move":
    func = _pass_move
elif func_name == "_remove_stones":
    func = _remove_stones
elif func_name == "_set_stone_next_to_oppo_ren":
    func = _set_stone_next_to_oppo_ren
elif func_name == "_update_legal_action":
    func = _update_legal_action
elif func_name == "_update_state_wo_legal_action":
    func = _update_state_wo_legal_action
elif func_name == "_legal_actions":
    func = _legal_actions
elif func_name == "step":
    func = step
else:
    print(func_name)
    assert False

test(func=func)
