import time

import jax
from _go import (
    _count_ji,
    _get_alphazero_features,
    _get_reward,
    _merge_ren,
    _not_pass_move,
    _pass_move,
    _remove_stones,
    _set_stone_next_to_oppo_ren,
    _update_state_wo_legal_action,
    init,
    legal_actions,
    step,
)

rng = jax.random.PRNGKey(0)
size = 19
_, state = init(rng, size)


def test(func):
    rng = jax.random.PRNGKey(0)
    size = 19
    _, state = init(rng, size)
    if func.__name__ == "legal_actions":
        time_sta = time.perf_counter()
        jax.jit(func, static_argnums=(1,))(state, 19)
        time_end = time.perf_counter()
        delta = (time_end - time_sta) * 1000
        print(f"| `{func.__name__}` | {delta:.1f}ms |")
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
        except ZeroDivisionError:
            time_sta = time.perf_counter()
            jax.jit(func, static_argnums=(1,))(state, 19)
            time_end = time.perf_counter()
            delta = (time_end - time_sta) * 1000
        except TypeError:
            time_sta = time.perf_counter()
            jax.jit(func, static_argnums=(2,))(state, 0, 19)
            time_end = time.perf_counter()
            delta = (time_end - time_sta) * 1000
    print(f"| `{func.__name__}` | {delta:.1f}ms |")


# fmt: off
functions = [
    step, _update_state_wo_legal_action, _pass_move, _not_pass_move,
    _merge_ren, _set_stone_next_to_oppo_ren, _remove_stones,
    legal_actions, _get_reward, _count_ji, _get_alphazero_features
]
# fmt: on

print("| function name | compile time |")
print("| :--- | ---: |")

for func in functions:
    test(func=func)
