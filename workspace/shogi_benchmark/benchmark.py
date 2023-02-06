import sys
import time

import jax
import jax.numpy as jnp

from pgx.shogi import (
    init, step, _dlaction_to_action,_action_to_dlaction,_piece_moves,_legal_actions, _init_legal_actions, _is_mate, ShogiAction, _board_status
)


def test(func_name):
    if func_name == "init":
        time_sta = time.perf_counter()
        jax.jit(init)()
        time_end = time.perf_counter()
        delta = (time_end - time_sta) * 1000
        exp = jax.make_jaxpr(init)()
    elif func_name == "step":
        state = init()
        time_sta = time.perf_counter()
        jax.jit(step)(state, 0)
        time_end = time.perf_counter()
        delta = (time_end - time_sta) * 1000
        exp = jax.make_jaxpr(step)(state, 0)
    elif func_name == "_dlaction_to_action":
        state = init()
        time_sta = time.perf_counter()
        jax.jit(_dlaction_to_action)(0, state)
        time_end = time.perf_counter()
        delta = (time_end - time_sta) * 1000
        exp = jax.make_jaxpr(_dlaction_to_action)(0, state)
    elif func_name == "_action_to_dlaction":
        time_sta = time.perf_counter()
        jax.jit(_action_to_dlaction)(ShogiAction(0,0,0,0,0,False), 0)
        time_end = time.perf_counter()
        delta = (time_end - time_sta) * 1000
        exp = jax.make_jaxpr(_action_to_dlaction)(ShogiAction(0,0,0,0,0,False), 0)
    elif func_name == "_piece_moves":
        state = init()
        time_sta = time.perf_counter()
        jax.jit(_piece_moves)(_board_status(state), 0, 0)
        time_end = time.perf_counter()
        delta = (time_end - time_sta) * 1000
        exp = jax.make_jaxpr(_piece_moves)(_board_status(state), 0, 0)
    elif func_name == "_legal_actions":
        state = init()
        time_sta = time.perf_counter()
        jax.jit(_legal_actions)(state)
        time_end = time.perf_counter()
        delta = (time_end - time_sta) * 1000
        exp = jax.make_jaxpr(_legal_actions)(state)
    elif func_name == "_init_legal_actions":
        state = init()
        time_sta = time.perf_counter()
        jax.jit(_init_legal_actions)(state)
        time_end = time.perf_counter()
        delta = (time_end - time_sta) * 1000
        exp = jax.make_jaxpr(_init_legal_actions)(state)
    elif func_name == "_is_mate":
        state = init()
        time_sta = time.perf_counter()
        jax.jit(_is_mate)(state)
        time_end = time.perf_counter()
        delta = (time_end - time_sta) * 1000
        exp = jax.make_jaxpr(_is_mate)(state)
    else:
        return
    n_line = len(str(exp).split('\n'))
    print(f"| `{func_name}` | {n_line} | {delta:.1f}ms |")
    return


test(func_name=sys.argv[1])
