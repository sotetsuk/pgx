import sys
import time
import jax
import jax.numpy as jnp
from pgx.backgammon import (
    init,
    step,
    observe,
    _change_turn,
    _legal_action_mask,
    _is_action_legal,
    _legal_action_mask_for_valid_single_dice,
    _calc_win_score,
    _is_all_on_home_board,
    _rear_distance,
    _distance_to_goal,
    _decompose_action,
    _update_by_action,
    _move,
    _winning_step,
    _no_winning_step,
    _normal_step
)
action_to_point = (19 + 2) * 6 + 1
def test(func):
    rng = jax.random.PRNGKey(0)
    _, state = init(rng)
    if func.__name__ == "init":
        time_sta = time.perf_counter()
        jax.jit(func)(rng)
        time_end = time.perf_counter()
        delta = (time_end - time_sta) * 1000
        exp = jax.make_jaxpr(func)(rng)
    elif func.__name__ in ["step", "_update_by_action", "_normal_step"]:
        time_sta = time.perf_counter()
        jax.jit(func)(state, action_to_point)
        time_end = time.perf_counter()
        delta = (time_end - time_sta) * 1000
        exp = jax.make_jaxpr(func)(state, action_to_point)
    elif func.__name__ == "_legal_action_mask":
        time_sta = time.perf_counter()
        jax.jit(func)(state.board, state.turn, state.dice)
        time_end = time.perf_counter()
        delta = (time_end - time_sta) * 1000
        exp = jax.make_jaxpr(func)(state.board, state.turn, state.dice)
    elif func.__name__ == "_legal_action_mask_for_valid_single_dice":
        time_sta = time.perf_counter()
        jax.jit(func)(state.board, state.turn, jnp.int32(4))
        time_end = time.perf_counter()
        delta = (time_end - time_sta) * 1000
        exp = jax.make_jaxpr(func)(state.board, state.turn, jnp.int32(4))
    elif func.__name__ in ["_is_action_legal", "_move"]:
        time_sta = time.perf_counter()
        jax.jit(func)(state.board, state.turn, action_to_point)
        time_end = time.perf_counter()
        delta = (time_end - time_sta) * 1000
        exp = jax.make_jaxpr(func)(state.board, state.turn, action_to_point)
    elif func.__name__ in ["_change_turn", "_winning_step", "_no_winning_step"]:
        time_sta = time.perf_counter()
        jax.jit(func)(state)
        time_end = time.perf_counter()
        delta = (time_end - time_sta) * 1000
        exp = jax.make_jaxpr(func)(state)
    elif func.__name__ == "observe":
        time_sta = time.perf_counter()
        jax.jit(func)(state, _)
        time_end = time.perf_counter()
        delta = (time_end - time_sta) * 1000
        exp = jax.make_jaxpr(func)(state, _)
    elif func.__name__ in ["_rear_distance", "_is_all_on_home_board", "_calc_win_score"]:
        time_sta = time.perf_counter()
        jax.jit(func)(state.board, state.turn)
        time_end = time.perf_counter()
        delta = (time_end - time_sta) * 1000
        exp = jax.make_jaxpr(func)(state.board, state.turn)
    n_line = len(str(exp).split('\n'))
    print(f"| `{func.__name__}` | {n_line} | {delta:.1f}ms |")
    return
func_name = sys.argv[1]
if func_name == "init":
    func = init
elif func_name == "step":
    func = step
elif func_name == "_normal_step":
    func = _normal_step
elif func_name == "_winning_step":
    func = _winning_step
elif func_name == "_no_winning_step":
    func = _no_winning_step
elif func_name == "observe":
    func = observe
elif func_name == "_legal_action_mask":
    func = _legal_action_mask
elif func_name == "_legal_action_mask_for_valid_single_dice":
    func = _legal_action_mask_for_valid_single_dice
elif func_name == "_is_action_legal":
    func = _is_action_legal
elif func_name == "_update_by_action":
    func = _update_by_action
elif func_name == "_move":
    func = _move
elif func_name == "_change_turn":
    func = _change_turn
elif func_name == "_is_all_on_home_board":
    func = _is_all_on_home_board
elif func_name == "_rear_distance":
    func = _rear_distance
elif func_name == "_calc_win_score":
    func = _calc_win_score
else:
    print(func_name)
    assert False
test(func=func)