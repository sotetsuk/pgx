from pgx.animal_shogi import init, JaxAnimalShogiState, _legal_actions, _action_to_dlaction, _turn_to_reward, _another_color, _dlaction_to_action, _is_try, _is_check, _drop, _move, _update_legal_drop_actions, _update_legal_move_actions
import jax
import jax.numpy as jnp
import time
from typing import Tuple


@jax.jit
def step(
    state: JaxAnimalShogiState, action: int
) -> Tuple[JaxAnimalShogiState, int, bool]:
    first_time = time.time()
    s = state
    reward = 0
    terminated = False
    s_time = time.time()
    legal_actions = _legal_actions(s)
    e_time = time.time()
    print("legal_actions:", (e_time - s_time)*1000)
    _action = _dlaction_to_action(action, s)
    reward = jax.lax.cond(
        (_action.from_[0] > 11)
        | (_action.from_[0] < 0)
        | (legal_actions[_action_to_dlaction(_action, s.turn[0])] == 0),
        lambda: _turn_to_reward(_another_color(s)),
        lambda: reward,
    )
    terminated = jax.lax.cond(
        (_action.from_[0] > 11)
        | (_action.from_[0] < 0)
        | (legal_actions[_action_to_dlaction(_action, s.turn[0])] == 0),
        lambda: True,
        lambda: terminated,
    )
    s_time = time.time()
    s = jax.lax.cond(
        terminated,
        lambda: s,
        lambda: jax.lax.cond(
            _action.is_drop[0] == 1,
            lambda: _drop(_update_legal_drop_actions(s, _action), _action),
            lambda: _move(_update_legal_move_actions(s, _action), _action),
        ),
    )
    e_time = time.time()
    print("update_state:", (e_time - s_time)*1000)
    s_time = time.time()
    reward = jax.lax.cond(
        (terminated is False) & _is_try(_action),
        lambda: _turn_to_reward(s.turn[0]),
        lambda: reward,
    )
    terminated = jax.lax.cond(
        (terminated is False) & _is_try(_action),
        lambda: True,
        lambda: terminated,
    )
    turn = jnp.zeros(1, dtype=jnp.int32).at[0].set(_another_color(s))
    s = JaxAnimalShogiState(
        turn=turn,
        board=s.board,
        hand=s.hand,
        legal_actions_black=s.legal_actions_black,
        legal_actions_white=s.legal_actions_white,
    )  # type: ignore
    e_time = time.time()
    print("is_try:", (e_time - s_time)*1000)
    no_checking_piece = jnp.zeros(12, dtype=jnp.int32)
    checking_piece = no_checking_piece.at[_action.to[0]].set(1)
    s_time = time.time()
    s = jax.lax.cond(
        (_is_check(s)) & (terminated is False),
        lambda: JaxAnimalShogiState(
            turn=s.turn,
            board=s.board,
            hand=s.hand,
            legal_actions_black=s.legal_actions_black,
            legal_actions_white=s.legal_actions_white,
            is_check=jnp.array([1]),
            checking_piece=checking_piece,
        ),  # type: ignore
        lambda: JaxAnimalShogiState(
            turn=s.turn,
            board=s.board,
            hand=s.hand,
            legal_actions_black=s.legal_actions_black,
            legal_actions_white=s.legal_actions_white,
            is_check=jnp.array([0]),
            checking_piece=no_checking_piece,
        ),  # type: ignore
    )
    e_time = time.time()
    print("is_check:", (e_time - s_time)*1000)
    finish_time = time.time()
    print("all:", (finish_time - first_time)*1000)
    return s, reward, terminated


@jax.jit
def rand_step(rng):
    rng, subkey = jax.random.split(rng)
    cp, state = init(rng=rng)
    legal_actions = _legal_actions(state)
    legal_actions = jnp.nonzero(legal_actions, size=180, fill_value=-1)[0]
    rng, subkey = jax.random.split(rng)
    action = jax.random.choice(subkey, legal_actions)
    s, r, t = step(state, action)
    return s


N = 100000
if __name__ == '__main__':
    s = time.time()
    vmap_step = jax.jit(jax.vmap(rand_step))
    rng = jax.random.PRNGKey(0)
    rngs = jax.random.split(rng, N)
    states = vmap_step(rngs)
    print("all:", time.time() - s)
