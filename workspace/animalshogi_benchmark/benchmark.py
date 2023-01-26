from pgx.animal_shogi import init, JaxAnimalShogiState, _legal_actions, _action_to_dlaction, _turn_to_reward, _another_color, _dlaction_to_action, _is_try, _is_check, _drop, _move, _update_legal_drop_actions, _update_legal_move_actions
import jax
import jax.numpy as jnp
import time
from typing import Tuple


@jax.jit
def step(
    state: JaxAnimalShogiState, action: int
) -> Tuple[JaxAnimalShogiState, int, bool]:
    s_time = time.time()
    s = state
    reward = 0
    terminated = False
    legal_actions = _legal_actions(s)
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
    l1_time = time.time()
    print("is legal action?")
    print((l1_time - s_time) * 1000)
    s = jax.lax.cond(
        terminated,
        lambda: s,
        lambda: jax.lax.cond(
            _action.is_drop[0] == 1,
            lambda: _drop(_update_legal_drop_actions(s, _action), _action),
            lambda: _move(_update_legal_move_actions(s, _action), _action),
        ),
    )
    l2_time = time.time()
    print("change board")
    print((l2_time - l1_time) * 1000)
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
    l3_time = time.time()
    print("is try?")
    print((l3_time - l2_time) * 1000)
    no_checking_piece = jnp.zeros(12, dtype=jnp.int32)
    checking_piece = no_checking_piece.at[_action.to[0]].set(1)
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
    l4_time = time.time()
    print("is check?")
    print((l4_time - l3_time) * 1000)
    return s, reward, terminated


@jax.jit
def rand_step(rng):
    rng, subkey = jax.random.split(rng)
    cp, state = init(rng=rng)
    legal_actions = jax.lax.cond(
        cp == 0,
        lambda: state.legal_actions_black,
        lambda: state.legal_actions_white
    )
    legal_actions = jnp.nonzero(legal_actions, size=180, fill_value=-1)[0]
    rng, subkey = jax.random.split(rng)
    action = jax.random.choice(subkey, legal_actions)
    s, r, t = step(state, action)
    return s


N = 1000
if __name__ == '__main__':
    vmap_step = jax.vmap(rand_step)
    rng = jax.random.PRNGKey(0)
    rngs = jax.random.split(rng, N)
    states = vmap_step(rngs)
