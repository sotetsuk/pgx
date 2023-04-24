import jax
import jax.numpy as jnp

import pgx
from pgx.bridge_bidding import BridgeBidding, State

env = BridgeBidding()


def _imp_reward(
    table_a_reward: jnp.ndarray, table_b_reward: jnp.ndarray
) -> jnp.ndarray:
    """Convert score reward to IMP reward

    >>> table_a_reward = jnp.array([0, 0, 0, 0])
    >>> table_b_reward = jnp.array([0, 0, 0, 0])
    >>> _imp_reward(table_a_reward, table_b_reward)
    Array([0., 0., 0., 0.], dtype=float32)
    >>> table_a_reward = jnp.array([0, 0, 0, 0])
    >>> table_b_reward = jnp.array([100, 100, -100, -100])
    >>> _imp_reward(table_a_reward, table_b_reward)
    Array([ 3.,  3., -3., -3.], dtype=float32)
    >>> table_a_reward = jnp.array([-100, -100, 100, 100])
    >>> table_b_reward = jnp.array([0, 0, 0, 0])
    >>> _imp_reward(table_a_reward, table_b_reward)
    Array([-3., -3.,  3.,  3.], dtype=float32)
    >>> table_a_reward = jnp.array([-100, -100, 100, 100])
    >>> table_b_reward = jnp.array([100, 100, -100, -100])
    >>> _imp_reward(table_a_reward, table_b_reward)
    Array([0., 0., 0., 0.], dtype=float32)
    >>> table_a_reward = jnp.array([-3500, -3500, 3500, 3500])
    >>> table_b_reward = jnp.array([0, 0, 0, 0])
    >>> _imp_reward(table_a_reward, table_b_reward)
    Array([-23., -23.,  23.,  23.], dtype=float32)
    >>> table_a_reward = jnp.array([2000, 2000, -2000, -2000])
    >>> table_b_reward = jnp.array([2000, 2000, -2000, -2000])
    >>> _imp_reward(table_a_reward, table_b_reward)
    Array([ 24.,  24., -24., -24.], dtype=float32)
    """
    # fmt: off
    IMP_LIST = jnp.array([20, 50, 90, 130, 170,
                          220, 270, 320, 370, 430,
                          500, 600, 750, 900, 1100,
                          1300, 1500, 1750, 2000, 2250,
                          2500, 3000, 3500, 4000], dtype=jnp.float32)
    # fmt: on
    win = jax.lax.cond(
        table_a_reward[0] + table_b_reward[0] >= 0, lambda: 1, lambda: -1
    )

    def condition_fun(imp_diff):
        imp, difference_point = imp_diff
        return (difference_point >= IMP_LIST[imp]) & (imp < 24)

    def body_fun(imp_diff):
        imp, difference_point = imp_diff
        imp += 1
        return (imp, difference_point)

    imp, difference_point = jax.lax.while_loop(
        condition_fun,
        body_fun,
        (0, abs(table_a_reward[0] + table_b_reward[0])),
    )
    return jnp.array(
        [imp * win, imp * win, -imp * win, -imp * win], dtype=jnp.float32
    )


@jax.jit
def _duplicate_init(
    state: State,
) -> State:
    """Make duplicated state where NSplayer and EWplayer are swapped

    >>> key = jax.random.PRNGKey(0)
    >>> state = env.init(key)
    >>> duplicate_state = _duplicate_init(state)
    >>> duplicate_state.shuffled_players
    Array([0, 2, 1, 3], dtype=int8)
    >>> duplicate_state.dealer
    Array(1, dtype=int32)
    >>> duplicate_state.current_player
    Array(2, dtype=int8)
    >>> state = env.step(state, 35)
    >>> duplicate_state = _duplicate_init(state)
    >>> duplicate_state.shuffled_players
    Array([0, 2, 1, 3], dtype=int8)
    >>> duplicate_state.dealer
    Array(1, dtype=int32)
    >>> duplicate_state.current_player
    Array(2, dtype=int8)
    >>> duplicate_state.pass_num
    Array(0, dtype=int32)

    >>> state = env.step(state, 0)
    >>> duplicate_state = _duplicate_init(state)
    >>> duplicate_state.shuffled_players
    Array([0, 2, 1, 3], dtype=int8)
    >>> duplicate_state.dealer
    Array(1, dtype=int32)
    >>> duplicate_state.current_player
    Array(2, dtype=int8)
    >>> duplicate_state.legal_action_mask
    Array([ True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
           False, False], dtype=bool)
    """
    ix = jnp.array([1, 0, 3, 2])
    shuffled_players = state.shuffled_players[ix]
    current_player = shuffled_players[state.dealer]
    legal_actions = jnp.ones(38, dtype=jnp.bool_)
    # 最初はdable, redoubleできない
    legal_actions = legal_actions.at[36].set(False)
    legal_actions = legal_actions.at[37].set(False)
    duplicated_state = State(  # type: ignore
        shuffled_players=state.shuffled_players[ix],
        current_player=current_player,
        hand=state.hand,
        dealer=state.dealer,
        vul_NS=state.vul_NS,
        vul_EW=state.vul_EW,
        legal_action_mask=legal_actions,
    )
    return duplicated_state


@jax.jit
def duplicate_step(
    state: pgx.State, action, table_a_reward, has_duplicate_result
):
    """step function to perform a DUPLICATE match"""
    state = env.step(state=state, action=action)
    return jax.lax.cond(
        ~state.terminated,
        lambda: (state, table_a_reward, has_duplicate_result),
        lambda: jax.lax.cond(
            has_duplicate_result,
            lambda: (
                state.replace(  # type: ignore
                    reward=_imp_reward(table_a_reward, state.reward)
                ),
                jnp.zeros(4, dtype=jnp.float32),
                jnp.bool_(True),
            ),
            lambda: (_duplicate_init(state), state.reward, jnp.bool_(True)),
        ),
    )
