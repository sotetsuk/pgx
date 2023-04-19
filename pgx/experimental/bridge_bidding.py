import jax
import jax.numpy as jnp
import pgx
from pgx.experimental.utils import act_randomly
from pgx.bridge_bidding import _player_position, State


def _imp_reward(
    table_a_reward: jnp.ndarray, table_b_reward: jnp.ndarray
) -> jnp.ndarray:
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
def duplicate_init(
    state: State,
) -> State:
    """Make duplicated state where NSplayer and EWplayer are swapped"""
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
def _duplicate_step(
    state: pgx.State, action, table_a_reward, has_duplicate_result
):
    state = env.step(state, action)
    return jax.lax.cond(
        ~state.terminated,
        lambda: (state, table_a_reward, has_duplicate_result),
        lambda: jax.lax.cond(
            has_duplicate_result,
            lambda: (
                state.replace(
                    reward=_imp_reward(table_a_reward, state.reward)
                ),
                table_a_reward,
                jnp.bool_(True),
            ),
            lambda: (duplicate_init(state), state.reward, jnp.bool_(True)),
        ),
    )
