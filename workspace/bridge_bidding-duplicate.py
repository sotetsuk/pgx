import jax
import jax.numpy as jnp
import pgx
from pgx.experimental.utils import act_randomly
from pgx.bridge_bidding import _player_position
from pgx.experimental.bridge_bidding import duplicate_step
import time

env_id: pgx.EnvId = "bridge_bidding"
time1 = time.time()
env = pgx.make(env_id, dds_hash_table_path="dds_hash_table")
time2 = time.time()
print(time2 - time1)
# run api test
pgx.v1_api_test(env, 100)

# jit
init = jax.jit(jax.vmap(env.init))
step = jax.jit(jax.vmap(env.step))
duplicate_step = jax.jit(jax.vmap(duplicate_step))
player_position = jax.vmap(_player_position)
act_randomly = jax.jit(act_randomly)

# show multi visualizations
N = 4
key = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key)
keys = jax.random.split(subkey, N)

state: pgx.State = init(keys)
# duplicate_state: pgx.State = duplicate_vmap(state)


i = 0
has_duplicate_result = jnp.zeros(N, dtype=jnp.bool_)
table_a_reward = state.rewards
while not state.terminated.all():
    key, subkey = jax.random.split(key)
    action = act_randomly(subkey, state)

    print("================")
    print(f"{i:04d}")
    print("================")
    print(f"curr_player: {state.current_player}\naction: {action}")
    print(
        f"curr_player_position: {player_position(state.current_player, state)}"
    )
    print(f"shuflled_players:\n{state.shuffled_players}")
    state.save_svg(f"test/{i:04d}.svg")
    state_check = step(state, action)
    state, table_a_reward, has_duplicate_result = duplicate_step(
        state, action, table_a_reward, has_duplicate_result
    )
    print(f"table a reward\n{table_a_reward}")
    print(f"score_reward:\n{state_check.rewards}")
    print(f"IMP_reward:\n{state.rewards}")
    print(f"has_duplicate_result:\n{has_duplicate_result}")
    i += 1
state.save_svg(f"test/{i:04d}.svg")


'''
def state_to_pbn(state):
    TO_CARD = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]
    for state_hand in state.hand:
        pbn = "N:"
        for i in range(4):  # player
            hand = jax.numpy.sort(state_hand[i * 13 : (i + 1) * 13])
            for j in range(4):  # suit
                card = [
                    TO_CARD[i % 13]
                    for i in hand
                    if (j * 13 <= i < (j + 1) * 13)
                ][::-1]
                if card != [] and card[-1] == "A":
                    card = card[-1:] + card[:-1]
                pbn += "".join(card)
                if j == 3:
                    if i != 3:
                        pbn += " "
                else:
                    pbn += "."
    return pbn


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
'''


"""
print("table a")
print(f"current_player: {state.current_player}")
print(f"dealer: {state.dealer}")
print(f"shuffled_players: {state.shuffled_players}")

print("table b")
print(f"current_player: {duplicate_state.current_player}")
print(f"dealer: {duplicate_state.dealer}")
print(f"shuffled_players: {duplicate_state.shuffled_players}")

# table a start
i = 0
reward = state.reward
while not state.terminated.all():
    key, subkey = jax.random.split(key)
    action = act_randomly(subkey, state)
    print("================")
    print(f"{i:04d}")
    print("================")
    print(f"curr_player: {state.current_player}\naction: {action}")
    print(
        f"curr_player_position: {player_position(state.current_player, state)}"
    )
    state.save_svg(f"test/table_a_{i:04d}.svg")
    state = step(state, action)
    print(f"reward:\n{state.reward}")
    i += 1
    reward = reward + state.reward
state.save_svg(f"test/table_a_{i:04d}.svg")

# table b start
i = 0
duplicate_reward = duplicate_state.reward
while not duplicate_state.terminated.all():
    key, subkey = jax.random.split(key)
    action = act_randomly(subkey, duplicate_state)
    print("================")
    print(f"{i:04d}")
    print("================")
    print(f"curr_player: {duplicate_state.current_player}\naction: {action}")
    print(
        f"curr_player_position: {player_position(duplicate_state.current_player, state)}"
    )
    duplicate_state.save_svg(f"test/table_b_{i:04d}.svg")
    duplicate_state = step(duplicate_state, action)
    print(f"reward:\n{duplicate_state.reward}")
    i += 1
    duplicate_reward = duplicate_reward + duplicate_state.reward
state.save_svg(f"test/table_b_{i:04d}.svg")
print("================")
print("================")
print(f"table a reward: {reward}")
print(f"table b reward: {duplicate_reward}")
print(f"duplicate reward:\n{imp_reward(reward, duplicate_reward)}")
"""
