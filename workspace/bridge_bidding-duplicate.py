import time
import jax
import jax.numpy as jnp
import pgx
from pgx.experimental.utils import act_randomly
from pgx.bridge_bidding import _player_position, duplicate


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


env_id: pgx.EnvId = "bridge_bidding"
env = pgx.make(env_id)
# run api test
pgx.api_test(env, 100)

# jit
init = jax.jit(jax.vmap(env.init))
step = jax.jit(jax.vmap(env.step))
duplicate_vmap = jax.jit(jax.vmap(duplicate))
imp_reward = jax.jit(jax.vmap(_imp_reward))
player_position = jax.vmap(_player_position)
act_randomly = jax.jit(act_randomly)

# show multi visualizations
N = 4
key = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key)
keys = jax.random.split(subkey, N)

state: pgx.State = init(keys)
duplicate_state: pgx.State = duplicate_vmap(state)

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
