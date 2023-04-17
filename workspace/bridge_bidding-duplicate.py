import sys
import json
import time
import jax
import pgx
from pgx.experimental.utils import act_randomly
from pgx.bridge_bidding import _player_position, _state_to_pbn

env_id: pgx.EnvId = sys.argv[1]
env = pgx.make(env_id)
player_position = jax.vmap(_player_position)
# state_to_pbn = jax.vmap(_state_to_pbn)
# run api test
pgx.api_test(env, 100)

# jit
init = jax.jit(jax.vmap(env.init))
step = jax.jit(jax.vmap(env.step))
act_randomly = jax.jit(act_randomly)

# show multi visualizations
N = 1
key = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key)
keys = jax.random.split(subkey, N)
state: pgx.State = init(keys)

TO_CARD = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]
print(state.hand)


for state_hand in state.hand:
    pbn = "N:"
    for i in range(4):  # player
        hand = jax.numpy.sort(state_hand[i * 13 : (i + 1) * 13])
        for j in range(4):  # suit
            card = [
                TO_CARD[i % 13] for i in hand if (j * 13 <= i < (j + 1) * 13)
            ][::-1]
            if card != [] and card[-1] == "A":
                card = card[-1:] + card[:-1]
            pbn += "".join(card)
            if j == 3:
                if i != 3:
                    pbn += " "
            else:
                pbn += "."
    print(pbn)
print(state.dealer)
print(state.shuffled_players)
i = 0
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
    state.save_svg(f"{i:04d}.svg")
    state = step(state, action)
    print(f"reward:\n{state.reward}")
    i += 1
state.save_svg(f"{i:04d}.svg")
