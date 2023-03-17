import csv
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

from pgx._bridge_bidding import (
    State,
    _calc_score,
    _calculate_dds_tricks,
    _contract,
    _key_to_hand,
    _load_sample_hash,
    _pbn_to_key,
    _player_position,
    _shuffle_players,
    _state_to_key,
    _state_to_pbn,
    _to_binary,
    _value_to_dds_tricks,
    duplicate,
    init,
    init_by_key,
    step,
)


def test_shuffle_players():
    key = jax.random.PRNGKey(0)
    for i in range(100):
        key, subkey = jax.random.split(key)
        shuffled_players = _shuffle_players(subkey)
        assert (shuffled_players[0] - shuffled_players[2]) % 2
        assert (shuffled_players[1] - shuffled_players[3]) % 2


def test_init():
    key = jax.random.PRNGKey(0)
    state = init(key)
    assert state.last_bid == -1
    assert state.last_bidder == -1
    assert not state.call_x
    assert not state.call_xx
    assert not state.pass_num
    assert _player_position(state.current_player, state) == state.dealer
    assert state.legal_action_mask[:-2].all()
    assert not state.legal_action_mask[-2:].all()


def test_duplicate():
    key = jax.random.PRNGKey(0)
    for i in range(1000):
        key, subkey = jax.random.split(key)
        init_state = init(subkey)
        duplicated_state = duplicate(init_state)
        assert (
            init_state.shuffled_players[0]
            == duplicated_state.shuffled_players[1]
        )
        assert (
            init_state.shuffled_players[1]
            == duplicated_state.shuffled_players[0]
        )
        assert (
            init_state.shuffled_players[2]
            == duplicated_state.shuffled_players[3]
        )
        assert (
            init_state.shuffled_players[3]
            == duplicated_state.shuffled_players[2]
        )


def test_step():
    #  player_id: 0 = N, 1 = S, 2 = W, 3 = E
    #   0  3  1  2
    #   N  E  S  W
    #  -----------
    #      P  P  P
    #  1C 2S  X  P
    #   P XX  P 5H
    #  5S  P 6C  X
    #  XX  P 7C  P
    #   P  P
    key = jax.random.PRNGKey(0)
    HASH_TABLE_SAMPLE_KEYS, HASH_TABLE_SAMPLE_VALUES = _load_sample_hash()
    state = init_by_key(HASH_TABLE_SAMPLE_KEYS[0], key)
    state = state.replace(
        dealer=jnp.int8(1),
        current_player=jnp.int8(3),
        shuffled_players=jnp.array([0, 3, 1, 2], dtype=jnp.int8),
        vul_NS=jnp.bool_(0),
        vul_EW=jnp.bool_(0),
    )
    bidding_history = jnp.full(319, -1, dtype=jnp.int8)
    legal_action_mask = jnp.ones(38, dtype=jnp.bool_)
    legal_action_mask = legal_action_mask.at[36].set(False)
    legal_action_mask = legal_action_mask.at[37].set(False)
    first_denomination_NS = jnp.full(5, -1, dtype=jnp.int8)
    first_denomination_EW = jnp.full(5, -1, dtype=jnp.int8)

    state = step(state, 35, HASH_TABLE_SAMPLE_KEYS, HASH_TABLE_SAMPLE_VALUES)
    #  player_id: 0 = N, 1 = S, 2 = W, 3 = E
    #   0  3  1  2
    #   N  E  S  W
    #  -----------
    #      P

    assert state.turn == 1
    assert state.current_player == 1
    assert _player_position(state.current_player, state) == 2
    assert not state.terminated
    bidding_history = bidding_history.at[0].set(35)
    assert jnp.all(state.bidding_history == bidding_history)
    assert jnp.all(state.legal_action_mask == legal_action_mask)
    assert state.last_bid == -1
    assert _player_position(state.last_bidder, state) == -1
    assert not state.call_x
    assert not state.call_xx
    assert jnp.all(state.first_denomination_NS == first_denomination_NS)
    assert jnp.all(state.first_denomination_EW == first_denomination_EW)
    assert state.pass_num == 1
    assert jnp.all(state.reward == np.zeros(4))

    state = step(state, 35, HASH_TABLE_SAMPLE_KEYS, HASH_TABLE_SAMPLE_VALUES)
    #  player_id: 0 = N, 1 = S, 2 = W, 3 = E
    #   0  3  1  2
    #   N  E  S  W
    #  -----------
    #      P  P

    assert state.turn == 2
    assert state.current_player == 2
    assert _player_position(state.current_player, state) == 3
    assert not state.terminated
    bidding_history = bidding_history.at[1].set(35)
    assert jnp.all(state.bidding_history == bidding_history)
    assert jnp.all(state.legal_action_mask == legal_action_mask)
    assert state.last_bid == -1
    assert _player_position(state.last_bidder, state) == -1
    assert not state.call_x
    assert not state.call_xx
    assert jnp.all(state.first_denomination_NS == first_denomination_NS)
    assert jnp.all(state.first_denomination_EW == first_denomination_EW)
    assert state.pass_num == 2
    assert jnp.all(state.reward == jnp.zeros(4))

    state = step(state, 35, HASH_TABLE_SAMPLE_KEYS, HASH_TABLE_SAMPLE_VALUES)
    #  player_id: 0 = N, 1 = S, 2 = W, 3 = E
    #   0  3  1  2
    #   N  E  S  W
    #  -----------
    #      P  P  P

    assert state.turn == 3
    assert state.current_player == 0
    assert _player_position(state.current_player, state) == 0
    assert not state.terminated
    bidding_history = bidding_history.at[2].set(35)
    assert jnp.all(state.bidding_history == bidding_history)
    assert jnp.all(state.legal_action_mask == legal_action_mask)
    assert state.last_bid == -1
    assert _player_position(state.last_bidder, state) == -1
    assert not state.call_x
    assert not state.call_xx
    assert jnp.all(state.first_denomination_NS == first_denomination_NS)
    assert jnp.all(state.first_denomination_EW == first_denomination_EW)
    assert state.pass_num == 3
    assert jnp.all(state.reward == jnp.zeros(4))

    state = step(state, 0, HASH_TABLE_SAMPLE_KEYS, HASH_TABLE_SAMPLE_VALUES)
    #  player_id: 0 = N, 1 = S, 2 = W, 3 = E
    #   0  3  1  2
    #   N  E  S  W
    #  -----------
    #      P  P  P
    #  1C
    first_denomination_NS = jnp.array([0, -1, -1, -1, -1])
    first_denomination_EW = jnp.array([-1, -1, -1, -1, -1])

    assert state.turn == 4
    assert state.current_player == 3
    assert _player_position(state.current_player, state) == 1
    assert not state.terminated
    bidding_history = bidding_history.at[3].set(0)
    assert jnp.all(state.bidding_history == bidding_history)
    legal_action_mask = legal_action_mask.at[0].set(False)
    legal_action_mask = legal_action_mask.at[36].set(True)
    assert jnp.all(state.legal_action_mask == legal_action_mask)
    assert state.last_bid == 0
    assert _player_position(state.last_bidder, state) == 0
    assert not state.call_x
    assert not state.call_xx
    assert jnp.all(state.first_denomination_NS == first_denomination_NS)
    assert jnp.all(state.first_denomination_EW == first_denomination_EW)
    assert state.pass_num == 0
    assert jnp.all(state.reward == jnp.zeros(4))

    state = step(state, 8, HASH_TABLE_SAMPLE_KEYS, HASH_TABLE_SAMPLE_VALUES)
    #  player_id: 0 = N, 1 = S, 2 = W, 3 = E
    #   0  3  1  2
    #   N  E  S  W
    #  -----------
    #      P  P  P
    #  1C 2S
    first_denomination_NS = jnp.array([0, -1, -1, -1, -1])
    first_denomination_EW = jnp.array([-1, -1, -1, 3, -1])

    assert state.turn == 5
    assert state.current_player == 1
    assert _player_position(state.current_player, state) == 2
    assert not state.terminated
    bidding_history = bidding_history.at[4].set(8)
    assert np.all(state.bidding_history == bidding_history)
    legal_action_mask = jnp.where(
        jnp.arange(38) < 9, False, state.legal_action_mask
    )
    legal_action_mask = legal_action_mask.at[36].set(True)
    assert jnp.all(state.legal_action_mask == legal_action_mask)
    assert state.last_bid == 8
    assert _player_position(state.last_bidder, state) == 1
    assert not state.call_x
    assert not state.call_xx
    assert jnp.all(state.first_denomination_NS == first_denomination_NS)
    assert jnp.all(state.first_denomination_EW == first_denomination_EW)
    assert state.pass_num == 0
    assert jnp.all(state.reward == jnp.zeros(4))

    state = step(state, 36, HASH_TABLE_SAMPLE_KEYS, HASH_TABLE_SAMPLE_VALUES)
    #  player_id: 0 = N, 1 = S, 2 = W, 3 = E
    #   0  3  1  2
    #   N  E  S  W
    #  -----------
    #      P  P  P
    #  1C 2S  X
    first_denomination_NS = jnp.array([0, -1, -1, -1, -1])
    first_denomination_EW = jnp.array([-1, -1, -1, 3, -1])

    assert state.turn == 6
    assert state.current_player == 2
    assert _player_position(state.current_player, state) == 3
    assert not state.terminated
    bidding_history = bidding_history.at[5].set(36)
    assert jnp.all(state.bidding_history == bidding_history)
    legal_action_mask = jnp.where(
        jnp.arange(38) < 9, False, state.legal_action_mask
    )
    legal_action_mask = legal_action_mask.at[36].set(False)
    legal_action_mask = legal_action_mask.at[37].set(True)
    assert jnp.all(state.legal_action_mask == legal_action_mask)
    assert state.last_bid == 8
    assert _player_position(state.last_bidder, state) == 1
    assert state.call_x
    assert not state.call_xx
    assert jnp.all(state.first_denomination_NS == first_denomination_NS)
    assert jnp.all(state.first_denomination_EW == first_denomination_EW)
    assert state.pass_num == 0
    assert jnp.all(state.reward == jnp.zeros(4))

    state = step(state, 35, HASH_TABLE_SAMPLE_KEYS, HASH_TABLE_SAMPLE_VALUES)
    #  player_id: 0 = N, 1 = S, 2 = W, 3 = E
    #   0  3  1  2
    #   N  E  S  W
    #  -----------
    #      P  P  P
    #  1C 2S  X  P
    first_denomination_NS = jnp.array([0, -1, -1, -1, -1])
    first_denomination_EW = jnp.array([-1, -1, -1, 3, -1])

    assert state.turn == 7
    assert state.current_player == 0
    assert _player_position(state.current_player, state) == 0
    assert not state.terminated
    bidding_history = bidding_history.at[6].set(35)
    assert np.all(state.bidding_history == bidding_history)
    legal_action_mask = jnp.ones(38, dtype=jnp.bool_)
    legal_action_mask = jnp.where(
        jnp.arange(38) < 9, False, state.legal_action_mask
    )
    legal_action_mask = legal_action_mask.at[36].set(False)
    legal_action_mask = legal_action_mask.at[37].set(False)
    assert jnp.all(state.legal_action_mask == legal_action_mask)
    assert state.last_bid == 8
    assert _player_position(state.last_bidder, state) == 1
    assert state.call_x
    assert not state.call_xx
    assert jnp.all(state.first_denomination_NS == first_denomination_NS)
    assert jnp.all(state.first_denomination_EW == first_denomination_EW)
    assert state.pass_num == 1
    assert jnp.all(state.reward == jnp.zeros(4))

    state = step(state, 35, HASH_TABLE_SAMPLE_KEYS, HASH_TABLE_SAMPLE_VALUES)
    #  player_id: 0 = N, 1 = S, 2 = W, 3 = E
    #   0  3  1  2
    #   N  E  S  W
    #  -----------
    #      P  P  P
    #  1C 2S  X  P
    #   P
    first_denomination_NS = jnp.array([0, -1, -1, -1, -1])
    first_denomination_EW = jnp.array([-1, -1, -1, 3, -1])

    assert state.turn == 8
    assert state.current_player == 3
    assert _player_position(state.current_player, state) == 1
    assert not state.terminated
    bidding_history = bidding_history.at[7].set(35)
    assert jnp.all(state.bidding_history == bidding_history)
    legal_action_mask = jnp.ones(38, dtype=jnp.bool_)
    legal_action_mask = jnp.where(
        jnp.arange(38) < 9, False, state.legal_action_mask
    )
    legal_action_mask = legal_action_mask.at[36].set(False)
    legal_action_mask = legal_action_mask.at[37].set(True)
    assert jnp.all(state.legal_action_mask == legal_action_mask)
    assert state.last_bid == 8
    assert _player_position(state.last_bidder, state) == 1
    assert state.call_x
    assert not state.call_xx
    assert jnp.all(state.first_denomination_NS == first_denomination_NS)
    assert jnp.all(state.first_denomination_EW == first_denomination_EW)
    assert state.pass_num == 2
    assert jnp.all(state.reward == jnp.zeros(4))

    state = step(state, 37, HASH_TABLE_SAMPLE_KEYS, HASH_TABLE_SAMPLE_VALUES)
    #  player_id: 0 = N, 1 = S, 2 = W, 3 = E
    #   0  3  1  2
    #   N  E  S  W
    #  -----------
    #      P  P  P
    #  1C 2S  X  P
    #   P XX
    first_denomination_NS = jnp.array([0, -1, -1, -1, -1])
    first_denomination_EW = jnp.array([-1, -1, -1, 3, -1])

    assert state.turn == 9
    assert state.current_player == 1
    assert _player_position(state.current_player, state) == 2
    assert not state.terminated
    bidding_history = bidding_history.at[8].set(37)
    assert jnp.all(state.bidding_history == bidding_history)
    legal_action_mask = jnp.ones(38, dtype=jnp.bool_)
    legal_action_mask = jnp.where(
        jnp.arange(38) < 9, False, state.legal_action_mask
    )
    legal_action_mask = legal_action_mask.at[36].set(False)
    legal_action_mask = legal_action_mask.at[37].set(False)
    assert jnp.all(state.legal_action_mask == legal_action_mask)
    assert state.last_bid == 8
    assert _player_position(state.last_bidder, state) == 1
    assert state.call_x
    assert state.call_xx
    assert jnp.all(state.first_denomination_NS == first_denomination_NS)
    assert jnp.all(state.first_denomination_EW == first_denomination_EW)
    assert state.pass_num == 0
    assert jnp.all(state.reward == jnp.zeros(4))

    state = step(state, 35, HASH_TABLE_SAMPLE_KEYS, HASH_TABLE_SAMPLE_VALUES)
    #  player_id: 0 = N, 1 = S, 2 = W, 3 = E
    #   0  3  1  2
    #   N  E  S  W
    #  -----------
    #      P  P  P
    #  1C 2S  X  P
    #   P XX  P
    first_denomination_NS = jnp.array([0, -1, -1, -1, -1])
    first_denomination_EW = jnp.array([-1, -1, -1, 3, -1])

    assert state.turn == 10
    assert state.current_player == 2
    assert _player_position(state.current_player, state) == 3
    assert not state.terminated
    bidding_history = bidding_history.at[9].set(35)
    assert jnp.all(state.bidding_history == bidding_history)
    legal_action_mask = jnp.ones(38, dtype=jnp.bool_)
    legal_action_mask = jnp.where(
        jnp.arange(38) < 9, False, state.legal_action_mask
    )
    legal_action_mask = legal_action_mask.at[36].set(False)
    legal_action_mask = legal_action_mask.at[37].set(False)
    assert jnp.all(state.legal_action_mask == legal_action_mask)
    assert state.last_bid == 8
    assert _player_position(state.last_bidder, state) == 1
    assert state.call_x
    assert state.call_xx
    assert jnp.all(state.first_denomination_NS == first_denomination_NS)
    assert jnp.all(state.first_denomination_EW == first_denomination_EW)
    assert state.pass_num == 1
    assert jnp.all(state.reward == jnp.zeros(4))

    state = step(state, 22, HASH_TABLE_SAMPLE_KEYS, HASH_TABLE_SAMPLE_VALUES)
    #  player_id: 0 = N, 1 = S, 2 = W, 3 = E
    #   0  3  1  2
    #   N  E  S  W
    #  -----------
    #      P  P  P
    #  1C 2S  X  P
    #   P XX  P 5H
    first_denomination_NS = jnp.array([0, -1, -1, -1, -1])
    first_denomination_EW = jnp.array([-1, -1, 2, 3, -1])

    assert state.turn == 11
    assert state.current_player == 0
    assert _player_position(state.current_player, state) == 0
    assert not state.terminated
    bidding_history = bidding_history.at[10].set(22)
    assert jnp.all(state.bidding_history == bidding_history)
    legal_action_mask = jnp.ones(38, dtype=jnp.bool_)
    legal_action_mask = jnp.where(
        jnp.arange(38) < 23, False, state.legal_action_mask
    )
    legal_action_mask = legal_action_mask.at[36].set(True)
    legal_action_mask = legal_action_mask.at[37].set(False)
    assert jnp.all(state.legal_action_mask == legal_action_mask)
    assert state.last_bid == 22
    assert _player_position(state.last_bidder, state) == 3
    assert not state.call_x
    assert not state.call_xx
    assert jnp.all(state.first_denomination_NS == first_denomination_NS)
    assert jnp.all(state.first_denomination_EW == first_denomination_EW)
    assert state.pass_num == 0
    assert jnp.all(state.reward == jnp.zeros(4))

    state = step(state, 23, HASH_TABLE_SAMPLE_KEYS, HASH_TABLE_SAMPLE_VALUES)
    #  player_id: 0 = N, 1 = S, 2 = W, 3 = E
    #   0  3  1  2
    #   N  E  S  W
    #  -----------
    #      P  P  P
    #  1C 2S  X  P
    #   P XX  P 5H
    #  5S
    first_denomination_NS = jnp.array([0, -1, -1, 0, -1])
    first_denomination_EW = jnp.array([-1, -1, 2, 3, -1])

    assert state.turn == 12
    assert state.current_player == 3
    assert _player_position(state.current_player, state) == 1
    assert not state.terminated
    bidding_history = bidding_history.at[11].set(23)
    assert jnp.all(state.bidding_history == bidding_history)
    legal_action_mask = jnp.ones(38, dtype=jnp.bool_)
    legal_action_mask = jnp.where(
        jnp.arange(38) < 24, False, state.legal_action_mask
    )
    legal_action_mask = legal_action_mask.at[36].set(True)
    legal_action_mask = legal_action_mask.at[37].set(False)
    assert jnp.all(state.legal_action_mask == legal_action_mask)
    assert state.last_bid == 23
    assert _player_position(state.last_bidder, state) == 0
    assert not state.call_x
    assert not state.call_xx
    assert jnp.all(state.first_denomination_NS == first_denomination_NS)
    assert jnp.all(state.first_denomination_EW == first_denomination_EW)
    assert state.pass_num == 0
    assert jnp.all(state.reward == jnp.zeros(4))

    state = step(state, 35, HASH_TABLE_SAMPLE_KEYS, HASH_TABLE_SAMPLE_VALUES)
    #  player_id: 0 = N, 1 = S, 2 = W, 3 = E
    #   0  3  1  2
    #   N  E  S  W
    #  -----------
    #      P  P  P
    #  1C 2S  X  P
    #   P XX  P 5H
    #  5S  P
    first_denomination_NS = jnp.array([0, -1, -1, 0, -1])
    first_denomination_EW = jnp.array([-1, -1, 2, 3, -1])

    assert state.turn == 13
    assert state.current_player == 1
    assert _player_position(state.current_player, state) == 2
    assert not state.terminated
    bidding_history = bidding_history.at[12].set(35)
    assert jnp.all(state.bidding_history == bidding_history)
    legal_action_mask = jnp.ones(38, dtype=jnp.bool_)
    legal_action_mask = jnp.where(
        jnp.arange(38) < 24, False, state.legal_action_mask
    )
    legal_action_mask = legal_action_mask.at[36].set(False)
    legal_action_mask = legal_action_mask.at[37].set(False)
    assert jnp.all(state.legal_action_mask == legal_action_mask)
    assert state.last_bid == 23
    assert _player_position(state.last_bidder, state) == 0
    assert not state.call_x
    assert not state.call_xx
    assert jnp.all(state.first_denomination_NS == first_denomination_NS)
    assert jnp.all(state.first_denomination_EW == first_denomination_EW)
    assert state.pass_num == 1
    assert jnp.all(state.reward == jnp.zeros(4))

    state = step(state, 25, HASH_TABLE_SAMPLE_KEYS, HASH_TABLE_SAMPLE_VALUES)
    #  player_id: 0 = N, 1 = S, 2 = W, 3 = E
    #   0  3  1  2
    #   N  E  S  W
    #  -----------
    #      P  P  P
    #  1C 2S  X  P
    #   P XX  P 5H
    #  5S  P 6C
    first_denomination_NS = jnp.array([0, -1, -1, 0, -1])
    first_denomination_EW = jnp.array([-1, -1, 2, 3, -1])

    assert state.turn == 14
    assert state.current_player == 2
    assert _player_position(state.current_player, state) == 3
    assert not state.terminated
    bidding_history = bidding_history.at[13].set(25)
    assert np.all(state.bidding_history == bidding_history)
    legal_action_mask = jnp.ones(38, dtype=jnp.bool_)
    legal_action_mask = jnp.where(
        jnp.arange(38) < 26, False, state.legal_action_mask
    )
    legal_action_mask = legal_action_mask.at[36].set(True)
    legal_action_mask = legal_action_mask.at[37].set(False)
    assert jnp.all(state.legal_action_mask == legal_action_mask)
    assert state.last_bid == 25
    assert _player_position(state.last_bidder, state) == 2
    assert not state.call_x
    assert not state.call_xx
    assert jnp.all(state.first_denomination_NS == first_denomination_NS)
    assert jnp.all(state.first_denomination_EW == first_denomination_EW)
    assert state.pass_num == 0
    assert jnp.all(state.reward == jnp.zeros(4))

    state = step(state, 36, HASH_TABLE_SAMPLE_KEYS, HASH_TABLE_SAMPLE_VALUES)
    #  player_id: 0 = N, 1 = S, 2 = W, 3 = E
    #   0  3  1  2
    #   N  E  S  W
    #  -----------
    #      P  P  P
    #  1C 2S  X  P
    #   P XX  P 5H
    #  5S  P 6C  X
    first_denomination_NS = jnp.array([0, -1, -1, 0, -1])
    first_denomination_EW = jnp.array([-1, -1, 2, 3, -1])

    assert state.turn == 15
    assert state.current_player == 0
    assert _player_position(state.current_player, state) == 0
    assert not state.terminated
    bidding_history = bidding_history.at[14].set(36)
    assert np.all(state.bidding_history == bidding_history)
    legal_action_mask = jnp.ones(38, dtype=jnp.bool_)
    legal_action_mask = jnp.where(
        jnp.arange(38) < 26, False, state.legal_action_mask
    )
    legal_action_mask = legal_action_mask.at[36].set(False)
    legal_action_mask = legal_action_mask.at[37].set(True)
    assert np.all(state.legal_action_mask == legal_action_mask)
    assert state.last_bid == 25
    assert _player_position(state.last_bidder, state) == 2
    assert state.call_x
    assert not state.call_xx
    assert np.all(state.first_denomination_NS == first_denomination_NS)
    assert np.all(state.first_denomination_EW == first_denomination_EW)
    assert state.pass_num == 0
    assert np.all(state.reward == np.zeros(4))

    state = step(state, 37, HASH_TABLE_SAMPLE_KEYS, HASH_TABLE_SAMPLE_VALUES)
    #  player_id: 0 = N, 1 = S, 2 = W, 3 = E
    #   0  3  1  2
    #   N  E  S  W
    #  -----------
    #      P  P  P
    #  1C 2S  X  P
    #   P XX  P 5H
    #  5S  P 6C  X
    #  XX
    first_denomination_NS = jnp.array([0, -1, -1, 0, -1])
    first_denomination_EW = jnp.array([-1, -1, 2, 3, -1])

    assert state.turn == 16
    assert state.current_player == 3
    assert _player_position(state.current_player, state) == 1
    assert not state.terminated
    bidding_history = bidding_history.at[15].set(37)
    assert jnp.all(state.bidding_history == bidding_history)
    legal_action_mask = jnp.ones(38, dtype=jnp.bool_)
    legal_action_mask = jnp.where(
        jnp.arange(38) < 26, False, state.legal_action_mask
    )
    legal_action_mask = legal_action_mask.at[36].set(False)
    legal_action_mask = legal_action_mask.at[37].set(False)
    assert jnp.all(state.legal_action_mask == legal_action_mask)
    assert state.last_bid == 25
    assert _player_position(state.last_bidder, state) == 2
    assert state.call_x
    assert state.call_xx
    assert jnp.all(state.first_denomination_NS == first_denomination_NS)
    assert jnp.all(state.first_denomination_EW == first_denomination_EW)
    assert state.pass_num == 0
    assert jnp.all(state.reward == jnp.zeros(4))

    state = step(state, 35, HASH_TABLE_SAMPLE_KEYS, HASH_TABLE_SAMPLE_VALUES)
    #  player_id: 0 = N, 1 = S, 2 = W, 3 = E
    #   0  3  1  2
    #   N  E  S  W
    #  -----------
    #      P  P  P
    #  1C 2S  X  P
    #   P XX  P 5H
    #  5S  P 6C  X
    #  XX  P
    first_denomination_NS = jnp.array([0, -1, -1, 0, -1])
    first_denomination_EW = jnp.array([-1, -1, 2, 3, -1])

    assert state.turn == 17
    assert state.current_player == 1
    assert _player_position(state.current_player, state) == 2
    assert not state.terminated
    bidding_history = bidding_history.at[16].set(35)
    assert jnp.all(state.bidding_history == bidding_history)
    legal_action_mask = jnp.ones(38, dtype=jnp.bool_)
    legal_action_mask = jnp.where(
        jnp.arange(38) < 26, False, state.legal_action_mask
    )
    legal_action_mask = legal_action_mask.at[36].set(False)
    legal_action_mask = legal_action_mask.at[37].set(False)
    assert np.all(state.legal_action_mask == legal_action_mask)
    assert state.last_bid == 25
    assert _player_position(state.last_bidder, state) == 2
    assert state.call_x
    assert state.call_xx
    assert jnp.all(state.first_denomination_NS == first_denomination_NS)
    assert jnp.all(state.first_denomination_EW == first_denomination_EW)
    assert state.pass_num == 1
    assert jnp.all(state.reward == jnp.zeros(4))

    state = step(state, 30, HASH_TABLE_SAMPLE_KEYS, HASH_TABLE_SAMPLE_VALUES)
    #  player_id: 0 = N, 1 = S, 2 = W, 3 = E
    #   0  3  1  2
    #   N  E  S  W
    #  -----------
    #      P  P  P
    #  1C 2S  X  P
    #   P XX  P 5H
    #  5S  P 6C  X
    #  XX  P 7C
    first_denomination_NS = jnp.array([0, -1, -1, 0, -1])
    first_denomination_EW = jnp.array([-1, -1, 2, 3, -1])

    assert state.turn == 18
    assert state.current_player == 2
    assert _player_position(state.current_player, state) == 3
    assert not state.terminated
    bidding_history = bidding_history.at[17].set(30)
    assert jnp.all(state.bidding_history == bidding_history)
    legal_action_mask = jnp.ones(38, dtype=jnp.bool_)
    legal_action_mask = jnp.where(
        jnp.arange(38) < 31, False, state.legal_action_mask
    )
    legal_action_mask = legal_action_mask.at[36].set(True)
    legal_action_mask = legal_action_mask.at[37].set(False)
    assert jnp.all(state.legal_action_mask == legal_action_mask)
    assert state.last_bid == 30
    assert _player_position(state.last_bidder, state) == 2
    assert not state.call_x
    assert not state.call_xx
    assert jnp.all(state.first_denomination_NS == first_denomination_NS)
    assert jnp.all(state.first_denomination_EW == first_denomination_EW)
    assert state.pass_num == 0
    assert jnp.all(state.reward == jnp.zeros(4))

    state = step(state, 35, HASH_TABLE_SAMPLE_KEYS, HASH_TABLE_SAMPLE_VALUES)
    #  player_id: 0 = N, 1 = S, 2 = W, 3 = E
    #   0  3  1  2
    #   N  E  S  W
    #  -----------
    #      P  P  P
    #  1C 2S  X  P
    #   P XX  P 5H
    #  5S  P 6C  X
    #  XX  P 7C  P
    first_denomination_NS = jnp.array([0, -1, -1, 0, -1])
    first_denomination_EW = jnp.array([-1, -1, 2, 3, -1])

    assert state.turn == 19
    assert state.current_player == 0
    assert _player_position(state.current_player, state) == 0
    assert not state.terminated
    bidding_history = bidding_history.at[18].set(35)
    assert np.all(state.bidding_history == bidding_history)
    legal_action_mask = jnp.ones(38, dtype=jnp.bool_)
    legal_action_mask = jnp.where(
        jnp.arange(38) < 31, False, state.legal_action_mask
    )
    legal_action_mask = legal_action_mask.at[36].set(False)
    legal_action_mask = legal_action_mask.at[37].set(False)
    assert jnp.all(state.legal_action_mask == legal_action_mask)
    assert state.last_bid == 30
    assert _player_position(state.last_bidder, state) == 2
    assert not state.call_x
    assert not state.call_xx
    assert jnp.all(state.first_denomination_NS == first_denomination_NS)
    assert jnp.all(state.first_denomination_EW == first_denomination_EW)
    assert state.pass_num == 1
    assert jnp.all(state.reward == jnp.zeros(4))

    state = step(state, 35, HASH_TABLE_SAMPLE_KEYS, HASH_TABLE_SAMPLE_VALUES)
    #  player_id: 0 = N, 1 = S, 2 = W, 3 = E
    #   0  3  1  2
    #   N  E  S  W
    #  -----------
    #      P  P  P
    #  1C 2S  X  P
    #   P XX  P 5H
    #  5S  P 6C  X
    #  XX  P 7C  P
    #   P
    first_denomination_NS = jnp.array([0, -1, -1, 0, -1])
    first_denomination_EW = jnp.array([-1, -1, 2, 3, -1])

    assert state.turn == 20
    assert state.current_player == 3
    assert _player_position(state.current_player, state) == 1
    assert not state.terminated
    bidding_history = bidding_history.at[19].set(35)
    assert jnp.all(state.bidding_history == bidding_history)
    legal_action_mask = jnp.ones(38, dtype=jnp.bool_)
    legal_action_mask = jnp.where(
        jnp.arange(38) < 31, False, state.legal_action_mask
    )
    legal_action_mask = legal_action_mask.at[36].set(True)
    legal_action_mask = legal_action_mask.at[37].set(False)
    assert jnp.all(state.legal_action_mask == legal_action_mask)
    assert state.last_bid == 30
    assert _player_position(state.last_bidder, state) == 2
    assert not state.call_x
    assert not state.call_xx
    assert jnp.all(state.first_denomination_NS == first_denomination_NS)
    assert jnp.all(state.first_denomination_EW == first_denomination_EW)
    assert state.pass_num == 2
    assert jnp.all(state.reward == jnp.zeros(4))

    state = step(state, 35, HASH_TABLE_SAMPLE_KEYS, HASH_TABLE_SAMPLE_VALUES)
    #  player_id: 0 = N, 1 = S, 2 = W, 3 = E
    #   0  3  1  2
    #   N  E  S  W
    #  -----------
    #      P  P  P
    #  1C 2S  X  P
    #   P XX  P 5H
    #  5S  P 6C  X
    #  XX  P 7C  P
    #   P  P
    # Passが3回続いたので終了
    assert state.terminated == True
    assert state.current_player == -1
    first_denomination_NS = jnp.array([0, -1, -1, 0, -1])
    first_denomination_EW = jnp.array([-1, -1, 2, 3, -1])

    assert state.turn == 20
    assert _player_position(state.current_player, state) == -1
    assert state.terminated
    bidding_history = bidding_history.at[20].set(35)
    assert jnp.all(state.bidding_history == bidding_history)
    legal_action_mask = jnp.ones(38, dtype=jnp.bool_)
    legal_action_mask = jnp.where(
        jnp.arange(38) < 31, False, state.legal_action_mask
    )
    legal_action_mask = legal_action_mask.at[36].set(True)
    legal_action_mask = legal_action_mask.at[37].set(False)
    assert jnp.all(state.legal_action_mask == legal_action_mask)
    assert state.last_bid == 30
    assert _player_position(state.last_bidder, state) == 2
    assert not state.call_x
    assert not state.call_xx
    assert jnp.all(state.first_denomination_NS == first_denomination_NS)
    assert jnp.all(state.first_denomination_EW == first_denomination_EW)
    assert state.pass_num == 3
    assert state.reward.shape == (4,)
    assert jnp.all(
        state.reward == jnp.array([-600, -600, 600, 600], dtype=jnp.int16)
    )
    declare_position, denomination, level, vul = _contract(state)
    assert declare_position == 0
    assert denomination == 0
    assert level == 7
    assert vul == 0


def max_action_length_agent(state: State) -> int:
    if (state.last_bid == -1 and state.pass_num != 3) or (
        state.last_bid != -1 and state.pass_num != 2
    ):
        return 35
    elif state.legal_action_mask[36]:
        return 36
    elif state.legal_action_mask[37]:
        return 37
    else:
        return int(state.last_bid) + 1


def test_max_action():
    key = jax.random.PRNGKey(0)
    HASH_TABLE_SAMPLE_KEYS, HASH_TABLE_SAMPLE_VALUES = _load_sample_hash()
    state = init(key)

    for i in range(319):
        if i < 318:
            state = step(
                state,
                max_action_length_agent(state),
                HASH_TABLE_SAMPLE_KEYS,
                HASH_TABLE_SAMPLE_VALUES,
            )
            assert not state.terminated
        else:
            state = step(
                state,
                max_action_length_agent(state),
                HASH_TABLE_SAMPLE_KEYS,
                HASH_TABLE_SAMPLE_VALUES,
            )
            assert state.terminated


def test_pass_out():
    #  player_id: 0 = N, 1 = S, 2 = W, 3 = E
    #   0  3  1  2
    #   N  E  S  W
    #  -----------
    #      P  P  P
    #   P
    key = jax.random.PRNGKey(0)
    HASH_TABLE_SAMPLE_KEYS, HASH_TABLE_SAMPLE_VALUES = _load_sample_hash()
    state = init_by_key(HASH_TABLE_SAMPLE_KEYS[1], key)
    state = state.replace(
        dealer=jnp.int8(1),
        current_player=jnp.int8(3),
        shuffled_players=jnp.array([0, 3, 1, 2], dtype=jnp.int8),
        vul_NS=jnp.bool_(0),
        vul_EW=jnp.bool_(0),
    )

    bidding_history = jnp.full(319, -1, dtype=jnp.int8)
    legal_action_mask = jnp.ones(38, dtype=jnp.bool_)
    legal_action_mask = legal_action_mask.at[36].set(False)
    legal_action_mask = legal_action_mask.at[37].set(False)
    first_denomination_NS = jnp.full(5, -1, dtype=jnp.int8)
    first_denomination_EW = jnp.full(5, -1, dtype=jnp.int8)

    state = step(state, 35, HASH_TABLE_SAMPLE_KEYS, HASH_TABLE_SAMPLE_VALUES)
    #  player_id: 0 = N, 1 = S, 2 = W, 3 = E
    #   0  3  1  2
    #   N  E  S  W
    #  -----------
    #      P

    assert state.turn == 1
    assert state.current_player == 1
    assert _player_position(state.current_player, state) == 2
    assert not state.terminated
    bidding_history = bidding_history.at[0].set(35)
    assert jnp.all(state.bidding_history == bidding_history)
    assert jnp.all(state.legal_action_mask == legal_action_mask)
    assert state.last_bid == -1
    assert _player_position(state.last_bidder, state) == -1
    assert not state.call_x
    assert not state.call_xx
    assert jnp.all(state.first_denomination_NS == first_denomination_NS)
    assert jnp.all(state.first_denomination_EW == first_denomination_EW)
    assert state.pass_num == 1
    assert jnp.all(state.reward == jnp.zeros(4))

    state = step(state, 35, HASH_TABLE_SAMPLE_KEYS, HASH_TABLE_SAMPLE_VALUES)
    #  player_id: 0 = N, 1 = S, 2 = W, 3 = E
    #   0  3  1  2
    #   N  E  S  W
    #  -----------
    #      P  P

    assert state.turn == 2
    assert state.current_player == 2
    assert _player_position(state.current_player, state) == 3
    assert not state.terminated
    bidding_history = bidding_history.at[1].set(35)
    assert jnp.all(state.bidding_history == bidding_history)
    assert jnp.all(state.legal_action_mask == legal_action_mask)
    assert state.last_bid == -1
    assert _player_position(state.last_bidder, state) == -1
    assert not state.call_x
    assert not state.call_xx
    assert jnp.all(state.first_denomination_NS == first_denomination_NS)
    assert jnp.all(state.first_denomination_EW == first_denomination_EW)
    assert state.pass_num == 2
    assert jnp.all(state.reward == jnp.zeros(4))

    state = step(state, 35, HASH_TABLE_SAMPLE_KEYS, HASH_TABLE_SAMPLE_VALUES)
    #  player_id: 0 = N, 1 = S, 2 = W, 3 = E
    #   0  3  1  2
    #   N  E  S  W
    #  -----------
    #      P  P  P

    assert state.turn == 3
    assert state.current_player == 0
    assert _player_position(state.current_player, state) == 0
    assert not state.terminated
    bidding_history = bidding_history.at[2].set(35)
    assert jnp.all(state.bidding_history == bidding_history)
    assert jnp.all(state.legal_action_mask == legal_action_mask)
    assert state.last_bid == -1
    assert _player_position(state.last_bidder, state) == -1
    assert not state.call_x
    assert not state.call_xx
    assert jnp.all(state.first_denomination_NS == first_denomination_NS)
    assert jnp.all(state.first_denomination_EW == first_denomination_EW)
    assert state.pass_num == 3
    assert jnp.all(state.reward == jnp.zeros(4))

    state = step(state, 35, HASH_TABLE_SAMPLE_KEYS, HASH_TABLE_SAMPLE_VALUES)
    #  player_id: 0 = N, 1 = S, 2 = W, 3 = E
    #   0  3  1  2
    #   N  E  S  W
    #  -----------
    #      P  P  P
    #   P

    assert state.terminated == True
    assert state.current_player == -1
    assert state.turn == 3
    bidding_history = bidding_history.at[3].set(35)
    assert jnp.all(state.bidding_history == bidding_history)
    assert jnp.all(state.legal_action_mask == legal_action_mask)
    assert state.last_bid == -1
    assert _player_position(state.last_bidder, state) == -1
    assert not state.call_x
    assert not state.call_xx
    assert jnp.all(state.first_denomination_NS == first_denomination_NS)
    assert jnp.all(state.first_denomination_EW == first_denomination_EW)
    assert state.pass_num == 4
    assert jnp.all(state.reward == jnp.zeros(4))


def test_calc_score():
    # http://web2.acbl.org/documentLibrary/play/InstantScorer.pdf
    #
    # SCORE_TABLE[n][taken_trick_num - bid_level - 6][m]
    # n =   0: 1 minor
    #       1: 1 major
    #       2: 1 NT
    #       3: 2 minor
    #       ...
    # (n = (bid_level - 1) * 3 + (0: minor, 1: major, 2: NT))
    #
    # m =   0: non vul
    #       1: non vul, doubled
    #       2: non vul, redoubled
    #       3: vul
    #       4: vul, double
    #       5: vul, redouble
    # fmt: off
    SCORE_TABLE = (((70, 140, 230, 70, 140, 230),
                    (90, 240, 430, 90, 340, 630),
                    (110, 340, 630, 110, 540, 1030),
                    (130, 440, 830, 130, 740, 1430),
                    (150, 540, 1030, 150, 940, 1830),
                    (170, 640, 1230, 170, 1140, 2230),
                    (190, 740, 1430, 190, 1340, 2630)),
                ((80, 160, 520, 80, 160, 720),
                    (110, 260, 720, 110, 360, 1120),
                    (140, 360, 920, 140, 560, 1520),
                    (170, 460, 1120, 170, 760, 1920),
                    (200, 560, 1320, 200, 960, 2320),
                    (230, 660, 1520, 230, 1160, 2720),
                    (260, 760, 1720, 260, 1360, 3120)),
                ((90, 180, 560, 90, 180, 760),
                    (120, 280, 760, 120, 380, 1160),
                    (150, 380, 960, 150, 580, 1560),
                    (180, 480, 1160, 180, 780, 1960),
                    (210, 580, 1360, 210, 980, 2360),
                    (240, 680, 1560, 240, 1180, 2760),
                    (270, 780, 1760, 270, 1380, 3160)),
                ((90, 180, 560, 90, 180, 760),
                    (110, 280, 760, 110, 380, 1160),
                    (130, 380, 960, 130, 580, 1560),
                    (150, 480, 1160, 150, 780, 1960),
                    (170, 580, 1360, 170, 980, 2360),
                    (190, 680, 1560, 190, 1180, 2760)),
                ((110, 470, 640, 110, 670, 840),
                    (140, 570, 840, 140, 870, 1240),
                    (170, 670, 1040, 170, 1070, 1640),
                    (200, 770, 1240, 200, 1270, 2040),
                    (230, 870, 1440, 230, 1470, 2440),
                    (260, 970, 1640, 260, 1670, 2840)),
                ((120, 490, 680, 120, 690, 880),
                    (150, 590, 880, 150, 890, 1280),
                    (180, 690, 1080, 180, 1090, 1680),
                    (210, 790, 1280, 210, 1290, 2080),
                    (240, 890, 1480, 240, 1490, 2480),
                    (270, 990, 1680, 270, 1690, 2880)),
                ((110, 470, 640, 110, 670, 840),
                    (130, 570, 840, 130, 870, 1240),
                    (150, 670, 1040, 150, 1070, 1640),
                    (170, 770, 1240, 170, 1270, 2040),
                    (190, 870, 1440, 190, 1470, 2440)),
                ((140, 530, 760, 140, 730, 960),
                    (170, 630, 960, 170, 930, 1360),
                    (200, 730, 1160, 200, 1130, 1760),
                    (230, 830, 1360, 230, 1330, 2160),
                    (260, 930, 1560, 260, 1530, 2560)),
                ((400, 550, 800, 600, 750, 1000),
                    (430, 650, 1000, 630, 950, 1400),
                    (460, 750, 1200, 660, 1150, 1800),
                    (490, 850, 1400, 690, 1350, 2200),
                    (520, 950, 1600, 720, 1550, 2600)),
                ((130, 510, 720, 130, 710, 920),
                    (150, 610, 920, 150, 910, 1320),
                    (170, 710, 1120, 170, 1110, 1720),
                    (190, 810, 1320, 190, 1310, 2120)),
                ((420, 590, 880, 620, 790, 1080),
                    (450, 690, 1080, 650, 990, 1480),
                    (480, 790, 1280, 680, 1190, 1880),
                    (510, 890, 1480, 710, 1390, 2280)),
                ((430, 610, 920, 630, 810, 1120),
                    (460, 710, 1120, 660, 1010, 1520),
                    (490, 810, 1320, 690, 1210, 1920),
                    (520, 910, 1520, 720, 1410, 2320)),
                ((400, 550, 800, 600, 750, 1000),
                    (420, 650, 1000, 620, 950, 1400),
                    (440, 750, 1200, 640, 1150, 1800)),
                ((450, 650, 1000, 650, 850, 1200),
                    (480, 750, 1200, 680, 1050, 1600),
                    (510, 850, 1400, 710, 1250, 2000)),
                ((460, 670, 1040, 660, 870, 1240),
                    (490, 770, 1240, 690, 1070, 1640),
                    (520, 870, 1440, 720, 1270, 2040)),
                ((920, 1090, 1380, 1370, 1540, 1830),
                    (940, 1190, 1580, 1390, 1740, 2230)),
                ((980, 1210, 1620, 1430, 1660, 2070),
                    (1010, 1310, 1820, 1460, 1860, 2470)),
                ((990, 1230, 1660, 1440, 1680, 2110),
                    (1020, 1330, 1860, 1470, 1880, 2510)),
                ((1440, 1630, 1960, 2140, 2330, 2660),),
                ((1510, 1770, 2240, 2210, 2470, 2940),),
                ((1520, 1790, 2280, 2220, 2490, 2980),))
    # fmt: on
    # DOWN_TABLE[down_num - 1][n]
    # n =   0: non vul
    #       1: non vul, doubled
    #       2: non vul, redoubled
    #       3: vul
    #       4: vul, double
    #       5: vul, redouble
    # fmt: off
    DOWN_TABLE = ((50, 100, 200, 100, 200, 400),
                (100, 300, 600, 200, 500, 1000,),
                (150, 500, 1000, 300, 800, 1600),
                (200, 800, 1600, 400, 1100, 2200),
                (250, 1100, 2200, 500, 1400, 2800),
                (300, 1400, 2800, 600, 1700, 3400),
                (350, 1700, 3400, 700, 2000, 4000),
                (400, 2000, 4000, 800, 2300, 4600),
                (450, 2300, 4600, 900, 2600, 5200),
                (500, 2600, 5200, 1000, 2900, 5800),
                (550, 2900, 5800, 1100, 3200, 6400),
                (600, 3200, 6400, 1200, 3500, 7000),
                (650, 3500, 7000, 1300, 3800, 7600))
    # fmt: on
    for denomination in range(5):
        for level in range(1, 7):
            for vul in range(2):
                for call_x in range(2):
                    for call_xx in range(2):
                        for trick in range(14):
                            overtrick_num = trick - level - 6
                            m = 0
                            if vul:
                                m += 3
                            if call_xx:
                                m += 2
                            elif call_x:
                                m += 1

                            if overtrick_num < 0:
                                down_num = -overtrick_num
                                expected_score = -DOWN_TABLE[down_num - 1][m]
                            else:
                                n = (level - 1) * 3
                                if denomination == 4:
                                    n += 2
                                elif 2 <= denomination <= 3:
                                    n += 1

                                expected_score = SCORE_TABLE[n][overtrick_num][
                                    m
                                ]

                            actural_score = _calc_score(
                                jnp.int16(denomination),
                                jnp.int16(level),
                                jnp.int16(vul),
                                jnp.int16(call_x),
                                jnp.int16(call_xx),
                                jnp.int16(trick),
                            )
                            assert actural_score == expected_score
                            assert actural_score.shape == ()

    # 1NT, 11 tricks
    score = _calc_score(4, 1, 0, 0, 0, 11)
    assert score.shape == ()
    assert score == 210
    # 1NT, 11 tricks, vul
    assert _calc_score(4, 1, 1, 0, 0, 11) == 210
    # 1NTx, 11 tricks
    assert _calc_score(4, 1, 0, 1, 0, 11) == 580
    # 1NTx, 11 tricks, vul
    assert _calc_score(4, 1, 1, 1, 0, 11) == 980
    # 1NTxx, 11 tricks
    assert _calc_score(4, 1, 0, 0, 1, 11) == 1360
    # 1NTxx, 11 tricks, vul
    assert _calc_score(4, 1, 1, 0, 1, 11) == 2360
    # 2H, 6 tricks, vul
    score = _calc_score(2, 2, 1, 0, 0, 6)
    assert score.shape == ()
    assert score == -200
    # 6Dxx, 13 tricks
    assert _calc_score(1, 6, 0, 0, 1, 13) == 1580
    # 4Sx, 10 tricks,
    assert _calc_score(3, 4, 0, 1, 0, 10) == 590
    # 7D, 13 tricks
    assert _calc_score(1, 7, 0, 0, 0, 13) == 1440
    # 6Dxx, 13 tricks
    assert _calc_score(1, 6, 0, 0, 1, 13) == 1580


def test_to_binary():
    x = jnp.arange(52, dtype=jnp.int8)[::-1].reshape((4, 13)) % 4
    y = _to_binary(x)

    assert jnp.all(
        y
        == jnp.array([60003219, 38686286, 20527417, 15000804], dtype=jnp.int32)
    )

    x = jnp.arange(52, dtype=jnp.int8).reshape((4, 13)) // 13
    y = _to_binary(x)
    assert jnp.all(
        y == jnp.array([0, 22369621, 44739242, 67108863], dtype=jnp.int32)
    )

    x = jnp.arange(52, dtype=jnp.int8)[::-1].reshape((4, 13)) // 13
    y = _to_binary(x)
    assert jnp.all(
        y == jnp.array([67108863, 44739242, 22369621, 0], dtype=jnp.int32)
    )


def test_state_to_pbn():
    key = jax.random.PRNGKey(0)
    state = init(key)
    state = state.replace(hand=jnp.arange(52, dtype=jnp.int8))
    pbn = _state_to_pbn(state)
    assert (
        pbn
        == "N:AKQJT98765432... .AKQJT98765432.. ..AKQJT98765432. ...AKQJT98765432"
    )

    state = state.replace(hand=jnp.arange(52, dtype=jnp.int8)[::-1])
    pbn = _state_to_pbn(state)
    assert (
        pbn
        == "N:...AKQJT98765432 ..AKQJT98765432. .AKQJT98765432.. AKQJT98765432..."
    )
    # fmt: off
    hand = jnp.array([
        12,9,8,6,3,2,13,24,22,16,15,36,45,
        10,7,4,21,37,31,51,50,49,47,43,41,40,
        11,1,25,23,19,18,17,35,34,33,48,44,42,
        0,5,20,14,26,38,32,30,29,28,27,39,46,
        ]
    )
    # fmt: on
    state = state.replace(hand=hand)
    pbn = _state_to_pbn(state)
    print(pbn)
    assert (
        pbn
        == "N:KT9743.AQT43.J.7 J85.9.Q6.KQJ9532 Q2.KJ765.T98.T64 A6.82.AK75432.A8"
    )


def test_state_to_key():
    rng = jax.random.PRNGKey(0)
    state = init(rng)

    state = state.replace(hand=jnp.arange(52, dtype=jnp.int8))
    key = _state_to_key(state)
    assert jnp.all(
        key == jnp.array([0, 22369621, 44739242, 67108863], dtype=jnp.int32)
    )

    state = state.replace(hand=jnp.arange(52, dtype=jnp.int8)[::-1])
    key = _state_to_key(state)
    assert jnp.all(
        key == jnp.array([67108863, 44739242, 22369621, 0], dtype=jnp.int32)
    )

    # fmt: off
    hand = jnp.array([
        12,9,8,6,3,2,13,24,22,16,15,36,45,
        10,7,4,21,37,31,51,50,49,47,43,41,40,
        11,1,25,23,19,18,17,35,34,33,48,44,42,
        0,5,20,14,26,38,32,30,29,28,27,39,46,
        ]
    )
    # fmt: on
    state = state.replace(hand=hand)
    key = _state_to_key(state)
    assert jnp.all(
        key
        == jnp.array([58835992, 12758306, 67074695, 56200597], dtype=jnp.int32)
    )


def test_key_to_hand():
    key = jnp.array([0, 22369621, 44739242, 67108863], dtype=jnp.int32)
    hand = _key_to_hand(key)
    assert jnp.all(hand == jnp.arange(52, dtype=jnp.int8))

    key = jnp.array([67108863, 44739242, 22369621, 0], dtype=jnp.int32)
    hand = _key_to_hand(key)
    correct_hand = jnp.arange(52, dtype=jnp.int8)[::-1]
    sorted_correct_hand = jnp.concatenate(
        [
            jnp.sort(correct_hand[:13]),
            jnp.sort(correct_hand[13:26]),
            jnp.sort(correct_hand[26:39]),
            jnp.sort(correct_hand[39:]),
        ]
    ).reshape(-1)
    assert jnp.all(hand == sorted_correct_hand)

    key = jnp.array([58835992, 12758306, 67074695, 56200597], dtype=jnp.int32)
    hand = _key_to_hand(key)
    # fmt: off
    correct_hand = jnp.array([
        12,9,8,6,3,2,13,24,22,16,15,36,45,
        10,7,4,21,37,31,51,50,49,47,43,41,40,
        11,1,25,23,19,18,17,35,34,33,48,44,42,
        0,5,20,14,26,38,32,30,29,28,27,39,46,
        ]
    )
    # fmt: on
    sorted_correct_hand = jnp.concatenate(
        [
            jnp.sort(correct_hand[:13]),
            jnp.sort(correct_hand[13:26]),
            jnp.sort(correct_hand[26:39]),
            jnp.sort(correct_hand[39:]),
        ]
    ).reshape(-1)
    print(hand)
    assert jnp.all(hand == sorted_correct_hand)


def test_state_to_key_cycle():
    rng = jax.random.PRNGKey(0)
    # state => key => st
    for _ in range(1000):
        rng1, rng2 = jax.random.split(rng)
        state = init(rng2)
        sorted_hand = jnp.concatenate(
            [
                jnp.sort(state.hand[:13]),
                jnp.sort(state.hand[13:26]),
                jnp.sort(state.hand[26:39]),
                jnp.sort(state.hand[39:]),
            ]
        ).reshape(-1)
        key = _state_to_key(state)
        reconst_hand = _key_to_hand(key)
        assert jnp.all(sorted_hand == reconst_hand)


def test_calcurate_dds_tricks():
    HASH_TABLE_SAMPLE_KEYS, HASH_TABLE_SAMPLE_VALUES = _load_sample_hash()
    samples = []
    with open("tests/assets/contractbridge-ddstable-sample100.csv", "r") as f:
        reader = csv.reader(f, delimiter=",")
        for i in reader:
            samples.append([i[0], np.array(i[1:]).astype(np.int8)])
    key = jax.random.PRNGKey(0)
    for i in range(len(HASH_TABLE_SAMPLE_KEYS)):
        key, subkey = jax.random.split(key)
        state = init(subkey)
        state = state.replace(hand=_key_to_hand(HASH_TABLE_SAMPLE_KEYS[i]))
        dds_tricks = _calculate_dds_tricks(
            state, HASH_TABLE_SAMPLE_KEYS, HASH_TABLE_SAMPLE_VALUES
        )
        # sample dataから、作成したhash tableを用いて、ddsの結果を計算
        # その結果とsample dataが一致しているか確認
        assert jnp.all(dds_tricks == samples[i][1])


def test_value_to_dds_tricks():
    value = jnp.array([4160, 904605, 4160, 904605])
    # fmt: off
    assert jnp.all(
        _value_to_dds_tricks(value)
        == jnp.array(
            [ 0,  1,  0,  4,  0, 13, 12, 13,  9, 13,  0,  1,  0,  4,  0, 13, 12,
           13,  9, 13],
            dtype=jnp.int8,
        )
    )
    # fmt: on


def to_value(sample: list) -> jnp.ndarray:
    """Convert sample to value
    >>> sample = ['0', '1', '0', '4', '0', '13', '12', '13', '9', '13', '0', '1', '0', '4', '0', '13', '12', '13', '9', '13']
    >>> to_value(sample)
    Array([  4160, 904605,   4160, 904605], dtype=int32)
    """
    jnp_sample = jnp.array([int(s) for s in sample], dtype=np.int8).reshape(
        4, 5
    )
    return to_binary(jnp_sample)


def to_binary(x) -> jnp.ndarray:
    """Convert dds information to value
    >>> jnp.array([16**i for i in range(5)], dtype=jnp.int32)[::-1]
    Array([65536,  4096,   256,    16,     1], dtype=int32)
    >>> x = jnp.arange(20, dtype=jnp.int32).reshape(4, 5) % 14
    >>> to_binary(x)
    Array([  4660, 354185, 703696,  74565], dtype=int32)
    """
    bases = jnp.array([16**i for i in range(5)], dtype=jnp.int32)[::-1]
    return (x * bases).sum(axis=1)  # shape = (4, )


def make_hash_table(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """make key and value of hash from samples"""
    samples = []
    with open(csv_path, "r") as f:
        reader = csv.reader(f, delimiter=",")
        for i in reader:
            samples.append(i)
    keys = []
    values = []
    for sample in samples:
        keys.append(_pbn_to_key(sample[0]))
        values.append(to_value(sample[1:]))
    return np.array(keys), np.array(values)
