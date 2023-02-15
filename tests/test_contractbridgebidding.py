import csv
from typing import Tuple

import numpy as np

from pgx.contractbridgebidding import (
    ContractBridgeBiddingState,
    _calculate_dds_tricks,
    _key_to_hand,
    _load_sample_hash,
    _pbn_to_key,
    _state_to_key,
    _state_to_pbn,
    _to_binary,
    init,
    step,
)


def test_init():
    curr_player, state = init()
    assert state.last_bid == -1
    assert state.last_bidder == -1
    assert not state.call_x
    assert not state.call_xx
    assert not state.pass_num
    assert state.curr_player == state.dealer
    assert state.legal_action_mask[:-2].all()
    assert not state.legal_action_mask[-2:].all()


def test_step():
    #   N  E  S  W
    #  -----------
    #      P  P  P
    #  1C 2S  X  P
    #   P XX  P 5H
    #  5S  P 6C  X
    #  XX  P 7C  P
    #   P  P
    _, state = init()
    state.dealer[0] = 1
    state.curr_player[0] = 1

    bidding_history = np.full(319, -1, dtype=np.int8)
    legal_action_mask = np.ones(38, dtype=np.bool8)
    legal_action_mask[-2:] = 0
    first_denomination_NS = np.full(5, -1, dtype=np.int8)
    first_denomination_EW = np.full(5, -1, dtype=np.int8)

    curr_player, state, rewards = step(state, 35)
    #   N  E  S  W
    #  -----------
    #      P

    assert state.turn == 1
    assert curr_player == 2
    assert not state.terminated
    bidding_history[0] = 35
    assert np.all(state.bidding_history == bidding_history)
    assert np.all(state.legal_action_mask == legal_action_mask)
    assert state.last_bid == -1
    assert state.last_bidder == -1
    assert not state.call_x
    assert not state.call_xx
    assert np.all(state.first_denomination_NS == first_denomination_NS)
    assert np.all(state.first_denomination_EW == first_denomination_EW)
    assert state.pass_num == 1

    curr_player, state, rewards = step(state, 35)
    #   N  E  S  W
    #  -----------
    #      P  P

    assert state.turn == 2
    assert curr_player == 3
    assert not state.terminated
    bidding_history[1] = 35
    assert np.all(state.bidding_history == bidding_history)
    assert np.all(state.legal_action_mask == legal_action_mask)
    assert state.last_bid == -1
    assert state.last_bidder == -1
    assert not state.call_x
    assert not state.call_xx
    assert np.all(state.first_denomination_NS == first_denomination_NS)
    assert np.all(state.first_denomination_EW == first_denomination_EW)
    assert state.pass_num == 2

    curr_player, state, rewards = step(state, 35)
    #   N  E  S  W
    #  -----------
    #      P  P  P

    assert state.turn == 3
    assert curr_player == 0
    assert not state.terminated
    bidding_history[2] = 35
    assert np.all(state.bidding_history == bidding_history)
    assert np.all(state.legal_action_mask == legal_action_mask)
    assert state.last_bid == -1
    assert state.last_bidder == -1
    assert not state.call_x
    assert not state.call_xx
    assert np.all(state.first_denomination_NS == first_denomination_NS)
    assert np.all(state.first_denomination_EW == first_denomination_EW)
    assert state.pass_num == 3

    curr_player, state, rewards = step(state, 0)
    #   N  E  S  W
    #  -----------
    #      P  P  P
    #  1C
    first_denomination_NS = np.array([0, -1, -1, -1, -1])
    first_denomination_EW = np.array([-1, -1, -1, -1, -1])

    assert state.turn == 4
    assert curr_player == 1
    assert not state.terminated
    bidding_history[3] = 0
    assert np.all(state.bidding_history == bidding_history)
    legal_action_mask[0] = 0
    legal_action_mask[36] = 1
    assert np.all(state.legal_action_mask == legal_action_mask)
    assert state.last_bid == 0
    assert state.last_bidder == 0
    assert not state.call_x
    assert not state.call_xx
    assert np.all(state.first_denomination_NS == first_denomination_NS)
    assert np.all(state.first_denomination_EW == first_denomination_EW)
    assert state.pass_num == 0

    curr_player, state, rewards = step(state, 8)
    #   N  E  S  W
    #  -----------
    #      P  P  P
    #  1C 2S
    first_denomination_NS = np.array([0, -1, -1, -1, -1])
    first_denomination_EW = np.array([-1, -1, -1, 1, -1])

    assert state.turn == 5
    assert curr_player == 2
    assert not state.terminated
    bidding_history[4] = 8
    assert np.all(state.bidding_history == bidding_history)
    legal_action_mask[0:9] = 0
    legal_action_mask[36] = 1
    assert np.all(state.legal_action_mask == legal_action_mask)
    assert state.last_bid == 8
    assert state.last_bidder == 1
    assert not state.call_x
    assert not state.call_xx
    assert np.all(state.first_denomination_NS == first_denomination_NS)
    assert np.all(state.first_denomination_EW == first_denomination_EW)
    assert state.pass_num == 0

    curr_player, state, rewards = step(state, 36)
    #   N  E  S  W
    #  -----------
    #      P  P  P
    #  1C 2S  X
    first_denomination_NS = np.array([0, -1, -1, -1, -1])
    first_denomination_EW = np.array([-1, -1, -1, 1, -1])

    assert state.turn == 6
    assert curr_player == 3
    assert not state.terminated
    bidding_history[5] = 36
    assert np.all(state.bidding_history == bidding_history)
    legal_action_mask[0:9] = 0
    legal_action_mask[36] = 0
    legal_action_mask[37] = 1
    assert np.all(state.legal_action_mask == legal_action_mask)
    assert state.last_bid == 8
    assert state.last_bidder == 1
    assert state.call_x
    assert not state.call_xx
    assert np.all(state.first_denomination_NS == first_denomination_NS)
    assert np.all(state.first_denomination_EW == first_denomination_EW)
    assert state.pass_num == 0

    curr_player, state, rewards = step(state, 35)
    #   N  E  S  W
    #  -----------
    #      P  P  P
    #  1C 2S  X  P
    first_denomination_NS = np.array([0, -1, -1, -1, -1])
    first_denomination_EW = np.array([-1, -1, -1, 1, -1])

    assert state.turn == 7
    assert curr_player == 0
    assert not state.terminated
    bidding_history[6] = 35
    assert np.all(state.bidding_history == bidding_history)
    legal_action_mask = np.ones(38, dtype=np.bool8)
    legal_action_mask[0:9] = 0
    legal_action_mask[36] = 0
    legal_action_mask[37] = 0
    assert np.all(state.legal_action_mask == legal_action_mask)
    assert state.last_bid == 8
    assert state.last_bidder == 1
    assert state.call_x
    assert not state.call_xx
    assert np.all(state.first_denomination_NS == first_denomination_NS)
    assert np.all(state.first_denomination_EW == first_denomination_EW)
    assert state.pass_num == 1

    curr_player, state, rewards = step(state, 35)
    #   N  E  S  W
    #  -----------
    #      P  P  P
    #  1C 2S  X  P
    #   P
    first_denomination_NS = np.array([0, -1, -1, -1, -1])
    first_denomination_EW = np.array([-1, -1, -1, 1, -1])

    assert state.turn == 8
    assert curr_player == 1
    assert not state.terminated
    bidding_history[7] = 35
    assert np.all(state.bidding_history == bidding_history)
    legal_action_mask = np.ones(38, dtype=np.bool8)
    legal_action_mask[0:9] = 0
    legal_action_mask[36] = 0
    legal_action_mask[37] = 1
    assert np.all(state.legal_action_mask == legal_action_mask)
    assert state.last_bid == 8
    assert state.last_bidder == 1
    assert state.call_x
    assert not state.call_xx
    assert np.all(state.first_denomination_NS == first_denomination_NS)
    assert np.all(state.first_denomination_EW == first_denomination_EW)
    assert state.pass_num == 2

    curr_player, state, rewards = step(state, 37)
    #   N  E  S  W
    #  -----------
    #      P  P  P
    #  1C 2S  X  P
    #   P XX
    first_denomination_NS = np.array([0, -1, -1, -1, -1])
    first_denomination_EW = np.array([-1, -1, -1, 1, -1])

    assert state.turn == 9
    assert curr_player == 2
    assert not state.terminated
    bidding_history[8] = 37
    assert np.all(state.bidding_history == bidding_history)
    legal_action_mask = np.ones(38, dtype=np.bool8)
    legal_action_mask[0:9] = 0
    legal_action_mask[36] = 0
    legal_action_mask[37] = 0
    assert np.all(state.legal_action_mask == legal_action_mask)
    assert state.last_bid == 8
    assert state.last_bidder == 1
    assert state.call_x
    assert state.call_xx
    assert np.all(state.first_denomination_NS == first_denomination_NS)
    assert np.all(state.first_denomination_EW == first_denomination_EW)
    assert state.pass_num == 0

    curr_player, state, rewards = step(state, 35)
    #   N  E  S  W
    #  -----------
    #      P  P  P
    #  1C 2S  X  P
    #   P XX  P
    first_denomination_NS = np.array([0, -1, -1, -1, -1])
    first_denomination_EW = np.array([-1, -1, -1, 1, -1])

    assert state.turn == 10
    assert curr_player == 3
    assert not state.terminated
    bidding_history[9] = 35
    assert np.all(state.bidding_history == bidding_history)
    legal_action_mask = np.ones(38, dtype=np.bool8)
    legal_action_mask[0:9] = 0
    legal_action_mask[36] = 0
    legal_action_mask[37] = 0
    assert np.all(state.legal_action_mask == legal_action_mask)
    assert state.last_bid == 8
    assert state.last_bidder == 1
    assert state.call_x
    assert state.call_xx
    assert np.all(state.first_denomination_NS == first_denomination_NS)
    assert np.all(state.first_denomination_EW == first_denomination_EW)
    assert state.pass_num == 1

    curr_player, state, rewards = step(state, 22)
    #   N  E  S  W
    #  -----------
    #      P  P  P
    #  1C 2S  X  P
    #   P XX  P 5H
    first_denomination_NS = np.array([0, -1, -1, -1, -1])
    first_denomination_EW = np.array([-1, -1, 3, 1, -1])

    assert state.turn == 11
    assert curr_player == 0
    assert not state.terminated
    bidding_history[10] = 22
    assert np.all(state.bidding_history == bidding_history)
    legal_action_mask = np.ones(38, dtype=np.bool8)
    legal_action_mask[0:23] = 0
    legal_action_mask[36] = 1
    legal_action_mask[37] = 0
    assert np.all(state.legal_action_mask == legal_action_mask)
    assert state.last_bid == 22
    assert state.last_bidder == 3
    assert not state.call_x
    assert not state.call_xx
    assert np.all(state.first_denomination_NS == first_denomination_NS)
    assert np.all(state.first_denomination_EW == first_denomination_EW)
    assert state.pass_num == 0

    curr_player, state, rewards = step(state, 23)
    #   N  E  S  W
    #  -----------
    #      P  P  P
    #  1C 2S  X  P
    #   P XX  P 5H
    #  5S
    first_denomination_NS = np.array([0, -1, -1, 0, -1])
    first_denomination_EW = np.array([-1, -1, 3, 1, -1])

    assert state.turn == 12
    assert curr_player == 1
    assert not state.terminated
    bidding_history[11] = 23
    assert np.all(state.bidding_history == bidding_history)
    legal_action_mask = np.ones(38, dtype=np.bool8)
    legal_action_mask[0:24] = 0
    legal_action_mask[36] = 1
    legal_action_mask[37] = 0
    assert np.all(state.legal_action_mask == legal_action_mask)
    assert state.last_bid == 23
    assert state.last_bidder == 0
    assert not state.call_x
    assert not state.call_xx
    assert np.all(state.first_denomination_NS == first_denomination_NS)
    assert np.all(state.first_denomination_EW == first_denomination_EW)
    assert state.pass_num == 0

    curr_player, state, rewards = step(state, 35)
    #   N  E  S  W
    #  -----------
    #      P  P  P
    #  1C 2S  X  P
    #   P XX  P 5H
    #  5S  P
    first_denomination_NS = np.array([0, -1, -1, 0, -1])
    first_denomination_EW = np.array([-1, -1, 3, 1, -1])

    assert state.turn == 13
    assert curr_player == 2
    assert not state.terminated
    bidding_history[12] = 35
    assert np.all(state.bidding_history == bidding_history)
    legal_action_mask = np.ones(38, dtype=np.bool8)
    legal_action_mask[0:24] = 0
    legal_action_mask[36] = 0
    legal_action_mask[37] = 0
    assert np.all(state.legal_action_mask == legal_action_mask)
    assert state.last_bid == 23
    assert state.last_bidder == 0
    assert not state.call_x
    assert not state.call_xx
    assert np.all(state.first_denomination_NS == first_denomination_NS)
    assert np.all(state.first_denomination_EW == first_denomination_EW)
    assert state.pass_num == 1

    curr_player, state, rewards = step(state, 25)
    #   N  E  S  W
    #  -----------
    #      P  P  P
    #  1C 2S  X  P
    #   P XX  P 5H
    #  5S  P 6C
    first_denomination_NS = np.array([0, -1, -1, 0, -1])
    first_denomination_EW = np.array([-1, -1, 3, 1, -1])

    assert state.turn == 14
    assert curr_player == 3
    assert not state.terminated
    bidding_history[13] = 25
    assert np.all(state.bidding_history == bidding_history)
    legal_action_mask = np.ones(38, dtype=np.bool8)
    legal_action_mask[0:26] = 0
    legal_action_mask[36] = 1
    legal_action_mask[37] = 0
    assert np.all(state.legal_action_mask == legal_action_mask)
    assert state.last_bid == 25
    assert state.last_bidder == 2
    assert not state.call_x
    assert not state.call_xx
    assert np.all(state.first_denomination_NS == first_denomination_NS)
    assert np.all(state.first_denomination_EW == first_denomination_EW)
    assert state.pass_num == 0

    curr_player, state, rewards = step(state, 36)
    #   N  E  S  W
    #  -----------
    #      P  P  P
    #  1C 2S  X  P
    #   P XX  P 5H
    #  5S  P 6C  X
    first_denomination_NS = np.array([0, -1, -1, 0, -1])
    first_denomination_EW = np.array([-1, -1, 3, 1, -1])

    assert state.turn == 15
    assert curr_player == 0
    assert not state.terminated
    bidding_history[14] = 36
    assert np.all(state.bidding_history == bidding_history)
    legal_action_mask = np.ones(38, dtype=np.bool8)
    legal_action_mask[0:26] = 0
    legal_action_mask[36] = 0
    legal_action_mask[37] = 1
    assert np.all(state.legal_action_mask == legal_action_mask)
    assert state.last_bid == 25
    assert state.last_bidder == 2
    assert state.call_x
    assert not state.call_xx
    assert np.all(state.first_denomination_NS == first_denomination_NS)
    assert np.all(state.first_denomination_EW == first_denomination_EW)
    assert state.pass_num == 0

    curr_player, state, rewards = step(state, 37)
    #   N  E  S  W
    #  -----------
    #      P  P  P
    #  1C 2S  X  P
    #   P XX  P 5H
    #  5S  P 6C  X
    #  XX
    first_denomination_NS = np.array([0, -1, -1, 0, -1])
    first_denomination_EW = np.array([-1, -1, 3, 1, -1])

    assert state.turn == 16
    assert curr_player == 1
    assert not state.terminated
    bidding_history[15] = 37
    assert np.all(state.bidding_history == bidding_history)
    legal_action_mask = np.ones(38, dtype=np.bool8)
    legal_action_mask[0:26] = 0
    legal_action_mask[36] = 0
    legal_action_mask[37] = 0
    assert np.all(state.legal_action_mask == legal_action_mask)
    assert state.last_bid == 25
    assert state.last_bidder == 2
    assert state.call_x
    assert state.call_xx
    assert np.all(state.first_denomination_NS == first_denomination_NS)
    assert np.all(state.first_denomination_EW == first_denomination_EW)
    assert state.pass_num == 0

    curr_player, state, rewards = step(state, 35)
    #   N  E  S  W
    #  -----------
    #      P  P  P
    #  1C 2S  X  P
    #   P XX  P 5H
    #  5S  P 6C  X
    #  XX  P
    first_denomination_NS = np.array([0, -1, -1, 0, -1])
    first_denomination_EW = np.array([-1, -1, 3, 1, -1])

    assert state.turn == 17
    assert curr_player == 2
    assert not state.terminated
    bidding_history[16] = 35
    assert np.all(state.bidding_history == bidding_history)
    legal_action_mask = np.ones(38, dtype=np.bool8)
    legal_action_mask[0:26] = 0
    legal_action_mask[36] = 0
    legal_action_mask[37] = 0
    assert np.all(state.legal_action_mask == legal_action_mask)
    assert state.last_bid == 25
    assert state.last_bidder == 2
    assert state.call_x
    assert state.call_xx
    assert np.all(state.first_denomination_NS == first_denomination_NS)
    assert np.all(state.first_denomination_EW == first_denomination_EW)
    assert state.pass_num == 1

    curr_player, state, rewards = step(state, 30)
    #   N  E  S  W
    #  -----------
    #      P  P  P
    #  1C 2S  X  P
    #   P XX  P 5H
    #  5S  P 6C  X
    #  XX  P 7C
    first_denomination_NS = np.array([0, -1, -1, 0, -1])
    first_denomination_EW = np.array([-1, -1, 3, 1, -1])

    assert state.turn == 18
    assert curr_player == 3
    assert not state.terminated
    bidding_history[17] = 30
    assert np.all(state.bidding_history == bidding_history)
    legal_action_mask = np.ones(38, dtype=np.bool8)
    legal_action_mask[0:31] = 0
    legal_action_mask[36] = 1
    legal_action_mask[37] = 0
    assert np.all(state.legal_action_mask == legal_action_mask)
    assert state.last_bid == 30
    assert state.last_bidder == 2
    assert not state.call_x
    assert not state.call_xx
    assert np.all(state.first_denomination_NS == first_denomination_NS)
    assert np.all(state.first_denomination_EW == first_denomination_EW)
    assert state.pass_num == 0

    curr_player, state, rewards = step(state, 35)
    #   N  E  S  W
    #  -----------
    #      P  P  P
    #  1C 2S  X  P
    #   P XX  P 5H
    #  5S  P 6C  X
    #  XX  P 7C  P
    first_denomination_NS = np.array([0, -1, -1, 0, -1])
    first_denomination_EW = np.array([-1, -1, 3, 1, -1])

    assert state.turn == 19
    assert curr_player == 0
    assert not state.terminated
    bidding_history[18] = 35
    assert np.all(state.bidding_history == bidding_history)
    legal_action_mask = np.ones(38, dtype=np.bool8)
    legal_action_mask[0:31] = 0
    legal_action_mask[36] = 0
    legal_action_mask[37] = 0
    assert np.all(state.legal_action_mask == legal_action_mask)
    assert state.last_bid == 30
    assert state.last_bidder == 2
    assert not state.call_x
    assert not state.call_xx
    assert np.all(state.first_denomination_NS == first_denomination_NS)
    assert np.all(state.first_denomination_EW == first_denomination_EW)
    assert state.pass_num == 1

    curr_player, state, rewards = step(state, 35)
    #   N  E  S  W
    #  -----------
    #      P  P  P
    #  1C 2S  X  P
    #   P XX  P 5H
    #  5S  P 6C  X
    #  XX  P 7C  P
    #   P
    first_denomination_NS = np.array([0, -1, -1, 0, -1])
    first_denomination_EW = np.array([-1, -1, 3, 1, -1])

    assert state.turn == 20
    assert curr_player == 1
    assert not state.terminated
    bidding_history[19] = 35
    assert np.all(state.bidding_history == bidding_history)
    legal_action_mask = np.ones(38, dtype=np.bool8)
    legal_action_mask[0:31] = 0
    legal_action_mask[36] = 1
    legal_action_mask[37] = 0
    assert np.all(state.legal_action_mask == legal_action_mask)
    assert state.last_bid == 30
    assert state.last_bidder == 2
    assert not state.call_x
    assert not state.call_xx
    assert np.all(state.first_denomination_NS == first_denomination_NS)
    assert np.all(state.first_denomination_EW == first_denomination_EW)
    assert state.pass_num == 2

    curr_player, state, rewards = step(state, 35)
    #   N  E  S  W
    #  -----------
    #      P  P  P
    #  1C 2S  X  P
    #   P XX  P 5H
    #  5S  P 6C  X
    #  XX  P 7C  P
    #   P  P
    # Passが3回続いたので終了
    # state.terminated = True
    # state.curr_player = -1
    first_denomination_NS = np.array([0, -1, -1, 0, -1])
    first_denomination_EW = np.array([-1, -1, 3, 1, -1])

    assert state.turn == 20
    assert curr_player == -1
    assert state.terminated
    bidding_history[20] = 35
    assert np.all(state.bidding_history == bidding_history)
    legal_action_mask = np.ones(38, dtype=np.bool8)
    legal_action_mask[0:31] = 0
    legal_action_mask[36] = 1
    legal_action_mask[37] = 0
    assert np.all(state.legal_action_mask == legal_action_mask)
    assert state.last_bid == 30
    assert state.last_bidder == 2
    assert not state.call_x
    assert not state.call_xx
    assert np.all(state.first_denomination_NS == first_denomination_NS)
    assert np.all(state.first_denomination_EW == first_denomination_EW)
    assert state.pass_num == 3


def max_action_length_agent(state: ContractBridgeBiddingState):
    if (state.last_bid == -1 and state.pass_num != 3) or (
        state.last_bid != -1 and state.pass_num != 2
    ):
        return 35
    elif state.legal_action_mask[36]:
        return 36
    elif state.legal_action_mask[37]:
        return 37
    else:
        return state.last_bid + 1


def test_max_action():
    _, state = init()
    state.turn = np.int16(0)
    state.terminated = np.bool8(0)
    state.bidding_history = np.full(319, -1, dtype=np.int8)
    state.legal_action_mask = np.ones(38, dtype=np.bool8)
    state.legal_action_mask[-2:] = 0
    state.first_denomination_NS = np.full(5, -1, dtype=np.int8)
    state.first_denomination_EW = np.full(5, -1, dtype=np.int8)
    state.call_x = np.bool8(0)
    state.call_xx = np.bool8(0)
    state.pass_num = np.bool8(0)
    state.last_bid = np.int8(-1)
    state.last_bidder = np.int8(-1)

    for i in range(319):
        if i < 318:
            _, state, _ = step(state, max_action_length_agent(state))
            assert not state.terminated
        else:
            _, state, _ = step(state, max_action_length_agent(state))
            assert state.terminated


def test_to_binary():
    x = np.arange(52, dtype=np.int8)[::-1].reshape((4, 13)) % 4
    y = _to_binary(x)

    assert np.all(
        y == np.array([60003219, 38686286, 20527417, 15000804], dtype=np.int32)
    )

    x = np.arange(52, dtype=np.int8).reshape((4, 13)) // 13
    y = _to_binary(x)
    assert np.all(
        y == np.array([0, 22369621, 44739242, 67108863], dtype=np.int32)
    )

    x = np.arange(52, dtype=np.int8)[::-1].reshape((4, 13)) // 13
    y = _to_binary(x)
    assert np.all(
        y == np.array([67108863, 44739242, 22369621, 0], dtype=np.int32)
    )


def test_state_to_pbn():
    _, state = init()
    state.hand = np.arange(52, dtype=np.int8)
    pbn = _state_to_pbn(state)
    assert (
        pbn
        == "N:AKQJT98765432... .AKQJT98765432.. ..AKQJT98765432. ...AKQJT98765432"
    )
    state.hand = np.arange(52, dtype=np.int8)[::-1]
    pbn = _state_to_pbn(state)
    assert (
        pbn
        == "N:...AKQJT98765432 ..AKQJT98765432. .AKQJT98765432.. AKQJT98765432..."
    )
    # fmt: off
    state.hand = np.array([
        12,9,8,6,3,2,13,24,22,16,15,36,45,
        10,7,4,21,37,31,51,50,49,47,43,41,40,
        11,1,25,23,19,18,17,35,34,33,48,44,42,
        0,5,20,14,26,38,32,30,29,28,27,39,46,
        ]
    )
    # fmt: on
    pbn = _state_to_pbn(state)
    print(pbn)
    assert (
        pbn
        == "N:KT9743.AQT43.J.7 J85.9.Q6.KQJ9532 Q2.KJ765.T98.T64 A6.82.AK75432.A8"
    )


def test_state_to_key():
    _, state = init()
    state.hand = np.arange(52, dtype=np.int8)
    key = _state_to_key(state)
    assert np.all(
        key == np.array([0, 22369621, 44739242, 67108863], dtype=np.int32)
    )

    state.hand = np.arange(52, dtype=np.int8)[::-1]
    key = _state_to_key(state)
    assert np.all(
        key == np.array([67108863, 44739242, 22369621, 0], dtype=np.int32)
    )

    # fmt: off
    state.hand = np.array([
        12,9,8,6,3,2,13,24,22,16,15,36,45,
        10,7,4,21,37,31,51,50,49,47,43,41,40,
        11,1,25,23,19,18,17,35,34,33,48,44,42,
        0,5,20,14,26,38,32,30,29,28,27,39,46,
        ]
    )
    # fmt: on
    key = _state_to_key(state)
    print(key)
    assert np.all(
        key
        == np.array([58835992, 12758306, 67074695, 56200597], dtype=np.int32)
    )


def test_key_to_hand():
    key = np.array([0, 22369621, 44739242, 67108863], dtype=np.int32)
    hand = _key_to_hand(key)
    assert np.all(hand == np.arange(52, dtype=np.int8))

    key = np.array([67108863, 44739242, 22369621, 0], dtype=np.int32)
    hand = _key_to_hand(key)
    correct_hand = np.arange(52, dtype=np.int8)[::-1]
    sorted_correct_hand = np.concatenate(
        [
            np.sort(correct_hand[:13]),
            np.sort(correct_hand[13:26]),
            np.sort(correct_hand[26:39]),
            np.sort(correct_hand[39:]),
        ]
    ).reshape(-1)
    assert np.all(hand == sorted_correct_hand)

    key = np.array([58835992, 12758306, 67074695, 56200597], dtype=np.int32)
    hand = _key_to_hand(key)
    # fmt: off
    correct_hand = np.array([
        12,9,8,6,3,2,13,24,22,16,15,36,45,
        10,7,4,21,37,31,51,50,49,47,43,41,40,
        11,1,25,23,19,18,17,35,34,33,48,44,42,
        0,5,20,14,26,38,32,30,29,28,27,39,46,
        ]
    )
    # fmt: on
    sorted_correct_hand = np.concatenate(
        [
            np.sort(correct_hand[:13]),
            np.sort(correct_hand[13:26]),
            np.sort(correct_hand[26:39]),
            np.sort(correct_hand[39:]),
        ]
    ).reshape(-1)
    print(hand)
    assert np.all(hand == sorted_correct_hand)


def test_state_to_key_cycle():
    # state => key => st
    for _ in range(1000):
        _, state = init()
        sorted_hand = np.concatenate(
            [
                np.sort(state.hand[:13]),
                np.sort(state.hand[13:26]),
                np.sort(state.hand[26:39]),
                np.sort(state.hand[39:]),
            ]
        ).reshape(-1)
        key = _state_to_key(state)
        reconst_hand = _key_to_hand(key)
        assert np.all(sorted_hand == reconst_hand)


def test_calcurate_dds_tricks():
    HASH_TABLE_SAMPLE_KEYS, HASH_TABLE_SAMPLE_VALUES = _load_sample_hash()
    samples = []
    with open("tests/assets/contractbridge-ddstable-sample100.csv", "r") as f:
        reader = csv.reader(f, delimiter=",")
        for i in reader:
            samples.append([i[0], np.array(i[1:]).astype(np.int8)])
    for i in range(len(HASH_TABLE_SAMPLE_KEYS)):
        _, state = init()
        state.hand = _key_to_hand(HASH_TABLE_SAMPLE_KEYS[i])
        dds_tricks = _calculate_dds_tricks(
            state, HASH_TABLE_SAMPLE_KEYS, HASH_TABLE_SAMPLE_VALUES
        )
        # sample dataから、作成したhash tableを用いて、ddsの結果を計算
        # その結果とsample dataが一致しているか確認
        assert np.all(dds_tricks == samples[i][1])


def to_value(sample: list) -> np.ndarray:
    """Convert sample to value
    >>> sample = ['0', '1', '0', '4', '0', '13', '12', '13', '9', '13', '0', '1', '0', '4', '0', '13', '12', '13', '9', '13']
    >>> to_value(sample)
    array([  4160, 904605,   4160, 904605])
    """
    np_sample = np.array(sample)
    np_sample = np_sample.astype(np.int8).reshape(4, 5)
    return to_binary(np_sample)


def to_binary(x) -> np.ndarray:
    """Convert dds information to value
    >>> np.array([16**i for i in range(5)], dtype=np.int32)[::-1]
    array([65536,  4096,   256,    16,     1], dtype=int32)
    >>> x = np.arange(20, dtype=np.int32).reshape(4, 5) % 14
    >>> to_binary(x)
    array([  4660, 354185, 703696,  74565])
    """
    bases = np.array([16**i for i in range(5)], dtype=np.int32)[::-1]
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
