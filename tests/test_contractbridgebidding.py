import numpy as np

from pgx.contractbridgebidding import ContractBridgeBiddingState, init, step


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
