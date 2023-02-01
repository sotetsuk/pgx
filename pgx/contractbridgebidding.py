import copy
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class ContractBridgeBiddingState:
    # turn 現在のターン数
    turn: np.ndarray = np.array(0, dtype=np.int16)
    # curr_player 現在のプレイヤーid
    curr_player: np.ndarray = np.array(-1, dtype=np.int8)
    # 終端状態
    terminated: np.ndarray = np.array(False, dtype=np.bool_)
    # hand 各プレイヤーの手札
    # index = 0 ~ 12がN, 13 ~ 25がE, 26 ~ 38がS, 39 ~ 51がWの持つ手札
    # 各要素にはカードを表す0 ~ 51の整数が格納される
    hand: np.ndarray = np.zeros(52, dtype=np.int8)
    # bidding_history 各プレイヤーのbidを時系列順に記憶
    # 最大の行動系列長 = 319
    # 各要素には、行動を表す整数が格納される
    # bidを表す0 ~ 34, passを表す35, doubleを表す36, redoubleを表す37, 行動が行われていない-1
    # 各ビッドがどのプレイヤーにより行われたかは、要素のindexから分かる（ix % 4）
    bidding_history: np.ndarray = np.full(319, -1, dtype=np.int8)
    # dealer どのプレイヤーがdealerかを表す
    # 0 = N, 1 = E, 2 = S, 3 = W
    # dealerは最初にbidを行うプレイヤー
    dealer: np.ndarray = np.zeros(4, dtype=np.int8)
    # vul_NS NSチームがvulかどうかを表す
    # 0 = non vul, 1 = vul
    vul_NS: np.ndarray = np.array(False, dtype=np.bool_)
    # vul_EW EWチームがvulかどうかを表す
    # 0 = non vul, 1 = vul
    vul_EW: np.ndarray = np.array(False, dtype=np.bool_)
    # last_bid 最後にされたbid
    # last_bidder 最後にbidをしたプレイヤー
    # call_x 最後にされたbidがdoubleされているか
    # call_xx 最後にされたbidがredoubleされているか
    last_bid: np.ndarray = np.array(-1, dtype=np.int8)
    last_bidder: np.ndarray = np.array(-1, dtype=np.int8)
    call_x: np.ndarray = np.array(False, dtype=np.bool_)
    call_xx: np.ndarray = np.array(False, dtype=np.bool_)
    # legal_actions プレイヤーの可能なbidの一覧
    legal_action_mask: np.ndarray = np.ones(38, dtype=np.bool_)
    # first_denominaton_NS NSチームにおいて、各denominationをどのプレイヤー
    # が最初にbidしたかを表す
    first_denomination_NS: np.ndarray = np.full(5, -1, dtype=np.int8)
    # first_denominaton_EW EWチームにおいて、各denominationをどのプレイヤー
    # が最初にbidしたかを表す
    first_denomination_EW: np.ndarray = np.full(5, -1, dtype=np.int8)
    # passの回数
    pass_num: np.ndarray = np.array(0, dtype=np.int8)


def init() -> Tuple[np.ndarray, ContractBridgeBiddingState]:
    hand = np.arange(0, 52)
    np.random.shuffle(hand)
    vul_NS = np.random.randint(0, 2, 1)
    vul_EW = np.random.randint(0, 2, 1)
    dealer = np.random.randint(0, 4, 1)
    curr_player = copy.deepcopy(dealer)
    legal_actions = np.ones(38, dtype=np.bool_)
    # 最初はdable, redoubleできない
    legal_actions[-2:] = 0
    state = ContractBridgeBiddingState(
        curr_player=curr_player,
        hand=hand,
        dealer=dealer,
        vul_NS=vul_NS,
        vul_EW=vul_EW,
        legal_action_mask=legal_actions,
    )
    return state.curr_player, state


def step(
    state: ContractBridgeBiddingState, action: int
) -> Tuple[np.ndarray, ContractBridgeBiddingState, np.ndarray]:
    state.bidding_history[state.turn] = action
    # 非合法手判断
    if not state.legal_action_mask[action]:
        return _illegal_step(state)
    # pass
    elif action == 35:
        state = _state_pass(state)
        # 終了判定
        if _is_terminated(state):
            return _terminated_step(state)
        else:
            return _continue_step(state)
    # double
    elif action == 36:
        state = _state_X(state)
        return _continue_step(state)
    # redouble
    elif action == 37:
        state = _state_XX(state)
        return _continue_step(state)
    # bid
    else:
        state = _state_bid(state, action)
        return _continue_step(state)


# ゲームが非合法手検知で終了した場合
def _illegal_step(
    state: ContractBridgeBiddingState,
) -> Tuple[np.ndarray, ContractBridgeBiddingState, np.ndarray]:
    state.terminated = np.array(True, dtype=np.bool_)
    state.curr_player = np.array(-1, dtype=np.int8)
    illegal_rewards = np.zeros(4)
    return state.curr_player, state, illegal_rewards


# ゲームが正常に終了した場合
def _terminated_step(
    state: ContractBridgeBiddingState,
) -> Tuple[np.ndarray, ContractBridgeBiddingState, np.ndarray]:
    state.terminated = np.array(True, dtype=np.bool_)
    state.curr_player = np.array(-1, dtype=np.int8)
    rewards = _calc_reward()
    return state.curr_player, state, rewards


# ゲームが継続する場合
def _continue_step(
    state: ContractBridgeBiddingState,
) -> Tuple[np.ndarray, ContractBridgeBiddingState, np.ndarray]:
    # 次ターンのプレイヤー、ターン数
    state.curr_player = (state.curr_player + 1) % 4
    state.turn += 1
    (
        state.legal_action_mask[36],
        state.legal_action_mask[37],
    ) = _update_legal_action_X_XX(
        state
    )  # 次のターンにX, XXが合法手か判断
    return state.curr_player, state, np.zeros(4)


# 終了判定　ビッドされていない場合はパスが４回連続（パスアウト）、それ以外はパスが3回連続
def _is_terminated(state: ContractBridgeBiddingState) -> bool:
    if (state.last_bid == -1 and state.pass_num == 4) or (
        state.last_bid != -1 and state.pass_num == 3
    ):
        return True
    else:
        return False


# コントラクトから報酬を計算
def _calc_reward() -> np.ndarray:
    return np.full(4, 0)


# passによるstateの変化
def _state_pass(
    state: ContractBridgeBiddingState,
) -> ContractBridgeBiddingState:
    state.pass_num += 1
    return state


# Xによるstateの変化
def _state_X(state: ContractBridgeBiddingState) -> ContractBridgeBiddingState:
    state.call_x = np.array(True, dtype=np.bool_)
    state.pass_num = np.array(0, dtype=np.int8)
    return state


# XXによるstateの変化
def _state_XX(state: ContractBridgeBiddingState) -> ContractBridgeBiddingState:
    state.call_xx = np.array(True, dtype=np.bool_)
    state.pass_num = np.array(0, dtype=np.int8)
    return state


# bidによるstateの変化
def _state_bid(
    state: ContractBridgeBiddingState, action: int
) -> ContractBridgeBiddingState:
    # 最後のbidとそのプレイヤーを保存
    state.last_bid = np.array(action, dtype=np.int8)
    state.last_bidder = state.curr_player
    # チーム内で各denominationを最初にbidしたプレイヤー
    denomination = _bid_to_denomination(action)
    team = _player_to_team(state.last_bidder)
    # team = 1ならEWチーム
    if team and (state.first_denomination_EW[denomination] == -1):
        state.first_denomination_EW[denomination] = state.last_bidder
    # team = 0ならNSチーム
    elif not team and (state.first_denomination_NS[denomination] == -1):
        state.first_denomination_NS[denomination] = state.last_bidder
    state.legal_action_mask[: action + 1] = 0
    state.call_x = np.array(False, dtype=np.bool_)
    state.call_xx = np.array(False, dtype=np.bool_)
    state.pass_num = np.array(0, dtype=np.int8)
    return state


# bidのdenominationを計算
def _bid_to_denomination(bid: int) -> int:
    return bid % 5


# playerのチームを判定　0: NSチーム, 1: EWチーム
def _player_to_team(player: np.ndarray) -> np.ndarray:
    return player % 2


# 次プレイヤーのX, XXが合法手かどうか
def _update_legal_action_X_XX(
    state: ContractBridgeBiddingState,
) -> Tuple[bool, bool]:
    if state.last_bidder != -1:
        return _is_legal_X(state), _is_legal_XX(state)
    else:
        return False, False


def _is_legal_X(state: ContractBridgeBiddingState) -> bool:
    if (
        (not state.call_x)
        and (not state.call_xx)
        and (
            not _is_partner(
                state.last_bidder,
                state.curr_player,
            )
        )
    ):
        return True
    else:
        return False


def _is_legal_XX(state: ContractBridgeBiddingState) -> bool:
    if (
        state.call_x
        and (not state.call_xx)
        and (
            _is_partner(
                state.last_bidder,
                state.curr_player,
            )
        )
    ):
        return True
    else:
        return False


# playerがパートナーか判断
def _is_partner(player1: np.ndarray, player2: np.ndarray) -> np.ndarray:
    return (abs(player1 - player2) + 1) % 2


# カードと数字の対応
# 0~12 spade, 13~25 heart, 26~38 diamond, 39~51 club
# それぞれのsuitにおいて以下の順で数字が並ぶ
TO_CARD = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]


def _state_to_pbn(state: ContractBridgeBiddingState) -> str:
    """Convert state to pbn format"""
    pbn = "N:"
    for i in range(4):  # player
        hand = np.sort(state.hand[i * 13 : (i + 1) * 13])
        for j in range(4):  # suit
            card = [
                TO_CARD[i % 13] for i in hand if j * 13 <= i < (j + 1) * 13
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


def _state_to_key(state: ContractBridgeBiddingState) -> np.ndarray:
    """Convert state to key of dds table"""
    hand = state.hand
    key = np.zeros(52, dtype=np.int8)
    for i in range(52):  # N: 0, E: 1, S: 2, W: 3
        if i // 13 == 0:
            key[hand[i]] = 0
        elif i // 13 == 1:
            key[hand[i]] = 1
        elif i // 13 == 2:
            key[hand[i]] = 2
        elif i // 13 == 3:
            key[hand[i]] = 3
    key = key.reshape(4, 13)
    return _to_binary(key)


def _pbn_to_key(pbn: str) -> np.ndarray:
    """Convert pbn to key of dds table"""
    key = np.zeros(52, dtype=np.int8)
    hands = pbn[2:]
    for player, hand in enumerate(list(hands.split())):  # for each player
        for suit, cards in enumerate(list(hand.split("."))):  # for each suit
            for card in cards:  # for each card
                card_num = _card_str_to_int(card) + suit * 13
                key[card_num] = player
    key = key.reshape(4, 13)
    return _to_binary(key)


def _to_binary(x: np.ndarray) -> np.ndarray:
    bases = np.array([4**i for i in range(13)], dtype=np.int32)[::-1]
    return (x * bases).sum(axis=1)  # shape = (4, )


def _card_str_to_int(card: str) -> int:
    if card == "K":
        return 12
    elif card == "Q":
        return 11
    elif card == "J":
        return 10
    elif card == "T":
        return 9
    elif card == "A":
        return 0
    else:
        return int(card) - 1


def _key_to_hand(key: np.ndarray) -> np.ndarray:
    """Convert key to hand"""
    cards = np.array(
        [int(i) for j in key for i in np.base_repr(j, 4).zfill(13)],
        dtype=np.int8,
    )
    return np.concatenate(
        [
            np.where(cards == 0),
            np.where(cards == 1),
            np.where(cards == 2),
            np.where(cards == 3),
        ],
        axis=1,
    ).reshape(-1)


def _value_to_dds_tricks(values: np.ndarray) -> np.ndarray:
    """Convert values to dds tricks
    >>> value = np.array([4160, 904605, 4160, 904605])
    >>> _value_to_dds_tricks(value)
    array([ 0,  1,  0,  4,  0, 13, 12, 13,  9, 13,  0,  1,  0,  4,  0, 13, 12,
           13,  9, 13], dtype=int8)
    """
    return np.array(
        [int(i, 16) for j in values for i in np.base_repr(j, 16).zfill(5)],
        dtype=np.int8,
    )


def _calculate_dds_tricks(
    state: ContractBridgeBiddingState,
    hash_keys: np.ndarray,
    hash_values: np.ndarray,
) -> np.ndarray:
    key = _state_to_key(state)
    return _value_to_dds_tricks(_find_key(key, hash_keys, hash_values))


def _find_key(key: np.ndarray, hash_keys: np.ndarray, hash_values: np.ndarray):
    """Find a value matching key without batch processing
    >>> VALUES = np.arange(20).reshape(5, 4)
    >>> KEYS = np.arange(20).reshape(5, 4)
    >>> key = np.arange(4, 8)
    >>> _find_key(key, KEYS, VALUES)
    array([4, 5, 6, 7])
    """
    ix = np.argmin(np.abs(hash_keys - key).sum(axis=1))
    return hash_values[ix]


def _find_key_2(
    key: np.ndarray, hash_keys: np.ndarray, hash_values: np.ndarray
):
    """Find a value matching key without batch processing
    >>> VALUES = np.arange(20).reshape(5, 4)
    >>> KEYS = np.arange(20).reshape(5, 4)
    >>> key = np.arange(4, 8)
    >>> _find_key_2(key, KEYS, VALUES)
    array([4, 5, 6, 7])
    """
    mask = np.where(
        np.all((hash_keys == key), axis=1),
        np.ones(1, dtype=np.bool_),
        np.zeros(1, dtype=np.bool_),
    )
    ix = np.argmax(mask)
    return hash_values[ix]
