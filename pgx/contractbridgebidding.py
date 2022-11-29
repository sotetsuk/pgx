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
    terminated: np.ndarray = np.array(False, dtype=np.bool8)
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
    vul_NS: np.ndarray = np.array(False, dtype=np.bool8)
    # vul_EW EWチームがvulかどうかを表す
    # 0 = non vul, 1 = vul
    vul_EW: np.ndarray = np.array(False, dtype=np.bool8)
    # last_bid 最後にされたbid
    # last_bidder 最後にbidをしたプレイヤー
    # call_x 最後にされたbidがdoubleされているか
    # call_xx 最後にされたbidがredoubleされているか
    last_bid: np.ndarray = np.array(-1, dtype=np.int8)
    last_bidder: np.ndarray = np.array(-1, dtype=np.int8)
    call_x: np.ndarray = np.array(False, dtype=np.bool8)
    call_xx: np.ndarray = np.array(False, dtype=np.bool8)
    # legal_actions プレイヤーの可能なbidの一覧
    legal_action_mask: np.ndarray = np.ones(38, dtype=np.bool8)
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
    legal_actions = np.ones(38, dtype=np.bool8)
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
    state.terminated = np.array(True, dtype=np.bool8)
    state.curr_player = np.array(-1, dtype=np.int8)
    illegal_rewards = np.zeros(4)
    return state.curr_player, state, illegal_rewards


# ゲームが正常に終了した場合
def _terminated_step(
    state: ContractBridgeBiddingState,
) -> Tuple[np.ndarray, ContractBridgeBiddingState, np.ndarray]:
    state.terminated = np.array(True, dtype=np.bool8)
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
    state.call_x = np.array(True, dtype=np.bool8)
    state.pass_num = np.array(0, dtype=np.int8)
    return state


# XXによるstateの変化
def _state_XX(state: ContractBridgeBiddingState) -> ContractBridgeBiddingState:
    state.call_xx = np.array(True, dtype=np.bool8)
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
    state.call_x = np.array(False, dtype=np.bool8)
    state.call_xx = np.array(False, dtype=np.bool8)
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
            state.legal_action_mask[36] = True
        else:
            state.legal_action_mask[36] = False
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
            state.legal_action_mask[37] = True
        else:
            state.legal_action_mask[37] = False
    else:
        state.legal_action_mask[36] = False
        state.legal_action_mask[37] = False
    return state.legal_action_mask[36], state.legal_action_mask[37]


# playerがパートナーか判断
def _is_partner(player1: np.ndarray, player2: np.ndarray) -> bool:
    return (abs(player1 - player2) + 1) % 2
