from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class ContractBridgeBiddingState:
    # turn 現在のターン数
    turn: np.ndarray = np.zeros(1, dtype=np.int16)
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
    vul_NS: np.ndarray = np.zeros(1, dtype=np.bool8)
    # vul_EW EWチームがvulかどうかを表す
    # 0 = non vul, 1 = vul
    vul_EW: np.ndarray = np.zeros(1, dtype=np.bool8)
    # last_bid 最後にされたbid
    # last_bidder 最後にbidをしたプレイヤー
    # call_x 最後にされたbidがdoubleされているか
    # call_xx 最後にされたbidがredoubleされているか
    last_bid: np.ndarray = np.full(1, -1, dtype=np.int8)
    last_bidder: np.ndarray = np.full(1, -1, dtype=np.int8)
    call_x: np.ndarray = np.zeros(1, dtype=np.bool8)
    call_xx: np.ndarray = np.zeros(1, dtype=np.bool8)
    # legal_actions プレイヤーの可能なbidの一覧
    legal_actions: np.ndarray = np.ones(38, dtype=np.bool8)
    # first_denominaton_NS NSチームにおいて、各denominationをどのプレイヤー
    # が最初にbidしたかを表す
    first_denomination_NS: np.ndarray = np.full(5, -1, dtype=np.int8)
    # first_denominaton_EW EWチームにおいて、各denominationをどのプレイヤー
    # が最初にbidしたかを表す
    first_denomination_EW: np.ndarray = np.full(5, -1, dtype=np.int8)
    # passの回数
    pass_num: np.ndarray = np.zeros(1, dtype=np.int8)


def init() -> ContractBridgeBiddingState:
    hand = np.arange(0, 52)
    np.random.shuffle(hand)
    vul_NS = np.random.randint(0, 2, 1)
    vul_EW = np.random.randint(0, 2, 1)
    dealer = np.random.randint(0, 4, 1)
    legal_actions = np.ones(38, dtype=np.bool8)
    # 最初はdable, redoubleできない
    legal_actions[-2:] = 0
    state = ContractBridgeBiddingState(
        hand=hand,
        dealer=dealer,
        vul_NS=vul_NS,
        vul_EW=vul_EW,
        legal_actions=legal_actions,
    )
    return state


def step(
    state: ContractBridgeBiddingState, action: int
) -> Tuple[ContractBridgeBiddingState, int, bool]:

    state.bidding_history[state.turn] = action
    # 非合法手判断
    if not state.legal_actions[action]:
        return state, -1, True
    # pass
    if action == 35:
        state.pass_num += 1
        # 終了判定
        if _is_over(state):
            reward = _calc_reward()
            return state, reward, True
    # double
    elif action == 36:
        state.call_x[0] = True
        state.pass_num[0] = 0
    # redouble
    elif action == 37:
        state.call_xx[0] = True
        state.pass_num[0] = 0
    # bid
    else:
        state = _state_bid(state, action)
    state = _update_legal_action_X_XX(state)  # 次のターンにX, XXが合法手か判断
    return state, 0, False


# 終了判定　ビッドされていない場合はパスが４回連続（パスアウト）、それ以外はパスが3回連続
def _is_over(state: ContractBridgeBiddingState) -> bool:
    if (state.last_bid[0] == -1 and state.pass_num == 4) or (
        state.last_bid[0] != -1 and state.pass_num == 3
    ):
        return True
    else:
        return False


# コントラクトから報酬を計算
def _calc_reward():
    return 20


# bidによるstateの変化
def _state_bid(
    state: ContractBridgeBiddingState, action: int
) -> ContractBridgeBiddingState:
    # 最後のbidとそのプレイヤーを保存
    state.last_bid[0] = action
    state.last_bidder[0] = _active_player(state.turn[0], state.dealer[0])
    # チーム内で各denominationを最初にbidしたプレイヤー
    denomination = _bid_to_denomination(action)
    team = _player_to_team(state.last_bidder[0])
    # team = 1ならEWチーム
    if team and (state.first_denomination_EW[denomination] == -1):
        state.first_denomination_EW[denomination] = state.last_bidder[0]
    # team = 0ならNSチーム
    elif not team and (state.first_denomination_NS[denomination] == -1):
        state.first_denomination_NS[denomination] = state.last_bidder[0]
    state.legal_actions[: action + 1] = 0
    state.call_x[0] = False
    state.call_xx[0] = False
    state.pass_num[0] = 0
    return state


# bidのdenominationを計算
def _bid_to_denomination(bid: int) -> int:
    return bid % 5


# playerのチームを判定　0: NSチーム, 1: EWチーム
def _player_to_team(player: int) -> int:
    return player % 2


# 次プレイヤーのX, XXが合法手かどうか
def _update_legal_action_X_XX(
    state: ContractBridgeBiddingState,
) -> ContractBridgeBiddingState:
    state.turn[0] += 1
    if state.last_bidder[0] != -1:
        if (
            (not state.call_x)
            and (not state.call_xx)
            and (
                not _is_partner(
                    state.last_bidder[0],
                    _active_player(state.turn[0], state.dealer[0]),
                )
            )
        ):
            state.legal_actions[36] = True
        else:
            state.legal_actions[36] = False
        if (
            state.call_x
            and (not state.call_xx)
            and (
                _is_partner(
                    state.last_bidder[0],
                    _active_player(state.turn[0], state.dealer[0]),
                )
            )
        ):
            state.legal_actions[37] = True
        else:
            state.legal_actions[37] = False
    else:
        state.legal_actions[36] = False
        state.legal_actions[37] = False
    return state


# playerがパートナーか判断
def _is_partner(player1: int, player2: int):
    return (abs(player1 - player2) + 1) % 2


# ターン数、dealerから現在のプレイヤーを計算
def _active_player(turn, dealer):
    return (dealer + turn) % 4


# debug用の最長に行動するエージェント
def max_action_length_agent(state: ContractBridgeBiddingState):
    if (state.last_bid[0] == -1 and state.pass_num != 3) or (
        state.last_bid[0] != -1 and state.pass_num != 2
    ):
        return 35
    elif state.legal_actions[36]:
        return 36
    elif state.legal_actions[37]:
        return 37
    else:
        return state.last_bid[0] + 1


if __name__ == "__main__":
    state = init()
    for i in range(319):
        state, _, is_over = step(state, max_action_length_agent(state))
        print(state)
        if is_over:
            print("over")
            break
