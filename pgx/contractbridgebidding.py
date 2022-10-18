from dataclasses import dataclass

import numpy as np


@dataclass
class ContractBridgeBiddingState:
    # turn 現在のターン数
    turn: np.ndarray = np.zeros(1, dtype=np.int8)
    # hand 各プレイヤーの手札
    # index = 0 ~ 12がN, 13 ~ 25がE, 26 ~ 38がS, 39 ~ 51がWの持つ手札
    # 各要素にはカードを表す0 ~ 51の整数が格納される
    hand: np.ndarray = np.zeros(52, dtype=np.int8)
    # bidding_history 各プレイヤーのbidを時系列順に記憶
    # 最大の行動系列長 = 525
    # 各要素には、bidを表す0 ~ 37の整数が格納される
    # 各ビッドがどのプレイヤーにより行われたかは、要素のindexから分かる（ix % 4）
    bidding_history: np.ndarray = np.zeros(525, dtype=np.int8)
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
    # legal_actions プレイヤーの可能なbidの一覧
    legal_actions: np.ndarray = np.zeros(38, dtype=np.bool8)
    # first_denominaton_NS NSチームにおいて、各denominationをどのプレイヤー
    # が最初にbidしたかを表す
    first_denomination_NS: np.ndarray = np.zeros(5, dtype=np.int8)
    # first_denominaton_EW EWチームにおいて、各denominationをどのプレイヤー
    # が最初にbidしたかを表す
    first_denomination_EW: np.ndarray = np.zeros(5, dtype=np.int8)


def init() -> ContractBridgeBiddingState:
    hand = _make_init_hand()
    dealer = _make_init_dealer()
    vul_NS = _make_init_vul()
    vul_EW = _make_init_vul()
    dealer = _make_init_dealer()
    state = ContractBridgeBiddingState(
        hand=hand,
        dealer=dealer,
        vul_NS=vul_NS,
        vul_EW=vul_EW
    )
    return state


def _make_init_hand() -> np.ndarray:
    hand = np.arange(0, 52)
    np.random.shuffle(hand)
    return hand


def _make_init_dealer() -> np.ndarray:
    return np.random.randint(0, 3, 1)


def _make_init_vul() -> np.ndarray:
    return np.random.randint(0, 1, 1)


if __name__ == "__main__":
    state = init()
    print(state)
