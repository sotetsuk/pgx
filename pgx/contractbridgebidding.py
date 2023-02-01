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
    return _value_to_dds_tricks(
        _find_value_from_key(key, hash_keys, hash_values)
    )


def _find_value_from_key(
    key: np.ndarray, hash_keys: np.ndarray, hash_values: np.ndarray
):
    """Find a value matching key without batch processing
    >>> VALUES = np.arange(20).reshape(5, 4)
    >>> KEYS = np.arange(20).reshape(5, 4)
    >>> key = np.arange(4, 8)
    >>> _find_value_from_key(key, KEYS, VALUES)
    array([4, 5, 6, 7])
    """
    mask = np.where(
        np.all((hash_keys == key), axis=1),
        np.ones(1, dtype=np.bool_),
        np.zeros(1, dtype=np.bool_),
    )
    ix = np.argmax(mask)
    return hash_values[ix]


def _load_sample_hash() -> Tuple[np.ndarray, np.ndarray]:
    # fmt: off
    return np.array([[32111356, 27911748, 64692765, 20711444], [14584782, 20176164, 26690286, 12659034], [10403651, 60544536, 59050327, 57548940], [43929222, 52419614, 18484397, 22790201], [2004469, 38150871, 36709054, 32297884], [22812141, 57594972, 54202090, 665657], [18078001, 32390796, 14992599, 35298473], [47144865, 23409442, 8777506, 35954455], [7905661, 26952482, 25730770, 33473852], [2582531, 50734700, 33226075, 45564745], [45225074, 24821169, 47205109, 15610689], [16933322, 12084092, 20113319, 62729556], [40367551, 36309501, 20867143, 17374720], [64429872, 47194286, 45768272, 39524849], [10331660, 2865357, 28119681, 62218108], [23269250, 12311378, 34451217, 52899413], [1142439, 26418751, 63255256, 51489670], [12202761, 9833369, 28339975, 60682725], [4958128, 23917289, 52506718, 55044485], [14299537, 29942901, 43000611, 47824886], [1486463, 59136123, 30117718, 2491203], [54413654, 41002115, 9352334, 24919131], [6283781, 39930418, 61637869, 53967713], [49346693, 52604437, 23034242, 32537986], [55042434, 25836292, 37440741, 16100702], [13253177, 28401541, 59317191, 27588636], [11908799, 5022756, 20342847, 25489225], [22766354, 66728979, 48199337, 664892], [49879249, 36231566, 38561980, 58514693], [44081513, 54298465, 37688427, 55689009], [54662962, 18533503, 3560087, 28982921], [8104071, 53165381, 66566952, 8733028], [15522735, 46051418, 21000517, 37587219], [35425318, 60256306, 30299595, 28103350], [54106042, 57422087, 25961681, 7003795], [35498181, 34959537, 3125181, 36464460], [12687102, 57247967, 18424019, 38722322], [64903998, 41252032, 60916591, 9865474], [59596985, 24799476, 4613648, 67087808], [57553989, 33401588, 12328996, 40168557], [55290345, 19695229, 20782283, 22741539], [47472477, 55091214, 23670715, 59560040], [3528557, 63476421, 14711993, 11154872], [27769021, 29362015, 56672351, 45782114], [14335103, 6510944, 51047649, 14078489], [9621476, 31895890, 34721085, 45510699], [62868522, 26335800, 23108881, 62441273], [30000845, 36558007, 7023734, 33769582], [2161157, 34295181, 6150878, 51304877], [47127969, 53230294, 28350058, 57987520], [34487952, 12797662, 12150969, 64235351], [13683599, 31615764, 49772224, 23304345], [63517564, 64105254, 9606426, 18723527], [2724304, 57558752, 48135465, 57978096], [66919782, 2505000, 10347293, 2276785], [35740127, 59770046, 18430732, 39753967], [52832176, 22130754, 4947177, 60783260], [21345468, 26998928, 60641222, 58896902], [59247259, 31183407, 15402317, 64272720], [44355427, 26372243, 26768291, 1390579], [5245311, 20981032, 65261179, 48651190], [30869640, 23122679, 46505534, 5802431], [16049633, 54624934, 52607684, 58167763], [2309147, 47091278, 31610747, 5793363], [17445766, 11270278, 2022680, 47538035], [50177504, 39562270, 25542862, 60557399], [27137260, 52566451, 55848344, 16758932], [43058351, 48221402, 52200782, 60703050], [60429828, 59611555, 13885901, 65651313], [39560960, 46491633, 26152548, 4386526], [18609860, 5257804, 15182667, 41063262], [23012067, 708183, 4536979, 14849789], [59927549, 44915116, 20496626, 12678409], [32369442, 6311327, 15899677, 26108487], [51518120, 3535318, 30796858, 58777115], [16727996, 37843501, 34092153, 40273254], [22566144, 61120722, 43941630, 3543807], [12860808, 42796566, 28441951, 52014734], [15076818, 40590571, 17586659, 41206311], [62587746, 4495077, 20050563, 30715551], [30463554, 11852026, 33108458, 59396556], [59982997, 41403215, 36897325, 23775962], [38593144, 48691174, 57950892, 5453893], [2566940, 19233977, 31396374, 57163682], [17351199, 15548779, 22245476, 61597611], [10478811, 40733952, 32646254, 27325669], [62408598, 56083202, 43724622, 38539897], [54399093, 51545635, 7318697, 16216425], [911659, 10517104, 65484115, 48631638], [915173, 34015963, 6463324, 28061003], [15570115, 48305729, 52011886, 46648931], [14409442, 6371241, 4397216, 60112869], [8843908, 30791165, 53445340, 40014347], [21522127, 57394342, 32968740, 41916460], [50901145, 39779070, 39076921, 18414019], [7231951, 24094228, 55746315, 3748552], [43988238, 30584220, 25475537, 14794528], [31454438, 40005816, 617974, 6326163], [63177133, 16759705, 59326533, 51539611], [33218465, 11670545, 9307389, 32876078]], dtype=np.int32), np.array([[4160, 904605, 4160, 904605], [546712, 357701, 546712, 357701], [283782, 493909, 279686, 559446], [398625, 501675, 398625, 435863], [153396, 755367, 153396, 755367], [71828, 767287, 71828, 767287], [226216, 617011, 226216, 616995], [502665, 336450, 502665, 336450], [358198, 550567, 358198, 550567], [771446, 137301, 771446, 137333], [611734, 231237, 611734, 231237], [549941, 354470, 549941, 354470], [156291, 747866, 156291, 752218], [514955, 393810, 514682, 393810], [489509, 349608, 489509, 349608], [550998, 292230, 550998, 292230], [341108, 567657, 341108, 563544], [619540, 285111, 554004, 285111], [410931, 432297, 410931, 432297], [481606, 427158, 350533, 427158], [279702, 559428, 279702, 624964], [166179, 738471, 166179, 738471], [226340, 616887, 226340, 616887], [488805, 419958, 488805, 419958], [628326, 280182, 628326, 280182], [328000, 576651, 328000, 576667], [226099, 682647, 226099, 682647], [493722, 414771, 493722, 415027], [506984, 401780, 506728, 401780], [541750, 367014, 541750, 366758], [266593, 637563, 266593, 637563], [341334, 432263, 341334, 432263], [566936, 341813, 566680, 341812], [615814, 292951, 615814, 292951], [279635, 559498, 279635, 563594], [763835, 144674, 763835, 144930], [350054, 493159, 350054, 493159], [219032, 619812, 284825, 619812], [341285, 563384, 341285, 563384], [71987, 767125, 71987, 767125], [201009, 707756, 201009, 707756], [103553, 805194, 103553, 805194], [297112, 607300, 297112, 607300], [205697, 637254, 205697, 637254], [358263, 550246, 292727, 550246], [620200, 284453, 620200, 284453], [209169, 633786, 209169, 629690], [350616, 558133, 350616, 558133], [619624, 285045, 619624, 284789], [423767, 484998, 424023, 484742], [752199, 152197, 752199, 152197], [484727, 419942, 419191, 419942], [161572, 681654, 161572, 681654], [702618, 205891, 702618, 205891], [304280, 604468, 304280, 604468], [230979, 546200, 231236, 546441], [488774, 354454, 488774, 419991], [221813, 686952, 221813, 686952], [467506, 441003, 467506, 441003], [419382, 489383, 419382, 489383], [361315, 407400, 365668, 472952], [476518, 362342, 476518, 362342], [161829, 681144, 161829, 681144], [697431, 207222, 697431, 207222], [838506, 70257, 838506, 70257], [563319, 279910, 563319, 279910], [357718, 551030, 357718, 551031], [616345, 292420, 616345, 292420], [348706, 494507, 348706, 494507], [682136, 226372, 616583, 226372], [506266, 402499, 506266, 402499], [353877, 554888, 353877, 554888], [292982, 545638, 292982, 545638], [406323, 498090, 406323, 498090], [541815, 366949, 541815, 366932], [366985, 541524, 366985, 537170], [370023, 407395, 501111, 407395], [693141, 215620, 693141, 211524], [625048, 283717, 625048, 283716], [402708, 505801, 402708, 505801], [629130, 279603, 629130, 279363], [141682, 697449, 141682, 697433], [830618, 78147, 830618, 78147], [541830, 366934, 541830, 366934], [410468, 498041, 410467, 498297], [292438, 616326, 292438, 616070], [427911, 410965, 493447, 410965], [165922, 742826, 165922, 742826], [837256, 71250, 837256, 71250], [288101, 620664, 288101, 620664], [559241, 283732, 559497, 283732], [502856, 340369, 502856, 340369], [726452, 107538, 726452, 107538], [423542, 484967, 423542, 484967], [546135, 296822, 546135, 297094], [214151, 694614, 214151, 694614], [345207, 563558, 344949, 563558], [506042, 402722, 506042, 402722], [209731, 633498, 209731, 633498], [357989, 550776, 353893, 550776]], dtype=np.int32)
    # fmt: on
