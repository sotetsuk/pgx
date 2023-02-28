import copy
from dataclasses import dataclass
from typing import Tuple

import numpy as np

# カードと数字の対応
# 0~12 spade, 13~25 heart, 26~38 diamond, 39~51 club
# それぞれのsuitにおいて以下の順で数字が並ぶ
TO_CARD = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]


@dataclass
class State:
    # turn 現在のターン数
    turn: np.ndarray = np.array(0, dtype=np.int16)
    # curr_player 現在のプレイヤーid
    curr_player: np.ndarray = np.array(-1, dtype=np.int8)
    # シャッフルされたプレイヤーの並び
    shuffled_players: np.ndarray = np.zeros(4, dtype=np.int8)
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
    # デノミネーションの順番は C, D, H, S, NT = 0, 1, 2, 3, 4
    first_denomination_NS: np.ndarray = np.full(5, -1, dtype=np.int8)
    # first_denominaton_EW EWチームにおいて、各denominationをどのプレイヤー
    # が最初にbidしたかを表す
    first_denomination_EW: np.ndarray = np.full(5, -1, dtype=np.int8)
    # passの回数
    pass_num: np.ndarray = np.array(0, dtype=np.int8)


def init() -> Tuple[np.ndarray, State]:
    hand = np.arange(0, 52)
    np.random.shuffle(hand)
    vul_NS = np.random.randint(0, 2, 1)
    vul_EW = np.random.randint(0, 2, 1)
    dealer = np.random.randint(0, 4, 1)
    # shuffled players and arrange in order of NESW
    shuffled_players = _shuffle_players()
    curr_player = shuffled_players[dealer]
    legal_actions = np.ones(38, dtype=np.bool_)
    # 最初はdable, redoubleできない
    legal_actions[-2:] = 0
    state = State(
        shuffled_players=shuffled_players,
        curr_player=curr_player,
        hand=hand,
        dealer=dealer,
        vul_NS=vul_NS,
        vul_EW=vul_EW,
        legal_action_mask=legal_actions,
    )
    return state.curr_player, state


def init_by_key(key):
    """Make init state from key"""
    hand = _key_to_hand(key)
    np.random.shuffle(hand)
    vul_NS = np.random.randint(0, 2, 1)
    vul_EW = np.random.randint(0, 2, 1)
    dealer = np.random.randint(0, 4, 1)
    # shuffled players and arrange in order of NESW
    shuffled_players = _shuffle_players()
    curr_player = shuffled_players[dealer]
    legal_actions = np.ones(38, dtype=np.bool_)
    # 最初はdable, redoubleできない
    legal_actions[-2:] = 0
    state = State(
        shuffled_players=shuffled_players,
        curr_player=curr_player,
        hand=hand,
        dealer=dealer,
        vul_NS=vul_NS,
        vul_EW=vul_EW,
        legal_action_mask=legal_actions,
    )
    return state.curr_player, state


def _shuffle_players() -> np.ndarray:
    """Randomly arranges player IDs in a list in NESW order.

    Returns:
        np.ndarray: A list of 4 player IDs randomly arranged in NESW order.

    Example:
        >>> np.random.seed(0)
        >>> _shuffle_players()
        array([1, 2, 0, 3], dtype=int8)
    """
    # player_id = 0, 1 -> team a
    team_a_players = np.random.permutation(np.arange(2, dtype=np.int8))
    # player_id = 2, 3 -> team b
    team_b_players = np.random.permutation(np.arange(2, 4, dtype=np.int8))
    # decide which team is on
    # Randomly determine NSteam and EWteam
    # Arrange in order of NESW
    if np.random.randint(2) == 1:
        shuffled_players = np.array(
            [
                team_a_players[0],
                team_b_players[0],
                team_a_players[1],
                team_b_players[1],
            ]
        )
    else:
        shuffled_players = np.array(
            [
                team_b_players[0],
                team_a_players[0],
                team_b_players[1],
                team_a_players[1],
            ]
        )
    return shuffled_players


def _player_position(player: np.ndarray, state: State) -> np.ndarray:
    if player != -1:
        return np.where(state.shuffled_players == player)[0]
    else:
        return np.full(1, -1, dtype=np.int8)


def step(
    state: State,
    action: int,
    hash_keys: np.ndarray,
    hash_values: np.ndarray,
) -> Tuple[np.ndarray, State, np.ndarray]:
    state.bidding_history[state.turn] = action
    # 非合法手判断
    if not state.legal_action_mask[action]:
        return _illegal_step(state)
    # pass
    elif action == 35:
        state = _state_pass(state)
        # 終了判定
        if _is_terminated(state):
            return _terminated_step(state, hash_keys, hash_values)
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


def duplicate(
    init_state: State,
) -> State:
    """Make duplicated state where NSplayer and EWplayer are swapped"""
    duplicated_state = copy.deepcopy(init_state)
    ix = np.array([1, 0, 3, 2])
    duplicated_state.shuffled_players = duplicated_state.shuffled_players[ix]
    return duplicated_state


# ゲームが非合法手検知で終了した場合
def _illegal_step(
    state: State,
) -> Tuple[np.ndarray, State, np.ndarray]:
    state.terminated = np.array(True, dtype=np.bool_)
    state.curr_player = np.array(-1, dtype=np.int8)
    illegal_rewards = np.zeros(4)
    return state.curr_player, state, illegal_rewards


# ゲームが正常に終了した場合
def _terminated_step(
    state: State,
    hash_keys: np.ndarray,
    hash_values: np.ndarray,
) -> Tuple[np.ndarray, State, np.ndarray]:
    state.terminated = np.array(True, dtype=np.bool_)
    state.curr_player = np.array(-1, dtype=np.int8)
    rewards = _reward(state, hash_keys, hash_values)
    return state.curr_player, state, rewards


# ゲームが継続する場合
def _continue_step(
    state: State,
) -> Tuple[np.ndarray, State, np.ndarray]:
    # 次ターンのプレイヤー、ターン数
    # state.curr_player = (state.curr_player + 1) % 4
    state.turn += 1
    state.curr_player = state.shuffled_players[(state.dealer + state.turn) % 4]
    (
        state.legal_action_mask[36],
        state.legal_action_mask[37],
    ) = _update_legal_action_X_XX(
        state
    )  # 次のターンにX, XXが合法手か判断
    return state.curr_player, state, np.zeros(4)


# 終了判定　ビッドされていない場合はパスが４回連続（パスアウト）、それ以外はパスが3回連続
def _is_terminated(state: State) -> bool:
    if (state.last_bid == -1 and state.pass_num == 4) or (
        state.last_bid != -1 and state.pass_num == 3
    ):
        return True
    else:
        return False


def _reward(
    state: State,
    hash_keys: np.ndarray,
    hash_values: np.ndarray,
) -> np.ndarray:
    """Calculate rewards for each player by dds results

    Returns:
        np.ndarray: A list of rewards for each player in playerID order
    """
    # pass out
    if state.last_bid == -1 and state.pass_num == 4:
        return np.zeros(4)
    else:
        # Extract contract from state
        declare_position, denomination, level, vul = _contract(state)
        # Calculate trick table from hash table
        dds_tricks = _calculate_dds_tricks(state, hash_keys, hash_values)
        # Calculate the tricks you could have accomplished with this contraption
        dds_trick = dds_tricks[int(declare_position) * 5 + int(denomination)]
        # Clculate score
        score = _calc_score(
            denomination,
            level,
            vul,
            state.call_x,
            state.call_xx,
            dds_trick,
        )
        # Make reward array in playerID order
        reward = np.array(
            [
                score
                if _is_partner(_player_position(i, state), declare_position)
                else -score
                for i in np.arange(4)
            ]
        )
        return reward


def _calc_score(
    denomination: np.ndarray,
    level: np.ndarray,
    vul: np.ndarray,
    call_x: np.ndarray,
    call_xx: np.ndarray,
    trick: np.ndarray,
) -> np.int16:
    """Calculate score from contract and trick
    Returns:
        np.ndarray: A score of declarer team
    """
    # fmt: off
    _MINOR = np.int16(20)
    _MAJOR = np.int16(30)
    _NT = np.int16(10)
    _MAKE = np.int16(50)
    _MAKE_X = np.int16(50)
    _MAKE_XX = np.int16(50)

    _GAME = np.int16(250)
    _GAME_VUL = np.int16(450)
    _SMALL_SLAM = np.int16(500)
    _SMALL_SLAM_VUL = np.int16(750)
    _GRAND_SLAM = np.int16(500)
    _GRAND_SLAM_VUL = np.int16(750)

    _OVERTRICK_X = np.int16(100)
    _OVERTRICK_X_VUL = np.int16(200)
    _OVERTRICK_XX = np.int16(200)
    _OVERTRICK_XX_VUL = np.int16(400)

    _DOWN = np.array([-50, -100, -150, -200, -250, -300, -350, -400, -450, -500, -550, -600, -650], dtype=np.int16)
    _DOWN_VUL = np.array([-100, -200, -300, -400, -500, -600, -700, -800, -900, -1000, -1100, -1200, -1300], dtype=np.int16)
    _DOWN_X = np.array([-100, -300, -500, -800, -1100, -1400, -1700, -2000, -2300, -2600, -2900, -3200, -3500], dtype=np.int16)
    _DOWN_X_VUL = np.array([-200, -500, -800, -1100, -1400, -1700, -2000, -2300, -2600, -2900, -3200, -3500, -3800], dtype=np.int16)
    _DOWN_XX = np.array([-200, -600, -1000, -1600, -2200, -2800, -3400, -4000, -4600, -5200, -5800, -6400, -7000], dtype=np.int16)
    _DOWN_XX_VUL = np.array([-400, -1000, -1600, -2200, -2800, -3400, -4000, -4600, -5200, -5800, -6400, -7000, -7600], dtype=np.int16)
    # fmt: on
    if level + 6 > trick:  # down
        under_trick = level + 6 - trick
        if call_xx:
            return (
                _DOWN_XX_VUL[under_trick - 1]
                if vul
                else _DOWN_XX[under_trick - 1]
            )
        elif call_x:
            return (
                _DOWN_X_VUL[under_trick - 1]
                if vul
                else _DOWN_X[under_trick - 1]
            )
        else:
            return (
                _DOWN_VUL[under_trick - 1] if vul else _DOWN[under_trick - 1]
            )
    else:  # make
        over_trick_score_per_trick = np.int16(0)
        over_trick = trick - level - np.int16(6)
        score = np.int16(0)
        if denomination <= 1:
            score += np.int16(_MINOR * level)
            over_trick_score_per_trick += _MINOR
        elif 2 <= denomination <= 3:
            score += np.int16(_MAJOR * level)
            over_trick_score_per_trick += _MAJOR
        elif denomination == 4:
            score += np.int16(_MAJOR * level + _NT)
            over_trick_score_per_trick += _MAJOR
        if call_xx:
            score *= np.int16(4)
        elif call_x:
            score *= np.int16(2)
        if score >= 100:  # game make bonus
            score += _GAME_VUL if vul else _GAME
            if level >= 6:  # small slam make bonus
                score += _SMALL_SLAM_VUL if vul else _SMALL_SLAM
                if level == 7:  # grand slam make bonus
                    score += _GRAND_SLAM_VUL if vul else _GRAND_SLAM
        score += _MAKE  # make bonus
        if call_x or call_xx:
            score += _MAKE_X
            if call_xx:
                score += _MAKE_XX
                over_trick_score_per_trick = (
                    _OVERTRICK_XX_VUL if vul else _OVERTRICK_XX
                )
            else:
                over_trick_score_per_trick = (
                    _OVERTRICK_X_VUL if vul else _OVERTRICK_X
                )
        score += over_trick_score_per_trick * over_trick
        return score


def _contract(
    state: State,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return Contract which has position of declare ,denomination, level"""
    denomination = state.last_bid % 5
    level = state.last_bid // 5 + 1
    if _position_to_team(_player_position(state.last_bidder, state)) == 0:
        declare_position = _player_position(
            state.first_denomination_NS[denomination], state
        )
        vul = state.vul_NS
    else:
        declare_position = _player_position(
            state.first_denomination_EW[denomination], state
        )
        vul = state.vul_EW
    return declare_position, denomination, level, vul


# passによるstateの変化
def _state_pass(
    state: State,
) -> State:
    state.pass_num += 1
    return state


# Xによるstateの変化
def _state_X(state: State) -> State:
    state.call_x = np.array(True, dtype=np.bool_)
    state.pass_num = np.array(0, dtype=np.int8)
    return state


# XXによるstateの変化
def _state_XX(state: State) -> State:
    state.call_xx = np.array(True, dtype=np.bool_)
    state.pass_num = np.array(0, dtype=np.int8)
    return state


# bidによるstateの変化
def _state_bid(state: State, action: int) -> State:
    # 最後のbidとそのプレイヤーを保存
    state.last_bid = np.array(action, dtype=np.int8)
    state.last_bidder = state.curr_player
    # チーム内で各denominationを最初にbidしたプレイヤー
    denomination = _bid_to_denomination(action)
    team = _position_to_team(_player_position(state.last_bidder, state))
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
def _position_to_team(position: np.ndarray) -> np.ndarray:
    return position % 2


# 次プレイヤーのX, XXが合法手かどうか
def _update_legal_action_X_XX(
    state: State,
) -> Tuple[bool, bool]:
    if state.last_bidder != -1:
        return _is_legal_X(state), _is_legal_XX(state)
    else:
        return False, False


def _is_legal_X(state: State) -> bool:
    if (
        (not state.call_x)
        and (not state.call_xx)
        and (
            not _is_partner(
                _player_position(state.last_bidder, state),
                _player_position(state.curr_player, state),
            )
        )
    ):
        return True
    else:
        return False


def _is_legal_XX(state: State) -> bool:
    if (
        state.call_x
        and (not state.call_xx)
        and (
            _is_partner(
                _player_position(state.last_bidder, state),
                _player_position(state.curr_player, state),
            )
        )
    ):
        return True
    else:
        return False


# playerがパートナーか判断
def _is_partner(position1: np.ndarray, position2: np.ndarray) -> np.ndarray:
    return (abs(position1 - position2) + 1) % 2


def _state_to_pbn(state: State) -> str:
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


def _state_to_key(state: State) -> np.ndarray:
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
    state: State,
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
        np.bool_(1),
        np.bool_(0),
    )
    ix = np.argmax(mask)
    return hash_values[ix]


def _load_sample_hash() -> Tuple[np.ndarray, np.ndarray]:
    # fmt: off
    return np.array([[19556549, 61212362, 52381660, 50424958], [53254536, 21854346, 37287883, 14009558], [44178585, 6709002, 23279217, 16304124], [36635659, 48114215, 13583653, 26208086], [44309474, 39388022, 28376136, 59735189], [61391908, 52173479, 29276467, 31670621], [34786519, 13802254, 57433417, 43152306], [48319039, 55845612, 44614774, 58169152], [47062227, 32289487, 12941848, 21338650], [36579116, 15643926, 64729756, 18678099], [62136384, 37064817, 59701038, 39188202], [13417016, 56577539, 25995845, 27248037], [61125047, 43238281, 23465183, 20030494], [7139188, 31324229, 58855042, 14296487], [2653767, 47502150, 35507905, 43823846], [31453323, 11605145, 6716808, 41061859], [21294711, 49709, 26110952, 50058629], [48130172, 3340423, 60445890, 7686579], [16041939, 27817393, 37167847, 9605779], [61154057, 17937858, 12254613, 12568801], [13796245, 46546127, 49123920, 51772041], [7195005, 45581051, 41076865, 17429796], [20635965, 14642724, 7001617, 45370595], [35616421, 19938131, 45131030, 16524847], [14559399, 15413729, 39188470, 535365], [48743216, 39672069, 60203571, 60210880], [63862780, 2462075, 23267370, 36595020], [11229980, 11616119, 20292263, 3695004], [24135854, 37532826, 54421444, 14130249], [42798085, 33026223, 2460251, 18566823], [49558558, 65537599, 14768519, 31103243], [44321156, 20075251, 42663767, 11615602], [20186726, 42678073, 11763300, 56739471], [57534601, 16703645, 6039937, 17088125], [50795278, 17350238, 11955835, 21538127], [45919621, 5520088, 27736513, 52674927], [13928720, 57324148, 28222453, 15480785], [910719, 47238830, 26345802, 56166394], [58841430, 1098476, 61890558, 26907706], [10379825, 8624220, 39701822, 29045990], [54444873, 50000486, 48563308, 55867521], [47291672, 22084522, 45484828, 32878832], [55350706, 23903891, 46142039, 11499952], [4708326, 27588734, 31010458, 11730972], [27078872, 59038086, 62842566, 51147874], [28922172, 32377861, 9109075, 10154350], [26104086, 62786977, 224865, 14335943], [20448626, 33187645, 34338784, 26382893], [29194006, 19635744, 24917755, 8532577], [64047742, 34885257, 5027048, 58399668], [27603972, 26820121, 44837703, 63748595], [60038456, 19611050, 7928914, 38555047], [13583610, 19626473, 22239272, 19888268], [28521006, 1743692, 31319264, 15168920], [64585849, 63931241, 57019799, 14189800], [2632453, 7269809, 60404342, 57986125], [1996183, 49918209, 49490468, 47760867], [6233580, 15318425, 51356120, 55074857], [15769884, 61654638, 8374039, 43685186], [44162419, 47272176, 62693156, 35359329], [36345796, 15667465, 53341561, 2978505], [1664472, 12761950, 34145519, 55197543], [37567005, 3228834, 6198166, 15646487], [63233399, 42640049, 12969011, 41620641], [22090925, 3386355, 56655568, 31631004], [16442787, 9420273, 48595545, 29770176], [49404288, 37823218, 58551818, 6772527], [36575583, 53847347, 32379432, 1630009], [9004247, 12999580, 48379959, 14252211], [25850203, 26136823, 64934025, 29362603], [10214276, 43557352, 33387586, 55512773], [45810841, 49561478, 41130845, 27034816], [34460081, 16560450, 57722793, 41007718], [53414778, 6845803, 15340368, 16647575], [30535873, 5193469, 43608154, 11391114], [20622004, 34424126, 31475211, 29619615], [10428836, 49656416, 7912677, 33427787], [57600861, 18251799, 46147432, 58946294], [6760779, 14675737, 42952146, 5480498], [46037552, 39969058, 30103468, 55330772], [64466305, 29376674, 49914839, 55269895], [36494113, 27010567, 65752150, 12395385], [49385632, 19550767, 39809394, 58806235], [20987521, 37444597, 49290126, 42326125], [37150229, 37487849, 28254397, 32949826], [9724895, 53813417, 19431235, 27438556], [42132748, 47073733, 19396568, 10026137], [3961481, 27204521, 62087205, 37602005], [22178323, 17505521, 42006207, 44143605], [12753258, 63063515, 61993175, 8920985], [10998000, 64833190, 6446892, 63676805], [66983817, 63684932, 18378359, 39946382], [63476803, 60000436, 19442420, 66417845], [38004446, 64752157, 42570179, 52844512], [1270809, 23735482, 17543294, 18795903], [4862706, 16352249, 57100612, 6219870], [63203206, 25630930, 35608240, 51357885], [59819625, 64662579, 50925335, 55670434], [29216830, 26446697, 52243336, 58475666], [43138915, 30592834, 43931516, 50628002]], dtype=np.int32), np.array([[71233, 771721, 71505, 706185], [289177, 484147, 358809, 484147], [359355, 549137, 359096, 549137], [350631, 558133, 350630, 554037], [370087, 538677, 370087, 538677], [4432, 899725, 4432, 904077], [678487, 229987, 678487, 229987], [423799, 480614, 423799, 480870], [549958, 284804, 549958, 280708], [423848, 480565, 423848, 480549], [489129, 283940, 554921, 283940], [86641, 822120, 86641, 822120], [206370, 702394, 206370, 567209], [500533, 407959, 500533, 407959], [759723, 79137, 759723, 79137], [563305, 345460, 559209, 345460], [231733, 611478, 231733, 611478], [502682, 406082, 498585, 406082], [554567, 288662, 554567, 288662], [476823, 427846, 476823, 427846], [488823, 415846, 488823, 415846], [431687, 477078, 431687, 477078], [419159, 424070, 415062, 424070], [493399, 345734, 493143, 345718], [678295, 230451, 678295, 230451], [496520, 342596, 496520, 346709], [567109, 276116, 567109, 276116], [624005, 284758, 624005, 284758], [420249, 484420, 420248, 484420], [217715, 621418, 217715, 621418], [344884, 493977, 344884, 493977], [550841, 292132, 550841, 292132], [284262, 558967, 284006, 558967], [152146, 756616, 152146, 756616], [144466, 698763, 144466, 694667], [284261, 624504, 284261, 624504], [288406, 620102, 288405, 620358], [301366, 607383, 301366, 607382], [468771, 435882, 468771, 435882], [555688, 283444, 555688, 283444], [485497, 414820, 485497, 414820], [633754, 275010, 633754, 275010], [419141, 489608, 419157, 489608], [694121, 214387, 694121, 214387], [480869, 427639, 481125, 427639], [489317, 419447, 489301, 419447], [152900, 747672, 152900, 747672], [348516, 494457, 348516, 494457], [534562, 370088, 534562, 370088], [371272, 537475, 371274, 537475], [144194, 760473, 144194, 760473], [567962, 275011, 567962, 275011], [493161, 350052, 493161, 350052], [490138, 348979, 490138, 348979], [328450, 506552, 328450, 506552], [148882, 759593, 148626, 755497], [642171, 266593, 642171, 266593], [685894, 218774, 685894, 218774], [674182, 234548, 674214, 234548], [756347, 152146, 690811, 86353], [612758, 291894, 612758, 291894], [296550, 612214, 296550, 612214], [363130, 475730, 363130, 475730], [691559, 16496, 691559, 16496], [340755, 502202, 336659, 502218], [632473, 210499, 628377, 210483], [564410, 266513, 564410, 266513], [427366, 481399, 427366, 481399], [493159, 349797, 493159, 415605], [331793, 576972, 331793, 576972], [416681, 492084, 416681, 492084], [813496, 95265, 813496, 91153], [695194, 213571, 695194, 213571], [436105, 407124, 436105, 407124], [836970, 6243, 902506, 6243], [160882, 747882, 160882, 747882], [493977, 414788, 489624, 414788], [29184, 551096, 29184, 616888], [903629, 4880, 899517, 4880], [351419, 553250, 351419, 553250], [75554, 767671, 75554, 767671], [279909, 563304, 279909, 563304], [215174, 628054, 215174, 628054], [361365, 481864, 361365, 481864], [424022, 484743, 358486, 484725], [271650, 633018, 271650, 633018], [681896, 226867, 616088, 226867], [222580, 686184, 222564, 686184], [144451, 698778, 209987, 698778], [532883, 310086, 532883, 310086], [628872, 279893, 628872, 279893], [533797, 374951, 533797, 374951], [91713, 817036, 91713, 817036], [427605, 477046, 431718, 477046], [145490, 689529, 145490, 689529], [551098, 291875, 551098, 291875], [349781, 558984, 349781, 558983], [205378, 703115, 205378, 703115], [362053, 546456, 362053, 546456], [612248, 226371, 678040, 226371]], dtype=np.int32)
    # fmt: on
