# Copyright 2023 The Pgx Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

import pgx.core as core
from pgx._flax.struct import dataclass

FALSE = jnp.bool_(False)

# カードと数字の対応
# 0~12 spade, 13~25 heart, 26~38 diamond, 39~51 club
# それぞれのsuitにおいて以下の順で数字が並ぶ
TO_CARD = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]


@dataclass
class State(core.State):
    _step_count: jnp.ndarray = jnp.int32(0)
    # turn 現在のターン数
    turn: jnp.ndarray = jnp.int16(0)
    # current_player 現在のプレイヤーid
    current_player: jnp.ndarray = jnp.int8(-1)
    # 各プレイヤーの観測
    observation: jnp.ndarray = jnp.zeros(52, dtype=jnp.bool_)
    # 報酬　player_id: 0, 1, 2, 3
    reward: jnp.ndarray = jnp.int16([0, 0, 0, 0])
    # シャッフルされたプレイヤーの並び
    shuffled_players: jnp.ndarray = jnp.zeros(4, dtype=jnp.int8)
    # 終端状態
    terminated: jnp.ndarray = FALSE
    truncated: jnp.ndarray = FALSE
    _rng_key: jax.random.KeyArray = jax.random.PRNGKey(0)
    # hand 各プレイヤーの手札
    # index = 0 ~ 12がN, 13 ~ 25がE, 26 ~ 38がS, 39 ~ 51がWの持つ手札
    # 各要素にはカードを表す0 ~ 51の整数が格納される
    hand: jnp.ndarray = jnp.zeros(52, dtype=jnp.int8)
    # bidding_history 各プレイヤーのbidを時系列順に記憶
    # 最大の行動系列長 = 319
    # 各要素には、行動を表す整数が格納される
    # bidを表す0 ~ 34, passを表す35, doubleを表す36, redoubleを表す37, 行動が行われていない-1
    # 各ビッドがどのプレイヤーにより行われたかは、要素のindexから分かる（ix % 4）
    bidding_history: jnp.ndarray = jnp.full(319, -1, dtype=jnp.int8)
    # dealer どのプレイヤーがdealerかを表す
    # 0 = N, 1 = E, 2 = S, 3 = W
    # dealerは最初にbidを行うプレイヤー
    dealer: jnp.ndarray = jnp.zeros(4, dtype=jnp.int8)
    # vul_NS NSチームがvulかどうかを表す
    # 0 = non vul, 1 = vul
    vul_NS: jnp.ndarray = jnp.bool_(False)
    # vul_EW EWチームがvulかどうかを表す
    # 0 = non vul, 1 = vul
    vul_EW: jnp.ndarray = jnp.bool_(False)
    # last_bid 最後にされたbid
    # last_bidder 最後にbidをしたプレイヤー
    # call_x 最後にされたbidがdoubleされているか
    # call_xx 最後にされたbidがredoubleされているか
    last_bid: jnp.ndarray = jnp.int8(-1)
    last_bidder: jnp.ndarray = jnp.int8(-1)
    call_x: jnp.ndarray = jnp.bool_(False)
    call_xx: jnp.ndarray = jnp.bool_(False)
    # legal_actions プレイヤーの可能なbidの一覧
    legal_action_mask: jnp.ndarray = jnp.ones(38, dtype=jnp.bool_)
    # first_denominaton_NS NSチームにおいて、各denominationをどのプレイヤー
    # が最初にbidしたかを表す
    # デノミネーションの順番は C, D, H, S, NT = 0, 1, 2, 3, 4
    first_denomination_NS: jnp.ndarray = jnp.full(5, -1, dtype=jnp.int8)
    # first_denominaton_EW EWチームにおいて、各denominationをどのプレイヤー
    # が最初にbidしたかを表す
    first_denomination_EW: jnp.ndarray = jnp.full(5, -1, dtype=jnp.int8)
    # passの回数
    pass_num: jnp.ndarray = jnp.array(0, dtype=jnp.int8)


@jax.jit
def init(rng: jax.random.KeyArray) -> State:
    rng1, rng2, rng3, rng4, rng5, rng6 = jax.random.split(rng, num=6)
    hand = jnp.arange(0, 52)
    hand = jax.random.permutation(rng2, hand)
    vul_NS = jax.random.randint(rng3, (1,), 0, 2)[0]
    vul_EW = jax.random.randint(rng4, (1,), 0, 2)[0]
    dealer = jax.random.randint(rng5, (1,), 0, 4)[0]
    # shuffled players and arrange in order of NESW
    shuffled_players = _shuffle_players(rng6)
    current_player = shuffled_players[dealer]
    legal_actions = jnp.ones(38, dtype=jnp.bool_)
    # 最初はdable, redoubleできない
    legal_actions = legal_actions.at[36].set(False)
    legal_actions = legal_actions.at[37].set(False)
    state = State(  # type: ignore
        shuffled_players=shuffled_players,
        current_player=current_player,
        hand=hand,
        dealer=dealer,
        vul_NS=vul_NS,
        vul_EW=vul_EW,
        legal_action_mask=legal_actions,
    )
    return state


@jax.jit
def init_by_key(key: jnp.ndarray, rng: jax.random.KeyArray) -> State:
    """Make init state from key"""
    rng1, rng2, rng3, rng4, rng5 = jax.random.split(rng, num=5)
    hand = _key_to_hand(key)
    vul_NS = jax.random.randint(rng2, (1,), 0, 2)[0]
    vul_EW = jax.random.randint(rng3, (1,), 0, 2)[0]
    dealer = jax.random.randint(rng4, (1,), 0, 4)[0]
    # shuffled players and arrange in order of NESW
    shuffled_players = _shuffle_players(rng5)
    current_player = shuffled_players[dealer]
    legal_actions = jnp.ones(38, dtype=jnp.bool_)
    # 最初はdable, redoubleできない
    legal_actions = legal_actions.at[36].set(False)
    legal_actions = legal_actions.at[37].set(False)
    state = State(  # type: ignore
        shuffled_players=shuffled_players,
        current_player=current_player,
        hand=hand,
        dealer=dealer,
        vul_NS=vul_NS,
        vul_EW=vul_EW,
        legal_action_mask=legal_actions,
    )
    return state


@jax.jit
def _shuffle_players(rng: jax.random.KeyArray) -> jnp.ndarray:
    """Randomly arranges player IDs in a list in NESW order.

    Returns:
        jnp.ndarray: A list of 4 player IDs randomly arranged in NESW order.

    Example:
        >>> key = jax.random.PRNGKey(0)
        >>> _shuffle_players(key)
        Array([0, 3, 1, 2], dtype=int8)
    """
    rng1, rng2, rng3, rng4 = jax.random.split(rng, num=4)
    # player_id = 0, 1 -> team a
    team_a_players = jax.random.permutation(
        rng2, jnp.arange(2, dtype=jnp.int8)
    )
    # player_id = 2, 3 -> team b
    team_b_players = jax.random.permutation(
        rng3, jnp.arange(2, 4, dtype=jnp.int8)
    )
    # decide which team is on
    # Randomly determine NSteam and EWteam
    # Arrange in order of NESW
    return jax.lax.cond(
        jax.random.randint(rng4, (1,), 1, 2)[0] == 1,
        lambda: jnp.array(
            [
                team_a_players[0],
                team_b_players[0],
                team_a_players[1],
                team_b_players[1],
            ]
        ),
        lambda: jnp.array(
            [
                team_b_players[0],
                team_a_players[0],
                team_b_players[1],
                team_a_players[1],
            ]
        ),
    )


@jax.jit
def _player_position(player: jnp.ndarray, state: State) -> jnp.ndarray:
    return jax.lax.cond(
        player != -1,
        lambda: jnp.int8(
            jnp.argmax(state.shuffled_players == player)
        ),  # playerと同じ要素のstate.shuffle_playersのindexを返す
        lambda: jnp.int8(-1),
    )


@jax.jit
def step(
    state: State,
    action: int,
    hash_keys: jnp.ndarray,
    hash_values: jnp.ndarray,
) -> State:
    # fmt: off
    state = state.replace(bidding_history=state.bidding_history.at[state.turn].set(action))  # type: ignore
    # fmt: on
    return jax.lax.cond(
        state.legal_action_mask[action] == 0,  # 非合法手判断
        lambda: _illegal_step(state),
        lambda: jax.lax.cond(
            action >= 35,
            lambda: jax.lax.switch(
                action - 35,
                [
                    lambda: jax.lax.cond(
                        _is_terminated(_state_pass(state)),
                        lambda: _terminated_step(
                            _state_pass(state), hash_keys, hash_values
                        ),
                        lambda: _continue_step(_state_pass(state)),
                    ),
                    lambda: _continue_step(_state_X(state)),
                    lambda: _continue_step(_state_XX(state)),
                ],
            ),
            lambda: _continue_step(_state_bid(state, action)),
        ),
    )


@jax.jit
def duplicate(
    init_state: State,
) -> State:
    """Make duplicated state where NSplayer and EWplayer are swapped"""
    duplicated_state = copy.deepcopy(init_state)
    ix = jnp.array([1, 0, 3, 2])
    # fmt: off
    duplicated_state = duplicated_state.replace(shuffled_players=duplicated_state.shuffled_players[ix])  # type: ignore
    # fmt: on
    return duplicated_state


@jax.jit
def _illegal_step(
    state: State,
) -> State:
    """Return state when an illegal move is detected"""
    illegal_rewards = jnp.zeros(4, dtype=jnp.int16)
    # fmt: off
    return state.replace(terminated=jnp.bool_(True), current_player=jnp.int8(-1), reward=illegal_rewards)  # type: ignore
    # fmt: on


@jax.jit
def _terminated_step(
    state: State,
    hash_keys: jnp.ndarray,
    hash_values: jnp.ndarray,
) -> State:
    """Return state if the game is successfully completed"""
    terminated = jnp.bool_(True)
    current_player = jnp.int8(-1)
    reward = _reward(state, hash_keys, hash_values)
    # fmt: off
    return state.replace(terminated=terminated, current_player=current_player, reward=reward)  # type: ignore
    # fmt: on


@jax.jit
def _continue_step(
    state: State,
) -> State:
    """Return state when the game continues"""
    # fmt: off
    # 次ターンのプレイヤー、ターン数
    state = state.replace(current_player=state.shuffled_players[(state.dealer + state.turn + 1) % 4], turn=state.turn + 1)  # type: ignore
    # 次のターンにX, XXが合法手か判断
    x_mask, xx_mask = _update_legal_action_X_XX(state)
    return state.replace(legal_action_mask=state.legal_action_mask.at[36].set(x_mask).at[37].set(xx_mask), reward=jnp.zeros(4, dtype=jnp.int16))  # type: ignore
    # fmt: on


@jax.jit
def _is_terminated(state: State) -> bool:
    """Check if the game is finished
    Four consecutive passes if not bid (pass out), otherwise three consecutive passes
    """
    return jax.lax.cond(
        ((state.last_bid == -1) & (state.pass_num == 4))
        | ((state.last_bid != -1) & (state.pass_num == 3)),
        lambda: True,
        lambda: False,
    )


@jax.jit
def _reward(
    state: State,
    hash_keys: jnp.ndarray,
    hash_values: jnp.ndarray,
) -> jnp.ndarray:
    """Return reward
    If pass out, 0 reward for everyone; if bid, calculate and return reward
    """
    return jax.lax.cond(
        (state.last_bid == -1) & (state.pass_num == 4),
        lambda: jnp.zeros(4, dtype=jnp.int16),  # pass out
        lambda: _make_reward(  # caluculate reward
            state, hash_keys, hash_values
        ),
    )


@jax.jit
def _make_reward(
    state: State,
    hash_keys: jnp.ndarray,
    hash_values: jnp.ndarray,
) -> jnp.ndarray:
    """Calculate rewards for each player by dds results

    Returns:
        np.ndarray: A list of rewards for each player in playerID order
    """
    # Extract contract from state
    declare_position, denomination, level, vul = _contract(state)
    # Calculate trick table from hash table
    dds_tricks = _calculate_dds_tricks(state, hash_keys, hash_values)
    # Calculate the tricks you could have accomplished with this contraption
    dds_trick = dds_tricks[declare_position * 5 + denomination]
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
    player_positions = jax.vmap(lambda i: _player_position(i, state))(
        jnp.arange(4)
    )
    partners = jax.vmap(lambda pos: _is_partner(pos, declare_position))(
        player_positions
    )

    return jnp.where(partners, score, -score)


@jax.jit
def _calc_score(
    denomination: jnp.ndarray,
    level: jnp.ndarray,
    vul: jnp.ndarray,
    call_x: jnp.ndarray,
    call_xx: jnp.ndarray,
    trick: jnp.ndarray,
) -> jnp.ndarray:
    """Calculate score from contract and trick
    Returns:
        np.ndarray: A score of declarer team
    """
    return jax.lax.cond(
        level + 6 > trick,
        lambda: _down_score(level, vul, call_x, call_xx, trick),
        lambda: _make_score(denomination, level, vul, call_x, call_xx, trick),
    )


@jax.jit
def _down_score(
    level: jnp.ndarray,
    vul: jnp.ndarray,
    call_x: jnp.ndarray,
    call_xx: jnp.ndarray,
    trick: jnp.ndarray,
) -> jnp.ndarray:
    """Calculate down score from contract and trick
    Returns:
        np.ndarray: A score of declarer team
    """
    # fmt: off
    _DOWN = jnp.array([-50, -100, -150, -200, -250, -300, -350, -400, -450, -500, -550, -600, -650], dtype=jnp.int16)
    _DOWN_VUL = jnp.array([-100, -200, -300, -400, -500, -600, -700, -800, -900, -1000, -1100, -1200, -1300], dtype=jnp.int16)
    _DOWN_X = jnp.array([-100, -300, -500, -800, -1100, -1400, -1700, -2000, -2300, -2600, -2900, -3200, -3500], dtype=jnp.int16)
    _DOWN_X_VUL = jnp.array([-200, -500, -800, -1100, -1400, -1700, -2000, -2300, -2600, -2900, -3200, -3500, -3800], dtype=jnp.int16)
    _DOWN_XX = jnp.array([-200, -600, -1000, -1600, -2200, -2800, -3400, -4000, -4600, -5200, -5800, -6400, -7000], dtype=jnp.int16)
    _DOWN_XX_VUL = jnp.array([-400, -1000, -1600, -2200, -2800, -3400, -4000, -4600, -5200, -5800, -6400, -7000, -7600], dtype=jnp.int16)
    # fmt: on
    under_trick = level + 6 - trick
    down = jax.lax.cond(
        vul,
        lambda: _DOWN_VUL[under_trick - 1],
        lambda: _DOWN[under_trick - 1],
    )
    down_x = jax.lax.cond(
        vul,
        lambda: _DOWN_X_VUL[under_trick - 1],
        lambda: _DOWN_X[under_trick - 1],
    )
    down_xx = jax.lax.cond(
        vul,
        lambda: _DOWN_XX_VUL[under_trick - 1],
        lambda: _DOWN_XX[under_trick - 1],
    )
    return jax.lax.cond(
        call_xx,
        lambda: down_xx,
        lambda: jax.lax.cond(call_x, lambda: down_x, lambda: down),
    )


@jax.jit
def _make_score(
    denomination: jnp.ndarray,
    level: jnp.ndarray,
    vul: jnp.ndarray,
    call_x: jnp.ndarray,
    call_xx: jnp.ndarray,
    trick: jnp.ndarray,
) -> jnp.ndarray:
    """Calculate make score from contract and trick
    Returns:
        np.ndarray: A score of declarer team
    """
    # fmt: off
    _MINOR = jnp.int16(20)
    _MAJOR = jnp.int16(30)
    _NT = jnp.int16(10)
    _MAKE = jnp.int16(50)
    _MAKE_X = jnp.int16(50)
    _MAKE_XX = jnp.int16(50)

    _GAME = jnp.int16(250)
    _GAME_VUL = jnp.int16(450)
    _SMALL_SLAM = jnp.int16(500)
    _SMALL_SLAM_VUL = jnp.int16(750)
    _GRAND_SLAM = jnp.int16(500)
    _GRAND_SLAM_VUL = jnp.int16(750)

    _OVERTRICK_X = jnp.int16(100)
    _OVERTRICK_X_VUL = jnp.int16(200)
    _OVERTRICK_XX = jnp.int16(200)
    _OVERTRICK_XX_VUL = jnp.int16(400)
    # fmt: on
    over_trick_score_per_trick = jnp.int16(0)
    over_trick = trick - level - jnp.int16(6)
    score = jnp.int16(0)
    score, over_trick_score_per_trick = jax.lax.switch(
        denomination,
        [
            lambda: (
                score + jnp.int16(_MINOR * level),
                over_trick_score_per_trick + _MINOR,
            ),
            lambda: (
                score + jnp.int16(_MINOR * level),
                over_trick_score_per_trick + _MINOR,
            ),
            lambda: (
                score + jnp.int16(_MAJOR * level),
                over_trick_score_per_trick + _MAJOR,
            ),
            lambda: (
                score + jnp.int16(_MAJOR * level),
                over_trick_score_per_trick + _MAJOR,
            ),
            lambda: (
                score + jnp.int16(_MAJOR * level + _NT),
                over_trick_score_per_trick + _MAJOR,
            ),
        ],
    )
    score = jax.lax.cond(
        call_xx,
        lambda: score * jnp.int16(4),
        lambda: jax.lax.cond(
            call_x, lambda: score * jnp.int16(2), lambda: score
        ),
    )
    game_bonus = jax.lax.cond(vul, lambda: _GAME_VUL, lambda: _GAME)
    small_slam_bonus = jax.lax.cond(
        vul, lambda: _SMALL_SLAM_VUL, lambda: _SMALL_SLAM
    )
    grand_slam_bonus = jax.lax.cond(
        vul, lambda: _GRAND_SLAM_VUL, lambda: _GRAND_SLAM
    )

    score = jax.lax.cond(
        score >= 100, lambda: score + game_bonus, lambda: score
    )
    score = jax.lax.cond(
        level >= 6, lambda: score + small_slam_bonus, lambda: score
    )
    score = jax.lax.cond(
        level == 7, lambda: score + grand_slam_bonus, lambda: score
    )
    score += _MAKE  # make bonus

    overtrick_x = jax.lax.cond(
        vul, lambda: _OVERTRICK_X_VUL, lambda: _OVERTRICK_X
    )
    overtrick_xx = jax.lax.cond(
        vul, lambda: _OVERTRICK_XX_VUL, lambda: _OVERTRICK_XX
    )
    score, over_trick_score_per_trick = jax.lax.cond(
        call_x | call_xx,
        lambda: jax.lax.cond(
            call_xx,
            lambda: (score + _MAKE_X + _MAKE_XX, overtrick_xx),
            lambda: (score + _MAKE_X, overtrick_x),
        ),
        lambda: (score, over_trick_score_per_trick),
    )
    return score + over_trick_score_per_trick * over_trick


@jax.jit
def _contract(
    state: State,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return Contract which has position of declare ,denomination, level"""
    denomination = state.last_bid % 5
    level = state.last_bid // 5 + 1
    declare_position, vul = jax.lax.cond(
        _position_to_team(_player_position(state.last_bidder, state)) == 0,
        lambda: (
            _player_position(state.first_denomination_NS[denomination], state),
            state.vul_NS,
        ),
        lambda: (
            _player_position(state.first_denomination_EW[denomination], state),
            state.vul_EW,
        ),
    )
    return declare_position, denomination, level, vul


@jax.jit
def _state_pass(
    state: State,
) -> State:
    """Change state if pass is taken"""
    return state.replace(pass_num=state.pass_num + 1)  # type: ignore


@jax.jit
def _state_X(state: State) -> State:
    """Change state if double(X) is taken"""
    return state.replace(call_x=jnp.bool_(True), pass_num=jnp.int8(0))  # type: ignore


@jax.jit
def _state_XX(state: State) -> State:
    """Change state if double(XX) is taken"""
    return state.replace(call_xx=jnp.bool_(True), pass_num=jnp.int8(0))  # type: ignore


@jax.jit
def _state_bid(state: State, action: int) -> State:
    """Change state if bid is taken"""
    # 最後のbidとそのプレイヤーを保存
    # fmt: off
    state = state.replace(last_bid=jnp.int8(action), last_bidder=state.current_player)  # type: ignore
    # fmt: on
    # チーム内で各denominationを最初にbidしたプレイヤー
    denomination = _bid_to_denomination(action)
    team = _position_to_team(_player_position(state.last_bidder, state))
    # fmt: off
    # team = 1ならEWチーム
    state = jax.lax.cond(team & (state.first_denomination_EW[denomination] == -1),
                         lambda: state.replace(first_denomination_EW=state.first_denomination_EW.at[denomination].set(state.last_bidder.astype(jnp.int8))),  # type: ignore
                         lambda: state)  # type: ignore
    # team = 0ならNSチーム
    state = jax.lax.cond((team == 0) & (state.first_denomination_NS[denomination] == -1),
                         lambda: state.replace(first_denomination_NS=state.first_denomination_NS.at[denomination].set(state.last_bidder.astype(jnp.int8))),  # type: ignore
                         lambda: state)  # type: ignore
    # fmt: on
    # 小さいbidを非合法手にする
    mask = jnp.arange(38) < action + 1
    return state.replace(legal_action_mask=jnp.where(mask, jnp.bool_(0), state.legal_action_mask), call_x=jnp.bool_(False), call_xx=jnp.bool_(False), pass_num=jnp.int8(0))  # type: ignore


@jax.jit
def _bid_to_denomination(bid: int) -> int:
    """Calcularete denomination of bid"""
    return bid % 5


@jax.jit
def _position_to_team(position: jnp.ndarray) -> jnp.ndarray:
    """Determine which team from the position
    0: NS team, 1: EW team
    """
    return position % 2


@jax.jit
def _update_legal_action_X_XX(
    state: State,
) -> Tuple[bool, bool]:
    """Determine if X or XX is a legal move for the next player"""
    return jax.lax.cond(
        state.last_bidder != -1,
        lambda: (_is_legal_X(state), _is_legal_XX(state)),
        lambda: (False, False),
    )


@jax.jit
def _is_legal_X(state: State) -> bool:
    return jax.lax.cond(
        (state.call_x == 0)
        & (state.call_xx == 0)
        & (
            _is_partner(
                _player_position(state.last_bidder, state),
                _player_position(state.current_player, state),
            )
            == 0
        ),
        lambda: True,
        lambda: False,
    )


@jax.jit
def _is_legal_XX(state: State) -> bool:
    return jax.lax.cond(
        state.call_x
        & (state.call_xx == 0)
        & (
            _is_partner(
                _player_position(state.last_bidder, state),
                _player_position(state.current_player, state),
            )
        ),
        lambda: True,
        lambda: False,
    )


@jax.jit
def _is_partner(position1: jnp.ndarray, position2: jnp.ndarray) -> jnp.ndarray:
    """Determine if positon1 and position2 belong to the same team"""
    return (abs(position1 - position2) + 1) % 2


def _state_to_pbn(state: State) -> str:
    """Convert state to pbn format"""
    pbn = "N:"
    for i in range(4):  # player
        hand = jnp.sort(state.hand[i * 13 : (i + 1) * 13])
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


# @jax.jit
def _state_to_key(state: State) -> jnp.ndarray:
    """Convert state to key of dds table"""
    hand = state.hand
    key = jnp.zeros(52, dtype=jnp.int8)
    for i in range(52):  # N: 0, E: 1, S: 2, W: 3
        key = key.at[hand[i]].set(i // 13)
    key = key.reshape(4, 13)
    return _to_binary(key)


def _pbn_to_key(pbn: str) -> jnp.ndarray:
    """Convert pbn to key of dds table"""
    key = jnp.zeros(52, dtype=jnp.int8)
    hands = pbn[2:]
    for player, hand in enumerate(list(hands.split())):  # for each player
        for suit, cards in enumerate(list(hand.split("."))):  # for each suit
            for card in cards:  # for each card
                card_num = _card_str_to_int(card) + suit * 13
                key[card_num] = player
    key = key.reshape(4, 13)
    return _to_binary(key)


@jax.jit
def _to_binary(x: jnp.ndarray) -> jnp.ndarray:
    bases = jnp.array([4**i for i in range(13)], dtype=jnp.int32)[::-1]
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


@jax.jit
def _key_to_hand(key: jnp.ndarray) -> jnp.ndarray:
    """Convert key to hand"""

    def _convert_quat(j):
        shifts = jnp.arange(24, -1, step=-2)
        quat_digits = (j >> shifts) & 0b11
        return quat_digits

    cards = jax.vmap(_convert_quat)(key).flatten()
    hand = jnp.zeros((4, 13), dtype=jnp.int8)
    for i in range(4):
        count = 0
        for j in range(52):
            hand, count = jax.lax.cond(
                cards[j] == i,
                lambda: (hand.at[i, count].set(j), count + 1),
                lambda: (hand, count),
            )
    return hand.flatten()


@jax.jit
def _value_to_dds_tricks(value: jnp.ndarray) -> jnp.ndarray:
    """Convert values to dds tricks
    >>> value = jnp.array([4160, 904605, 4160, 904605])
    >>> _value_to_dds_tricks(value)
    Array([ 0,  1,  0,  4,  0, 13, 12, 13,  9, 13,  0,  1,  0,  4,  0, 13, 12,
           13,  9, 13], dtype=int8)
    """

    def _convert_hex(j):
        shifts = jnp.arange(16, -1, step=-4)
        hex_digits = (j >> shifts) & 0xF
        return hex_digits

    hex_digits = jax.vmap(_convert_hex)(value).flatten()
    return jnp.array(hex_digits, dtype=jnp.int8)


@jax.jit
def _calculate_dds_tricks(
    state: State,
    hash_keys: jnp.ndarray,
    hash_values: jnp.ndarray,
) -> jnp.ndarray:
    key = _state_to_key(state)
    return _value_to_dds_tricks(
        _find_value_from_key(key, hash_keys, hash_values)
    )


@jax.jit
def _find_value_from_key(
    key: jnp.ndarray, hash_keys: jnp.ndarray, hash_values: jnp.ndarray
):
    """Find a value matching key without batch processing
    >>> VALUES = jnp.arange(20).reshape(5, 4)
    >>> KEYS = jnp.arange(20).reshape(5, 4)
    >>> key = jnp.arange(4, 8)
    >>> _find_value_from_key(key, KEYS, VALUES)
    Array([4, 5, 6, 7], dtype=int32)
    """
    mask = jnp.where(
        jnp.all((hash_keys == key), axis=1),
        jnp.ones(1, dtype=np.bool_),
        jnp.zeros(1, dtype=np.bool_),
    )
    ix = jnp.argmax(mask)
    return hash_values[ix]


def _load_sample_hash() -> Tuple[jnp.ndarray, jnp.ndarray]:
    # fmt: off
    return jnp.array([[19556549, 61212362, 52381660, 50424958], [53254536, 21854346, 37287883, 14009558], [44178585, 6709002, 23279217, 16304124], [36635659, 48114215, 13583653, 26208086], [44309474, 39388022, 28376136, 59735189], [61391908, 52173479, 29276467, 31670621], [34786519, 13802254, 57433417, 43152306], [48319039, 55845612, 44614774, 58169152], [47062227, 32289487, 12941848, 21338650], [36579116, 15643926, 64729756, 18678099], [62136384, 37064817, 59701038, 39188202], [13417016, 56577539, 25995845, 27248037], [61125047, 43238281, 23465183, 20030494], [7139188, 31324229, 58855042, 14296487], [2653767, 47502150, 35507905, 43823846], [31453323, 11605145, 6716808, 41061859], [21294711, 49709, 26110952, 50058629], [48130172, 3340423, 60445890, 7686579], [16041939, 27817393, 37167847, 9605779], [61154057, 17937858, 12254613, 12568801], [13796245, 46546127, 49123920, 51772041], [7195005, 45581051, 41076865, 17429796], [20635965, 14642724, 7001617, 45370595], [35616421, 19938131, 45131030, 16524847], [14559399, 15413729, 39188470, 535365], [48743216, 39672069, 60203571, 60210880], [63862780, 2462075, 23267370, 36595020], [11229980, 11616119, 20292263, 3695004], [24135854, 37532826, 54421444, 14130249], [42798085, 33026223, 2460251, 18566823], [49558558, 65537599, 14768519, 31103243], [44321156, 20075251, 42663767, 11615602], [20186726, 42678073, 11763300, 56739471], [57534601, 16703645, 6039937, 17088125], [50795278, 17350238, 11955835, 21538127], [45919621, 5520088, 27736513, 52674927], [13928720, 57324148, 28222453, 15480785], [910719, 47238830, 26345802, 56166394], [58841430, 1098476, 61890558, 26907706], [10379825, 8624220, 39701822, 29045990], [54444873, 50000486, 48563308, 55867521], [47291672, 22084522, 45484828, 32878832], [55350706, 23903891, 46142039, 11499952], [4708326, 27588734, 31010458, 11730972], [27078872, 59038086, 62842566, 51147874], [28922172, 32377861, 9109075, 10154350], [26104086, 62786977, 224865, 14335943], [20448626, 33187645, 34338784, 26382893], [29194006, 19635744, 24917755, 8532577], [64047742, 34885257, 5027048, 58399668], [27603972, 26820121, 44837703, 63748595], [60038456, 19611050, 7928914, 38555047], [13583610, 19626473, 22239272, 19888268], [28521006, 1743692, 31319264, 15168920], [64585849, 63931241, 57019799, 14189800], [2632453, 7269809, 60404342, 57986125], [1996183, 49918209, 49490468, 47760867], [6233580, 15318425, 51356120, 55074857], [15769884, 61654638, 8374039, 43685186], [44162419, 47272176, 62693156, 35359329], [36345796, 15667465, 53341561, 2978505], [1664472, 12761950, 34145519, 55197543], [37567005, 3228834, 6198166, 15646487], [63233399, 42640049, 12969011, 41620641], [22090925, 3386355, 56655568, 31631004], [16442787, 9420273, 48595545, 29770176], [49404288, 37823218, 58551818, 6772527], [36575583, 53847347, 32379432, 1630009], [9004247, 12999580, 48379959, 14252211], [25850203, 26136823, 64934025, 29362603], [10214276, 43557352, 33387586, 55512773], [45810841, 49561478, 41130845, 27034816], [34460081, 16560450, 57722793, 41007718], [53414778, 6845803, 15340368, 16647575], [30535873, 5193469, 43608154, 11391114], [20622004, 34424126, 31475211, 29619615], [10428836, 49656416, 7912677, 33427787], [57600861, 18251799, 46147432, 58946294], [6760779, 14675737, 42952146, 5480498], [46037552, 39969058, 30103468, 55330772], [64466305, 29376674, 49914839, 55269895], [36494113, 27010567, 65752150, 12395385], [49385632, 19550767, 39809394, 58806235], [20987521, 37444597, 49290126, 42326125], [37150229, 37487849, 28254397, 32949826], [9724895, 53813417, 19431235, 27438556], [42132748, 47073733, 19396568, 10026137], [3961481, 27204521, 62087205, 37602005], [22178323, 17505521, 42006207, 44143605], [12753258, 63063515, 61993175, 8920985], [10998000, 64833190, 6446892, 63676805], [66983817, 63684932, 18378359, 39946382], [63476803, 60000436, 19442420, 66417845], [38004446, 64752157, 42570179, 52844512], [1270809, 23735482, 17543294, 18795903], [4862706, 16352249, 57100612, 6219870], [63203206, 25630930, 35608240, 51357885], [59819625, 64662579, 50925335, 55670434], [29216830, 26446697, 52243336, 58475666], [43138915, 30592834, 43931516, 50628002]], dtype=jnp.int32), jnp.array([[71233, 771721, 71505, 706185], [289177, 484147, 358809, 484147], [359355, 549137, 359096, 549137], [350631, 558133, 350630, 554037], [370087, 538677, 370087, 538677], [4432, 899725, 4432, 904077], [678487, 229987, 678487, 229987], [423799, 480614, 423799, 480870], [549958, 284804, 549958, 280708], [423848, 480565, 423848, 480549], [489129, 283940, 554921, 283940], [86641, 822120, 86641, 822120], [206370, 702394, 206370, 567209], [500533, 407959, 500533, 407959], [759723, 79137, 759723, 79137], [563305, 345460, 559209, 345460], [231733, 611478, 231733, 611478], [502682, 406082, 498585, 406082], [554567, 288662, 554567, 288662], [476823, 427846, 476823, 427846], [488823, 415846, 488823, 415846], [431687, 477078, 431687, 477078], [419159, 424070, 415062, 424070], [493399, 345734, 493143, 345718], [678295, 230451, 678295, 230451], [496520, 342596, 496520, 346709], [567109, 276116, 567109, 276116], [624005, 284758, 624005, 284758], [420249, 484420, 420248, 484420], [217715, 621418, 217715, 621418], [344884, 493977, 344884, 493977], [550841, 292132, 550841, 292132], [284262, 558967, 284006, 558967], [152146, 756616, 152146, 756616], [144466, 698763, 144466, 694667], [284261, 624504, 284261, 624504], [288406, 620102, 288405, 620358], [301366, 607383, 301366, 607382], [468771, 435882, 468771, 435882], [555688, 283444, 555688, 283444], [485497, 414820, 485497, 414820], [633754, 275010, 633754, 275010], [419141, 489608, 419157, 489608], [694121, 214387, 694121, 214387], [480869, 427639, 481125, 427639], [489317, 419447, 489301, 419447], [152900, 747672, 152900, 747672], [348516, 494457, 348516, 494457], [534562, 370088, 534562, 370088], [371272, 537475, 371274, 537475], [144194, 760473, 144194, 760473], [567962, 275011, 567962, 275011], [493161, 350052, 493161, 350052], [490138, 348979, 490138, 348979], [328450, 506552, 328450, 506552], [148882, 759593, 148626, 755497], [642171, 266593, 642171, 266593], [685894, 218774, 685894, 218774], [674182, 234548, 674214, 234548], [756347, 152146, 690811, 86353], [612758, 291894, 612758, 291894], [296550, 612214, 296550, 612214], [363130, 475730, 363130, 475730], [691559, 16496, 691559, 16496], [340755, 502202, 336659, 502218], [632473, 210499, 628377, 210483], [564410, 266513, 564410, 266513], [427366, 481399, 427366, 481399], [493159, 349797, 493159, 415605], [331793, 576972, 331793, 576972], [416681, 492084, 416681, 492084], [813496, 95265, 813496, 91153], [695194, 213571, 695194, 213571], [436105, 407124, 436105, 407124], [836970, 6243, 902506, 6243], [160882, 747882, 160882, 747882], [493977, 414788, 489624, 414788], [29184, 551096, 29184, 616888], [903629, 4880, 899517, 4880], [351419, 553250, 351419, 553250], [75554, 767671, 75554, 767671], [279909, 563304, 279909, 563304], [215174, 628054, 215174, 628054], [361365, 481864, 361365, 481864], [424022, 484743, 358486, 484725], [271650, 633018, 271650, 633018], [681896, 226867, 616088, 226867], [222580, 686184, 222564, 686184], [144451, 698778, 209987, 698778], [532883, 310086, 532883, 310086], [628872, 279893, 628872, 279893], [533797, 374951, 533797, 374951], [91713, 817036, 91713, 817036], [427605, 477046, 431718, 477046], [145490, 689529, 145490, 689529], [551098, 291875, 551098, 291875], [349781, 558984, 349781, 558983], [205378, 703115, 205378, 703115], [362053, 546456, 362053, 546456], [612248, 226371, 678040, 226371]], dtype=jnp.int32)
    # fmt: on
