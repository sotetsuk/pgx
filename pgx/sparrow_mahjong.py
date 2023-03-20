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

"""Sparrow Mahjong

すずめ雀のルール
  * 2-6人
  * 牌はソウズと發中のみの11種44枚
  * 手牌は基本5枚、ツモ直後6枚
  * 順子か刻子を2つ完成で手牌完成
  * チーポンカンはなし
  * ドラは表示牌がそのままドラ
  * 中はすべて赤ドラ、發は赤ドラなし、各牌ひとつ赤ドラがある
  * フリテンは自分が捨てた牌はあがれないが、他の牌ではあがれる

Pgx実装での違い
  * 3人のみ
  * 一局のみ
  * 行動選択は打牌選択のみ
    * 打牌選択時、赤牌とそうでない牌がある場合には、常に赤牌でないほうを打牌する
    * ロン・ツモは自動判定
  * 誰も一つも行動を取らずにエピソードが終わるのを避けるため、親の第一ツモでの和了の場合は配牌し直し（天和を避けている）
"""

import jax
import jax.lax as lax
import jax.numpy as jnp

import pgx.core as core
from pgx._flax.struct import dataclass

TRUE = jnp.bool_(True)
FALSE = jnp.bool_(False)
NUM_TILES = 44
NUM_TILE_TYPES = 11
N_PLAYER = 3
MAX_RIVER_LENGTH = 10
NUM_CACHE = 160
WIN_HANDS = jnp.int32([18, 78, 90, 378, 390, 450, 778, 790, 850, 1150, 1550, 1878, 1890, 1950, 2250, 2650, 3878, 3890, 3950, 4250, 4650, 5750, 7750, 9378, 9390, 9450, 9750, 10150, 11250, 13250, 19378, 19390, 19450, 19750, 20150, 21250, 23250, 28750, 38750, 46878, 46890, 46950, 47250, 47650, 48750, 50750, 56250, 66250, 96878, 96890, 96950, 97250, 97650, 98750, 100750, 106250, 116250, 143750, 193750, 234378, 234390, 234450, 234750, 235150, 236250, 238250, 243750, 253750, 281250, 331250, 484378, 484390, 484450, 484750, 485150, 486250, 488250, 493750, 503750, 531250, 581250, 718750, 968750, 1171878, 1171890, 1171950, 1172250, 1172650, 1173750, 1175750, 1181250, 1191250, 1218750, 1268750, 1406250, 1656250, 2421878, 2421890, 2421950, 2422250, 2422650, 2423750, 2425750, 2431250, 2441250, 2468750, 2518750, 2656250, 2906250, 3593750, 4843750, 5859378, 5859390, 5859450, 5859750, 5860150, 5861250, 5863250, 5868750, 5878750, 5906250, 5956250, 6093750, 6343750, 7031250, 8281250, 12109378, 12109390, 12109450, 12109750, 12110150, 12111250, 12113250, 12118750, 12128750, 12156250, 12206250, 12343750, 12593750, 13281250, 14531250, 17968750, 24218750, 29296878, 29296890, 29296950, 29297250, 29297650, 29298750, 29300750, 29306250, 29316250, 29343750, 29393750, 29531250, 29781250, 30468750, 31718750, 35156250, 41406250])  # type: ignore
BASE_SCORES = jnp.int32([4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 2, 4, 4, 4, 4, 3, 3, 3, 3, 3, 2, 3, 2, 4, 4, 4, 4, 3, 4, 3, 3, 3, 3, 3, 2, 3, 2, 3, 2, 4, 4, 4, 4, 3, 4, 3, 4, 3, 3, 3, 3, 3, 2, 3, 2, 3, 2, 3, 2, 4, 4, 4, 4, 3, 4, 3, 4, 3, 4, 3, 3, 3, 3, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 4, 4, 4, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 3, 3, 3, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 4, 4, 4, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 3, 3, 3, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 4, 4, 4, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3])  # type: ignore
YAKU_SCORES = jnp.int32([15, 15, 15, 0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 10, 0, 10, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 10, 0, 10, 0, 1, 1, 10, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 10, 0, 10, 0, 1, 1, 10, 1, 1, 1, 10, 1, 0, 10, 0, 10, 0, 1, 1, 10, 1, 1, 1, 10, 1, 10, 10, 0, 10, 0, 10, 0, 1, 1, 10, 1, 1, 1, 10, 1, 10, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 15, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # type: ignore
MAX_SCORE = 26  # 親の中含むスーパーレッド自摸和了 (1 + 2 + 20 + 2) // 2 * 2


@dataclass
class State(core.State):
    current_player: jnp.ndarray = jnp.int8(0)
    observation: jnp.ndarray = jnp.zeros((15, 11), dtype=jnp.bool_)
    reward: jnp.ndarray = jnp.zeros(3, dtype=jnp.float32)
    terminated: jnp.ndarray = FALSE
    truncated: jnp.ndarray = FALSE
    legal_action_mask: jnp.ndarray = jnp.zeros(9, dtype=jnp.bool_)
    _rng_key: jax.random.KeyArray = jax.random.PRNGKey(0)
    _step_count: jnp.ndarray = jnp.int32(0)
    # --- Sparrow Mahjong specific ---
    turn: jnp.ndarray = jnp.int32(0)  # 0 = dealer
    rivers: jnp.ndarray = -jnp.ones(
        (N_PLAYER, MAX_RIVER_LENGTH), dtype=jnp.int32
    )  # tile type (0~10) is set
    last_discard: jnp.ndarray = jnp.int32(-1)  # tile type (0~10) is set
    hands: jnp.ndarray = jnp.zeros(
        (N_PLAYER, NUM_TILE_TYPES), dtype=jnp.int32
    )  # tile type (0~10) is set
    n_red_in_hands: jnp.ndarray = jnp.zeros(
        (N_PLAYER, NUM_TILE_TYPES), dtype=jnp.int32
    )
    is_red_in_river: jnp.ndarray = jnp.zeros(
        (N_PLAYER, MAX_RIVER_LENGTH), dtype=jnp.bool_
    )
    wall: jnp.ndarray = jnp.zeros(
        NUM_TILES, dtype=jnp.int32
    )  # tile id (0~43) is set
    draw_ix: jnp.ndarray = jnp.int32(N_PLAYER * 5)
    shuffled_players: jnp.ndarray = jnp.zeros(
        N_PLAYER, dtype=jnp.int8
    )  # 0: dealer, ...
    dora: jnp.ndarray = jnp.int32(0)  # tile type (0~10) is set
    scores: jnp.ndarray = jnp.zeros(3, dtype=jnp.int32)  # 0 = dealer


class SparrowMahjong(core.Env):
    def __init__(
        self,
    ):
        super().__init__()

    def _init(self, key: jax.random.KeyArray) -> State:
        key, subkey = jax.random.split(key)
        state = _init(subkey)

        def f(x):
            k, _subkey = jax.random.split(x[0])
            s = _init(_subkey)
            return k, s

        # avoid tenhou
        key, state = lax.while_loop(
            lambda x: x[-1].terminated, f, (key, state)
        )
        return state

    def _step(self, state: core.State, action: jnp.ndarray) -> State:
        assert isinstance(state, State)
        # discard tile
        hands = state.hands.at[state.turn % N_PLAYER, action].add(-1)
        is_red_discarded = (
            hands[state.turn % N_PLAYER, action]
            < state.n_red_in_hands[state.turn % N_PLAYER, action]
        )
        n_red_in_hands = state.n_red_in_hands.at[
            state.turn % N_PLAYER, action
        ].add(-is_red_discarded.astype(jnp.int32))
        rivers = state.rivers.at[
            state.turn % N_PLAYER, state.turn // N_PLAYER
        ].set(action)
        is_red_in_river = state.is_red_in_river.at[
            state.turn % N_PLAYER, state.turn // N_PLAYER
        ].set(is_red_discarded)
        last_discard = action
        state = state.replace(  # type: ignore
            hands=hands,
            n_red_in_hands=n_red_in_hands,
            rivers=rivers,
            is_red_in_river=is_red_in_river,
            last_discard=last_discard,
        )

        scores = _hands_to_score(state)  # type: ignore
        winning_players = _check_ron(state, scores)  # type: ignore
        return lax.cond(
            jnp.any(winning_players),
            lambda: _step_by_ron(state, scores, winning_players),  # type: ignore
            lambda: lax.cond(
                jnp.bool_(NUM_TILES - 1 <= state.draw_ix),  # type: ignore
                lambda: _step_by_tie(state),
                lambda: _step_non_tied(state, scores),  # type: ignore
            ),
        )

    def _observe(
        self, state: core.State, player_id: jnp.ndarray
    ) -> jnp.ndarray:
        assert isinstance(state, State)
        return _observe(state, player_id)

    @property
    def name(self) -> str:
        return "SparrowMahjong"

    @property
    def version(self) -> str:
        return "alpha"

    @property
    def num_players(self) -> int:
        return 3


def _init(rng: jax.random.KeyArray):
    # shuffle players and wall
    key1, key2 = jax.random.split(rng)
    shuffled_players = jnp.arange(N_PLAYER, dtype=jnp.int8)
    shuffled_players = jax.random.permutation(
        key1, shuffled_players, independent=True
    )
    wall = jnp.arange(NUM_TILES, dtype=jnp.int32)
    wall = jax.random.permutation(key2, wall, independent=True)
    current_player = shuffled_players[0]  # dealer
    dora = wall[-1] // 4
    # set hands (hands[0] = dealer's hand)
    hands = jnp.zeros((N_PLAYER, NUM_TILE_TYPES), dtype=jnp.int32)
    hands = lax.fori_loop(
        0, N_PLAYER * 5, lambda i, x: x.at[i // 5, wall[i] // 4].add(1), hands
    )
    n_red_in_hands = jnp.zeros((N_PLAYER, NUM_TILE_TYPES), dtype=jnp.int32)
    n_red_in_hands = lax.fori_loop(
        0,
        N_PLAYER * 5,
        lambda i, x: x.at[i // 5, wall[i] // 4].add(
            ((wall[i] % 4 == 0) & (wall[i] != 36)) | (wall[i] >= 40)
        ),
        n_red_in_hands,
    )
    # first draw
    draw_ix = jnp.int32(N_PLAYER * 5)
    hands = hands.at[0, wall[draw_ix] // 4].add(1)
    draw_ix += 1
    legal_action_mask = hands[0] > 0
    state = State(
        current_player=current_player,
        legal_action_mask=legal_action_mask,
        hands=hands,
        n_red_in_hands=n_red_in_hands,
        wall=wall,
        draw_ix=draw_ix,
        shuffled_players=shuffled_players,
        dora=dora,
    )  # type: ignore

    # check tenhou
    scores = _hands_to_score(state)
    is_tsumo = _check_tsumo(state, scores)
    state = lax.cond(
        is_tsumo,
        lambda: _step_by_tsumo(state, scores),
        lambda: state,
    )

    return state


def _to_base5(hand: jnp.ndarray):
    b = jnp.int32(
        [9765625, 1953125, 390625, 78125, 15625, 3125, 625, 125, 25, 5, 1]
    )
    return (hand * b).sum()


def _is_completed(hand: jnp.ndarray):
    return jnp.any(_to_base5(hand) == WIN_HANDS)


def _hand_to_score(hand: jnp.ndarray):
    # behavior for incomplete hand is undefined
    ix = jnp.argmin(jnp.abs(WIN_HANDS - _to_base5(hand)))
    return BASE_SCORES[ix], YAKU_SCORES[ix]


def _hands_to_score(state: State) -> jnp.ndarray:
    scores = jnp.zeros(3, dtype=jnp.int32)
    for i in range(N_PLAYER):
        hand = state.hands[i]
        hand = jax.lax.cond(
            hand.sum() == 5,
            lambda: hand.at[state.last_discard].add(1),
            lambda: hand,
        )
        bs, ys = _hand_to_score(hand)
        n_doras = hand[state.dora]
        n_red_doras = state.n_red_in_hands[i].sum().astype(jnp.int32)
        is_super_red = n_red_doras >= 6
        ys = lax.cond(is_super_red, lambda: jnp.int32(20), lambda: ys)
        s = lax.cond(
            ys >= 10,  # yakuman
            lambda: bs + ys,
            lambda: bs + ys + n_doras + n_red_doras,
        )
        scores = scores.at[i].set(s)
    return scores


def _check_ron(state: State, scores) -> jnp.ndarray:
    winning_players = jax.lax.fori_loop(
        0,
        N_PLAYER,
        lambda i, x: x.at[i].set(
            _is_completed(state.hands.at[i, state.last_discard].add(1)[i])
        ),
        jnp.zeros(N_PLAYER, dtype=jnp.bool_),
    )
    winning_players = winning_players.at[state.turn].set(False)
    is_furiten = (state.rivers == state.last_discard).sum(axis=1) > 0
    winning_players = winning_players & jnp.logical_not(is_furiten)
    winning_players = winning_players & (scores >= 5)
    return winning_players


def _check_tsumo(state: State, scores) -> jnp.ndarray:
    return _is_completed(state.hands[state.turn]) & (scores[state.turn] >= 0)


def _order_by_player_idx(x, shuffled_players):
    return lax.fori_loop(
        0,
        N_PLAYER,
        lambda i, e: e.at[shuffled_players[i]].set(x[i]),
        jnp.zeros_like(x),
    )


def _step_by_ron(state: State, scores, winning_players):
    scores = scores.at[0].add(2)
    scores = scores * winning_players
    scores = scores.at[state.turn % N_PLAYER].set(-scores.sum())
    state = state.replace(  # type: ignore
        terminated=jnp.bool_(True),
        legal_action_mask=jnp.zeros_like(state.legal_action_mask),
        scores=scores,
    )
    r = (
        _order_by_player_idx(scores, state.shuffled_players).astype(
            jnp.float32
        )
        / MAX_SCORE
    )
    return state.replace(reward=r)  # type: ignore


def _step_by_tsumo(state: State, scores):
    scores = scores.at[0].add(2)
    winner_score = scores[state.turn]
    loser_score = jnp.ceil(winner_score / (N_PLAYER - 1)).astype(jnp.int32)
    winner_score = loser_score * (N_PLAYER - 1)
    scores = -jnp.ones(N_PLAYER, dtype=jnp.int32) * loser_score
    scores = scores.at[state.turn % N_PLAYER].set(winner_score)
    state = state.replace(  # type: ignore
        terminated=jnp.bool_(True),
        legal_action_mask=jnp.zeros_like(state.legal_action_mask),
        scores=scores,
    )
    r = (
        _order_by_player_idx(scores, state.shuffled_players).astype(
            jnp.float32
        )
        / MAX_SCORE
    )
    return state.replace(reward=r)  # type: ignore


def _step_by_tie(state):
    state = state.replace(  # type: ignore
        terminated=jnp.bool_(True),
        legal_action_mask=jnp.zeros_like(state.legal_action_mask),
    )
    return state.replace(reward=jnp.zeros(3, dtype=jnp.float32))  # type: ignore


def _draw_tile(state: State) -> State:
    turn = state.turn + 1
    current_player = state.shuffled_players[turn % N_PLAYER]
    tile_id = state.wall[state.draw_ix]
    tile_type = tile_id // 4
    # gd=[36,37,38,39], rd=[40,41,42,43]
    is_red = ((tile_id % 4 == 0) & (tile_id != 36)) | (tile_id >= 40)
    hands = state.hands.at[turn % N_PLAYER, tile_type].add(1)
    n_red_in_hands = state.n_red_in_hands.at[turn % N_PLAYER, tile_type].add(
        is_red
    )
    draw_ix = state.draw_ix + 1
    legal_action_mask = hands[turn % N_PLAYER] > 0
    state = state.replace(  # type: ignore
        turn=turn,
        current_player=current_player,
        hands=hands,
        n_red_in_hands=n_red_in_hands,
        draw_ix=draw_ix,
        legal_action_mask=legal_action_mask,
        terminated=jnp.bool_(False),
    )
    return state


def _step_non_terminal(state: State):
    r = jnp.zeros(3, dtype=jnp.float32)
    return state.replace(reward=r)  # type: ignore


def _step_non_tied(state: State, scores):
    state = _draw_tile(state)
    scores = _hands_to_score(state)
    is_tsumo = _check_tsumo(state, scores)
    return lax.cond(
        is_tsumo,
        lambda: _step_by_tsumo(state, scores),
        lambda: _step_non_terminal(state),
    )


def _observe(state: State, player_id: jnp.ndarray):
    """
    * [binary 4x11] tile type in the player's hand (private info)
    * [binary 1x11] has red doras
    * [binary 1x11] dora
    * [binary 3x11] all discarded tiles by (self, next, after the next)
    * [binary 3x2x11] discarded tiles in the last 3 steps (next, after the next)
    """
    N = 3
    n_feat = 4 + 1 + 1 + 3 + N * 2
    turn = jnp.abs(state.shuffled_players - player_id).argmin()
    obs = jnp.zeros((n_feat, NUM_TILE_TYPES), dtype=jnp.bool_)
    zeros = jnp.zeros(NUM_TILE_TYPES, dtype=jnp.bool_)
    ones = jnp.ones(NUM_TILE_TYPES, dtype=jnp.bool_)
    # hand
    obs = obs.at[0].set(jnp.where(state.hands[turn] >= 1, ones, zeros))
    obs = obs.at[1].set(jnp.where(state.hands[turn] >= 2, ones, zeros))
    obs = obs.at[2].set(jnp.where(state.hands[turn] >= 3, ones, zeros))
    obs = obs.at[3].set(jnp.where(state.hands[turn] >= 4, ones, zeros))
    # red dora
    obs = obs.at[4].set(
        jnp.where(state.n_red_in_hands[turn] >= 1, ones, zeros)
    )
    # dora
    obs = obs.at[5, state.dora].set(True)
    # all discards
    obs = obs.at[6].set(
        lax.fori_loop(
            0,
            MAX_RIVER_LENGTH,
            lambda i, x: lax.cond(
                state.rivers[turn, i] >= 0,
                lambda: x.at[state.rivers[turn, i]].set(True),
                lambda: x,
            ),
            zeros,
        )
    )
    obs = obs.at[7].set(
        lax.fori_loop(
            0,
            MAX_RIVER_LENGTH,
            lambda i, x: lax.cond(
                state.rivers[(turn + 1) % N_PLAYER, i] >= 0,
                lambda: x.at[state.rivers[(turn + 1) % N_PLAYER, i]].set(True),
                lambda: x,
            ),
            zeros,
        )
    )
    obs = obs.at[8].set(
        lax.fori_loop(
            0,
            MAX_RIVER_LENGTH,
            lambda i, x: lax.cond(
                state.rivers[(turn + 2) % N_PLAYER, i] >= 0,
                lambda: x.at[state.rivers[(turn + 2) % N_PLAYER, i]].set(True),
                lambda: x,
            ),
            zeros,
        )
    )
    # last N discards (N=3)
    zeros = jnp.zeros(MAX_RIVER_LENGTH, dtype=jnp.bool_)
    ones = jnp.ones(MAX_RIVER_LENGTH, dtype=jnp.bool_)
    ix = (
        jnp.where(state.rivers[(turn + 1) % N_PLAYER] >= 0, ones, zeros)
    ).sum()
    obs = lax.cond(
        ix - 1 >= 0,
        lambda: obs.at[9, state.rivers[(turn + 1) % N_PLAYER, ix - 1]].set(
            True
        ),
        lambda: obs,
    )
    obs = lax.cond(
        ix - 2 >= 0,
        lambda: obs.at[10, state.rivers[(turn + 1) % N_PLAYER, ix - 2]].set(
            True
        ),
        lambda: obs,
    )
    obs = lax.cond(
        ix - 3 >= 0,
        lambda: obs.at[11, state.rivers[(turn + 1) % N_PLAYER, ix - 3]].set(
            True
        ),
        lambda: obs,
    )
    ix = (
        jnp.where(state.rivers[(turn + 2) % N_PLAYER] >= 0, ones, zeros)
    ).sum()
    obs = lax.cond(
        ix - 1 >= 0,
        lambda: obs.at[12, state.rivers[(turn + 2) % N_PLAYER, ix - 1]].set(
            True
        ),
        lambda: obs,
    )
    obs = lax.cond(
        ix - 2 >= 0,
        lambda: obs.at[13, state.rivers[(turn + 2) % N_PLAYER, ix - 2]].set(
            True
        ),
        lambda: obs,
    )
    obs = lax.cond(
        ix - 3 >= 0,
        lambda: obs.at[14, state.rivers[(turn + 2) % N_PLAYER, ix - 3]].set(
            True
        ),
        lambda: obs,
    )
    return obs


def _tile_type_to_str(tile_type) -> str:
    if tile_type < 9:
        s = str(tile_type + 1)
    elif tile_type == 9:
        s = "g"
    elif tile_type == 10:
        s = "r"
    return s


def _hand_to_str(hand: jnp.ndarray, n_red_in_hands: jnp.ndarray) -> str:
    s = ""
    for i in range(NUM_TILE_TYPES):
        for j in range(hand[i]):
            s += _tile_type_to_str(i)
            if j < n_red_in_hands[i]:
                s += "*"
            else:
                s += " "
    return s.ljust(12)


def _river_to_str(river: jnp.ndarray, is_red_in_river: jnp.ndarray) -> str:
    s = ""
    for i in range(MAX_RIVER_LENGTH):
        tile_type = river[i]
        if tile_type >= 0:
            s += _tile_type_to_str(tile_type)
            s += "*" if is_red_in_river[i] else " "
        else:
            s += "_ "
    return s.ljust(20)


def _to_str(state: State):
    s = f"{'[terminated]' if state. terminated else ''} dora: {_tile_type_to_str(state.dora)}\n"
    for i in range(N_PLAYER):
        s += f"{'*' if not state.terminated and state.turn % N_PLAYER == i else ' '}[{state.shuffled_players[i]}] "
        s += f"{_hand_to_str(state.hands[i], state.n_red_in_hands[i])}: "
        s += f"{_river_to_str(state.rivers[i], state.is_red_in_river[i])} "
        s += "\n"
    return s


def _validate(state: State) -> bool:
    if state.dora < 0 or 10 < state.dora:
        assert False
    if 10 < state.last_discard:
        assert False
    if state.last_discard < 0 and state.rivers[0, 0] >= 0:
        assert False
    if jnp.any(state.hands < 0):
        assert False
    counts = state.hands.sum(axis=0)
    for i in range(N_PLAYER):
        for j in range(MAX_RIVER_LENGTH):
            if state.rivers[i, j] >= 0:
                counts = counts.at[state.rivers[i, j]].add(1)
    if jnp.any(counts > 4):
        assert False

    # check legal_action_mask
    if not state.terminated:
        for i in range(NUM_TILE_TYPES):
            if (
                state.legal_action_mask[i]
                and state.hands[state.turn % N_PLAYER, i] <= 0
            ):
                assert (
                    False
                ), f"\n{state.legal_action_mask[i]}\n{state.hands[state.turn % N_PLAYER, i]}\n{_to_str(state)}"
            if (
                not state.legal_action_mask[i]
                and state.hands[state.turn % N_PLAYER, i] > 0
            ):
                assert (
                    False
                ), f"\n{state.legal_action_mask}\n{state.hands[state.turn % N_PLAYER]}\n{_to_str(state)}"

    if not jnp.all(state.n_red_in_hands[:, :-2] <= 1):
        assert False
    if (state.n_red_in_hands.sum() + state.is_red_in_river.sum()) > 14:
        assert (
            False
        ), f"\n{state.n_red_in_hands}\n{state.is_red_in_river}\n{_to_str(state)}"

    assert state.scores.sum() == 0, f"\n{state.scores}\n{_to_str(state)}"
    return True
