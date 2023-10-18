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

import jax
import jax.numpy as jnp

import pgx.v1 as v1
from pgx.mahjong.action import Action
from pgx.mahjong.hand import Hand
from pgx.mahjong.meld import Meld
from pgx.mahjong.yaku import Yaku
from pgx._src.struct import dataclass

FALSE = jnp.bool_(False)
TRUE = jnp.bool_(True)
NUM_ACTION = 79


@dataclass
class State(v1.State):
    current_player: jnp.ndarray = jnp.int8(0)  # actionを行うplayer
    observation: jnp.ndarray = jnp.int8(0)
    rewards: jnp.ndarray = jnp.zeros(4, dtype=jnp.float32)  # （百の位から）
    terminated: jnp.ndarray = FALSE
    truncated: jnp.ndarray = FALSE
    legal_action_mask: jnp.ndarray = jnp.zeros(NUM_ACTION, dtype=jnp.bool_)
    _rng_key: jax.random.KeyArray = jax.random.PRNGKey(0)
    _step_count: jnp.ndarray = jnp.int32(0)
    # --- Mahjong specific ---
    _round: jnp.ndarray = jnp.int8(0)
    honba: jnp.ndarray = jnp.int8(0)

    # 東1局の親
    # 各局の親は (state.oya+round)%4
    oya: jnp.ndarray = jnp.int8(0)

    # 点数（百の位から）
    score: jnp.ndarray = jnp.full(4, 250, dtype=jnp.float32)

    #      嶺上 ドラ  カンドラ
    # ... 13 11  9  7  5  3  1
    # ... 12 10  8  6  4  2  0
    deck: jnp.ndarray = jnp.zeros(136, dtype=jnp.int8)

    # 次に引く牌のindex
    next_deck_ix: jnp.ndarray = jnp.int8(135 - 13 * 4)

    # 各playerの手牌. 長さ34で、数字は持っている牌の数
    hand: jnp.ndarray = jnp.zeros((4, 34), dtype=jnp.int8)

    # 河
    # int8
    # 0b  0     0    0 0 0 0 0 0
    #    灰色|リーチ|   牌(0-33)
    river: jnp.ndarray = 34 * jnp.ones((4, 4 * 6), dtype=jnp.uint8)

    # 各playerの河の数
    n_river: jnp.ndarray = jnp.zeros(4, dtype=jnp.int8)

    # ドラ
    doras: jnp.ndarray = jnp.zeros(5, dtype=jnp.int8)

    # カンの回数=追加ドラ枚数
    n_kan: jnp.ndarray = jnp.int8(0)

    # 直前に捨てられてron,pon,chiの対象になっている牌. 存在しなければ-1
    target: jnp.ndarray = jnp.int8(-1)

    # 手牌が3n+2枚のplayerが直前に引いた牌. 存在しなければ-1
    last_draw: jnp.ndarray = jnp.int8(0)

    # 最後のプレイヤー.  ron,pon,chiの対象
    last_player: jnp.ndarray = jnp.int8(0)

    # 打牌の後に競合する副露が生じた場合用
    #     pon player, kan player, chi player
    # b0       00         00         00
    furo_check_num: jnp.ndarray = jnp.uint8(0)

    # state.current_player がリーチ宣言してから, その直後の打牌が通るまでTrue
    riichi_declared: jnp.ndarray = FALSE

    # 各playerのリーチが成立しているかどうか
    riichi: jnp.ndarray = jnp.zeros(4, dtype=jnp.bool_)

    # 各playerの副露回数
    n_meld: jnp.ndarray = jnp.zeros(4, dtype=jnp.int8)

    # melds[i][j]: player i のj回目の副露(j=1,2,3,4). 存在しなければ0
    melds: jnp.ndarray = jnp.zeros((4, 4), dtype=jnp.int32)

    is_menzen: jnp.ndarray = jnp.zeros(4, dtype=jnp.bool_)

    # pon[i][j]: player i がjをポンを所有している場合, src << 2 | index. or 0
    pon: jnp.ndarray = jnp.zeros((4, 34), dtype=jnp.int32)

    @property
    def env_id(self) -> v1.EnvId:
        # TODO add envid
        return "mahjong"  # type:ignore

    @property
    def json(self):
        import json

        class NumpyEncoder(json.JSONEncoder):
            """Special json encoder for numpy types"""

            def default(self, obj):
                if isinstance(obj, jnp.integer):
                    return int(obj)
                elif isinstance(obj, jnp.floating):
                    return float(obj)
                elif isinstance(obj, jnp.ndarray):
                    return obj.tolist()
                return json.JSONEncoder.default(self, obj)

        return json.dumps(self.__dict__, cls=NumpyEncoder)

    @classmethod
    def from_json(cls, path):
        import json

        def decode_state(data: dict):
            return cls(  # type:ignore
                current_player=jnp.int8(data["current_player"]),
                observation=jnp.int8(data["observation"]),
                rewards=jnp.array(data["rewards"], dtype=jnp.float32),
                terminated=jnp.bool_(data["terminated"]),
                truncated=jnp.bool_(data["truncated"]),
                legal_action_mask=jnp.array(
                    data["legal_action_mask"], dtype=jnp.bool_
                ),
                _rng_key=jnp.array(data["_rng_key"]),
                _step_count=jnp.int32(data["_step_count"]),
                _round=jnp.int8(data["_round"]),
                honba=jnp.int8(data["honba"]),
                oya=jnp.int8(data["oya"]),
                score=jnp.array(data["score"], dtype=jnp.float32),
                deck=jnp.array(data["deck"], dtype=jnp.int8),
                next_deck_ix=jnp.int8(data["next_deck_ix"]),
                hand=jnp.array(data["hand"], dtype=jnp.int8),
                river=jnp.array(data["river"], dtype=jnp.uint8),
                n_river=jnp.array(data["n_river"], dtype=jnp.int8),
                doras=jnp.array(data["doras"], dtype=jnp.int8),
                n_kan=jnp.int8(data["n_kan"]),
                target=jnp.int8(data["target"]),
                last_draw=jnp.int8(data["last_draw"]),
                last_player=jnp.int8(data["last_player"]),
                furo_check_num=jnp.uint8(data["furo_check_num"]),
                riichi_declared=jnp.bool_(data["riichi_declared"]),
                riichi=jnp.array(data["riichi"], dtype=jnp.bool_),
                n_meld=jnp.array(data["n_meld"], dtype=jnp.int8),
                melds=jnp.array(data["melds"], dtype=jnp.int32),
                is_menzen=jnp.array(data["is_menzen"], dtype=jnp.bool_),
                pon=jnp.array(data["pon"], dtype=jnp.int32),
            )

        with open(path, "r") as f:
            state = json.load(f, object_hook=decode_state)

        return state

    def __eq__(self, b):
        a = self
        return (
            a.current_player == b.current_player
            and a.observation == b.observation
            and (a.rewards == b.rewards).all()
            and a.terminated == b.terminated
            and a.truncated == b.truncated
            and (a.legal_action_mask == b.legal_action_mask).all()
            and (a._rng_key == b._rng_key).all()
            and a._step_count == b._step_count
            and a._round == b._round
            and a.honba == b.honba
            and a.oya == b.oya
            and (a.score == b.score).all()
            and (a.deck == b.deck).all()
            and a.next_deck_ix == b.next_deck_ix
            and (a.hand == b.hand).all()
            and (a.river == b.river).all()
            and (a.doras == b.doras).all()
            and a.n_kan == b.n_kan
            and a.target == b.target
            and a.last_draw == b.last_draw
            and a.last_player == b.last_player
            and a.furo_check_num == b.furo_check_num
            and a.riichi_declared == b.riichi_declared
            and (a.riichi == b.riichi).all()
            and (a.n_meld == b.n_meld).all()
            and (a.melds == b.melds).all()
            and (a.is_menzen == b.is_menzen).all()
            and (a.pon == b.pon).all()
        )


class Mahjong(v1.Env):
    def __init__(self):
        super().__init__()

    def _init(self, key: jax.Array) -> State:
        return _init(key)

    def _step(self, state: v1.State, action: jnp.ndarray) -> State:
        assert isinstance(state, State)
        return _step(state, action)

    def _observe(self, state: v1.State, player_id: jnp.ndarray) -> jnp.ndarray:
        assert isinstance(state, State)
        return _observe(state, player_id)

    @property
    def id(self) -> v1.EnvId:
        # TODO add envid
        return "mahjong"  # type:ignore

    @property
    def version(self) -> str:
        return "alpha"

    @property
    def num_players(self) -> int:
        return 4


def _init(rng: jax.random.KeyArray) -> State:
    rng, subkey = jax.random.split(rng)
    current_player = jnp.int8(jax.random.bernoulli(rng))
    last_player = jnp.int8(-1)
    deck = jax.random.permutation(rng, jnp.arange(136, dtype=jnp.int8) // 4)
    init_hand = Hand.make_init_hand(deck)
    doras = jnp.array([deck[9], -1, -1, -1, -1], dtype=jnp.int8)
    state = State(  # type:ignore
        current_player=current_player,
        oya=current_player,
        last_player=last_player,
        deck=deck,
        doras=doras,
        hand=init_hand,
        _rng_key=subkey,
    )
    return _draw(state)


def _step(state: State, action) -> State:
    # TODO
    # - Actionの処理
    #   - ron, tsumo
    # - 勝利条件確認
    # - playerどうするか

    discard = (action < 34) | (action == 68)
    self_kan = (34 <= action) & (action < 68)
    action_ix = action - 69
    return jax.lax.cond(
        discard,
        lambda: _discard(state, action),
        lambda: jax.lax.cond(
            self_kan,
            lambda: _selfkan(state, action),
            lambda: jax.lax.switch(
                action_ix,
                [
                    lambda: _riichi(state),
                    lambda: _tsumo(state),
                    lambda: _ron(state),
                    lambda: _pon(state),
                    lambda: _minkan(state),
                    lambda: _chi(state, Action.CHI_L),
                    lambda: _chi(state, Action.CHI_M),
                    lambda: _chi(state, Action.CHI_R),
                    lambda: _pass(state),
                    lambda: _next_game(state),
                ],
            ),
        ),
    )


def _draw(state: State):
    state = _accept_riichi(state)
    c_p = state.current_player
    new_tile = state.deck[state.next_deck_ix]
    next_deck_ix = state.next_deck_ix - 1
    hand = state.hand.at[c_p].set(Hand.add(state.hand[c_p], new_tile))

    legal_action_mask = jax.lax.select(
        state.riichi[c_p],
        _make_legal_action_mask_w_riichi(state, hand, c_p, new_tile),
        _make_legal_action_mask(state, hand, c_p, new_tile),
    )

    return state.replace(  # type:ignore
        target=jnp.int8(-1),
        next_deck_ix=next_deck_ix,
        hand=hand,
        last_draw=new_tile,
        last_player=c_p,
        legal_action_mask=legal_action_mask,
    )


def _make_legal_action_mask(state: State, hand, c_p, new_tile):
    legal_action_mask = jnp.zeros(NUM_ACTION, dtype=jnp.bool_)
    legal_action_mask = legal_action_mask.at[:34].set(hand[c_p] > 0)
    legal_action_mask = legal_action_mask.at[new_tile].set(FALSE)
    legal_action_mask = legal_action_mask.at[Action.TSUMOGIRI].set(TRUE)
    legal_action_mask = legal_action_mask.at[new_tile + 34].set(
        Hand.can_ankan(hand[c_p], new_tile)
        | (
            Hand.can_kakan(hand[c_p], new_tile) & state.pon[(c_p, new_tile)]
            > 0
        )
    )
    legal_action_mask = legal_action_mask.at[Action.RIICHI].set(
        jax.lax.cond(
            state.riichi[c_p]
            | state.is_menzen[c_p]
            | (state.next_deck_ix < 13 + 4),
            lambda: FALSE,
            lambda: Hand.can_riichi(hand[c_p]),
        )
    )
    legal_action_mask = legal_action_mask.at[Action.TSUMO].set(
        Hand.can_tsumo(hand[c_p])
        & Yaku.judge(
            state.hand[c_p],
            state.melds[c_p],
            state.n_meld[c_p],
            state.last_draw,
            state.riichi[c_p],
            FALSE,
            _dora_array(state, state.riichi[c_p]),
        )[0].any()
    )
    return legal_action_mask


def _make_legal_action_mask_w_riichi(state, hand, c_p, new_tile):
    legal_action_mask = jnp.zeros(NUM_ACTION, dtype=jnp.bool_)
    legal_action_mask = legal_action_mask.at[Action.TSUMOGIRI].set(TRUE)
    legal_action_mask = legal_action_mask.at[Action.TSUMO].set(
        Hand.can_tsumo(hand[c_p])
        & Yaku.judge(
            state.hand[c_p],
            state.melds[c_p],
            state.n_meld[c_p],
            state.last_draw,
            state.riichi[c_p],
            FALSE,
            _dora_array(state, state.riichi[c_p]),
        )[0].any()
    )
    return legal_action_mask


def _discard(state: State, tile: jnp.ndarray):
    c_p = state.current_player
    tile = jnp.where(tile == 68, state.last_draw, tile)
    _tile = jnp.where(
        state.riichi_declared, tile | jnp.uint8(0b01000000), tile
    )
    river = state.river.at[c_p, state.n_river[c_p]].set(jnp.uint8(_tile))
    n_river = state.n_river.at[c_p].add(1)
    hand = state.hand.at[c_p].set(Hand.sub(state.hand[c_p], tile))
    state = state.replace(  # type:ignore
        last_draw=jnp.int8(-1),
        hand=hand,
        river=river,
        n_river=n_river,
    )

    # ポンとかチーとかがあるか
    meld_type = 0  # none < chi_L = chi_M = chi_R < pon = kan < ron の中で最大のものを探す
    pon_player = kan_player = ron_player = c_p
    chi_player = (c_p + 1) % 4
    can_chi = (
        Hand.can_chi(state.hand[chi_player], tile, Action.CHI_L)
        | Hand.can_chi(state.hand[chi_player], tile, Action.CHI_M)
        | Hand.can_chi(state.hand[chi_player], tile, Action.CHI_R)
    )
    meld_type = jax.lax.cond(
        can_chi,
        lambda: jnp.max(jnp.array([1, meld_type])),
        lambda: meld_type,
    )

    def search(i, tpl):
        # iは相対位置
        meld_type, pon_player, kan_player, ron_player = tpl
        player = (c_p + 1 + i) % 4  # 絶対位置
        pon_player, meld_type = jax.lax.cond(
            Hand.can_pon(state.hand[player], tile),
            lambda: (i, jnp.max(jnp.array([2, meld_type]))),
            lambda: (pon_player, meld_type),
        )
        kan_player, meld_type = jax.lax.cond(
            Hand.can_minkan(hand[player], tile),
            lambda: (i, jnp.max(jnp.array([3, meld_type]))),
            lambda: (kan_player, meld_type),
        )
        ron_player, meld_type = jax.lax.cond(
            Hand.can_ron(state.hand[player], tile)
            & Yaku.judge(
                state.hand[player],
                state.melds[player],
                state.n_meld[player],
                state.last_draw,
                state.riichi[player],
                FALSE,
                _dora_array(state, state.riichi[player]),
            )[0].any(),
            lambda: (i, jnp.max(jnp.array([4, meld_type]))),
            lambda: (ron_player, meld_type),
        )
        return (meld_type, pon_player, kan_player, ron_player)

    meld_type, pon_player, kan_player, ron_player = jax.lax.fori_loop(
        jnp.int8(0),
        jnp.int8(3),
        search,
        (meld_type, pon_player, kan_player, ron_player),
    )
    furo_num = jnp.uint8(
        c_p << 6 | kan_player << 4 | pon_player << 2 | chi_player
    )
    pon_player, kan_player, ron_player = (
        (c_p + 1 + pon_player) % 4,
        (c_p + 1 + kan_player) % 4,
        (c_p + 1 + ron_player) % 4,
    )

    rewards = jnp.float32([Hand.is_tenpai(hand) * 100 for hand in state.hand])
    no_meld_state = jax.lax.cond(
        _is_ryukyoku(state),
        lambda: state.replace(  # type:ignore
            terminated=TRUE,
            rewards=rewards,
        ),
        lambda: _draw(
            state.replace(  # type:ignore
                current_player=(c_p + 1) % 4,
                target=jnp.int8(-1),
            )
        ),
    )

    return jax.lax.switch(
        meld_type,
        [
            lambda: no_meld_state,
            lambda: state.replace(  # type:ignore
                current_player=chi_player,
                last_player=c_p,
                target=jnp.int8(tile),
                furo_check_num=furo_num & 0b11111100,
                legal_action_mask=jnp.zeros(NUM_ACTION, dtype=jnp.bool_)
                .at[Action.CHI_L]
                .set(Hand.can_chi(state.hand[chi_player], tile, Action.CHI_L))
                .at[Action.CHI_M]
                .set(Hand.can_chi(state.hand[chi_player], tile, Action.CHI_M))
                .at[Action.CHI_R]
                .set(Hand.can_chi(state.hand[chi_player], tile, Action.CHI_R))
                .at[Action.PASS]
                .set(TRUE),
            ),
            lambda: state.replace(  # type:ignore
                current_player=pon_player,
                last_player=c_p,
                target=jnp.int8(tile),
                furo_check_num=furo_num & 0b11110011,
                legal_action_mask=jnp.zeros(NUM_ACTION, dtype=jnp.bool_)
                .at[Action.PON]
                .set(TRUE)
                .at[Action.PASS]
                .set(TRUE),
            ),
            lambda: state.replace(  # type:ignore
                current_player=kan_player,
                last_player=c_p,
                target=jnp.int8(tile),
                furo_check_num=furo_num & 0b11001111,
                legal_action_mask=jnp.zeros(NUM_ACTION, dtype=jnp.bool_)
                .at[Action.MINKAN]
                .set(Hand.can_minkan(hand[kan_player], tile))
                .at[Action.PASS]
                .set(TRUE),
            ),
            lambda: state.replace(  # type:ignore
                current_player=ron_player,
                last_player=c_p,
                target=jnp.int8(tile),
                furo_check_num=furo_num,
                legal_action_mask=jnp.zeros(NUM_ACTION, dtype=jnp.bool_)
                .at[Action.RON]
                .set(TRUE)
                .at[Action.PASS]
                .set(TRUE),
            ),
        ],
    )


def _append_meld(state: State, meld, player):
    melds = state.melds.at[(player, state.n_meld[player])].set(meld)
    n_meld = state.n_meld.at[player].add(1)
    return state.replace(melds=melds, n_meld=n_meld)  # type:ignore


def _selfkan(state: State, action):
    target = action - 34
    pon = state.pon[(state.current_player, target)]
    state = jax.lax.cond(
        pon == 0,
        lambda: _ankan(state, target),
        lambda: _kakan(state, target, pon >> 2, pon & 0b11),
    )

    # 嶺上牌
    rinshan_tile = state.deck[state.next_deck_ix]
    next_deck_ix = state.next_deck_ix - 1
    hand = state.hand.at[state.current_player].set(
        Hand.add(state.hand[state.current_player], rinshan_tile)
    )
    legal_action_mask = jnp.zeros(NUM_ACTION, dtype=jnp.bool_)
    legal_action_mask = legal_action_mask.at[0:34].set(
        hand[state.current_player] > 0
    )
    legal_action_mask = legal_action_mask.at[rinshan_tile].set(FALSE)
    legal_action_mask = legal_action_mask.at[Action.TSUMOGIRI].set(TRUE)

    return state.replace(  # type:ignore
        next_deck_ix=next_deck_ix,
        last_draw=rinshan_tile,
        hand=hand,
        legal_action_mask=legal_action_mask,
        n_kan=state.n_kan + 1,
        doras=state.doras.at[state.n_kan + 1].set(
            state.deck[9 - 2 * (state.n_kan + 1)]
        ),
    )


def _ankan(state: State, target):
    curr_player = state.current_player
    meld = Meld.init(target + 34, target, src=0)
    state = _append_meld(state, meld, curr_player)
    hand = state.hand.at[curr_player].set(
        Hand.ankan(state.hand[curr_player], target)
    )
    # TODO: 国士無双ロンの受付

    return state.replace(  # type:ignore
        hand=hand,
    )


def _kakan(state: State, target, pon_src, pon_idx):
    melds = state.melds.at[(state.current_player, pon_idx)].set(
        Meld.init(target + 34, target, pon_src)
    )
    hand = state.hand.at[state.current_player].set(
        Hand.kakan(state.hand[state.current_player], target)
    )
    pon = state.pon.at[(state.current_player, target)].set(0)
    # TODO: 槍槓の受付

    return state.replace(melds=melds, hand=hand, pon=pon)  # type:ignore


def _accept_riichi(state: State) -> State:
    l_p = state.last_player
    score = state.score.at[l_p].add(
        jnp.where(~state.riichi[l_p] & state.riichi_declared, -10, 0)
    )
    riichi = state.riichi.at[l_p].set(
        state.riichi[l_p] | state.riichi_declared
    )
    return state.replace(  # type:ignore
        riichi=riichi, riichi_declared=FALSE, score=score
    )


def _minkan(state: State):
    c_p = state.current_player
    l_p = state.last_player
    state = _accept_riichi(state)
    src = (l_p - c_p) % 4
    meld = Meld.init(Action.MINKAN, state.target, src)
    state = _append_meld(state, meld, c_p)
    hand = state.hand.at[c_p].set(Hand.minkan(state.hand[c_p], state.target))
    state = state.replace(hand=hand)  # type:ignore
    is_menzen = state.is_menzen.at[c_p].set(FALSE)

    rinshan_tile = state.deck[state.next_deck_ix]
    next_deck_ix = state.next_deck_ix - 1
    hand = state.hand.at[c_p].set(Hand.add(state.hand[c_p], rinshan_tile))

    legal_action_mask = jnp.zeros(NUM_ACTION, dtype=jnp.bool_)
    legal_action_mask = legal_action_mask.at[0:34].set(hand[c_p] > 0)
    legal_action_mask = legal_action_mask.at[rinshan_tile].set(FALSE)
    legal_action_mask = legal_action_mask.at[Action.TSUMOGIRI].set(TRUE)

    # 半透明処理
    river = state.river.at[l_p, state.n_river[l_p] - 1].set(
        state.river[l_p, state.n_river[l_p] - 1] | jnp.uint8(0b10000000)
    )

    return state.replace(  # type:ignore
        target=jnp.int8(-1),
        is_menzen=is_menzen,
        next_deck_ix=next_deck_ix,
        last_draw=rinshan_tile,
        hand=hand,
        legal_action_mask=legal_action_mask,
        river=river,
        n_kan=state.n_kan + 1,
        doras=state.doras.at[state.n_kan + 1].set(
            state.deck[9 - 2 * (state.n_kan + 1)]
        ),
    )


def _pon(state: State):
    c_p = state.current_player
    l_p = state.last_player
    state = _accept_riichi(state)
    src = (l_p - c_p) % 4
    meld = Meld.init(Action.PON, state.target, src)
    state = _append_meld(state, meld, c_p)
    hand = state.hand.at[c_p].set(Hand.pon(state.hand[c_p], state.target))
    is_menzen = state.is_menzen.at[c_p].set(FALSE)
    pon = state.pon.at[(c_p, state.target)].set(
        src << 2 | state.n_meld[c_p] - 1
    )

    # 半透明処理
    river = state.river.at[l_p, state.n_river[l_p] - 1].set(
        state.river[l_p, state.n_river[l_p] - 1] | jnp.uint8(0b10000000)
    )

    legal_action_mask = jnp.zeros(NUM_ACTION, dtype=jnp.bool_)
    legal_action_mask = legal_action_mask.at[:34].set(hand[c_p] > 0)

    return state.replace(  # type:ignore
        target=jnp.int8(-1),
        is_menzen=is_menzen,
        pon=pon,
        hand=hand,
        legal_action_mask=legal_action_mask,
        river=river,
    )


def _chi(state: State, action):
    c_p = state.current_player
    tar_p = (c_p + 3) % 4
    tar = state.target
    state = _accept_riichi(state)
    meld = Meld.init(action, tar, src=jnp.int32(3))
    state = _append_meld(state, meld, c_p)
    hand = state.hand.at[c_p].set(Hand.chi(state.hand[c_p], tar, action))
    is_menzen = state.is_menzen.at[c_p].set(FALSE)
    legal_action_mask = jnp.zeros(NUM_ACTION, dtype=jnp.bool_)
    legal_action_mask = legal_action_mask.at[:34].set(hand[c_p] > 0)

    # 半透明処理
    river = state.river.at[tar_p, state.n_river[tar_p] - 1].set(
        state.river[tar_p, state.n_river[tar_p] - 1] | jnp.uint8(0b10000000)
    )

    return state.replace(  # type:ignore
        target=jnp.int8(-1),
        is_menzen=is_menzen,
        hand=hand,
        legal_action_mask=legal_action_mask,
        river=river,
    )


def _pass(state: State):
    last_player = (state.furo_check_num & 0b11000000) >> 6
    kan_player = (state.furo_check_num & 0b00110000) >> 4
    pon_player = (state.furo_check_num & 0b00001100) >> 2
    chi_player = state.furo_check_num & 0b00000011
    return jax.lax.cond(
        kan_player > 0,
        lambda: state.replace(  # type:ignore
            current_player=jnp.int8(last_player + 1 + kan_player) % 4,
            furo_check_num=jnp.uint8(state.furo_check_num & 0b11001111),
            legal_action_mask=jnp.zeros(NUM_ACTION, dtype=jnp.bool_)
            .at[Action.MINKAN]
            .set(Hand.can_minkan(state.hand[kan_player], state.target))
            .at[Action.PASS]
            .set(TRUE),
        ),
        lambda: jax.lax.cond(
            pon_player > 0,
            lambda: state.replace(  # type:ignore
                current_player=jnp.int8(last_player + 1 + pon_player) % 4,
                furo_check_num=jnp.uint8(state.furo_check_num & 0b11110011),
                legal_action_mask=jnp.zeros(NUM_ACTION, dtype=jnp.bool_)
                .at[Action.PON]
                .set(Hand.can_pon(state.hand[pon_player], state.target))
                .at[Action.PASS]
                .set(TRUE),
            ),
            lambda: jax.lax.cond(
                chi_player > 0,
                lambda: state.replace(  # type:ignore
                    current_player=jnp.int8(last_player + 1 + chi_player) % 4,
                    furo_check_num=jnp.uint8(
                        state.furo_check_num & 0b11111100
                    ),
                    legal_action_mask=jnp.zeros(NUM_ACTION, dtype=jnp.bool_)
                    .at[Action.CHI_L]
                    .set(
                        Hand.can_chi(
                            state.hand[(last_player + 1 + chi_player) % 4],
                            state.target,
                            Action.CHI_L,
                        )
                    )
                    .at[Action.CHI_M]
                    .set(
                        Hand.can_chi(
                            state.hand[(last_player + 1 + chi_player) % 4],
                            state.target,
                            Action.CHI_M,
                        )
                    )
                    .at[Action.CHI_R]
                    .set(
                        Hand.can_chi(
                            state.hand[(last_player + 1 + chi_player) % 4],
                            state.target,
                            Action.CHI_R,
                        )
                    )
                    .at[Action.PASS]
                    .set(TRUE),
                ),
                lambda: _draw(
                    state.replace(  # type: ignore
                        current_player=jnp.int8(last_player + 1) % 4,
                    )
                ),
            ),
        ),
    )


def _riichi(state: State):
    c_p = state.current_player
    # リーチ宣言直後の打牌
    legal_action_mask = (
        jax.lax.fori_loop(
            0,
            34,
            lambda i, arr: arr.at[i].set(
                jax.lax.cond(
                    state.hand[c_p][i] > (i == state.last_draw),
                    lambda: Hand.is_tenpai(Hand.sub(state.hand[c_p], i)),
                    lambda: FALSE,
                )
            ),
            jnp.zeros(NUM_ACTION, dtype=jnp.bool_),
        )
        .at[Action.TSUMOGIRI]
        .set(Hand.is_tenpai(Hand.sub(state.hand[c_p], state.last_draw)))
    )

    return state.replace(  # type:ignore
        riichi_declared=TRUE, legal_action_mask=legal_action_mask
    )


def _tsumo(state: State):
    c_p = state.current_player

    score = Yaku.score(
        state.hand[c_p],
        state.melds[c_p],
        state.n_meld[c_p],
        state.target,
        state.riichi[c_p],
        is_ron=FALSE,
        dora=_dora_array(state, state.riichi[c_p]),
    )
    s1 = score + (-score) % 100
    s2 = (score * 2) + (-(score * 2)) % 100

    oya = (state.oya + state._round) % 4
    reward = jax.lax.cond(
        oya == c_p,
        lambda: jnp.full(4, -s2, dtype=jnp.int32).at[c_p].set(s2 * 3),
        lambda: jnp.full(4, -s1, dtype=jnp.int32)
        .at[oya]
        .set(-s2)
        .at[c_p]
        .set(s1 * 2 + s2),
    )

    # 供託
    reward -= 1000 * state.riichi
    reward = reward.at[c_p].set(reward[c_p] + 1000 * jnp.sum(state.riichi))
    return state.replace(  # type:ignore
        terminated=TRUE, rewards=jnp.float32(reward)
    )


def _ron(state: State):
    c_p = state.current_player
    score = Yaku.score(
        state.hand[c_p],
        state.melds[c_p],
        state.n_meld[c_p],
        state.target,
        state.riichi[c_p],
        is_ron=TRUE,
        dora=_dora_array(state, state.riichi[c_p]),
    )
    score = jax.lax.cond(
        (state.oya + state._round) % 4 == c_p,
        lambda: score * 6,
        lambda: score * 4,
    )
    score += -score % 100
    reward = (
        jnp.zeros(4, dtype=jnp.int32)
        .at[c_p]
        .set(score)
        .at[state.last_player]
        .set(-score)
    )

    # 供託
    reward -= 1000 * state.riichi
    reward = reward.at[c_p].set(reward[c_p] + 1000 * jnp.sum(state.riichi))
    return state.replace(  # type:ignore
        terminated=TRUE, rewards=jnp.float32(reward)
    )


def _is_ryukyoku(state: State):
    return state.next_deck_ix == 13


def _next_game(state: State):
    # TODO
    return _pass(state)


def _observe(state: State, player_id: jnp.ndarray) -> jnp.ndarray:
    return jnp.int8(0)


def _dora_array(state: State, riichi):
    def next(tile):
        return jax.lax.cond(
            tile < 27,
            lambda: tile // 9 + (tile + 1) % 9,
            lambda: jax.lax.cond(
                tile < 31,
                lambda: 27 + (tile + 1) % 4,
                lambda: 31 + (tile + 1) % 3,
            ),
        )

    dora = jnp.zeros(34, dtype=jnp.bool_)
    return jax.lax.cond(
        riichi,
        lambda: jax.lax.fori_loop(
            0,
            state.n_kan + 1,
            lambda i, arr: arr.at[next(state.deck[5 + 2 * i])]
            .set(TRUE)
            .at[next(state.doras[4 + 2 * i])]
            .set(TRUE),
            dora,
        ),
        lambda: jax.lax.fori_loop(
            0,
            state.n_kan + 1,
            lambda i, arr: arr.at[next(state.doras[5 + 2 * i])].set(TRUE),
            dora,
        ),
    )


# For debug
def _show_legal_action(legal_action):
    S = ["F", "T"]
    m = "m:  " + " ".join([S[int(tf)] for tf in legal_action[0:9]]) + "\n"
    p = "p:  " + " ".join([S[int(tf)] for tf in legal_action[9:18]]) + "\n"
    s = "s:  " + " ".join([S[int(tf)] for tf in legal_action[18:27]]) + "\n"
    z = "z:  " + " ".join([S[int(tf)] for tf in legal_action[27:34]]) + "\n"
    km = "km: " + " ".join([S[int(tf)] for tf in legal_action[34:43]]) + "\n"
    kp = "kp: " + " ".join([S[int(tf)] for tf in legal_action[43:52]]) + "\n"
    ks = "ks: " + " ".join([S[int(tf)] for tf in legal_action[52:61]]) + "\n"
    kz = "kz: " + " ".join([S[int(tf)] for tf in legal_action[61:68]]) + "\n"
    other = (
        f"TSUOGIRI:{S[int(legal_action[68])]}  RIICHI:{S[int(legal_action[69])]}  TSUMO:{S[int(legal_action[70])]}  "
        f"RON:{S[int(legal_action[71])]}  PON:{S[int(legal_action[72])]}  MINKAN:{S[int(legal_action[73])]}  "
        f"CHI_L:{S[int(legal_action[74])]}  CHI_M:{S[int(legal_action[75])]}  CHI_R:{S[int(legal_action[76])]}  "
        f"PASS:{S[int(legal_action[77])]}"
    )
    print(s + m + p + s + z + km + kp + ks + kz + other)
