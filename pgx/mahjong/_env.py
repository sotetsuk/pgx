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

import pgx.core as core
from pgx._src.struct import dataclass
from pgx._src.types import Array, PRNGKey
from pgx.mahjong._action import Action
from pgx.mahjong._hand import Hand
from pgx.mahjong._meld import Meld
from pgx.mahjong._yaku import Yaku

FALSE = jnp.bool_(False)
TRUE = jnp.bool_(True)
NUM_ACTION = 79


@dataclass
class State(core.State):
    current_player: Array = jnp.int8(0)  # actionを行うplayer
    observation: Array = jnp.int8(0)
    rewards: Array = jnp.zeros(4, dtype=jnp.float32)  # （百の位から）
    terminated: Array = FALSE
    truncated: Array = FALSE
    legal_action_mask: Array = jnp.zeros(NUM_ACTION, dtype=jnp.bool_)
    _rng_key: PRNGKey = jax.random.PRNGKey(0)
    _step_count: Array = jnp.int32(0)
    # --- Mahjong specific ---
    _round: Array = jnp.int8(0)
    _honba: Array = jnp.int8(0)

    # 東1局の親
    # 各局の親は (state._oya+round)%4
    _oya: Array = jnp.int8(0)

    # 点数（百の位から）
    _score: Array = jnp.full(4, 250, dtype=jnp.float32)

    #      嶺上 ドラ  カンドラ
    # ... 13 11  9  7  5  3  1
    # ... 12 10  8  6  4  2  0
    _deck: Array = jnp.zeros(136, dtype=jnp.int8)

    # 次に引く牌のindex
    _next_deck_ix: Array = jnp.int8(135 - 13 * 4)

    # 各playerの手牌. 長さ34で, 数字は持っている牌の数
    _hand: Array = jnp.zeros((4, 34), dtype=jnp.int8)

    # 河
    # int8
    # 0b  0     0    0 0 0 0 0 0
    #    灰色|リーチ|   牌(0-33)
    _river: Array = 34 * jnp.ones((4, 4 * 6), dtype=jnp.uint8)

    # 各playerの河の数
    _n_river: Array = jnp.zeros(4, dtype=jnp.int8)

    # ドラ
    _doras: Array = jnp.zeros(5, dtype=jnp.int8)

    # カンの回数=追加ドラ枚数
    _n_kan: Array = jnp.int8(0)

    # 直前に捨てられてron,pon,chiの対象になっている牌. 存在しなければ-1
    _target: Array = jnp.int8(-1)

    # 手牌が3n+2枚のplayerが直前に引いた牌. 存在しなければ-1
    _last_draw: Array = jnp.int8(0)

    # 最後のプレイヤー.  ron,pon,chiの対象
    _last_player: Array = jnp.int8(0)

    # 打牌の後に競合する副露が生じた場合用
    #     pon player, kan player, chi player
    # b0       00         00         00
    _furo_check_num: Array = jnp.uint8(0)

    # state.current_player がリーチ宣言してから, その直後の打牌が通るまでTrue
    _riichi_declared: Array = FALSE

    # 各playerのリーチが成立しているかどうか
    _riichi: Array = jnp.zeros(4, dtype=jnp.bool_)

    # 各playerの副露回数
    _n_meld: Array = jnp.zeros(4, dtype=jnp.int8)

    # melds[i][j]: player i のj回目の副露(j=1,2,3,4). 存在しなければ0
    _melds: Array = jnp.zeros((4, 4), dtype=jnp.int32)

    _is_menzen: Array = jnp.zeros(4, dtype=jnp.bool_)

    # pon[i][j]: player i がjをポンを所有している場合, src << 2 | index. or 0
    _pon: Array = jnp.zeros((4, 34), dtype=jnp.int32)

    @property
    def env_id(self) -> core.EnvId:
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
                legal_action_mask=jnp.array(data["legal_action_mask"], dtype=jnp.bool_),
                _rng_key=jnp.array(data["_rng_key"]),
                _step_count=jnp.int32(data["_step_count"]),
                _round=jnp.int8(data["_round"]),
                _honba=jnp.int8(data["_honba"]),
                _oya=jnp.int8(data["_oya"]),
                _score=jnp.array(data["_score"], dtype=jnp.float32),
                _deck=jnp.array(data["_deck"], dtype=jnp.int8),
                _next_deck_ix=jnp.int8(data["_next_deck_ix"]),
                _hand=jnp.array(data["_hand"], dtype=jnp.int8),
                _river=jnp.array(data["_river"], dtype=jnp.uint8),
                _n_river=jnp.array(data["_n_river"], dtype=jnp.int8),
                _doras=jnp.array(data["_doras"], dtype=jnp.int8),
                _n_kan=jnp.int8(data["_n_kan"]),
                _target=jnp.int8(data["_target"]),
                _last_draw=jnp.int8(data["_last_draw"]),
                _last_player=jnp.int8(data["_last_player"]),
                _furo_check_num=jnp.uint8(data["_furo_check_num"]),
                _riichi_declared=jnp.bool_(data["_riichi_declared"]),
                _riichi=jnp.array(data["_riichi"], dtype=jnp.bool_),
                _n_meld=jnp.array(data["_n_meld"], dtype=jnp.int8),
                _melds=jnp.array(data["_melds"], dtype=jnp.int32),
                _is_menzen=jnp.array(data["_is_menzen"], dtype=jnp.bool_),
                _pon=jnp.array(data["_pon"], dtype=jnp.int32),
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
            and a._honba == b._honba
            and a._oya == b._oya
            and (a._score == b._score).all()
            and (a._deck == b._deck).all()
            and a._next_deck_ix == b._next_deck_ix
            and (a._hand == b._hand).all()
            and (a._river == b._river).all()
            and (a._doras == b._doras).all()
            and a._n_kan == b._n_kan
            and a._target == b._target
            and a._last_draw == b._last_draw
            and a._last_player == b._last_player
            and a._furo_check_num == b._furo_check_num
            and a._riichi_declared == b._riichi_declared
            and (a._riichi == b._riichi).all()
            and (a._n_meld == b._n_meld).all()
            and (a._melds == b._melds).all()
            and (a._is_menzen == b._is_menzen).all()
            and (a._pon == b._pon).all()
        )


class Mahjong(core.Env):
    def __init__(self):
        super().__init__()

    def _init(self, key: PRNGKey) -> State:
        return _init(key)

    def _step(self, state: core.State, action: Array, key) -> State:
        del key
        assert isinstance(state, State)
        return _step(state, action)

    def _observe(self, state: core.State, player_id: Array) -> Array:
        assert isinstance(state, State)
        return _observe(state, player_id)

    @property
    def id(self) -> core.EnvId:
        # TODO add envid
        return "mahjong"  # type:ignore

    @property
    def version(self) -> str:
        return "beta"

    @property
    def num_players(self) -> int:
        return 4


def _init(rng: PRNGKey) -> State:
    rng, subkey = jax.random.split(rng)
    current_player = jnp.int8(jax.random.bernoulli(rng))
    last_player = jnp.int8(-1)
    deck = jax.random.permutation(rng, jnp.arange(136, dtype=jnp.int8) // 4)
    init_hand = Hand.make_init_hand(deck)
    doras = jnp.array([deck[9], -1, -1, -1, -1], dtype=jnp.int8)
    state = State(  # type:ignore
        current_player=current_player,
        _oya=current_player,
        _last_player=last_player,
        _deck=deck,
        _doras=doras,
        _hand=init_hand,
        _rng_key=subkey,
    )
    return _draw(state)


def _step(state: State, action) -> State:
    """
    actionに応じて処理を分岐
    actionの種類はmahjong/_action.pyを参照
    """

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
    """
    - deckから1枚ツモする
    - ツモしたプレイヤーのlegal_actionを生成する
    """
    state = _accept_riichi(state)
    c_p = state.current_player
    new_tile = state._deck[state._next_deck_ix]
    next_deck_ix = state._next_deck_ix - 1
    hand = state._hand.at[c_p].set(Hand.add(state._hand[c_p], new_tile))

    legal_action_mask = jax.lax.select(
        state._riichi[c_p],
        _make_legal_action_mask_w_riichi(state, hand, c_p),
        _make_legal_action_mask(state, hand, c_p, new_tile),
    )

    return state.replace(  # type:ignore
        _target=jnp.int8(-1),
        _next_deck_ix=next_deck_ix,
        _hand=hand,
        _last_draw=new_tile,
        _last_player=c_p,
        legal_action_mask=legal_action_mask,
    )


def _make_legal_action_mask(state: State, hand, c_p, new_tile):
    """
    - ツモ直後のlegal_actionを生成する
    """
    legal_action_mask = jnp.zeros(NUM_ACTION, dtype=jnp.bool_)
    legal_action_mask = legal_action_mask.at[:34].set(hand[c_p] > 0)
    legal_action_mask = legal_action_mask.at[new_tile].set(FALSE)
    legal_action_mask = legal_action_mask.at[Action.TSUMOGIRI].set(TRUE)
    legal_action_mask = legal_action_mask.at[new_tile + 34].set(
        Hand.can_ankan(hand[c_p], new_tile) | (Hand.can_kakan(hand[c_p], new_tile) & state._pon[(c_p, new_tile)] > 0)
    )
    legal_action_mask = legal_action_mask.at[Action.RIICHI].set(
        jax.lax.cond(
            state._riichi[c_p] | state._is_menzen[c_p] | (state._next_deck_ix < 13 + 4),
            lambda: FALSE,
            lambda: Hand.can_riichi(hand[c_p]),
        )
    )
    legal_action_mask = legal_action_mask.at[Action.TSUMO].set(
        Hand.can_tsumo(hand[c_p])
        & Yaku.judge(
            state._hand[c_p],
            state._melds[c_p],
            state._n_meld[c_p],
            state._last_draw,
            state._riichi[c_p],
            FALSE,
            _dora_array(state, state._riichi[c_p]),
        )[0].any()
    )
    return legal_action_mask


def _make_legal_action_mask_w_riichi(state, hand, c_p):
    """
    - リーチ状態でのツモ直後のlegal_actionを生成する
    """
    legal_action_mask = jnp.zeros(NUM_ACTION, dtype=jnp.bool_)
    legal_action_mask = legal_action_mask.at[Action.TSUMOGIRI].set(TRUE)
    legal_action_mask = legal_action_mask.at[Action.TSUMO].set(
        Hand.can_tsumo(hand[c_p])
        & Yaku.judge(
            state._hand[c_p],
            state._melds[c_p],
            state._n_meld[c_p],
            state._last_draw,
            state._riichi[c_p],
            FALSE,
            _dora_array(state, state._riichi[c_p]),
        )[0].any()
    )
    return legal_action_mask


def _discard(state: State, tile: Array):
    """
    - 手牌から選択された牌を河へ移動させる
    - 捨てられた牌に対して他のプレイヤーが鳴けるか探索する
    - もし鳴けるなら, そのプレイヤーを次の手番に設定する
    """
    c_p = state.current_player
    tile = jnp.where(tile == 68, state._last_draw, tile)
    _tile = jnp.where(state._riichi_declared, tile | jnp.uint8(0b01000000), tile)
    river = state._river.at[c_p, state._n_river[c_p]].set(jnp.uint8(_tile))
    n_river = state._n_river.at[c_p].add(1)
    hand = state._hand.at[c_p].set(Hand.sub(state._hand[c_p], tile))
    state = state.replace(  # type:ignore
        _last_draw=jnp.int8(-1),
        _hand=hand,
        _river=river,
        _n_river=n_river,
    )

    (
        meld_type,
        furo_num,
        chi_player,
        pon_player,
        kan_player,
        ron_player,
    ) = _get_next_player_after_pass(state, tile)

    no_meld_state = jax.lax.cond(
        _is_ryukyoku(state),
        lambda: state.replace(  # type:ignore
            terminated=TRUE,
            rewards=jnp.float32([Hand.is_tenpai(hand) * 100 for hand in state._hand]),
        ),
        lambda: _draw(
            state.replace(  # type:ignore
                current_player=(c_p + 1) % 4,
                _target=jnp.int8(-1),
            )
        ),
    )

    return jax.lax.switch(
        meld_type,
        [
            lambda: no_meld_state,
            lambda: state.replace(  # type:ignore
                current_player=chi_player,
                _last_player=c_p,
                _target=jnp.int8(tile),
                _furo_check_num=furo_num & 0b11111100,
                legal_action_mask=jnp.zeros(NUM_ACTION, dtype=jnp.bool_)
                .at[Action.CHI_L]
                .set(Hand.can_chi(state._hand[chi_player], tile, Action.CHI_L))
                .at[Action.CHI_M]
                .set(Hand.can_chi(state._hand[chi_player], tile, Action.CHI_M))
                .at[Action.CHI_R]
                .set(Hand.can_chi(state._hand[chi_player], tile, Action.CHI_R))
                .at[Action.PASS]
                .set(TRUE),
            ),
            lambda: state.replace(  # type:ignore
                current_player=pon_player,
                _last_player=c_p,
                _target=jnp.int8(tile),
                _furo_check_num=furo_num & 0b11110011,
                legal_action_mask=jnp.zeros(NUM_ACTION, dtype=jnp.bool_)
                .at[Action.PON]
                .set(TRUE)
                .at[Action.PASS]
                .set(TRUE),
            ),
            lambda: state.replace(  # type:ignore
                current_player=kan_player,
                _last_player=c_p,
                _target=jnp.int8(tile),
                _furo_check_num=furo_num & 0b11001111,
                legal_action_mask=jnp.zeros(NUM_ACTION, dtype=jnp.bool_)
                .at[Action.MINKAN]
                .set(TRUE)
                .at[Action.PASS]
                .set(TRUE),
            ),
            lambda: state.replace(  # type:ignore
                current_player=ron_player,
                _last_player=c_p,
                _target=jnp.int8(tile),
                _furo_check_num=furo_num,
                legal_action_mask=jnp.zeros(NUM_ACTION, dtype=jnp.bool_)
                .at[Action.RON]
                .set(TRUE)
                .at[Action.PASS]
                .set(TRUE),
            ),
        ],
    )


def _get_next_player_after_pass(state, tile):
    """
    discardの後にron,kan,pon,chiなどが可能なプレイヤーが複数いる場合,
    次にそのプレイヤーに手番を変えて, 実行するかパスするかを選択させる
    パスした場合にはその次にron,kan,pon,chiが可能なプレイヤーを探索しないといけない

    - どのプレイヤーが副露可能か探索する
    - それぞれの副露が可能なプレイヤーは高々1人なので, 8bit整数で上2桁ずつ
        現在のp kan可能なp pon可能なp chi可能なp
      を入れて, stateで保持しておく
    """

    meld_type = 0  # none < chi_L = chi_M = chi_R < pon = kan < ron の中で最大のものを探す
    pon_player = kan_player = ron_player = c_p = state.current_player
    chi_player = (c_p + 1) % 4
    can_chi = (
        Hand.can_chi(state._hand[chi_player], tile, Action.CHI_L)
        | Hand.can_chi(state._hand[chi_player], tile, Action.CHI_M)
        | Hand.can_chi(state._hand[chi_player], tile, Action.CHI_R)
    )
    meld_type = jax.lax.cond(
        can_chi,
        lambda: jnp.max(jnp.array([1, meld_type])),
        lambda: meld_type,
    )

    def search(i, tpl):
        """
        - 相対位置iのプレイヤーがpon/kan/chi可能か確認し, もし可能ならiのプレイヤーを設定する
        """
        meld_type, pon_player, kan_player, ron_player = tpl
        player = (c_p + 1 + i) % 4  # 絶対位置
        pon_player, meld_type = jax.lax.cond(
            Hand.can_pon(state._hand[player], tile),
            lambda: (i, jnp.max(jnp.array([2, meld_type]))),
            lambda: (pon_player, meld_type),
        )
        kan_player, meld_type = jax.lax.cond(
            Hand.can_minkan(state._hand[player], tile),
            lambda: (i, jnp.max(jnp.array([3, meld_type]))),
            lambda: (kan_player, meld_type),
        )
        ron_player, meld_type = jax.lax.cond(
            Hand.can_ron(state._hand[player], tile)
            & Yaku.judge(
                state._hand[player],
                state._melds[player],
                state._n_meld[player],
                state._last_draw,
                state._riichi[player],
                FALSE,
                _dora_array(state, state._riichi[player]),
            )[0].any(),
            lambda: (i, jnp.max(jnp.array([4, meld_type]))),
            lambda: (ron_player, meld_type),
        )
        return (meld_type, pon_player, kan_player, ron_player)

    # 各プレイヤーについて, どのアクションが可能かforiで調べる
    meld_type, pon_player, kan_player, ron_player = jax.lax.fori_loop(
        jnp.int8(0),
        jnp.int8(3),
        search,
        (meld_type, pon_player, kan_player, ron_player),
    )
    furo_num = jnp.uint8(c_p << 6 | kan_player << 4 | pon_player << 2 | chi_player)
    pon_player, kan_player, ron_player = (
        (c_p + 1 + pon_player) % 4,
        (c_p + 1 + kan_player) % 4,
        (c_p + 1 + ron_player) % 4,
    )

    return meld_type, furo_num, chi_player, pon_player, kan_player, ron_player


def _append_meld(state: State, meld, player):
    """
    - 与えられたmeldをstateに保持させる
    """
    melds = state._melds.at[(player, state._n_meld[player])].set(meld)
    n_meld = state._n_meld.at[player].add(1)
    return state.replace(_melds=melds, _n_meld=n_meld)  # type:ignore


def _selfkan(state: State, action):
    """
    - ankanかkakanを判断し, 分岐させる
    - deckから嶺上牌をツモする
    - 嶺上牌をツモした直後のlegal_actionを設定する
    """
    target = action - 34
    pon = state._pon[(state.current_player, target)]
    state = jax.lax.cond(
        pon == 0,
        lambda: _ankan(state, target),
        lambda: _kakan(state, target, pon >> 2, pon & 0b11),
    )

    # 嶺上牌
    rinshan_tile = state._deck[state._next_deck_ix]
    next_deck_ix = state._next_deck_ix - 1
    hand = state._hand.at[state.current_player].set(Hand.add(state._hand[state.current_player], rinshan_tile))
    legal_action_mask = jnp.zeros(NUM_ACTION, dtype=jnp.bool_)
    legal_action_mask = legal_action_mask.at[0:34].set(hand[state.current_player] > 0)
    legal_action_mask = legal_action_mask.at[rinshan_tile].set(FALSE)
    legal_action_mask = legal_action_mask.at[Action.TSUMOGIRI].set(TRUE)

    return state.replace(  # type:ignore
        _next_deck_ix=next_deck_ix,
        _last_draw=rinshan_tile,
        _hand=hand,
        legal_action_mask=legal_action_mask,
        _n_kan=state._n_kan + 1,
        _doras=state._doras.at[state._n_kan + 1].set(state._deck[9 - 2 * (state._n_kan + 1)]),
    )


def _ankan(state: State, target):
    """
    - targetからmeldを生成する
    - stateのhand,meldを更新する
    """
    curr_player = state.current_player
    meld = Meld.init(target + 34, target, src=0)
    state = _append_meld(state, meld, curr_player)
    hand = state._hand.at[curr_player].set(Hand.ankan(state._hand[curr_player], target))
    # TODO: 国士無双ロンの受付

    return state.replace(  # type:ignore
        _hand=hand,
    )


def _kakan(state: State, target, pon_src, pon_idx):
    """
    - targetからmeldを生成する
    - stateのhand,meldを更新する
    """
    melds = state._melds.at[(state.current_player, pon_idx)].set(Meld.init(target + 34, target, pon_src))
    hand = state._hand.at[state.current_player].set(Hand.kakan(state._hand[state.current_player], target))
    pon = state._pon.at[(state.current_player, target)].set(0)
    # TODO: 槍槓の受付

    return state.replace(_melds=melds, _hand=hand, _pon=pon)  # type:ignore


def _accept_riichi(state: State) -> State:
    """
    - リーチが通ったflagを立てる
    - 行動したプレイヤーのスコアを減らす
    """
    l_p = state._last_player
    _score = state._score.at[l_p].add(jnp.where(~state._riichi[l_p] & state._riichi_declared, -10, 0))
    riichi = state._riichi.at[l_p].set(state._riichi[l_p] | state._riichi_declared)
    return state.replace(  # type:ignore
        _riichi=riichi, _riichi_declared=FALSE, _score=_score
    )


def _minkan(state: State):
    """
    - targetからmeldを生成する
    - stateのhand,meldを更新する
    - deckから嶺上牌をツモする
    - 嶺上牌をツモした直後のlegal_actionを設定する
    """
    c_p = state.current_player
    l_p = state._last_player
    state = _accept_riichi(state)
    src = (l_p - c_p) % 4
    meld = Meld.init(Action.MINKAN, state._target, src)
    state = _append_meld(state, meld, c_p)
    hand = state._hand.at[c_p].set(Hand.minkan(state._hand[c_p], state._target))
    state = state.replace(_hand=hand)  # type:ignore
    is_menzen = state._is_menzen.at[c_p].set(FALSE)

    rinshan_tile = state._deck[state._next_deck_ix]
    _next_deck_ix = state._next_deck_ix - 1
    hand = state._hand.at[c_p].set(Hand.add(state._hand[c_p], rinshan_tile))

    legal_action_mask = jnp.zeros(NUM_ACTION, dtype=jnp.bool_)
    legal_action_mask = legal_action_mask.at[0:34].set(hand[c_p] > 0)
    legal_action_mask = legal_action_mask.at[rinshan_tile].set(FALSE)
    legal_action_mask = legal_action_mask.at[Action.TSUMOGIRI].set(TRUE)

    # 半透明処理
    river = state._river.at[l_p, state._n_river[l_p] - 1].set(
        state._river[l_p, state._n_river[l_p] - 1] | jnp.uint8(0b10000000)
    )

    return state.replace(  # type:ignore
        _target=jnp.int8(-1),
        _is_menzen=is_menzen,
        _next_deck_ix=_next_deck_ix,
        _last_draw=rinshan_tile,
        _hand=hand,
        legal_action_mask=legal_action_mask,
        _river=river,
        _n_kan=state._n_kan + 1,
        _doras=state._doras.at[state._n_kan + 1].set(state._deck[9 - 2 * (state._n_kan + 1)]),
    )


def _pon(state: State):
    """
    - targetからmeldを生成する
    - stateのhand,meldを更新する
    """
    c_p = state.current_player
    l_p = state._last_player
    state = _accept_riichi(state)
    src = (l_p - c_p) % 4
    meld = Meld.init(Action.PON, state._target, src)
    state = _append_meld(state, meld, c_p)
    hand = state._hand.at[c_p].set(Hand.pon(state._hand[c_p], state._target))
    is_menzen = state._is_menzen.at[c_p].set(FALSE)
    pon = state._pon.at[(c_p, state._target)].set(src << 2 | state._n_meld[c_p] - 1)

    # 半透明処理
    river = state._river.at[l_p, state._n_river[l_p] - 1].set(
        state._river[l_p, state._n_river[l_p] - 1] | jnp.uint8(0b10000000)
    )

    legal_action_mask = jnp.zeros(NUM_ACTION, dtype=jnp.bool_)
    legal_action_mask = legal_action_mask.at[:34].set(hand[c_p] > 0)

    return state.replace(  # type:ignore
        _target=jnp.int8(-1),
        _is_menzen=is_menzen,
        _pon=pon,
        _hand=hand,
        legal_action_mask=legal_action_mask,
        _river=river,
    )


def _chi(state: State, action):
    """
    - targetからmeldを生成する
    - stateのhand,meldを更新する
    """
    c_p = state.current_player
    tar_p = (c_p + 3) % 4
    tar = state._target
    state = _accept_riichi(state)
    meld = Meld.init(action, tar, src=jnp.int32(3))
    state = _append_meld(state, meld, c_p)
    hand = state._hand.at[c_p].set(Hand.chi(state._hand[c_p], tar, action))
    is_menzen = state._is_menzen.at[c_p].set(FALSE)
    legal_action_mask = jnp.zeros(NUM_ACTION, dtype=jnp.bool_)
    legal_action_mask = legal_action_mask.at[:34].set(hand[c_p] > 0)

    # 半透明処理
    river = state._river.at[tar_p, state._n_river[tar_p] - 1].set(
        state._river[tar_p, state._n_river[tar_p] - 1] | jnp.uint8(0b10000000)
    )

    return state.replace(  # type:ignore
        _target=jnp.int8(-1),
        _is_menzen=is_menzen,
        _hand=hand,
        legal_action_mask=legal_action_mask,
        _river=river,
    )


def _pass(state: State):
    """
    discard() を参照
    ron,kan,pon,chi可能なプレイヤーがパスした場合, 予めStateに保持してある探索データから次にron,kan,pon,chi可能なプレイヤーがいるか読み取る
    state._furo_check_numには8bit整数で上2桁ずつに
      現在のp kan可能なp pon可能なp chi可能なp
    を入れてあるので, ここから復元する

    - もしchi可能なプレイヤーがいれば, そのプレイヤーを次の手番とし, legal_actionを生成する
    - もしpon可能なプレイヤーがいれば, そのプレイヤーを次の手番とし, legal_actionを生成する
    - もしkan可能なプレイヤーがいれば, そのプレイヤーを次の手番とし, legal_actionを生成する

    # TODO
    - もしron可能なプレイヤーがいれば(ダブロン), そのプレイヤーを次の手番とし, legal_actionを生成する
    """
    last_player = (state._furo_check_num & 0b11000000) >> 6
    kan_player = (state._furo_check_num & 0b00110000) >> 4
    pon_player = (state._furo_check_num & 0b00001100) >> 2
    chi_player = state._furo_check_num & 0b00000011
    _state = state

    state = jax.lax.cond(
        chi_player > 0,
        lambda: state.replace(  # type:ignore
            current_player=jnp.int8(last_player + 1 + chi_player) % 4,
            _furo_check_num=jnp.uint8(state._furo_check_num & 0b11111100),
            legal_action_mask=jnp.zeros(NUM_ACTION, dtype=jnp.bool_)
            .at[Action.CHI_L]
            .set(
                Hand.can_chi(
                    state._hand[(last_player + 1 + chi_player) % 4],
                    state._target,
                    Action.CHI_L,
                )
            )
            .at[Action.CHI_M]
            .set(
                Hand.can_chi(
                    state._hand[(last_player + 1 + chi_player) % 4],
                    state._target,
                    Action.CHI_M,
                )
            )
            .at[Action.CHI_R]
            .set(
                Hand.can_chi(
                    state._hand[(last_player + 1 + chi_player) % 4],
                    state._target,
                    Action.CHI_R,
                )
            )
            .at[Action.PASS]
            .set(TRUE),
        ),
        lambda: _draw(
            _state.replace(  # type: ignore
                current_player=jnp.int8(last_player + 1) % 4,
            )
        ),
    )
    state = jax.lax.cond(
        pon_player > 0,
        lambda: _state.replace(  # type:ignore
            current_player=jnp.int8(last_player + 1 + pon_player) % 4,
            _furo_check_num=jnp.uint8(_state._furo_check_num & 0b11110011),
            legal_action_mask=jnp.zeros(NUM_ACTION, dtype=jnp.bool_)
            .at[Action.PON]
            .set(Hand.can_pon(_state._hand[pon_player], _state._target))
            .at[Action.PASS]
            .set(TRUE),
        ),
        lambda: state,
    )
    state = jax.lax.cond(
        kan_player > 0,
        lambda: _state.replace(  # type:ignore
            current_player=jnp.int8(last_player + 1 + kan_player) % 4,
            _furo_check_num=jnp.uint8(_state._furo_check_num & 0b11001111),
            legal_action_mask=jnp.zeros(NUM_ACTION, dtype=jnp.bool_)
            .at[Action.MINKAN]
            .set(Hand.can_minkan(_state._hand[kan_player], _state._target))
            .at[Action.PASS]
            .set(TRUE),
        ),
        lambda: state,
    )
    return state


def _riichi(state: State):
    """
    - リーチ宣言フラグを立てる
    - リーチ直後のlegal_actionを生成する
    """
    c_p = state.current_player
    legal_action_mask = (
        jax.lax.fori_loop(
            0,
            34,
            lambda i, arr: arr.at[i].set(
                jax.lax.cond(
                    state._hand[c_p][i] > (i == state._last_draw),
                    lambda: Hand.is_tenpai(Hand.sub(state._hand[c_p], i)),
                    lambda: FALSE,
                )
            ),
            jnp.zeros(NUM_ACTION, dtype=jnp.bool_),
        )
        .at[Action.TSUMOGIRI]
        .set(Hand.is_tenpai(Hand.sub(state._hand[c_p], state._last_draw)))
    )

    return state.replace(  # type:ignore
        _riichi_declared=TRUE, legal_action_mask=legal_action_mask
    )


def _tsumo(state: State):
    """
    - 勝者の得点を計算する
    - 得点を更新する
    - terminated=true
    """
    c_p = state.current_player

    score = Yaku.score(
        state._hand[c_p],
        state._melds[c_p],
        state._n_meld[c_p],
        state._target,
        state._riichi[c_p],
        is_ron=FALSE,
        dora=_dora_array(state, state._riichi[c_p]),
    )
    s1 = score + (-score) % 100
    s2 = (score * 2) + (-(score * 2)) % 100

    _oya = (state._oya + state._round) % 4
    reward = jax.lax.cond(
        _oya == c_p,
        lambda: jnp.full(4, -s2, dtype=jnp.int32).at[c_p].set(s2 * 3),
        lambda: jnp.full(4, -s1, dtype=jnp.int32).at[_oya].set(-s2).at[c_p].set(s1 * 2 + s2),
    )

    # 供託
    reward -= 1000 * state._riichi
    reward = reward.at[c_p].set(reward[c_p] + 1000 * jnp.sum(state._riichi))
    return state.replace(  # type:ignore
        terminated=TRUE, rewards=jnp.float32(reward)
    )


def _ron(state: State):
    """
    - 勝者の得点を計算する
    - 得点を更新する
    - terminated=true
    """
    c_p = state.current_player
    score = Yaku.score(
        state._hand[c_p],
        state._melds[c_p],
        state._n_meld[c_p],
        state._target,
        state._riichi[c_p],
        is_ron=TRUE,
        dora=_dora_array(state, state._riichi[c_p]),
    )
    score = jax.lax.cond(
        (state._oya + state._round) % 4 == c_p,
        lambda: score * 6,
        lambda: score * 4,
    )
    score += -score % 100
    reward = jnp.zeros(4, dtype=jnp.int32).at[c_p].set(score).at[state._last_player].set(-score)

    # 供託
    reward -= 1000 * state._riichi
    reward = reward.at[c_p].set(reward[c_p] + 1000 * jnp.sum(state._riichi))
    return state.replace(  # type:ignore
        terminated=TRUE, rewards=jnp.float32(reward)
    )


def _is_ryukyoku(state: State):
    return state._next_deck_ix == 13


def _next_game(state: State):
    # TODO
    return _pass(state)


def _observe(state: State, player_id: Array) -> Array:
    return jnp.int8(0)


def _dora_array(state: State, riichi):
    """
    - 得点計算用に, 長さ34で, ドラ牌のindexに枚数が格納されている配列を作成する
    """

    def next(tile):
        """
        - ドラ牌を算出する
        """
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
            state._n_kan + 1,
            lambda i, arr: arr.at[next(state._deck[5 + 2 * i])].set(TRUE).at[next(state._doras[4 + 2 * i])].set(TRUE),
            dora,
        ),
        lambda: jax.lax.fori_loop(
            0,
            state._n_kan + 1,
            lambda i, arr: arr.at[next(state._doras[5 + 2 * i])].set(TRUE),
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
