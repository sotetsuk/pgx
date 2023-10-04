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

import pgx.core as v1
from pgx._mahjong._action import Action
from pgx._mahjong._hand import Hand
from pgx._mahjong._meld import Meld
from pgx._src.struct import dataclass

FALSE = jnp.bool_(False)
TRUE = jnp.bool_(True)
NUM_ACTION = 78


@dataclass
class State(v1.State):
    current_player: jnp.ndarray = jnp.int8(0)  # actionを行うplayer
    observation: jnp.ndarray = jnp.int8(0)
    rewards: jnp.ndarray = jnp.float32([0.0, 0.0, 0.0, 0.0])
    terminated: jnp.ndarray = FALSE
    truncated: jnp.ndarray = FALSE
    legal_action_mask: jnp.ndarray = jnp.zeros(NUM_ACTION, dtype=jnp.bool_)
    _rng_key: jax.random.KeyArray = jax.random.PRNGKey(0)
    _step_count: jnp.ndarray = jnp.int32(0)
    # --- Mahjong specific ---
    deck: jnp.ndarray = jnp.zeros(136, dtype=jnp.int8)

    # 次に引く牌のindex
    next_deck_ix: jnp.ndarray = jnp.int8(135)

    # 各プレイヤーの手牌. 長さ34で、数字は持っている牌の数
    hand: jnp.ndarray = jnp.zeros((4, 34), dtype=jnp.int8)

    doras: jnp.ndarray = jnp.zeros(4, dtype=jnp.int8)

    num_kan: jnp.ndarray = jnp.int8(0)

    # 直前に捨てられてron,pon,chi,kanの対象になっている牌. 存在しなければ-1
    target: jnp.ndarray = jnp.int8(0)

    # 手牌が3n+2枚のplayerが直前に引いた牌. 存在しなければ-1
    last_draw: jnp.ndarray = jnp.int8(0)

    # 最後にプレイしたプレイヤ（ron,pon,chi,kanの対象）.
    last_player: jnp.ndarray = jnp.int8(0)

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


class Mahjong(v1.Env):
    def __init__(self):
        super().__init__()

    def _init(self, key: jax.random.KeyArray) -> State:
        return _init(key)

    def _step(self, state: v1.State, action: jnp.ndarray) -> State:
        assert isinstance(state, State)
        return _step(state, action, player=state.current_player)

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
    current_player = jnp.int8(jax.random.bernoulli(subkey))
    last_player = jnp.int8(-1)
    deck = jax.random.permutation(rng, jnp.arange(136) // 4)
    init_hand = Hand.make_init_hand(deck)
    state = State(  # type:ignore
        current_player=current_player,
        last_player=last_player,
        deck=deck,
        hand=init_hand,
        next_deck_ix=135 - 13 * 4,
    )
    return _draw(state)


def _step(state: State, action: jnp.ndarray, player: jnp.ndarray) -> State:
    # TODO
    # - Actionの処理
    #   - meld
    #   - riichi
    #   - ron, tsumo
    # - 勝利条件確認
    # - playerどうするか
    # - lax.switch使った方が良いんだろうけど、簡単なうちはcondで分岐させる

    state = state.replace(current_player=player)  # type:ignore

    discard = (action < 34) | (action == 68)
    self_kan = ~discard & (action < 68)
    state = jax.lax.cond(
        discard,
        lambda: _discard(state, action),
        lambda: state,
    )
    state = jax.lax.cond(
        self_kan,
        lambda: _selfkan(state, action),
        lambda: state,
    )
    state = jax.lax.cond(
        (~discard) & (~self_kan),
        lambda: jax.lax.switch(
            action - 68,
            [
                lambda: _discard(state, action),
                lambda: _riichi(state),
                lambda: _tsumo(state),
                lambda: _ron(state),
                lambda: _pon(state),
                lambda: _minkan(state),
                lambda: _chi(state, player, Action.CHI_L),
                lambda: _chi(state, player, Action.CHI_M),
                lambda: _chi(state, player, Action.CHI_R),
                lambda: _draw(state),
            ],
        ),
        lambda: state,
    )

    return state


def _draw(state: State):
    new_tile = state.deck[state.next_deck_ix]
    next_deck_ix = state.next_deck_ix - 1
    hand = state.hand.at[state.current_player].set(
        Hand.add(state.hand[state.current_player], new_tile)
    )

    legal_action_mask = jnp.zeros(NUM_ACTION, dtype=jnp.bool_)
    legal_action_mask = legal_action_mask.at[:34].set(
        hand[state.current_player] > 0
    )
    legal_action_mask = legal_action_mask.at[new_tile].set(FALSE)
    legal_action_mask = legal_action_mask.at[Action.TSUMOGIRI].set(TRUE)

    return state.replace(  # type:ignore
        current_player=state.current_player,
        next_deck_ix=next_deck_ix,
        hand=hand,
        last_draw=new_tile,
        legal_action_mask=legal_action_mask,
    )


def _discard(state: State, tile: jnp.ndarray):
    tile = jax.lax.select(tile == 68, state.last_draw, tile)
    hand = state.hand.at[state.current_player].set(
        Hand.sub(state.hand[state.current_player], tile)
    )
    state = state.replace(  # type:ignore
        target=jnp.int8(tile), last_draw=-1, hand=hand
    )

    return _draw(state)


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
    return state.replace(  # type:ignore
        next_deck_ix=next_deck_ix, last_draw=rinshan_tile, hand=hand
    )


def _ankan(state: State, target):
    curr_player = state.current_player
    meld = Meld.init(target + 34, target, src=0)
    state = _append_meld(state, meld, curr_player)
    hand = state.hand.at[curr_player].set(
        Hand.ankan(state.hand[curr_player], target)
    )
    # TODO: 国士無双ロンの受付

    return state.replace(hand=hand)  # type:ignore


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
    riichi = state.riichi.at[state.current_player].set(
        state.riichi[state.current_player] | state.riichi_declared
    )
    return state.replace(riichi=riichi, riichi_declared=FALSE)  # type:ignore


def _minkan(state: State):
    state = _accept_riichi(state)
    src = (state.last_player - state.current_player) % 4
    meld = Meld.init(Action.MINKAN, state.target, src)
    state = _append_meld(state, meld, state.current_player)
    hand = state.hand.at[state.current_player].set(
        Hand.minkan(state.hand[state.current_player], state.target)
    )
    is_menzen = state.is_menzen.at[state.current_player].set(FALSE)

    rinshan_tile = state.deck[state.next_deck_ix]
    next_deck_ix = state.next_deck_ix - 1
    hand = hand.at[state.current_player].set(
        Hand.add(state.hand[state.current_player], rinshan_tile)
    )
    return state.replace(  # type:ignore
        target=jnp.int8(-1),
        is_menzen=is_menzen,
        next_deck_ix=next_deck_ix,
        last_draw=rinshan_tile,
        hand=hand,
    )


def _pon(state: State):
    state = _accept_riichi(state)
    src = (state.last_player - state.current_player) % 4
    meld = Meld.init(Action.PON, state.target, src)
    state = _append_meld(state, meld, state.current_player)
    hand = state.hand.at[state.current_player].set(
        Hand.pon(state.hand[state.current_player], state.target)
    )
    is_menzen = state.is_menzen.at[state.current_player].set(FALSE)
    pon = state.pon.at[(state.current_player, state.target)].set(
        src << 2 | state.n_meld[state.current_player] - 1
    )

    return state.replace(  # type:ignore
        target=jnp.int8(-1),
        is_menzen=is_menzen,
        pon=pon,
        hand=hand,
    )


def _chi(state: State, player, action: int):
    state = _accept_riichi(state)
    meld = Meld.init(action, state.target, src=3)
    state = _append_meld(state, meld, player)
    hand = state.hand.at[player].set(
        Hand.chi(state.hand[player], state.target, action)
    )
    is_menzen = state.is_menzen.at[player].set(False)
    legal_action_mask = jnp.zeros(NUM_ACTION, dtype=jnp.bool_)
    legal_action_mask = legal_action_mask.at[:34].set(hand[player] > 0)

    return state.replace(  # type:ignore
        current_player=player,
        target=jnp.int8(-1),
        is_menzen=is_menzen,
        hand=hand,
        legal_action_mask=legal_action_mask,
    )


def _pass(state: State):
    # pon -> chi

    # ponでpassした場合

    # chiでpassした場合

    # kanでpassした場合

    ...


def _riichi(state: State):
    ...


def _tsumo(state: State):
    ...


def _ron(state: State):
    ...


def _observe(state: State, player_id: jnp.ndarray) -> jnp.ndarray:
    ...
