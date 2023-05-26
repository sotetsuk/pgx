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

# from pgx._mahjong._action import Action
from pgx._mahjong._hand import Hand

# from pgx._mahjong._meld import Meld
from pgx._src.struct import dataclass

FALSE = jnp.bool_(False)
TRUE = jnp.bool_(True)


@dataclass
class State(v1.State):
    current_player: jnp.ndarray = jnp.int8(0)  # 手牌が3n+2枚, もしくは直前に牌を捨てたplayer
    observation: jnp.ndarray = jnp.int8(0)
    rewards: jnp.ndarray = jnp.float32([0.0, 0.0, 0.0, 0.0])
    terminated: jnp.ndarray = FALSE
    truncated: jnp.ndarray = FALSE
    legal_action_mask: jnp.ndarray = jnp.ones(9, dtype=jnp.bool_)
    _rng_key: jax.random.KeyArray = jax.random.PRNGKey(0)
    _step_count: jnp.ndarray = jnp.int32(0)
    # --- Mahjong specific ---
    deck: jnp.ndarray = jnp.zeros(136)

    # 次に引く牌のindex
    next_deck_ix: jnp.ndarray = jnp.int8(135)

    # 各プレイヤーの手牌. 長さ34で、数字は持っている牌の数
    hand: jnp.ndarray = jnp.zeros((4, 34))

    # 直前に捨てられてron,pon,chi の対象になっている牌. 存在しなければ-1
    target: jnp.ndarray = jnp.int8(0)

    # 手牌が3n+2枚のplayerが直前に引いた牌. 存在しなければ-1
    last_draw: jnp.ndarray = jnp.int8(0)

    # state.current_player がリーチ宣言してから, その直後の打牌が通るまでTrue
    riichi_declared: jnp.ndarray = FALSE

    # 各playerのリーチが成立しているかどうか
    riichi: jnp.ndarray = FALSE

    # 各playerの副露回数
    n_meld: jnp.ndarray = jnp.zeros(4)

    # melds[i][j]: player i のj回目の副露(j=1,2,3,4). 存在しなければ0
    melds: jnp.ndarray = jnp.zeros((4, 4))

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
    current_player = jnp.int8(jax.random.bernoulli(subkey))
    deck = jax.random.permutation(rng, jnp.arange(136) // 4)
    first_tile = deck[135 - 13 * 4]
    init_hand = Hand.make_init_hand(deck)
    init_hand = init_hand.at[current_player].set(
        Hand.add(init_hand[current_player], first_tile)
    )
    return State(
        current_player=current_player,
        deck=deck,
        hand=init_hand,
        next_deck_ix=135 - 13 * 4 - 1,
        last_draw=first_tile,
    )  # type:ignore


def _step(state: State, action: jnp.ndarray) -> State:
    # TODO
    # - Actionの処理
    #   - meld
    #   - riichi
    #   - ron, tsumo
    # - 勝利条件確認

    state = _discard(state, action)

    # ポンとかチーとかない場合はdrawへ
    state = _draw(state)

    return state


def _draw(state: State):
    current_player = (state.current_player + 1) % 4
    new_tile = state.deck[state.next_deck_ix]
    next_deck_ix = state.next_deck_ix - 1
    hand = state.hand.at[current_player].set(
        Hand.add(state.hand[current_player], new_tile)
    )

    return state.replace(  # type:ignore
        current_player=current_player, next_deck_ix=next_deck_ix, hand=hand
    )


def _discard(state: State, tile: jnp.ndarray):
    hand = state.hand.at[state.current_player].set(
        Hand.sub(state.hand[state.current_player], tile)
    )
    return state.replace(  # type:ignore
        target=jnp.int8(tile), last_draw=-1, hand=hand
    )


def _observe(state: State, player_id: jnp.ndarray) -> jnp.ndarray:
    ...
