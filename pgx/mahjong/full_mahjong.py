from __future__ import annotations

import json
import os
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import jit, tree_util


class Tile:
    @staticmethod
    def to_str(tile: int) -> str:
        suit, num = tile // 9, tile % 9 + 1
        return str(num) + ["m", "p", "s", "z"][suit]

    @staticmethod
    @jit
    def is_outside(tile: int) -> bool:
        num = tile % 9
        return (tile >= 27) | (num == 0) | (num == 8)


class Action:
    # 手出し: 0~33
    TSUMOGIRI = 34
    RIICHI = 35
    TSUMO = 36

    RON = 37
    PON = 38
    CHI_R = 39  # 45[6]
    CHI_M = 40  # 4[5]6
    CHI_L = 41  # [4]56
    PASS = 42

    NONE = 43


@dataclass
class Deck:
    idx: int
    arr: jnp.ndarray

    @jit
    def is_empty(self) -> bool:
        return self.size() == 0

    @jit
    def size(self) -> int:
        return 122 - self.idx

    def _tree_flatten(self):
        children = (self.idx, self.arr)
        aux_data = {}
        return (children, aux_data)

    @staticmethod
    @jit
    def init(key: jnp.ndarray) -> Deck:
        arr = jax.random.permutation(key, jnp.arange(136) // 4)
        return Deck(0, arr)

    @staticmethod
    @jit
    def draw(deck: Deck) -> tuple[Deck, jnp.ndarray]:
        # -> tuple[Deck, int]
        # assert not deck.is_empty()
        tile = deck.arr[deck.idx]
        deck.idx += 1
        return deck, tile

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)


tree_util.register_pytree_node(Deck, Deck._tree_flatten, Deck._tree_unflatten)


class CacheLoader:
    DIR = os.path.join(os.path.dirname(__file__), "cache")

    @staticmethod
    @jit
    def load_hand_cache():
        with open(os.path.join(CacheLoader.DIR, "hand_cache.json")) as f:
            return jnp.array(json.load(f), dtype=jnp.uint32)

    @staticmethod
    def load_yaku_cache():
        with open(os.path.join(CacheLoader.DIR, "yaku_cache.json")) as f:
            return jnp.array(json.load(f), dtype=jnp.int32)


class Hand:
    CACHE = CacheLoader.load_hand_cache()

    @staticmethod
    @jit
    def cache(code: int) -> int:
        return (Hand.CACHE[code >> 5] >> (code & 0b11111)) & 1

    @staticmethod
    @jit
    def can_ron(hand: jnp.ndarray, tile: int) -> bool:
        # assert jnp.sum(hand) % 3 == 1
        # assert hand[tile] < 4
        return Hand.can_tsumo(Hand.add(hand, tile))

    @staticmethod
    @jit
    def can_riichi(hand: jnp.ndarray) -> bool:
        # assert: hand is menzen
        return jax.lax.fori_loop(
            0,
            34,
            lambda i, sum: jax.lax.cond(
                hand[i] == 0,
                lambda: sum,
                lambda: sum | Hand.is_tenpai(Hand.sub(hand, i)),
            ),
            False,
        )

    @staticmethod
    @jit
    def is_tenpai(hand: jnp.ndarray) -> bool:
        # assert jnp.sum(hand) % 3 == 1
        return jax.lax.fori_loop(
            0,
            34,
            lambda tile, sum: jax.lax.cond(
                hand[tile] == 4,
                lambda: False,
                lambda: sum | Hand.can_ron(hand, tile),
            ),
            False,
        )

    @staticmethod
    @jit
    def can_tsumo(hand: jnp.ndarray) -> bool:
        # assert jnp.sum(hand) % 3 == 2
        heads = 0
        valid = True

        for i in range(3):
            heads, valid, code, size = jax.lax.fori_loop(
                0,
                9,
                lambda j, tpl: jax.lax.cond(
                    hand[9 * i + j] == 0,
                    lambda: (
                        tpl[0] + (tpl[3] % 3 == 2),
                        tpl[1] & (Hand.cache(tpl[2]) != 0),
                        0,
                        0,
                    ),
                    lambda: (
                        tpl[0],
                        tpl[1],
                        ((tpl[2] << 1) + 1)
                        << (hand[9 * i + j].astype(int) - 1),
                        tpl[3] + hand[9 * i + j].astype(int),
                    ),
                ),
                (heads, valid, 0, 0),
            )

            heads += size % 3 == 2
            valid &= Hand.cache(code) != 0

        heads, valid = jax.lax.fori_loop(
            27,
            34,
            lambda i, tpl: (
                tpl[0] + (hand[i] == 2),
                tpl[1] & (hand[i] != 1) & (hand[i] != 4),
            ),
            (heads, valid),
        )

        return valid & (heads == 1)

    @staticmethod
    @jit
    def can_pon(hand: jnp.ndarray, tile: int) -> jnp.ndarray:
        # -> bool
        return hand[tile] >= 2

    @staticmethod
    @jit
    def can_chi(hand: jnp.ndarray, tile: int, action: int) -> bool:
        # assert jnp.sum(hand) % 3 == 1
        # assert action is Action.CHI_R, CHI_M or CHI_L
        return jax.lax.switch(
            action - Action.CHI_R,
            [
                lambda: (
                    (tile < 27)
                    & (tile % 9 > 1)
                    & (hand[tile - 2] > 0)
                    & (hand[tile - 1] > 0)
                ),
                lambda: (
                    (tile < 27)
                    & (tile % 9 > 0)
                    & (tile % 9 < 8)
                    & (hand[tile - 1] > 0)
                    & (hand[tile + 1] > 0)
                ),
                lambda: (
                    (tile < 27)
                    & (tile % 9 < 7)
                    & (hand[tile + 1] > 0)
                    & (hand[tile + 2] > 0)
                ),
            ],
        )

    @staticmethod
    @jit
    def add(hand: jnp.ndarray, tile: int, x: int = 1) -> jnp.ndarray:
        # assert 0 <= hand[tile] + x <= 4
        return hand.at[tile].set(hand[tile] + x)

    @staticmethod
    @jit
    def sub(hand: jnp.ndarray, tile: int, x: int = 1) -> jnp.ndarray:
        # assert 0 <= hand[tile] - x <= 4
        return Hand.add(hand, tile, -x)

    @staticmethod
    @jit
    def pon(hand: jnp.ndarray, tile: int) -> jnp.ndarray:
        # assert Hand.can_pon(hand, tile)
        return Hand.sub(hand, tile, 2)

    @staticmethod
    @jit
    def chi(hand: jnp.ndarray, tile: int, action: int) -> jnp.ndarray:
        # assert Hand.can_chi(hand, tile, action)
        # assert action is Action.CHI_R, CHI_M or CHI_L
        return jax.lax.switch(
            action - Action.CHI_R,
            [
                lambda: Hand.sub(Hand.sub(hand, tile - 2), tile - 1),
                lambda: Hand.sub(Hand.sub(hand, tile - 1), tile + 1),
                lambda: Hand.sub(Hand.sub(hand, tile + 1), tile + 2),
            ],
        )

    @staticmethod
    def to_str(hand: jnp.ndarray) -> str:
        s = ""
        for i in range(4):
            t = ""
            for j in range(9 if i < 3 else 7):
                t += str(j + 1) * hand[9 * i + j]
            if t:
                t += ["m", "p", "s", "z"][i]
            s += t
        return s

    @staticmethod
    def from_str(s: str) -> jnp.ndarray:
        base = 0
        hand = jnp.zeros(34, dtype=jnp.uint8)
        for c in reversed(s):
            if c == "m":
                base = 0
            elif c == "p":
                base = 9
            elif c == "s":
                base = 18
            elif c == "z":
                base = 27
            else:
                hand = Hand.add(hand, ord(c) - ord("1") + base)
        return hand


@dataclass
class Observation:
    hand: jnp.ndarray
    target: int
    last_draw: int


class Meld:
    @staticmethod
    @jit
    def init(action: int, target: int, src: int) -> int:
        return (src << 12) | (target << 6) | action

    @staticmethod
    def to_str(meld: int) -> str:
        action = Meld.action(meld)
        target = Meld.target(meld)
        suit, num = target // 9, target % 9 + 1
        if action == Action.PON:
            return "{}{}{}{}".format(num, num, num, ["m", "p", "s", "z"][suit])
        if action == Action.CHI_R:
            return "{}{}{}{}".format(
                num - 2, num - 1, num, ["m", "p", "s", "z"][suit]
            )
        if action == Action.CHI_M:
            return "{}{}{}{}".format(
                num - 1, num, num + 1, ["m", "p", "s", "z"][suit]
            )
        if action == Action.CHI_L:
            return "{}{}{}{}".format(
                num, num + 1, num + 2, ["m", "p", "s", "z"][suit]
            )
        return ""

    @staticmethod
    @jit
    def target(meld: int) -> int:
        return (meld >> 6) & 0b111111

    @staticmethod
    @jit
    def action(meld: int) -> int:
        return meld & 0b111111

    @staticmethod
    @jit
    def is_outside(meld: int) -> int:
        target = Meld.target(meld)
        return jax.lax.switch(
            Meld.action(meld) - Action.PON,
            [
                lambda: Tile.is_outside(target),
                lambda: Tile.is_outside(target - 2) | Tile.is_outside(target),
                lambda: Tile.is_outside(target - 1)
                | Tile.is_outside(target + 1),
                lambda: Tile.is_outside(target) | Tile.is_outside(target + 2),
            ],
        )


class Yaku:
    CACHE = CacheLoader.load_yaku_cache()
    MAX_PATTERNS = 4

    断么九 = 0
    平和 = 1
    一盃口 = 2
    二盃口 = 3
    混全帯么九 = 4
    純全帯么九 = 5

    FAN = jnp.array(
        [
            [1, 0, 0, 0, 1, 2],  # 副露
            [1, 1, 1, 3, 2, 3],  # 面前
        ]
    )

    @staticmethod
    @jit
    def shift(code: int) -> int:
        code -= (code >> 12 > 1) << 12
        code -= (code >> 9 > 1) << 9
        code -= (code >> 6 > 1) << 6
        code -= (code >> 3 > 1) << 3
        code -= code > 1
        return code

    @staticmethod
    @jit
    def head(code: int) -> jnp.ndarray:
        return Yaku.CACHE[code] & 0b1111

    @staticmethod
    @jit
    def chow(code: int) -> jnp.ndarray:
        return Yaku.CACHE[code] >> 4 & 0b1111111

    @staticmethod
    @jit
    def pung(code: int) -> jnp.ndarray:
        return Yaku.CACHE[code] >> 11 & 0b111111111

    @staticmethod
    @jit
    def double_chows(code: int) -> jnp.ndarray:
        return Yaku.CACHE[code] >> 20 & 0b11

    @staticmethod
    @jit
    def is_outside(code: int, begin: int, end: int) -> jnp.ndarray:
        outside = Yaku.CACHE[code] >> 22 & 0b11
        return (
            (code == 0)
            | (begin % 9 == 0) & (outside & 1)
            | (end % 9 == 0) & (outside >> 1 & 1)
        ) == 1

    @staticmethod
    @jit
    def is_pinfu(code: int, begin: int, end: int, last: int) -> jnp.ndarray:
        len = end - begin
        left = Yaku.chow(code)
        right = Yaku.chow(code) << 3
        open_end = (left ^ (left & 1) * (begin % 9 == 0)) << 2 | (
            right ^ (right >> len & 1) * (end % 9 == 0) << len
        ) >> 3

        pos = last - begin  # WARNING: may be negative
        in_range = (0 <= pos) & (pos < len)
        pos *= in_range
        return ((in_range == 0) | (open_end >> pos & 1) == 1) & (
            Yaku.pung(code) == 0
        )

    @staticmethod
    @jit
    def judge(
        hand: jnp.ndarray,
        melds: jnp.ndarray,
        meld_num: int,
        last: int,
    ) -> jnp.ndarray:
        # assert Hand.can_tsumo(hand)

        is_pinfu = jnp.full(
            Yaku.MAX_PATTERNS,
            jnp.all(hand[28:31] < 3)
            & (hand[27] == 0)
            & jnp.all(hand[31:34] == 0),
        )
        # NOTE: 南,西,北: オタ風扱い

        is_outside = jnp.full(
            Yaku.MAX_PATTERNS,
            jax.lax.fori_loop(
                0,
                meld_num,
                lambda i, valid: valid & Meld.is_outside(melds[i]),
                True,
            ),
        )
        double_chows = jnp.full(Yaku.MAX_PATTERNS, 0)

        for suit in range(3):
            code = 0
            begin = 9 * suit
            for tile in range(9 * suit, 9 * (suit + 1)):
                # print(code, begin, tile)
                code, is_pinfu, is_outside, double_chows, begin = jax.lax.cond(
                    hand[tile] == 0,
                    lambda: (
                        0,
                        is_pinfu & Yaku.is_pinfu(code, begin, tile, last),
                        is_outside & Yaku.is_outside(code, begin, tile),
                        double_chows + Yaku.double_chows(code),
                        tile + 1,
                    ),
                    lambda: (
                        ((code << 1) + 1) << (hand[tile].astype(int) - 1),
                        is_pinfu,
                        is_outside,
                        double_chows,
                        begin,
                    ),
                )

            is_pinfu &= Yaku.is_pinfu(code, begin, 9 * (suit + 1), last)
            is_outside &= Yaku.is_outside(code, begin, 9 * (suit + 1))
            double_chows += Yaku.double_chows(code)

        flatten = Yaku.flatten(hand, melds, meld_num)
        yaku = (
            jnp.full((Yaku.FAN.shape[1], Yaku.MAX_PATTERNS), False)
            .at[Yaku.平和]
            .set(is_pinfu)
            .at[Yaku.一盃口]
            .set(double_chows == 1)
            .at[Yaku.二盃口]
            .set(double_chows == 2)
            .at[Yaku.混全帯么九]
            .set(is_outside & jnp.any(flatten[27:] > 0))
            .at[Yaku.純全帯么九]
            .set(is_outside & jnp.all(flatten[27:] == 0))
        )

        is_menzen = jax.lax.cond(meld_num == 0, lambda: 1, lambda: 0)
        yaku = yaku.T[jnp.argmax(jnp.dot(Yaku.FAN[is_menzen], yaku))]

        return yaku.at[Yaku.断么九].set(Yaku._is_tanyao(flatten))

    @staticmethod
    @jit
    def flatten(
        hand: jnp.ndarray, melds: jnp.ndarray, meld_num: int
    ) -> jnp.ndarray:
        return jax.lax.fori_loop(
            0, meld_num, lambda i, arr: Yaku._flatten(arr, melds[i]), hand
        )

    @staticmethod
    @jit
    def _flatten(hand: jnp.ndarray, meld: int) -> jnp.ndarray:
        target, action = Meld.target(meld), Meld.action(meld)
        return jax.lax.switch(
            action - Action.PON,
            [
                lambda: Hand.add(hand, target, 3),
                lambda: Hand.add(
                    Hand.add(Hand.add(hand, target - 2), target - 1), target
                ),
                lambda: Hand.add(
                    Hand.add(Hand.add(hand, target - 1), target + 1), target
                ),
                lambda: Hand.add(
                    Hand.add(Hand.add(hand, target + 1), target + 2), target
                ),
            ],
        )

    @staticmethod
    @jit
    def _is_tanyao(hand: jnp.ndarray) -> bool:
        return (
            (hand[0] == 0)
            & (hand[8] == 0)
            & (hand[9] == 0)
            & (hand[17] == 0)
            & (hand[18] == 0)
            & jnp.all(hand[26:] == 0)
        )


@dataclass
class State:
    deck: Deck
    hand: jnp.ndarray
    turn: int  # 手牌が3n+2枚, もしくは直前に牌を捨てたplayer
    target: int  # 直前に捨てられてron,pon,chi の対象になっている牌. 存在しなければ-1
    last_draw: int  # 手牌が3n+2枚のplayerが直前に引いた牌. 存在しなければ-1
    riichi_declared: bool  # state.turn がリーチ宣言してから, その直後の打牌が通るまでTrue
    riichi: jnp.ndarray  # 各playerのリーチが成立しているかどうか
    meld_num: jnp.ndarray  # 各playerの副露回数
    melds: jnp.ndarray
    # melds[i][j]: player i のj回目の副露(j=1,2,3,4). 存在しなければ0

    # reward:
    # - player0 がplayer1 からロン => [ 2,-2, 0, 0]
    # - player0 がツモ             => [ 2,-2,-2,-2]
    # - 流局時 全員聴牌            => [ 1, 1, 1, 1]
    # - 流局時 全員ノー聴          => [-1,-1,-1,-1]
    # - 流局時 player0 だけ聴牌    => [ 1,-1,-1,-1]

    @jit
    def legal_actions(self) -> jnp.ndarray:
        legal_actions = jnp.full((4, 43), False)

        # リーチ
        legal_actions = jax.lax.cond(
            (self.last_draw == -1)
            | self.riichi_declared
            | self.riichi[self.turn]
            | (self.deck.size() < 4)
            # | (jnp.sum(self.hand[self.turn]) < 14),
            | (self.meld_num[self.turn]),
            lambda: legal_actions,
            lambda: legal_actions.at[(self.turn, Action.RIICHI)].set(
                Hand.can_riichi(self.hand[self.turn])
            ),
        )

        # リーチ宣言直後の打牌
        legal_actions = jax.lax.cond(
            (self.last_draw != -1) & self.riichi_declared,
            lambda arr: jax.lax.fori_loop(
                0,
                34,
                lambda i, arr: arr.at[(self.turn, i)].set(
                    jax.lax.cond(
                        self.hand[self.turn][i] > (i == self.last_draw),
                        lambda: Hand.is_tenpai(
                            Hand.sub(self.hand[self.turn], i)
                        ),
                        lambda: False,
                    )
                ),
                arr,
            )
            .at[(self.turn, Action.TSUMOGIRI)]
            .set(
                Hand.is_tenpai(Hand.sub(self.hand[self.turn], self.last_draw))
            ),
            lambda arr: arr,
            legal_actions,
        )

        # ツモ切り, ツモ
        legal_actions = jax.lax.cond(
            (self.last_draw == -1) | self.riichi_declared,
            lambda: legal_actions,
            lambda: legal_actions.at[(self.turn, Action.TSUMOGIRI)]
            .set(True)
            .at[(self.turn, Action.TSUMO)]
            .set(Hand.can_tsumo(self.hand[self.turn])),
        )

        # 手出し, 鳴いた後の手出し
        legal_actions = jax.lax.cond(
            (self.target != -1)
            | self.riichi_declared
            | self.riichi[self.turn],
            lambda arr: arr,
            lambda arr: jax.lax.fori_loop(
                0,
                34,
                lambda i, arr: arr.at[(self.turn, i)].set(
                    self.hand[self.turn][i] > (i == self.last_draw),
                ),
                arr,
            ),
            legal_actions,
        )

        for player in range(4):
            # ロン
            legal_actions = jax.lax.cond(
                (self.target == -1) | (player == self.turn),
                lambda: legal_actions,
                lambda: legal_actions.at[(player, Action.RON)].set(
                    Hand.can_ron(self.hand[player], self.target)
                ),
            )
            # ポン
            legal_actions = jax.lax.cond(
                (self.target == -1)
                | (player == self.turn)
                | self.deck.is_empty()
                | self.riichi[player],
                lambda: legal_actions,
                lambda: legal_actions.at[(player, Action.PON)].set(
                    Hand.can_pon(self.hand[player], self.target)
                ),
            )
            # チー
            legal_actions = jax.lax.cond(
                (self.target == -1)
                | (player != (self.turn + 1) % 4)
                | self.deck.is_empty()
                | self.riichi[player],
                lambda: legal_actions,
                lambda: legal_actions.at[(player, Action.CHI_R)]
                .set(
                    Hand.can_chi(self.hand[player], self.target, Action.CHI_R)
                )
                .at[(player, Action.CHI_M)]
                .set(
                    Hand.can_chi(self.hand[player], self.target, Action.CHI_M)
                )
                .at[(player, Action.CHI_L)]
                .set(
                    Hand.can_chi(self.hand[player], self.target, Action.CHI_L)
                ),
            )
            legal_actions = legal_actions.at[(player, Action.PASS)].set(
                (player != self.turn) & jnp.any(legal_actions[player])
            )

        return legal_actions

    def observe(self, player: int) -> Observation:
        return Observation(self.hand[player], self.target, self.last_draw)

    @staticmethod
    @jit
    def init(key) -> State:
        deck = Deck.init(key)
        hand = jnp.zeros((4, 34), dtype=jnp.uint8)

        for i in range(4):
            for _ in range(13):
                deck, tile = Deck.draw(deck)
                hand = hand.at[i].set(Hand.add(hand[i], tile))

        deck, tile = Deck.draw(deck)
        hand = hand.at[0].set(Hand.add(hand[0], tile))

        turn = 0
        target = -1
        last_draw = tile
        riichi_declared = False
        riichi = jnp.full(4, False)
        meld_num = jnp.zeros(4, dtype=jnp.uint8)
        melds = jnp.zeros((4, 4), dtype=jnp.uint32)
        return State(
            deck,
            hand,
            turn,
            target,
            last_draw,
            riichi_declared,
            riichi,
            meld_num,
            melds,
        )

    @staticmethod
    @jit
    def step(
        state: State, actions: jnp.ndarray
    ) -> tuple[State, jnp.ndarray, bool]:
        player = jnp.argmin(actions)
        return State._step(state, player, actions[player])

    @staticmethod
    @jit
    def _step(
        state: State, player: int, action: int
    ) -> tuple[State, jnp.ndarray, bool]:
        return jax.lax.cond(
            action < 34,
            lambda: State._discard(state, action),
            lambda: jax.lax.switch(
                action - 34,
                [
                    lambda: State._tsumogiri(state),
                    lambda: State._riichi(state),
                    lambda: State._tsumo(state),
                    lambda: State._ron(state, player),
                    lambda: State._pon(state, player),
                    lambda: State._chi(state, player, Action.CHI_R),
                    lambda: State._chi(state, player, Action.CHI_M),
                    lambda: State._chi(state, player, Action.CHI_L),
                    lambda: State._try_draw(state),
                ],
            ),
        )

    @staticmethod
    @jit
    def _tsumogiri(state: State) -> tuple[State, jnp.ndarray, bool]:
        return State._discard(state, state.last_draw)

    @staticmethod
    @jit
    def _discard(state: State, tile: int) -> tuple[State, jnp.ndarray, bool]:
        state.hand = state.hand.at[state.turn].set(
            Hand.sub(state.hand[state.turn], tile)
        )
        state.target = tile
        state.last_draw = -1
        return jax.lax.cond(
            jnp.any(state.legal_actions()),
            lambda: (state, jnp.full(4, 0), False),
            lambda: State._try_draw(state),
        )

    @staticmethod
    @jit
    def _try_draw(state: State) -> tuple[State, jnp.ndarray, bool]:
        state.target = -1
        return jax.lax.cond(
            state.deck.is_empty(),
            lambda: State._ryukyoku(state),
            lambda: State._draw(state),
        )

    @staticmethod
    @jit
    def _accept_riichi(state: State) -> State:
        state.riichi = state.riichi.at[state.turn].set(
            state.riichi[state.turn] | state.riichi_declared
        )
        state.riichi_declared = False
        return state

    @staticmethod
    @jit
    def _draw(state: State) -> tuple[State, jnp.ndarray, bool]:
        state = State._accept_riichi(state)
        state.turn += 1
        state.turn %= 4
        state.deck, tile = Deck.draw(state.deck)
        state.last_draw = tile
        state.hand = state.hand.at[state.turn].set(
            Hand.add(state.hand[state.turn], tile)
        )
        return state, jnp.full(4, 0), False

    @staticmethod
    @jit
    def _ryukyoku(state: State) -> tuple[State, jnp.ndarray, bool]:
        reward = jnp.array(
            [2 * Hand.is_tenpai(state.hand[i]) - 1 for i in range(4)]
        )
        return state, reward, True

    @staticmethod
    @jit
    def _ron(state: State, player: int) -> tuple[State, jnp.ndarray, bool]:
        return (
            state,
            jnp.full(4, 0).at[state.turn].set(-2).at[player].set(2),
            True,
        )

    @staticmethod
    @jit
    def _append_meld(state: State, meld: int, player: int) -> State:
        state.melds = state.melds.at[(player, state.meld_num[player])].set(
            meld
        )
        state.meld_num = state.meld_num.at[player].set(
            state.meld_num[player] + 1
        )
        return state

    @staticmethod
    @jit
    def _pon(state: State, player: int) -> tuple[State, jnp.ndarray, bool]:
        state = State._accept_riichi(state)
        meld = Meld.init(Action.PON, state.target, state.turn)
        state = State._append_meld(state, meld, player)
        state.hand = state.hand.at[player].set(
            Hand.pon(state.hand[player], state.target)
        )
        state.target = -1
        state.turn = player
        return state, jnp.full(4, 0), False

    @staticmethod
    @jit
    def _chi(
        state: State, player: int, action: int
    ) -> tuple[State, jnp.ndarray, bool]:
        state = State._accept_riichi(state)
        meld = Meld.init(action, state.target, state.turn)
        state = State._append_meld(state, meld, player)
        state.hand = state.hand.at[player].set(
            Hand.chi(state.hand[player], state.target, action)
        )
        state.target = -1
        state.turn = player
        return state, jnp.full(4, 0), False

    @staticmethod
    @jit
    def _tsumo(state: State) -> tuple[State, jnp.ndarray, bool]:
        return state, jnp.full(4, -2).at[state.turn].set(2), True

    @staticmethod
    @jit
    def _riichi(state: State) -> tuple[State, jnp.ndarray, bool]:
        state.riichi_declared = True
        return state, jnp.full(4, 0), False

    def _tree_flatten(self):
        children = (
            self.deck,
            self.hand,
            self.turn,
            self.target,
            self.last_draw,
            self.riichi_declared,
            self.riichi,
            self.meld_num,
            self.melds,
        )
        aux_data = {}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)


tree_util.register_pytree_node(
    State, State._tree_flatten, State._tree_unflatten
)


@jit
def init(key) -> State:
    return State.init(key)


@jit
def step(
    state: State, actions: jnp.ndarray
) -> tuple[State, jnp.ndarray, bool]:
    return State.step(state, actions)
