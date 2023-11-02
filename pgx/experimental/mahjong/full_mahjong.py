from __future__ import annotations

import json
import os
import re
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
    def from_str(s: str) -> int:
        return (int(s[0]) - 1) + 9 * ["m", "p", "s", "z"].index(s[1])

    @staticmethod
    @jit
    def is_outside(tile: int) -> bool:
        num = tile % 9
        return (tile >= 27) | (num == 0) | (num == 8)


class Action:
    # 手出し: 0~33
    # 暗/加槓: 34~67
    TSUMOGIRI = 68
    RIICHI = 69
    TSUMO = 70

    RON = 71
    PON = 72
    MINKAN = 73
    CHI_L = 74  # [4]56
    CHI_M = 75  # 4[5]6
    CHI_R = 76  # 45[6]
    PASS = 77

    NONE = 78

    @staticmethod
    @jit
    def is_selfkan(action: int) -> bool:
        return (34 <= action) & (action < 68)


class CacheLoader:
    DIR = os.path.join(os.path.dirname(__file__), "cache")

    @staticmethod
    @jit
    def load_hand_cache():
        with open(os.path.join(CacheLoader.DIR, "hand_cache.json")) as f:
            return jnp.array(json.load(f), dtype=jnp.uint32)

    @staticmethod
    @jit
    def load_yaku_cache():
        with open(os.path.join(CacheLoader.DIR, "yaku_cache.json")) as f:
            return jnp.array(json.load(f), dtype=jnp.int32)

    @staticmethod
    @jit
    def load_shanten_cache():
        with open(os.path.join(CacheLoader.DIR, "shanten_cache.json")) as f:
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
        return Hand.can_tsumo(Hand.add(hand, tile))

    @staticmethod
    @jit
    def can_riichi(hand: jnp.ndarray) -> bool:
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
    def can_tsumo(hand: jnp.ndarray):
        thirteen_orphan = (
            (hand[0] > 0)
            & (hand[8] > 0)
            & (hand[9] > 0)
            & (hand[17] > 0)
            & (hand[18] > 0)
            & jnp.all(hand[26:] > 0)
            & (
                (
                    hand[0]
                    + hand[8]
                    + hand[9]
                    + hand[17]
                    + hand[18]
                    + jnp.sum(hand[26:])
                )
                == 14
            )
        )
        seven_pairs = jnp.sum(hand == 2) == 7

        heads, valid = jnp.int32(0), jnp.int32(1)
        for suit in range(3):
            valid &= Hand.cache(
                jax.lax.fori_loop(
                    9 * suit,
                    9 * (suit + 1),
                    lambda i, code: code * 5 + hand[i].astype(int),
                    0,
                )
            )
            heads += jnp.sum(hand[9 * suit : 9 * (suit + 1)]) % 3 == 2

        heads, valid = jax.lax.fori_loop(
            27,
            34,
            lambda i, tpl: (
                tpl[0] + (hand[i] == 2),
                tpl[1] & (hand[i] != 1) & (hand[i] != 4),
            ),
            (heads, valid),
        )

        return ((valid & (heads == 1)) | thirteen_orphan | seven_pairs) == 1

    @staticmethod
    @jit
    def can_pon(hand: jnp.ndarray, tile: int) -> bool:
        return hand[tile] >= 2  # type: ignore

    @staticmethod
    @jit
    def can_minkan(hand: jnp.ndarray, tile: int) -> bool:
        return hand[tile] == 3  # type: ignore

    @staticmethod
    @jit
    def can_kakan(hand: jnp.ndarray, tile: int) -> bool:
        return hand[tile] == 1  # type: ignore

    @staticmethod
    @jit
    def can_ankan(hand: jnp.ndarray, tile: int) -> bool:
        return hand[tile] == 4  # type: ignore

    @staticmethod
    @jit
    def can_chi(hand: jnp.ndarray, tile: int, action: int) -> bool:
        return jax.lax.cond(
            (tile >= 27) | (action < Action.CHI_L) | (Action.CHI_R < action),
            lambda: False,
            lambda: jax.lax.switch(
                action - Action.CHI_L,
                [
                    lambda: jax.lax.cond(
                        tile % 9 < 7,
                        lambda: (hand[tile + 1] > 0) & (hand[tile + 2] > 0),
                        lambda: False,
                    ),
                    lambda: jax.lax.cond(
                        (tile % 9 < 8) & (tile % 9 > 0),
                        lambda: (hand[tile - 1] > 0) & (hand[tile + 1] > 0),
                        lambda: False,
                    ),
                    lambda: jax.lax.cond(
                        tile % 9 > 1,
                        lambda: (hand[tile - 2] > 0) & (hand[tile - 1] > 0),
                        lambda: False,
                    ),
                ],
            ),
        )

    @staticmethod
    @jit
    def add(hand: jnp.ndarray, tile: int, x: int = 1) -> jnp.ndarray:
        return hand.at[tile].set(hand[tile] + x)

    @staticmethod
    @jit
    def sub(hand: jnp.ndarray, tile: int, x: int = 1) -> jnp.ndarray:
        return Hand.add(hand, tile, -x)

    @staticmethod
    @jit
    def pon(hand: jnp.ndarray, tile: int) -> jnp.ndarray:
        return Hand.sub(hand, tile, 2)

    @staticmethod
    @jit
    def minkan(hand: jnp.ndarray, tile: int) -> jnp.ndarray:
        return Hand.sub(hand, tile, 3)

    @staticmethod
    @jit
    def kakan(hand: jnp.ndarray, tile: int) -> jnp.ndarray:
        return Hand.sub(hand, tile)

    @staticmethod
    @jit
    def ankan(hand: jnp.ndarray, tile: int) -> jnp.ndarray:
        return Hand.sub(hand, tile, 4)

    @staticmethod
    @jit
    def chi(hand: jnp.ndarray, tile: int, action: int) -> jnp.ndarray:
        return jax.lax.switch(
            action - Action.CHI_L,
            [
                lambda: Hand.sub(Hand.sub(hand, tile + 1), tile + 2),
                lambda: Hand.sub(Hand.sub(hand, tile - 1), tile + 1),
                lambda: Hand.sub(Hand.sub(hand, tile - 2), tile - 1),
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
        hand = jnp.zeros(34, dtype=jnp.uint32)
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
class Deck:
    arr: jnp.ndarray
    idx: int = 135
    end: int = 13
    doras: int = 1

    # 0   2 | 4 - 12 | 14 - 134
    # 1   3 | 5 - 13 | 15 - 135
    # -------------------------
    # 嶺上牌| ドラ   | to be used
    #       | 裏ドラ |

    @jit
    def is_empty(self) -> bool:
        return self.size() == 0

    @jit
    def size(self) -> int:
        return self.idx - self.end

    @staticmethod
    @jit
    def init(key: jnp.ndarray) -> Deck:
        return Deck(jax.random.permutation(key, jnp.arange(136) // 4))

    @staticmethod
    @jit
    def deal(deck: Deck) -> tuple[Deck, jnp.ndarray, int]:
        hand = jnp.zeros((4, 34), dtype=jnp.uint32)
        for i in range(3):
            for j in range(4):
                hand = hand.at[j].set(
                    jax.lax.fori_loop(
                        0,
                        4,
                        lambda k, h: Hand.add(
                            h, deck.arr[-(16 * i + 4 * j + k + 1)]
                        ),
                        hand[j],
                    )
                )
        for j in range(4):
            hand = hand.at[j].set(
                Hand.add(hand[j], deck.arr[-(16 * 3 + j + 1)])
            )

        last_draw = deck.arr[-(16 * 3 + 4 + 1)].astype(int)
        hand = hand.at[0].set(Hand.add(hand[0], last_draw))

        deck.idx -= 53

        return deck, hand, last_draw  # type: ignore

    @staticmethod
    @jit
    def draw(deck: Deck, is_kan: bool = False) -> tuple[Deck, int]:
        tile = deck.arr[
            deck.idx * (is_kan is False) | (deck.doras - 1) * is_kan
        ]
        deck.idx -= is_kan is False
        deck.end += is_kan
        deck.doras += is_kan  # NOTE: 先めくりで統一

        return deck, tile  # type: ignore

    def _tree_flatten(self):
        children = (self.arr, self.idx, self.end, self.doras)
        aux_data = {}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)


tree_util.register_pytree_node(Deck, Deck._tree_flatten, Deck._tree_unflatten)


@dataclass
class Observation:
    hand: jnp.ndarray
    target: int
    last_draw: int
    riichi: jnp.ndarray


class Meld:
    @staticmethod
    @jit
    def init(action: int, target: int, src: int) -> int:
        # src: 相対位置
        # - 0: 自分(暗槓の場合のみ)
        # - 1: 下家
        # - 2: 対面
        # - 3: 上家
        return (src << 13) | (target << 7) | action

    @staticmethod
    def to_str(meld: int) -> str:
        action = Meld.action(meld)
        target = Meld.target(meld)
        src = Meld.src(meld)
        suit, num = target // 9, target % 9 + 1

        if action == Action.PON:
            if src == 1:
                return "{}{}[{}]{}".format(
                    num, num, num, ["m", "p", "s", "z"][suit]
                )
            elif src == 2:
                return "{}[{}]{}{}".format(
                    num, num, num, ["m", "p", "s", "z"][suit]
                )
            elif src == 3:
                return "[{}]{}{}{}".format(
                    num, num, num, ["m", "p", "s", "z"][suit]
                )
        elif Action.is_selfkan(action):
            if src == 0:
                return "{}{}{}{}{}".format(
                    num, num, num, num, ["m", "p", "s", "z"][suit]
                )
            if src == 1:
                return "{}{}[{}{}]{}".format(
                    num, num, num, num, ["m", "p", "s", "z"][suit]
                )
            elif src == 2:
                return "{}[{}{}]{}{}".format(
                    num, num, num, num, ["m", "p", "s", "z"][suit]
                )
            elif src == 3:
                return "[{}{}]{}{}{}".format(
                    num, num, num, num, ["m", "p", "s", "z"][suit]
                )
        elif action == Action.MINKAN:
            if src == 1:
                return "{}{}{}[{}]{}".format(
                    num, num, num, num, ["m", "p", "s", "z"][suit]
                )
            elif src == 2:
                return "{}[{}]{}{}{}".format(
                    num, num, num, num, ["m", "p", "s", "z"][suit]
                )
            elif src == 3:
                return "[{}]{}{}{}{}".format(
                    num, num, num, num, ["m", "p", "s", "z"][suit]
                )
        elif Action.CHI_L <= action <= Action.CHI_R:
            assert src == 3
            pos = action - Action.CHI_L
            t = [num - pos + i for i in range(3)]
            t.insert(0, t.pop(pos))
            return "[{}]{}{}{}".format(*t, ["m", "p", "s", "z"][suit])
        assert False

    @staticmethod
    def from_str(s: str) -> int:
        l3 = re.match(r"^\[(\d)\](\d)(\d)([mpsz])$", s)
        m3 = re.match(r"^(\d)\[(\d)\](\d)([mpsz])$", s)
        r3 = re.match(r"^(\d)(\d)\[(\d)\]([mpsz])$", s)
        l4 = re.match(r"^\[(\d)\](\d)(\d)(\d)([mpsz])$", s)
        m4 = re.match(r"^(\d)\[(\d)\](\d)(\d)([mpsz])$", s)
        r4 = re.match(r"^(\d)(\d)(\d)\[(\d)\]([mpsz])$", s)
        ll4 = re.match(r"^\[(\d)](\d)\](\d)(\d)([mpsz])$", s)
        mm4 = re.match(r"^(\d)\[(\d)(\d)\](\d)([mpsz])$", s)
        rr4 = re.match(r"^(\d)(\d)\[(\d)(\d)\]([mpsz])$", s)
        ankan = re.match(r"^(\d)(\d)(\d)(\d)([mpsz])$", s)
        if l3:
            num = list(map(int, [l3[1], l3[2], l3[3]]))
            target = (num[0] - 1) + 9 * ["m", "p", "s", "z"].index(l3[4])
            src = 3
            if num[0] == num[1] and num[1] == num[2]:
                return Meld.init(Action.PON, target, src)
            if num[0] + 1 == num[1] and num[1] + 1 == num[2]:
                return Meld.init(Action.CHI_L, target, src)
            if num[0] == num[1] + 1 and num[1] + 2 == num[2]:
                return Meld.init(Action.CHI_M, target, src)
            if num[0] == num[1] + 2 and num[1] + 1 == num[2]:
                return Meld.init(Action.CHI_R, target, src)
        if m3:
            assert m3[1] == m3[2] and m3[2] == m3[3]
            target = (int(m3[1]) - 1) + 9 * ["m", "p", "s", "z"].index(m3[4])
            return Meld.init(Action.PON, target, src=2)
        if r3:
            assert r3[1] == r3[2] and r3[2] == r3[3]
            target = (int(r3[1]) - 1) + 9 * ["m", "p", "s", "z"].index(r3[4])
            return Meld.init(Action.PON, target, src=1)
        if l4:
            assert l4[1] == l4[2] and l4[2] == l4[3] and l4[4] == l4[4]
            target = (int(l4[1]) - 1) + 9 * ["m", "p", "s", "z"].index(l4[5])
            return Meld.init(Action.MINKAN, target, src=3)
        if m4:
            assert m4[1] == m4[2] and m4[2] == m4[3] and m4[4] == m4[4]
            target = (int(m4[1]) - 1) + 9 * ["m", "p", "s", "z"].index(m4[5])
            return Meld.init(Action.MINKAN, target, src=2)
        if r4:
            assert r4[1] == r4[2] and r4[2] == r4[3] and r4[4] == r4[4]
            target = (int(r4[1]) - 1) + 9 * ["m", "p", "s", "z"].index(r4[5])
            return Meld.init(Action.MINKAN, target, src=1)
        if ll4:
            target = (int(ll4[1]) - 1) + 9 * ["m", "p", "s", "z"].index(ll4[5])
            assert ll4[1] == ll4[2] and ll4[2] == ll4[3] and ll4[4] == ll4[4]
            return Meld.init(target + 34, target, src=3)
        if mm4:
            target = (int(mm4[1]) - 1) + 9 * ["m", "p", "s", "z"].index(mm4[5])
            assert mm4[1] == mm4[2] and mm4[2] == mm4[3] and mm4[4] == mm4[4]
            return Meld.init(target + 34, target, src=2)
        if rr4:
            target = (int(rr4[1]) - 1) + 9 * ["m", "p", "s", "z"].index(rr4[5])
            assert rr4[1] == rr4[2] and rr4[2] == rr4[3] and rr4[4] == rr4[4]
            return Meld.init(target + 34, target, src=1)
        if ankan:
            target = int(ankan[1]) - 1
            target = (int(ankan[1]) - 1) + 9 * ["m", "p", "s", "z"].index(
                ankan[5]
            )
            assert (
                ankan[1] == ankan[2]
                and ankan[2] == ankan[3]
                and ankan[4] == ankan[4]
            )
            return Meld.init(target + 34, target, src=0)

        assert False

    @staticmethod
    @jit
    def src(meld: int) -> int:
        return meld >> 13 & 0b11

    @staticmethod
    @jit
    def target(meld: int) -> int:
        return (meld >> 7) & 0b111111

    @staticmethod
    @jit
    def action(meld: int) -> int:
        return meld & 0b1111111

    @staticmethod
    @jit
    def suited_pung(meld: int) -> int:
        action = Meld.action(meld)
        target = Meld.target(meld)
        is_pung = (
            (action == Action.PON)
            | (action == Action.MINKAN)
            | Action.is_selfkan(action)
        )
        is_suited_pon = is_pung & (target < 27)

        return is_suited_pon << target

    @staticmethod
    @jit
    def chow(meld: int) -> int:
        action = Meld.action(meld)
        is_chi = (Action.CHI_L <= action) & (action <= Action.CHI_R)
        pos = Meld.target(meld) - (
            action - Action.CHI_L
        )  # WARNING: may be negative

        pos *= is_chi
        return is_chi << pos

    @staticmethod
    @jit
    def is_outside(meld: int) -> int:
        action = Meld.action(meld)
        target = Meld.target(meld)
        is_chi = (Action.CHI_L <= action) & (action <= Action.CHI_R)

        return jax.lax.cond(
            is_chi,
            lambda: Tile.is_outside(target - (action - Action.CHI_L))
            | Tile.is_outside(target - (action - Action.CHI_L) + 2),
            lambda: Tile.is_outside(target),
        )

    @staticmethod
    @jit
    def fu(meld: int) -> int:
        action = Meld.action(meld)

        fu = (
            (action == Action.PON) * 2
            + (action == Action.MINKAN) * 8
            + (Action.is_selfkan(action) * 8 * (1 + (Meld.src(meld) == 0)))
        )

        return fu * (1 + (Tile.is_outside(Meld.target(meld))))


@dataclass
class State:
    deck: Deck
    hand: jnp.ndarray
    turn: int  # 手牌が3n+2枚, もしくは直前に牌を捨てたplayer
    target: int  # 直前に捨てられてron,pon,chi の対象になっている牌. 存在しなければ-1
    last_draw: int  # 手牌が3n+2枚のplayerが直前に引いた牌. 存在しなければ-1
    riichi_declared: bool  # state.turn がリーチ宣言してから, その直後の打牌が通るまでTrue
    riichi: jnp.ndarray  # 各playerのリーチが成立しているかどうか
    n_meld: jnp.ndarray  # 各playerの副露回数
    melds: jnp.ndarray
    # melds[i][j]: player i のj回目の副露(j=1,2,3,4). 存在しなければ0

    # 以下計算効率のために保持
    is_menzen: jnp.ndarray
    pon: jnp.ndarray
    # pon[i][j]: player i がjをポンを所有している場合, src << 2 | index. or 0

    @jit
    def can_kakan(self, tile: int) -> bool:
        return (self.pon[(self.turn, tile)] > 0) & Hand.can_kakan(
            self.hand[self.turn], tile
        )

    @jit
    def can_tsumo(self) -> bool:
        return jax.lax.cond(
            Hand.can_tsumo(self.hand[self.turn]),
            lambda: jnp.any(
                Yaku.judge(
                    self.hand[self.turn],
                    self.melds[self.turn],
                    self.n_meld[self.turn],
                    self.last_draw,
                    self.riichi[self.turn],
                    False,
                )[0]
            ),
            lambda: False,
        )

    @jit
    def can_ron(self, player: int) -> bool:
        return jax.lax.cond(
            Hand.can_ron(self.hand[player], self.target),
            lambda: jnp.any(
                Yaku.judge(
                    Hand.add(self.hand[player], self.target),
                    self.melds[player],
                    self.n_meld[player],
                    self.last_draw,
                    self.riichi[player],
                    True,
                )[0]
            ),
            lambda: False,
        )

    @jit
    def legal_actions(self) -> jnp.ndarray:
        legal_actions = jnp.full((4, Action.NONE), False)

        # 暗/加槓, ツモ切り, ツモ, リーチ
        legal_actions = jax.lax.cond(
            (self.last_draw == -1) | self.riichi_declared,
            lambda: legal_actions,
            lambda: jax.lax.fori_loop(
                0,
                34,
                lambda tile, arr: arr.at[(self.turn, tile + 34)].set(
                    (
                        Hand.can_ankan(self.hand[self.turn], tile)
                        | self.can_kakan(tile)
                    )
                    & (self.deck.doras < 5)  # 5回目の槓はできない
                    & (self.deck.is_empty() == 0)
                ),
                legal_actions,
            )
            .at[(self.turn, Action.TSUMOGIRI)]
            .set(True)
            .at[(self.turn, Action.TSUMO)]
            .set(self.can_tsumo())
            .at[(self.turn, Action.RIICHI)]
            .set(
                jax.lax.cond(
                    self.riichi[self.turn]
                    | (self.is_menzen[self.turn] == 0)
                    | (self.deck.size() < 4),
                    lambda: False,
                    lambda: Hand.can_riichi(self.hand[self.turn]),
                )
            ),
        )

        # リーチ宣言直後の打牌
        legal_actions = jax.lax.cond(
            (self.last_draw == -1) | (self.riichi_declared == 0),
            lambda: legal_actions,
            lambda: jax.lax.fori_loop(
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
                legal_actions,
            )
            .at[(self.turn, Action.TSUMOGIRI)]
            .set(
                Hand.is_tenpai(Hand.sub(self.hand[self.turn], self.last_draw))
            ),
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

        # 他家の行動
        legal_actions = jax.lax.cond(
            self.target == -1,
            lambda: legal_actions,
            lambda: jax.lax.fori_loop(
                0,
                4,
                lambda player, legal_actions: jax.lax.cond(
                    (player == self.turn) | (self.target == -1),
                    lambda: legal_actions,
                    lambda: jax.lax.cond(
                        self.deck.is_empty() | self.riichi[player],
                        lambda legal_actions: legal_actions,
                        lambda legal_actions: jax.lax.cond(
                            (player - self.turn) % 4 != 1,
                            lambda legal_actions: legal_actions,
                            lambda legal_actions: legal_actions.at[
                                (player, Action.CHI_L)
                            ]
                            .set(
                                Hand.can_chi(
                                    self.hand[player],
                                    self.target,
                                    Action.CHI_L,
                                )
                            )
                            .at[(player, Action.CHI_M)]
                            .set(
                                Hand.can_chi(
                                    self.hand[player],
                                    self.target,
                                    Action.CHI_M,
                                )
                            )
                            .at[(player, Action.CHI_R)]
                            .set(
                                Hand.can_chi(
                                    self.hand[player],
                                    self.target,
                                    Action.CHI_R,
                                )
                            ),
                            legal_actions.at[(player, Action.PON)]
                            .set(Hand.can_pon(self.hand[player], self.target))
                            .at[(player, Action.MINKAN)]
                            .set(
                                Hand.can_minkan(self.hand[player], self.target)
                                & (self.deck.doras < 5)
                            ),
                        ),
                        legal_actions.at[(player, Action.RON)].set(
                            self.can_ron(player)
                        ),
                    ),
                ),
                legal_actions,
            ),
        )

        legal_actions = jax.lax.fori_loop(
            0,
            4,
            lambda player, legal_actions: legal_actions.at[
                (player, Action.PASS)
            ].set((player != self.turn) & jnp.any(legal_actions[player])),
            legal_actions,
        )

        return legal_actions

    def observe(self, player: int) -> Observation:
        return Observation(
            self.hand[player],
            self.target,
            self.last_draw,
            self.riichi,
        )

    @staticmethod
    @jit
    def init(key: jnp.ndarray) -> State:
        return State.init_with_deck(Deck.init(key))

    @staticmethod
    @jit
    def init_with_deck_arr(arr: jnp.ndarray) -> State:
        return State.init_with_deck(Deck(arr))

    @staticmethod
    @jit
    def init_with_deck(deck: Deck) -> State:
        deck, hand, last_draw = Deck.deal(deck)

        turn = 0
        target = -1
        riichi_declared = False
        riichi = jnp.full(4, False)
        n_meld = jnp.zeros(4, dtype=jnp.int32)
        melds = jnp.zeros((4, 4), dtype=jnp.int32)
        is_menzen = jnp.full(4, True)
        pon = jnp.zeros((4, 34), dtype=jnp.int32)
        return State(
            deck,
            hand,
            turn,
            target,
            last_draw,
            riichi_declared,
            riichi,
            n_meld,
            melds,
            is_menzen,
            pon,
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
            lambda: jax.lax.cond(
                action < 68,
                lambda: State._selfkan(state, action),
                lambda: jax.lax.switch(
                    action - 68,
                    [
                        lambda: State._tsumogiri(state),
                        lambda: State._riichi(state),
                        lambda: State._tsumo(state),
                        lambda: State._ron(state, player),
                        lambda: State._pon(state, player),
                        lambda: State._minkan(state, player),
                        lambda: State._chi(state, player, Action.CHI_L),
                        lambda: State._chi(state, player, Action.CHI_M),
                        lambda: State._chi(state, player, Action.CHI_R),
                        lambda: State._try_draw(state),
                    ],
                ),
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
            lambda: (state, jnp.zeros(4, dtype=jnp.int32), False),
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
        return state, jnp.zeros(4, dtype=jnp.int32), False

    @staticmethod
    @jit
    def _ryukyoku(state: State) -> tuple[State, jnp.ndarray, bool]:
        is_tenpai = jax.lax.map(
            lambda i: Hand.is_tenpai(state.hand[i]), jnp.arange(4)
        )
        tenpais = jnp.sum(is_tenpai)
        plus, minus = jax.lax.switch(
            tenpais,
            [
                lambda: (0, 0),
                lambda: (3000, 1000),
                lambda: (1500, 1500),
                lambda: (1000, 3000),
                lambda: (0, 0),
            ],
        )
        reward = jax.lax.map(
            lambda i: jax.lax.cond(
                is_tenpai[i],
                lambda: plus,
                lambda: -minus,
            ),
            jnp.arange(4),
        )

        # 供託
        reward -= 1000 * state.riichi
        top = jnp.argmax(reward)
        reward = reward.at[top].set(reward[top] + 1000 * jnp.sum(state.riichi))

        return state, reward, True

    @staticmethod
    @jit
    def _ron(state: State, player: int) -> tuple[State, jnp.ndarray, bool]:
        score = Yaku.score(
            state.hand[player],
            state.melds[player],
            state.n_meld[player],
            state.target,
            state.riichi[player],
            is_ron=True,
        )
        score = jax.lax.cond(
            player == 0,
            lambda: score * 6,
            lambda: score * 4,
        )
        score += -score % 100
        reward = (
            jnp.zeros(4, dtype=jnp.int32)
            .at[player]
            .set(score)
            .at[state.turn]
            .set(-score)
        )

        # 供託
        reward -= 1000 * state.riichi
        reward = reward.at[player].set(
            reward[player] + 1000 * jnp.sum(state.riichi)
        )

        return state, reward, True

    @staticmethod
    @jit
    def _append_meld(state: State, meld: int, player: int) -> State:
        state.melds = state.melds.at[(player, state.n_meld[player])].set(meld)
        state.n_meld = state.n_meld.at[player].set(state.n_meld[player] + 1)
        return state

    @staticmethod
    @jit
    def _minkan(state: State, player: int) -> tuple[State, jnp.ndarray, bool]:
        state = State._accept_riichi(state)
        meld = Meld.init(
            Action.MINKAN, state.target, (state.turn - player) % 4
        )
        state = State._append_meld(state, meld, player)
        state.hand = state.hand.at[player].set(
            Hand.minkan(state.hand[player], state.target)
        )
        state.target = -1
        state.turn = player
        state.is_menzen = state.is_menzen.at[player].set(False)

        # 嶺上牌
        state.deck, tile = Deck.draw(state.deck, is_kan=True)
        state.last_draw = tile
        state.hand = state.hand.at[state.turn].set(
            Hand.add(state.hand[state.turn], tile)
        )
        return state, jnp.zeros(4, dtype=jnp.int32), False

    @staticmethod
    @jit
    def _selfkan(state: State, action: int) -> tuple[State, jnp.ndarray, bool]:
        target = action - 34
        pon = state.pon[(state.turn, target)]
        return jax.lax.cond(
            pon == 0,
            lambda: State._ankan(state, target),
            lambda: State._kakan(state, target, pon >> 2, pon & 0b11),
        )

    @staticmethod
    @jit
    def _ankan(state: State, target: int) -> tuple[State, jnp.ndarray, bool]:
        meld = Meld.init(target + 34, target, src=0)
        state = State._append_meld(state, meld, state.turn)
        state.hand = state.hand.at[state.turn].set(
            Hand.ankan(state.hand[state.turn], target)
        )
        # TODO: 国士無双ロンの受付

        # 嶺上牌
        state.deck, tile = Deck.draw(state.deck, is_kan=True)
        state.last_draw = tile
        state.hand = state.hand.at[state.turn].set(
            Hand.add(state.hand[state.turn], tile)
        )
        return state, jnp.zeros(4, dtype=jnp.int32), False

    @staticmethod
    @jit
    def _kakan(
        state: State, target: int, pon_src: int, pon_idx: int
    ) -> tuple[State, jnp.ndarray, bool]:
        state.melds = state.melds.at[(state.turn, pon_idx)].set(
            Meld.init(target + 34, target, pon_src)
        )
        state.hand = state.hand.at[state.turn].set(
            Hand.kakan(state.hand[state.turn], target)
        )
        state.pon = state.pon.at[(state.turn, target)].set(0)
        # TODO: 槍槓の受付

        # 嶺上牌
        state.deck, tile = Deck.draw(state.deck, is_kan=True)
        state.last_draw = tile
        state.hand = state.hand.at[state.turn].set(
            Hand.add(state.hand[state.turn], tile)
        )
        return state, jnp.zeros(4, dtype=jnp.int32), False

    @staticmethod
    @jit
    def _pon(state: State, player: int) -> tuple[State, jnp.ndarray, bool]:
        state = State._accept_riichi(state)
        src = (state.turn - player) % 4
        meld = Meld.init(Action.PON, state.target, src)
        state = State._append_meld(state, meld, player)
        state.hand = state.hand.at[player].set(
            Hand.pon(state.hand[player], state.target)
        )
        state.is_menzen = state.is_menzen.at[player].set(False)
        state.pon = state.pon.at[(player, state.target)].set(
            src << 2 | state.n_meld[player] - 1
        )
        state.target = -1
        state.turn = player
        return state, jnp.zeros(4, dtype=jnp.int32), False

    @staticmethod
    @jit
    def _chi(
        state: State, player: int, action: int
    ) -> tuple[State, jnp.ndarray, bool]:
        state = State._accept_riichi(state)
        meld = Meld.init(action, state.target, src=3)
        state = State._append_meld(state, meld, player)
        state.hand = state.hand.at[player].set(
            Hand.chi(state.hand[player], state.target, action)
        )
        state.is_menzen = state.is_menzen.at[player].set(False)
        state.target = -1
        state.turn = player
        return state, jnp.zeros(4, dtype=jnp.int32), False

    @staticmethod
    @jit
    def _tsumo(state: State) -> tuple[State, jnp.ndarray, bool]:
        score = Yaku.score(
            state.hand[state.turn],
            state.melds[state.turn],
            state.n_meld[state.turn],
            state.target,
            state.riichi[state.turn],
            is_ron=False,
        )
        s1 = score + (-score) % 100
        s2 = (score * 2) + (-(score * 2)) % 100

        reward = jax.lax.cond(
            state.turn == 0,
            lambda: jnp.full(4, -s2, dtype=jnp.int32)
            .at[state.turn]
            .set(s2 * 3),
            lambda: jnp.full(4, -s1, dtype=jnp.int32)
            .at[0]
            .set(-s2)
            .at[state.turn]
            .set(s1 * 2 + s2),
        )

        # 供託
        reward -= 1000 * state.riichi
        reward = reward.at[state.turn].set(
            reward[state.turn] + 1000 * jnp.sum(state.riichi)
        )

        return state, reward, True

    @staticmethod
    @jit
    def _riichi(state: State) -> tuple[State, jnp.ndarray, bool]:
        state.riichi_declared = True
        return state, jnp.zeros(4, dtype=jnp.int32), False

    def _tree_flatten(self):
        children = (
            self.deck,
            self.hand,
            self.turn,
            self.target,
            self.last_draw,
            self.riichi_declared,
            self.riichi,
            self.n_meld,
            self.melds,
            self.is_menzen,
            self.pon,
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


def step(
    state: State, actions: jnp.ndarray
) -> tuple[State, jnp.ndarray, bool]:
    legal_actions = state.legal_actions()
    for i in range(4):
        if actions[i] == Action.NONE:
            continue
        assert legal_actions[(i, actions[i])]
    return State.step(state, actions)


class Yaku:
    CACHE = CacheLoader.load_yaku_cache()
    MAX_PATTERNS = 3

    平和 = 0
    一盃口 = 1
    二盃口 = 2
    混全帯么九 = 3
    純全帯么九 = 4
    一気通貫 = 5
    三色同順 = 6
    三色同刻 = 7
    対々和 = 8
    三暗刻 = 9
    七対子 = 10
    断么九 = 11
    混一色 = 12
    清一色 = 13
    混老頭 = 14
    小三元 = 15
    白 = 16
    發 = 17
    中 = 18
    場風 = 19
    自風 = 20
    門前清自摸和 = 21
    立直 = 22

    大三元 = 23
    小四喜 = 24
    大四喜 = 25
    九蓮宝燈 = 26
    国士無双 = 27
    清老頭 = 28
    字一色 = 29
    緑一色 = 30
    四暗刻 = 31

    # fmt: off
    FAN = jnp.array([
        [0,0,0,1,2,1,1,2,2,2,0,1,2,5,2,2,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0],  # noqa
        [1,1,3,2,3,2,2,2,2,2,2,1,3,6,2,2,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],  # noqa
    ])
    YAKUMAN = jnp.array([
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,2,1,1,1,1,1,1  # noqa
    ])
    # fmt: on

    @staticmethod
    @jit
    def score(
        hand: jnp.ndarray,
        melds: jnp.ndarray,
        n_meld: int,
        last: int,
        riichi: bool,
        is_ron: bool,
    ) -> int:
        yaku, fan, fu = Yaku.judge(hand, melds, n_meld, last, riichi, is_ron)
        score = fu << (fan + 2)
        return jax.lax.cond(
            fu == 0,
            lambda: 8000 * jnp.dot(yaku, Yaku.YAKUMAN),
            lambda: jax.lax.cond(
                score < 2000,
                lambda: score,
                lambda: jax.lax.switch(
                    fan - 5,
                    [
                        # 5翻以下
                        lambda: 2000,
                        # 6-7翻
                        lambda: 3000,
                        lambda: 3000,
                        # 8-10翻
                        lambda: 4000,
                        lambda: 4000,
                        lambda: 4000,
                        # 11-12翻
                        lambda: 6000,
                        lambda: 6000,
                        # 13翻以上
                        lambda: 8000,
                    ],
                ),
            ),
        )

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
    def n_pung(code: int) -> jnp.ndarray:
        return Yaku.CACHE[code] >> 20 & 0b111

    @staticmethod
    @jit
    def n_double_chow(code: int) -> jnp.ndarray:
        return Yaku.CACHE[code] >> 23 & 0b11

    @staticmethod
    @jit
    def outside(code: int) -> jnp.ndarray:
        return Yaku.CACHE[code] >> 25 & 1

    @staticmethod
    @jit
    def nine_gates(code: int) -> jnp.ndarray:
        return Yaku.CACHE[code] >> 26

    @staticmethod
    @jit
    def is_pure_straight(chow: jnp.ndarray) -> jnp.ndarray:
        return (
            ((chow & 0b1001001) == 0b1001001)
            | ((chow >> 9 & 0b1001001) == 0b1001001)
            | ((chow >> 18 & 0b1001001) == 0b1001001)
        ) == 1

    @staticmethod
    @jit
    def is_triple_chow(chow: jnp.ndarray) -> jnp.ndarray:
        return (
            ((chow & 0b1000000001000000001) == 0b1000000001000000001)
            | ((chow >> 1 & 0b1000000001000000001) == 0b1000000001000000001)
            | ((chow >> 2 & 0b1000000001000000001) == 0b1000000001000000001)
            | ((chow >> 3 & 0b1000000001000000001) == 0b1000000001000000001)
            | ((chow >> 4 & 0b1000000001000000001) == 0b1000000001000000001)
            | ((chow >> 5 & 0b1000000001000000001) == 0b1000000001000000001)
            | ((chow >> 6 & 0b1000000001000000001) == 0b1000000001000000001)
        ) == 1

    @staticmethod
    @jit
    def is_triple_pung(pung: jnp.ndarray) -> jnp.ndarray:
        return (
            ((pung & 0b1000000001000000001) == 0b1000000001000000001)
            | ((pung >> 1 & 0b1000000001000000001) == 0b1000000001000000001)
            | ((pung >> 2 & 0b1000000001000000001) == 0b1000000001000000001)
            | ((pung >> 3 & 0b1000000001000000001) == 0b1000000001000000001)
            | ((pung >> 4 & 0b1000000001000000001) == 0b1000000001000000001)
            | ((pung >> 5 & 0b1000000001000000001) == 0b1000000001000000001)
            | ((pung >> 6 & 0b1000000001000000001) == 0b1000000001000000001)
            | ((pung >> 7 & 0b1000000001000000001) == 0b1000000001000000001)
            | ((pung >> 8 & 0b1000000001000000001) == 0b1000000001000000001)
        ) == 1

    @staticmethod
    @jit
    def update(
        is_pinfu: jnp.ndarray,
        is_outside: jnp.ndarray,
        n_double_chow: jnp.ndarray,
        all_chow: jnp.ndarray,
        all_pung: jnp.ndarray,
        n_concealed_pung: jnp.ndarray,
        nine_gates: jnp.ndarray,
        fu: jnp.ndarray,
        code: int,
        suit: int,
        last: int,
        is_ron: bool,
    ):
        chow = Yaku.chow(code)
        pung = Yaku.pung(code)

        open_end = (chow ^ (chow & 1)) << 2 | (chow ^ (chow & 0b1000000))
        # リャンメン待ちにできる位置

        in_range = suit == last // 9
        pos = last % 9

        is_pinfu &= (in_range == 0) | (open_end >> pos & 1) == 1
        is_pinfu &= pung == 0

        is_outside &= Yaku.outside(code) == 1

        n_double_chow += Yaku.n_double_chow(code)
        all_chow |= chow << 9 * suit
        all_pung |= pung << 9 * suit

        n_pung = Yaku.n_pung(code)
        # 刻子の数

        chow_range = chow | chow << 1 | chow << 2

        loss = (
            is_ron
            & in_range
            & ((chow_range >> pos & 1) == 0)
            & (pung >> pos & 1)
        )
        # ロンして明刻扱いになってしまう場合

        n_concealed_pung += n_pung - loss

        nine_gates |= Yaku.nine_gates(code) == 1

        outside_pung = pung & 0b100000001

        strong = (
            in_range
            & (
                (1 << Yaku.head(code))
                | ((chow & 1) << 2)
                | (chow & 0b1000000)
                | (chow << 1)
            )
            >> pos
            & 1
        )
        # 強い待ち(カンチャン, ペンチャン, 単騎)にできるか

        loss <<= outside_pung >> pos & 1

        fu += 4 * (n_pung + (outside_pung > 0)) - 2 * loss + 2 * strong

        return (
            is_pinfu,
            is_outside,
            n_double_chow,
            all_chow,
            all_pung,
            n_concealed_pung,
            nine_gates,
            fu,
        )

    @staticmethod
    @jit
    def judge(
        hand: jnp.ndarray,
        melds: jnp.ndarray,
        n_meld: int,
        last: int,
        riichi: bool,
        is_ron: bool,
    ) -> tuple[jnp.ndarray, int, int]:
        is_menzen = jax.lax.fori_loop(
            0,
            n_meld,
            lambda i, menzen: menzen
            & (
                Action.is_selfkan(Meld.action(melds[i]))
                & (Meld.src(melds[i]) == 0)
            ),
            True,
        )

        is_pinfu = jnp.full(
            Yaku.MAX_PATTERNS,
            is_menzen
            & jnp.all(hand[28:31] < 3)
            & (hand[27] == 0)
            & jnp.all(hand[31:34] == 0),
        )
        # NOTE: 東場東家

        is_outside = jnp.full(
            Yaku.MAX_PATTERNS,
            jax.lax.fori_loop(
                0,
                n_meld,
                lambda i, valid: valid & Meld.is_outside(melds[i]),
                True,
            ),
        )
        n_double_chow = jnp.full(Yaku.MAX_PATTERNS, 0)

        all_chow = jnp.full(
            Yaku.MAX_PATTERNS,
            jax.lax.fori_loop(
                0, n_meld, lambda i, chow: chow | Meld.chow(melds[i]), 0
            ),
        )
        all_pung = jnp.full(
            Yaku.MAX_PATTERNS,
            jax.lax.fori_loop(
                0,
                n_meld,
                lambda i, pung: pung | Meld.suited_pung(melds[i]),
                0,
            ),
        )

        n_concealed_pung = jnp.full(Yaku.MAX_PATTERNS, 0)
        nine_gates = jnp.full(Yaku.MAX_PATTERNS, False)

        fu = jnp.full(
            Yaku.MAX_PATTERNS,
            2 * (is_ron == 0)
            + jax.lax.fori_loop(
                0, n_meld, lambda i, sum: sum + Meld.fu(melds[i]), 0
            )
            + (hand[27] == 2) * 4
            + jnp.any(hand[31:] == 2) * 2
            + (hand[27] == 3) * 4 * (2 - (is_ron & (27 == last)))
            + (hand[31] == 3) * 4 * (2 - (is_ron & (31 == last)))
            + (hand[32] == 3) * 4 * (2 - (is_ron & (32 == last)))
            + (hand[33] == 3) * 4 * (2 - (is_ron & (33 == last)))
            # NOTE: 東場東家
            + ((27 <= last) & (hand[last] == 2)),
        )

        for suit in range(3):
            code = jax.lax.fori_loop(
                9 * suit,
                9 * (suit + 1),
                lambda i, code: code * 5 + hand[i].astype(int),
                0,
            )
            (
                is_pinfu,
                is_outside,
                n_double_chow,
                all_chow,
                all_pung,
                n_concealed_pung,
                nine_gates,
                fu,
            ) = Yaku.update(
                is_pinfu,
                is_outside,
                n_double_chow,
                all_chow,
                all_pung,
                n_concealed_pung,
                nine_gates,
                fu,
                code,
                suit,
                last,
                is_ron,
            )

        n_concealed_pung += jnp.sum(hand[27:] >= 3) - (
            is_ron & (last >= 27) & (hand[last] >= 3)
        )

        fu *= is_pinfu == 0
        fu += 20 + 10 * (is_menzen & is_ron)
        fu += 10 * ((is_menzen == 0) & (fu == 20))

        flatten = Yaku.flatten(hand, melds, n_meld)

        four_winds = jnp.sum(flatten[27:31] >= 3)
        three_dragons = jnp.sum(flatten[31:34] >= 3)

        has_tanyao = (
            jnp.any(flatten[1:8])
            | jnp.any(flatten[10:17])
            | jnp.any(flatten[19:26])
        )
        has_honor = jnp.any(flatten[27:] > 0)
        is_flush = (
            jnp.any(flatten[0:9] > 0).astype(int)
            + jnp.any(flatten[9:18] > 0).astype(int)
            + jnp.any(flatten[18:27] > 0).astype(int)
        ) == 1

        has_outside = (
            (flatten[0] > 0)
            | (flatten[8] > 0)
            | (flatten[9] > 0)
            | (flatten[17] > 0)
            | (flatten[18] > 0)
            | (flatten[26] > 0)
        )

        yaku = (
            jnp.full((Yaku.FAN.shape[1], Yaku.MAX_PATTERNS), False)
            .at[Yaku.平和]
            .set(is_pinfu)
            .at[Yaku.一盃口]
            .set(is_menzen & (n_double_chow == 1))
            .at[Yaku.二盃口]
            .set(n_double_chow == 2)
            .at[Yaku.混全帯么九]
            .set(is_outside & has_honor & has_tanyao)
            .at[Yaku.純全帯么九]
            .set(is_outside & (has_honor == 0))
            .at[Yaku.一気通貫]
            .set(Yaku.is_pure_straight(all_chow))
            .at[Yaku.三色同順]
            .set(Yaku.is_triple_chow(all_chow))
            .at[Yaku.三色同刻]
            .set(Yaku.is_triple_pung(all_pung))
            .at[Yaku.対々和]
            .set(all_chow == 0)
            .at[Yaku.三暗刻]
            .set(n_concealed_pung == 3)
        )

        fan = Yaku.FAN[jax.lax.cond(is_menzen, lambda: 1, lambda: 0)]

        best_pattern = jnp.argmax(jnp.dot(fan, yaku) * 200 + fu)

        yaku_best = yaku.T[best_pattern]
        fu_best = fu[best_pattern]
        fu_best += -fu_best % 10

        yaku_best, fu_best = jax.lax.cond(
            yaku_best[Yaku.二盃口] | (jnp.sum(hand == 2) < 7),
            lambda: (yaku_best, fu_best),
            lambda: (
                jnp.full(Yaku.FAN.shape[1], False).at[Yaku.七対子].set(True),
                25,
            ),
        )

        yaku_best = (
            yaku_best.at[Yaku.断么九]
            .set((has_honor | has_outside) == 0)
            .at[Yaku.混一色]
            .set(is_flush & has_honor)
            .at[Yaku.清一色]
            .set(is_flush & (has_honor == 0))
            .at[Yaku.混老頭]
            .set(has_tanyao == 0)
            .at[Yaku.白]
            .set(flatten[31] >= 3)
            .at[Yaku.發]
            .set(flatten[32] >= 3)
            .at[Yaku.中]
            .set(flatten[33] >= 3)
            .at[Yaku.小三元]
            .set(jnp.all(flatten[31:34] >= 2) & (three_dragons >= 2))
            .at[Yaku.場風]
            .set(flatten[27] >= 3)
            .at[Yaku.自風]
            .set(flatten[27] >= 3)
            .at[Yaku.門前清自摸和]
            .set(is_menzen & (is_ron == 0))
            .at[Yaku.立直]
            .set(riichi)
        )

        yakuman = (
            jnp.full(Yaku.FAN.shape[1], False)
            .at[Yaku.大三元]
            .set(three_dragons == 3)
            .at[Yaku.小四喜]
            .set(jnp.all(flatten[27:31] >= 2) & (four_winds == 3))
            .at[Yaku.大四喜]
            .set(four_winds == 4)
            .at[Yaku.九蓮宝燈]
            .set(jnp.any(nine_gates))
            .at[Yaku.国士無双]
            .set(
                (hand[0] > 0)
                & (hand[8] > 0)
                & (hand[9] > 0)
                & (hand[17] > 0)
                & (hand[18] > 0)
                & jnp.all(hand[26:] > 0)
                & (has_tanyao == 0)
            )
            .at[Yaku.清老頭]
            .set((has_tanyao == 0) & (has_honor == 0))
            .at[Yaku.字一色]
            .set(jnp.all(flatten[0:27] == 0))
            .at[Yaku.緑一色]
            .set(
                jnp.all(flatten[0:19] == 0)
                & (flatten[22] == 0)
                & (flatten[24] == 0)
                & jnp.all(flatten[26:32] == 0)
                & (flatten[33] == 0)
            )
            .at[Yaku.四暗刻]
            .set(jnp.any(n_concealed_pung == 4))
        )

        return jax.lax.cond(
            jnp.any(yakuman),
            lambda: (yakuman, 0, 0),
            lambda: (yaku_best, jnp.dot(fan, yaku_best), fu_best),
        )

    @staticmethod
    @jit
    def flatten(
        hand: jnp.ndarray, melds: jnp.ndarray, n_meld: int
    ) -> jnp.ndarray:
        return jax.lax.fori_loop(
            0, n_meld, lambda i, arr: Yaku._flatten(arr, melds[i]), hand
        )

    @staticmethod
    @jit
    def _flatten(hand: jnp.ndarray, meld: int) -> jnp.ndarray:
        target, action = Meld.target(meld), Meld.action(meld)
        return jax.lax.switch(
            action - Action.PON + 1,
            [
                lambda: Hand.add(hand, target, 4),
                lambda: Hand.add(hand, target, 3),
                lambda: Hand.add(hand, target, 4),
                lambda: Hand.add(
                    Hand.add(Hand.add(hand, target + 1), target + 2), target
                ),
                lambda: Hand.add(
                    Hand.add(Hand.add(hand, target - 1), target + 1), target
                ),
                lambda: Hand.add(
                    Hand.add(Hand.add(hand, target - 2), target - 1), target
                ),
            ],
        )


class Shanten:
    # See the link below for the algorithm details.
    # https://github.com/sotetsuk/pgx/pull/123

    CACHE = CacheLoader.load_shanten_cache()

    @staticmethod
    @jit
    def discard(hand: jnp.ndarray) -> jnp.ndarray:
        return jax.lax.map(
            lambda i: jax.lax.cond(
                hand[i] == 0,
                lambda: 0,
                lambda: Shanten.number(hand.at[i].set(hand[i] - 1)),
            ),
            jnp.arange(34),
        )

    @staticmethod
    @jit
    def number(hand: jnp.ndarray):
        return jnp.min(
            jnp.array(
                [
                    Shanten.normal(hand),
                    Shanten.seven_pairs(hand),
                    Shanten.thirteen_orphan(hand),
                ]
            )
        )

    @staticmethod
    @jit
    def seven_pairs(hand: jnp.ndarray):
        n_pair = jnp.sum(hand >= 2)
        n_kind = jnp.sum(hand > 0)
        return 7 - n_pair + jax.lax.max(7 - n_kind, 0)

    @staticmethod
    @jit
    def thirteen_orphan(hand: jnp.ndarray):
        n_pair = (
            (hand[0] >= 2).astype(int)
            + (hand[8] >= 2).astype(int)
            + (hand[9] >= 2).astype(int)
            + (hand[17] >= 2).astype(int)
            + (hand[18] >= 2).astype(int)
            + jnp.sum(hand[26:34] >= 2)
        )
        n_kind = (
            (hand[0] > 0).astype(int)
            + (hand[8] > 0).astype(int)
            + (hand[9] > 0).astype(int)
            + (hand[17] > 0).astype(int)
            + (hand[18] > 0).astype(int)
            + jnp.sum(hand[26:34] > 0)
        )
        return 14 - n_kind - (n_pair > 0)

    @staticmethod
    @jit
    def normal(hand: jnp.ndarray):
        code = jax.lax.map(
            lambda suit: jax.lax.cond(
                suit == 3,
                lambda: jax.lax.fori_loop(
                    27,
                    34,
                    lambda i, code: code * 5 + hand[i].astype(int),
                    0,
                )
                + 1953125,
                lambda: jax.lax.fori_loop(
                    9 * suit,
                    9 * (suit + 1),
                    lambda i, code: code * 5 + hand[i].astype(int),
                    0,
                ),
            ),
            jnp.arange(4),
        )

        n_set = jnp.sum(hand).astype(int) // 3

        return jnp.min(
            jax.lax.map(
                lambda suit: Shanten._normal(code, n_set, suit), jnp.arange(4)
            )
        )

    @staticmethod
    @jit
    def _normal(code: jnp.ndarray, n_set: int, head_suit: int) -> int:
        cost = Shanten.CACHE[code[head_suit]][4]
        idx = jnp.full(4, 0).at[head_suit].set(5)
        cost, idx = jax.lax.fori_loop(
            0,
            n_set,
            lambda _, tpl: Shanten._update(code, *tpl),
            (cost, idx),
        )
        return cost

    @staticmethod
    @jit
    def _update(
        code: jnp.ndarray, cost: int, idx: jnp.ndarray
    ) -> tuple[int, jnp.ndarray]:
        i = jnp.argmin(Shanten.CACHE[code][[0, 1, 2, 3], idx])
        cost += Shanten.CACHE[code][i][idx[i]]
        idx = idx.at[i].set(idx[i] + 1)
        return (cost, idx)
