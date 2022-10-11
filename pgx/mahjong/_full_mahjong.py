from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

import numpy as np

np.random.seed(0)


class Tile:
    @staticmethod
    def to_str(tile: int) -> str:
        suit, num = tile // 9, tile % 9 + 1
        return str(num) + ["m", "p", "s", "z"][suit]

    @staticmethod
    def from_str(s: str) -> int:
        return (int(s[0]) - 1) + 9 * ["m", "p", "s", "z"].index(s[1])

    @staticmethod
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
    CHI_L = 74  # [4]56m
    CHI_M = 75  # [5]46m
    CHI_R = 76  # [6]45m
    PASS = 77

    NONE = 78

    @staticmethod
    def is_selfkan(action: int) -> bool:
        return (34 <= action) & (action < 68)


class CacheLoader:
    DIR = os.path.join(os.path.dirname(__file__), "cache")

    @staticmethod
    def load_hand_cache():
        with open(os.path.join(CacheLoader.DIR, "hand_cache.json")) as f:
            return np.array(json.load(f), dtype=np.uint32)

    @staticmethod
    def load_yaku_cache():
        with open(os.path.join(CacheLoader.DIR, "yaku_cache.json")) as f:
            return np.array(json.load(f), dtype=np.int32)


class Hand:
    CACHE = CacheLoader.load_hand_cache()

    @staticmethod
    def cache(code: int) -> int:
        return (Hand.CACHE[code >> 5] >> (code & 0b11111)) & 1

    @staticmethod
    def can_ron(hand: np.ndarray, tile: int) -> bool:
        assert np.sum(hand) % 3 == 1
        return Hand.can_tsumo(Hand.add(hand, tile))

    @staticmethod
    def can_riichi(hand: np.ndarray) -> bool:
        assert np.sum(hand) % 3 == 2
        for tile in range(34):
            if hand[tile] > 0 and Hand.is_tenpai(Hand.sub(hand, tile)):
                return True
        return False

    @staticmethod
    def is_tenpai(hand: np.ndarray) -> bool:
        assert np.sum(hand) % 3 == 1
        for tile in range(34):
            if hand[tile] < 4 and Hand.can_ron(hand, tile):
                return True
        return False

    @staticmethod
    def can_tsumo(hand: np.ndarray) -> bool:
        assert np.sum(hand) % 3 == 2
        thirteen_orphan = (
            (hand[0] > 0)
            & (hand[8] > 0)
            & (hand[9] > 0)
            & (hand[17] > 0)
            & (hand[18] > 0)
            & np.all(hand[26:] > 0)
            & (
                (
                    hand[0]
                    + hand[8]
                    + hand[9]
                    + hand[17]
                    + hand[18]
                    + np.sum(hand[26:])
                )
                == 14
            )
        )
        seven_pairs = np.sum(hand == 2) == 7

        heads, valid = 0, 1
        for suit in range(3):
            code, sum = 0, 0
            for i in range(9 * suit, 9 * (suit + 1)):
                code = code * 5 + hand[i]
                sum += hand[i]
            valid &= Hand.cache(code)
            heads += sum % 3 == 2

        for i in range(27, 34):
            valid &= (hand[i] != 1) & (hand[i] != 4)
            heads += hand[i] == 2

        return ((valid & (heads == 1)) | thirteen_orphan | seven_pairs) == 1

    @staticmethod
    def can_pon(hand: np.ndarray, tile: int) -> bool:
        assert np.sum(hand) % 3 == 1
        return hand[tile] >= 2

    @staticmethod
    def can_minkan(hand: np.ndarray, tile: int) -> bool:
        assert np.sum(hand) % 3 == 1
        return hand[tile] == 3

    @staticmethod
    def can_kakan(hand: np.ndarray, tile: int) -> bool:
        assert np.sum(hand) % 3 == 2
        return hand[tile] == 1

    @staticmethod
    def can_ankan(hand: np.ndarray, tile: int) -> bool:
        assert np.sum(hand) % 3 == 2
        return hand[tile] == 4

    @staticmethod
    def can_chi(hand: np.ndarray, tile: int, action: int) -> bool:
        assert np.sum(hand) % 3 == 1
        if tile >= 27:
            return False
        if action == Action.CHI_L:
            if tile % 9 >= 7:
                return False
            return (hand[tile + 1] > 0) & (hand[tile + 2] > 0)
        if action == Action.CHI_M:
            if (tile % 9 == 8) | (tile % 9 == 0):
                return False
            return (hand[tile - 1] > 0) & (hand[tile + 1] > 0)
        if action == Action.CHI_R:
            if tile % 9 <= 1:
                return False
            return (hand[tile - 2] > 0) & (hand[tile - 1] > 0)
        return False

    @staticmethod
    def add(hand: np.ndarray, tile: int, x: int = 1) -> np.ndarray:
        assert 0 <= hand[tile] + x <= 4
        tmp = hand.copy()
        tmp[tile] += x
        return tmp

    @staticmethod
    def sub(hand: np.ndarray, tile: int, x: int = 1) -> np.ndarray:
        return Hand.add(hand, tile, -x)

    @staticmethod
    def pon(hand: np.ndarray, tile: int) -> np.ndarray:
        assert Hand.can_pon(hand, tile)
        return Hand.sub(hand, tile, 2)

    @staticmethod
    def minkan(hand: np.ndarray, tile: int) -> np.ndarray:
        assert Hand.can_minkan(hand, tile)
        return Hand.sub(hand, tile, 3)

    @staticmethod
    def kakan(hand: np.ndarray, tile: int) -> np.ndarray:
        assert Hand.can_kakan(hand, tile)
        return Hand.sub(hand, tile)

    @staticmethod
    def ankan(hand: np.ndarray, tile: int) -> np.ndarray:
        assert Hand.can_ankan(hand, tile)
        return Hand.sub(hand, tile, 4)

    @staticmethod
    def chi(hand: np.ndarray, tile: int, action: int) -> np.ndarray:
        assert Hand.can_chi(hand, tile, action)
        if action == Action.CHI_L:
            return Hand.sub(Hand.sub(hand, tile + 1), tile + 2)
        if action == Action.CHI_M:
            return Hand.sub(Hand.sub(hand, tile - 1), tile + 1)
        if action == Action.CHI_R:
            return Hand.sub(Hand.sub(hand, tile - 2), tile - 1)
        assert False

    @staticmethod
    def to_str(hand: np.ndarray) -> str:
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
    def from_str(s: str) -> np.ndarray:
        base = 0
        hand = np.zeros(34, dtype=np.uint8)
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
    arr: np.ndarray
    idx: int = 135
    end: int = 13
    doras: int = 1

    # 0   2 | 4 - 12 | 14 - 134
    # 1   3 | 5 - 13 | 15 - 135
    # -------------------------
    # 嶺上牌| ドラ   | to be used
    #       | 裏ドラ |

    def is_empty(self) -> bool:
        return self.size() == 0

    def size(self) -> int:
        return self.idx - self.end

    @staticmethod
    def init_arr() -> np.ndarray:
        return np.random.permutation(np.arange(136) // 4)

    @staticmethod
    def deal(deck: Deck) -> tuple[Deck, np.ndarray, int]:
        hand = np.zeros((4, 34), dtype=np.uint8)
        for i in range(3):
            for j in range(4):
                for k in range(4):
                    hand[j] = Hand.add(
                        hand[j], deck.arr[-(16 * i + 4 * j + k + 1)]
                    )
        for j in range(4):
            hand[j] = Hand.add(hand[j], deck.arr[-(16 * 3 + j + 1)])

        last_draw = deck.arr[-(16 * 3 + 4 + 1)]
        hand[0] = Hand.add(hand[0], last_draw)

        deck.idx -= 53

        return deck, hand, last_draw

    @staticmethod
    def draw(deck: Deck, is_kan: bool = False) -> tuple[Deck, int]:
        assert not deck.is_empty()
        tile = deck.arr[
            deck.idx * (is_kan is False) | (deck.doras - 1) * is_kan
        ]
        deck.idx -= is_kan is False
        deck.end += is_kan
        deck.doras += is_kan  # NOTE: 先めくりで統一
        return deck, tile


@dataclass
class Observation:
    hand: np.ndarray
    target: int
    last_draw: int


class Meld:
    @staticmethod
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
    def src(meld: int) -> int:
        return meld >> 13 & 0b11

    @staticmethod
    def target(meld: int) -> int:
        return (meld >> 7) & 0b111111

    @staticmethod
    def action(meld: int) -> int:
        return meld & 0b1111111

    @staticmethod
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
    def chow(meld: int) -> int:
        action = Meld.action(meld)
        is_chi = (Action.CHI_L <= action) & (action <= Action.CHI_R)
        pos = Meld.target(meld) - (
            action - Action.CHI_L
        )  # WARNING: may be negative

        pos *= is_chi
        return is_chi << pos

    @staticmethod
    def is_outside(meld: int) -> int:
        action = Meld.action(meld)
        target = Meld.target(meld)
        is_chi = (Action.CHI_L <= action) & (action <= Action.CHI_R)

        if is_chi:
            return Tile.is_outside(
                target - (action - Action.CHI_L)
            ) | Tile.is_outside(target - (action - Action.CHI_L) + 2)
        else:
            return Tile.is_outside(target)

    @staticmethod
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
    hand: np.ndarray
    turn: int  # 手牌が3n+2枚, もしくは直前に牌を捨てたplayer
    target: int  # 直前に捨てられてron,pon,chi の対象になっている牌. 存在しなければ-1
    last_draw: int  # 手牌が3n+2枚のplayerが直前に引いた牌. 存在しなければ-1
    riichi_declared: bool  # state.turn がリーチ宣言してから, その直後の打牌が通るまでTrue
    riichi: np.ndarray  # 各playerのリーチが成立しているかどうか
    n_meld: np.ndarray  # 各playerの副露回数
    melds: np.ndarray
    # melds[i][j]: player i のj回目の副露(j=1,2,3,4). 存在しなければ0

    # 以下計算効率のために保持
    is_menzen: np.ndarray
    pon: np.ndarray
    # pon[i][j]: player i がjをポンを所有している場合, src << 2 | index. or 0

    def can_kakan(self, tile: int) -> bool:
        return (self.pon[(self.turn, tile)] > 0) & Hand.can_kakan(
            self.hand[self.turn], tile
        )

    def can_tsumo(self) -> bool:
        if Hand.can_tsumo(self.hand[self.turn]):
            yaku = Yaku.judge(
                self.hand[self.turn],
                self.melds[self.turn],
                self.n_meld[self.turn],
                self.last_draw,
                self.riichi[self.turn],
                False,
            )[0]
            return bool(np.any(yaku))
        return False

    def can_ron(self, player: int) -> bool:
        if Hand.can_ron(self.hand[player], self.target):
            yaku = Yaku.judge(
                Hand.add(self.hand[player], self.target),
                self.melds[player],
                self.n_meld[player],
                self.last_draw,
                self.riichi[player],
                True,
            )[0]
            return bool(np.any(yaku))
        return False

    def legal_actions(self) -> np.ndarray:
        legal_actions = np.full((4, Action.NONE), False)

        # 暗/加槓, ツモ切り, ツモ, リーチ
        if (self.last_draw != -1) & (not self.riichi_declared):
            for tile in range(34):
                legal_actions[(self.turn, tile + 34)] = (
                    (
                        Hand.can_ankan(self.hand[self.turn], tile)
                        | self.can_kakan(tile)
                    )
                    & (self.deck.doras < 5)  # 5回目の槓はできない
                    & (self.deck.is_empty() == 0)
                )
            legal_actions[(self.turn, Action.TSUMOGIRI)] = True
            legal_actions[(self.turn, Action.TSUMO)] = self.can_tsumo()
            legal_actions[(self.turn, Action.RIICHI)] = (
                (not self.riichi[self.turn])
                & self.is_menzen[self.turn]
                & (self.deck.size() >= 4)
                & Hand.can_riichi(self.hand[self.turn])
            )

        # リーチ宣言直後の打牌
        if (self.last_draw != -1) & self.riichi_declared:
            for tile in range(34):
                if self.hand[self.turn][tile] > (tile == self.last_draw):
                    legal_actions[(self.turn, tile)] = Hand.is_tenpai(
                        Hand.sub(self.hand[self.turn], tile)
                    )
            legal_actions[(self.turn, Action.TSUMOGIRI)] = Hand.is_tenpai(
                Hand.sub(self.hand[self.turn], self.last_draw)
            )

        # 手出し, 鳴いた後の手出し
        if (
            (self.target == -1)
            & (not self.riichi_declared)
            & (not self.riichi[self.turn])
        ):
            for tile in range(34):
                legal_actions[(self.turn, tile)] = self.hand[self.turn][
                    tile
                ] > (tile == self.last_draw)

        # 他家の行動
        if self.target != -1:
            for player in range(4):
                if player == self.turn:
                    continue

                legal_actions[(player, Action.RON)] = Hand.can_ron(
                    self.hand[player], self.target
                )

                if self.deck.is_empty() | self.riichi[player]:
                    continue

                legal_actions[(player, Action.PON)] = Hand.can_pon(
                    self.hand[player], self.target
                )

                legal_actions[(player, Action.MINKAN)] = Hand.can_minkan(
                    self.hand[player], self.target
                ) & (self.deck.doras < 5)

                if (player - self.turn) % 4 != 1:
                    continue
                legal_actions[(player, Action.CHI_L)] = Hand.can_chi(
                    self.hand[player], self.target, Action.CHI_L
                )
                legal_actions[(player, Action.CHI_M)] = Hand.can_chi(
                    self.hand[player], self.target, Action.CHI_M
                )
                legal_actions[(player, Action.CHI_R)] = Hand.can_chi(
                    self.hand[player], self.target, Action.CHI_R
                )

            for player in range(4):
                if player == self.turn:
                    continue
                legal_actions[(player, Action.PASS)] = np.any(
                    legal_actions[player]
                )

        return legal_actions

    def observe(self, player: int) -> Observation:
        return Observation(self.hand[player], self.target, self.last_draw)

    @staticmethod
    def init() -> State:
        return State.init_with_deck(Deck.init_arr())

    @staticmethod
    def init_with_deck(arr: np.ndarray) -> State:
        deck, hand, last_draw = Deck.deal(Deck(arr))

        turn = 0
        target = -1
        riichi_declared = False
        riichi = np.full(4, False)
        n_meld = np.zeros(4, dtype=np.int32)
        melds = np.zeros((4, 4), dtype=np.int32)
        is_menzen = np.full(4, True)
        pon = np.zeros((4, 34), dtype=np.int32)
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
    def step(
        state: State, actions: np.ndarray
    ) -> tuple[State, np.ndarray, bool]:
        player = int(np.argmin(actions))
        return State._step(state, player, actions[player])

    @staticmethod
    def _step(
        state: State, player: int, action: int
    ) -> tuple[State, np.ndarray, bool]:
        if action < 34:
            return State._discard(state, action)
        if action < 68:
            return State._selfkan(state, action)
        if action == Action.TSUMOGIRI:
            return State._tsumogiri(state)
        if action == Action.RIICHI:
            return State._riichi(state)
        if action == Action.TSUMO:
            return State._tsumo(state)
        if action == Action.RON:
            return State._ron(state, player)
        if action == Action.PON:
            return State._pon(state, player)
        if action == Action.MINKAN:
            return State._minkan(state, player)
        if action == Action.CHI_L:
            return State._chi(state, player, Action.CHI_L)
        if action == Action.CHI_M:
            return State._chi(state, player, Action.CHI_M)
        if action == Action.CHI_R:
            return State._chi(state, player, Action.CHI_R)
        if action == Action.PASS:
            return State._try_draw(state)
        assert False

    @staticmethod
    def _tsumogiri(state: State) -> tuple[State, np.ndarray, bool]:
        return State._discard(state, state.last_draw)

    @staticmethod
    def _discard(state: State, tile: int) -> tuple[State, np.ndarray, bool]:
        state.hand[state.turn] = Hand.sub(state.hand[state.turn], tile)
        state.target = tile
        state.last_draw = -1
        if np.any(state.legal_actions()):
            return (state, np.zeros(4, dtype=np.int32), False)
        return State._try_draw(state)

    @staticmethod
    def _try_draw(state: State) -> tuple[State, np.ndarray, bool]:
        state.target = -1
        if state.deck.is_empty():
            return State._ryukyoku(state)
        return State._draw(state)

    @staticmethod
    def _accept_riichi(state: State) -> State:
        state.riichi[state.turn] |= state.riichi_declared
        state.riichi_declared = False
        return state

    @staticmethod
    def _draw(state: State) -> tuple[State, np.ndarray, bool]:
        state = State._accept_riichi(state)
        state.turn += 1
        state.turn %= 4
        state.deck, tile = Deck.draw(state.deck)
        state.last_draw = tile
        state.hand[state.turn] = Hand.add(state.hand[state.turn], tile)
        return state, np.zeros(4, dtype=np.int32), False

    @staticmethod
    def _ryukyoku(state: State) -> tuple[State, np.ndarray, bool]:
        is_tenpai = np.array(
            [Hand.is_tenpai(state.hand[i]) for i in range(4)], dtype=np.int32
        )
        tenpais = np.sum(is_tenpai)
        if tenpais == 0:
            plus, minus = (0, 0)
        elif tenpais == 1:
            plus, minus = (3000, 1000)
        elif tenpais == 2:
            plus, minus = (1500, 1500)
        elif tenpais == 3:
            plus, minus = (1000, 3000)
        else:
            plus, minus = (0, 0)

        reward = np.array(
            [plus if is_tenpai[i] else -minus for i in range(4)],
            dtype=np.int32,
        )

        # 供託
        reward -= 1000 * state.riichi
        top = np.argmax(reward)
        reward[top] += 1000 * np.sum(state.riichi)

        return state, reward, True

    @staticmethod
    def _ron(state: State, player: int) -> tuple[State, np.ndarray, bool]:
        score = Yaku.score(
            state.hand[player],
            state.melds[player],
            state.n_meld[player],
            state.target,
            state.riichi[player],
            is_ron=True,
        )
        score *= 6 if player == 0 else 4
        score += -score % 100

        reward = np.zeros(4, dtype=np.int32)
        reward[player] += score
        reward[state.turn] -= score

        # 供託
        reward -= 1000 * state.riichi
        reward[player] += 1000 * np.sum(state.riichi)

        return state, reward, True

    @staticmethod
    def _append_meld(state: State, meld: int, player: int) -> State:
        state.melds[(player, state.n_meld[player])] = meld
        state.n_meld[player] += 1
        return state

    @staticmethod
    def _minkan(state: State, player: int) -> tuple[State, np.ndarray, bool]:
        state = State._accept_riichi(state)
        meld = Meld.init(
            Action.MINKAN, state.target, (state.turn - player) % 4
        )
        state = State._append_meld(state, meld, player)
        state.hand[player] = Hand.minkan(state.hand[player], state.target)
        state.target = -1
        state.turn = player
        state.is_menzen[player] = False

        # 嶺上牌
        state.deck, tile = Deck.draw(state.deck, is_kan=True)
        state.last_draw = tile
        state.hand[state.turn] = Hand.add(state.hand[state.turn], tile)
        return state, np.zeros(4, dtype=np.int32), False

    @staticmethod
    def _selfkan(state: State, action: int) -> tuple[State, np.ndarray, bool]:
        target = action - 34
        pon = state.pon[(state.turn, target)]
        if pon == 0:
            return State._ankan(state, target)
        else:
            return State._kakan(state, target, pon >> 2, pon & 0b11)

    @staticmethod
    def _ankan(state: State, target: int) -> tuple[State, np.ndarray, bool]:
        meld = Meld.init(target + 34, target, src=0)
        state = State._append_meld(state, meld, state.turn)
        state.hand[state.turn] = Hand.ankan(state.hand[state.turn], target)
        # TODO: 国士無双ロンの受付

        # 嶺上牌
        state.deck, tile = Deck.draw(state.deck, is_kan=True)
        state.last_draw = tile
        state.hand[state.turn] = Hand.add(state.hand[state.turn], tile)
        return state, np.zeros(4, dtype=np.int32), False

    @staticmethod
    def _kakan(
        state: State, target: int, pon_src: int, pon_idx: int
    ) -> tuple[State, np.ndarray, bool]:
        state.melds[(state.turn, pon_idx)] = Meld.init(
            target + 34, target, pon_src
        )
        state.hand[state.turn] = Hand.kakan(state.hand[state.turn], target)
        state.pon[(state.turn, target)] = 0
        # TODO: 槍槓の受付

        # 嶺上牌
        state.deck, tile = Deck.draw(state.deck, is_kan=True)
        state.last_draw = tile
        state.hand[state.turn] = Hand.add(state.hand[state.turn], tile)
        return state, np.zeros(4, dtype=np.int32), False

    @staticmethod
    def _pon(state: State, player: int) -> tuple[State, np.ndarray, bool]:
        state = State._accept_riichi(state)
        src = (state.turn - player) % 4
        meld = Meld.init(Action.PON, state.target, src)
        state = State._append_meld(state, meld, player)
        state.hand[player] = Hand.pon(state.hand[player], state.target)
        state.is_menzen[player] = False
        state.pon[(player, state.target)] = src << 2 | state.n_meld[player] - 1
        state.target = -1
        state.turn = player
        return state, np.zeros(4, dtype=np.int32), False

    @staticmethod
    def _chi(
        state: State, player: int, action: int
    ) -> tuple[State, np.ndarray, bool]:
        state = State._accept_riichi(state)
        meld = Meld.init(action, state.target, src=3)
        state = State._append_meld(state, meld, player)
        state.hand[player] = Hand.chi(state.hand[player], state.target, action)
        state.is_menzen[player] = False
        state.target = -1
        state.turn = player
        return state, np.zeros(4, dtype=np.int32), False

    @staticmethod
    def _tsumo(state: State) -> tuple[State, np.ndarray, bool]:
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

        if state.turn == 0:
            reward = np.full(4, -s2, dtype=np.int32)
            reward[0] = s2 * 3
        else:
            reward = np.full(4, -s1, dtype=np.int32)
            reward[0] = -s2
            reward[state.turn] = s1 * 2 + s2

        # 供託
        reward -= 1000 * state.riichi
        reward[state.turn] += 1000 * np.sum(state.riichi)

        return state, reward, True

    @staticmethod
    def _riichi(state: State) -> tuple[State, np.ndarray, bool]:
        state.riichi_declared = True
        return state, np.zeros(4, dtype=np.int32), False


def init() -> State:
    return State.init()


def step(state: State, actions: np.ndarray) -> tuple[State, np.ndarray, bool]:
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
    FAN = np.array([
        [0,0,0,1,2,1,1,2,2,2,0,1,2,5,2,2,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0],  # noqa
        [1,1,3,2,3,2,2,2,2,2,2,1,3,6,2,2,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],  # noqa
    ])
    YAKUMAN = np.array([
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,2,1,1,1,1,1,1  # noqa
    ])
    # fmt: on

    @staticmethod
    def score(
        hand: np.ndarray,
        melds: np.ndarray,
        n_meld: int,
        last: int,
        riichi: bool,
        is_ron: bool,
    ) -> int:
        yaku, fan, fu = Yaku.judge(hand, melds, n_meld, last, riichi, is_ron)
        score = fu << (fan + 2)
        if fu == 0:
            return 8000 * np.dot(yaku, Yaku.YAKUMAN)
        if score < 2000:
            return score
        if fan <= 5:
            return 2000
        if fan <= 7:
            return 3000
        if fan <= 10:
            return 4000
        if fan <= 12:
            return 6000
        return 8000

    @staticmethod
    def head(code: int) -> np.ndarray:
        return Yaku.CACHE[code] & 0b1111

    @staticmethod
    def chow(code: int) -> np.ndarray:
        return Yaku.CACHE[code] >> 4 & 0b1111111

    @staticmethod
    def pung(code: int) -> np.ndarray:
        return Yaku.CACHE[code] >> 11 & 0b111111111

    @staticmethod
    def n_pung(code: int) -> np.ndarray:
        return Yaku.CACHE[code] >> 20 & 0b111

    @staticmethod
    def n_double_chow(code: int) -> np.ndarray:
        return Yaku.CACHE[code] >> 23 & 0b11

    @staticmethod
    def outside(code: int) -> np.ndarray:
        return Yaku.CACHE[code] >> 25 & 1

    @staticmethod
    def nine_gates(code: int) -> np.ndarray:
        return Yaku.CACHE[code] >> 26

    @staticmethod
    def is_pure_straight(chow: np.ndarray) -> np.ndarray:
        return (
            ((chow & 0b1001001) == 0b1001001)
            | ((chow >> 9 & 0b1001001) == 0b1001001)
            | ((chow >> 18 & 0b1001001) == 0b1001001)
        ) == 1

    @staticmethod
    def is_triple_chow(chow: np.ndarray) -> np.ndarray:
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
    def is_triple_pung(pung: np.ndarray) -> np.ndarray:
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
    def update(
        is_pinfu: np.ndarray,
        is_outside: np.ndarray,
        n_double_chow: np.ndarray,
        all_chow: np.ndarray,
        all_pung: np.ndarray,
        n_concealed_pung: np.ndarray,
        nine_gates: np.ndarray,
        fu: np.ndarray,
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
    def judge(
        hand: np.ndarray,
        melds: np.ndarray,
        n_meld: int,
        last: int,
        riichi: bool,
        is_ron: bool,
    ) -> tuple[np.ndarray, int, int]:

        is_menzen = True
        for i in range(n_meld):
            is_menzen &= Action.is_selfkan(Meld.action(melds[i])) & (
                Meld.src(melds[i]) == 0
            )

        is_pinfu = np.full(
            Yaku.MAX_PATTERNS,
            is_menzen
            & np.all(hand[28:31] < 3)
            & (hand[27] == 0)
            & np.all(hand[31:34] == 0),
        )
        # NOTE: 東場東家

        is_outside = np.full(Yaku.MAX_PATTERNS, True)
        for i in range(n_meld):
            is_outside &= Meld.is_outside(melds[i])

        n_double_chow = np.full(Yaku.MAX_PATTERNS, 0)

        all_chow = np.full(Yaku.MAX_PATTERNS, 0)
        for i in range(n_meld):
            all_chow |= Meld.chow(melds[i])

        all_pung = np.full(Yaku.MAX_PATTERNS, 0)
        for i in range(n_meld):
            all_pung |= Meld.suited_pung(melds[i])

        n_concealed_pung = np.full(Yaku.MAX_PATTERNS, 0)
        nine_gates = np.full(Yaku.MAX_PATTERNS, False)

        fu = np.full(Yaku.MAX_PATTERNS, 0)
        for i in range(n_meld):
            fu += Meld.fu(melds[i])
        fu += (
            2 * (is_ron == 0)
            + (hand[27] == 2) * 4
            + np.any(hand[31:] == 2) * 2
            + (hand[27] == 3) * 4 * (2 - (is_ron & (27 == last)))
            + (hand[31] == 3) * 4 * (2 - (is_ron & (31 == last)))
            + (hand[32] == 3) * 4 * (2 - (is_ron & (32 == last)))
            + (hand[33] == 3) * 4 * (2 - (is_ron & (33 == last)))
            # NOTE: 東場東家
            + ((27 <= last) & (hand[last] == 2)),
        )

        for suit in range(3):
            code = 0
            for i in range(9 * suit, 9 * (suit + 1)):
                code = code * 5 + hand[i]
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

        n_concealed_pung += np.sum(hand[27:] >= 3) - (
            is_ron & (last >= 27) & (hand[last] >= 3)
        )

        fu *= is_pinfu == 0
        fu += 20 + 10 * (is_menzen & is_ron)
        fu += 10 * ((is_menzen == 0) & (fu == 20))

        flatten = Yaku.flatten(hand, melds, n_meld)

        four_winds = np.sum(flatten[27:31] >= 3)
        three_dragons = np.sum(flatten[31:34] >= 3)

        has_tanyao = (
            np.any(flatten[1:8])
            | np.any(flatten[10:17])
            | np.any(flatten[19:26])
        )
        has_honor = np.any(flatten[27:] > 0)
        is_flush = (
            np.any(flatten[0:9] > 0).astype(int)
            + np.any(flatten[9:18] > 0).astype(int)
            + np.any(flatten[18:27] > 0).astype(int)
        ) == 1

        has_outside = (
            (flatten[0] > 0)
            | (flatten[8] > 0)
            | (flatten[9] > 0)
            | (flatten[17] > 0)
            | (flatten[18] > 0)
            | (flatten[26] > 0)
        )

        yaku = np.full((Yaku.FAN.shape[1], Yaku.MAX_PATTERNS), False)
        yaku[Yaku.平和] = is_pinfu
        yaku[Yaku.一盃口] = is_menzen & (n_double_chow == 1)
        yaku[Yaku.二盃口] = n_double_chow == 2
        yaku[Yaku.混全帯么九] = is_outside & has_honor & has_tanyao
        yaku[Yaku.純全帯么九] = is_outside & (has_honor == 0)
        yaku[Yaku.一気通貫] = Yaku.is_pure_straight(all_chow)
        yaku[Yaku.三色同順] = Yaku.is_triple_chow(all_chow)
        yaku[Yaku.三色同刻] = Yaku.is_triple_pung(all_pung)
        yaku[Yaku.対々和] = all_chow == 0
        yaku[Yaku.三暗刻] = n_concealed_pung == 3

        fan = Yaku.FAN[1 if is_menzen else 0]

        best_pattern = np.argmax(np.dot(fan, yaku) * 200 + fu)

        yaku_best = yaku.T[best_pattern]
        fu_best = fu[best_pattern]
        fu_best += -fu_best % 10

        if (not yaku_best[Yaku.二盃口]) & (np.sum(hand == 2) == 7):
            yaku_best &= False
            yaku_best[Yaku.七対子] = True
            fu_best = 25

        yaku_best[Yaku.断么九] = (has_honor | has_outside) == 0
        yaku_best[Yaku.混一色] = is_flush & has_honor
        yaku_best[Yaku.清一色] = is_flush & (has_honor == 0)
        yaku_best[Yaku.混老頭] = has_tanyao == 0
        yaku_best[Yaku.白] = flatten[31] >= 3
        yaku_best[Yaku.發] = flatten[32] >= 3
        yaku_best[Yaku.中] = flatten[33] >= 3
        yaku_best[Yaku.小三元] = np.all(flatten[31:34] >= 2) & (
            three_dragons >= 2
        )
        yaku_best[Yaku.場風] = flatten[27] >= 3
        yaku_best[Yaku.自風] = flatten[27] >= 3
        yaku_best[Yaku.門前清自摸和] = is_menzen & (is_ron == 0)
        yaku_best[Yaku.立直] = riichi

        yakuman = np.full(Yaku.FAN.shape[1], False)

        yakuman[Yaku.大三元] = three_dragons == 3
        yakuman[Yaku.小四喜] = np.all(flatten[27:31] >= 2) & (four_winds == 3)
        yakuman[Yaku.大四喜] = four_winds == 4
        yakuman[Yaku.九蓮宝燈] = np.any(nine_gates)
        yakuman[Yaku.国士無双] = (
            (hand[0] > 0)
            & (hand[8] > 0)
            & (hand[9] > 0)
            & (hand[17] > 0)
            & (hand[18] > 0)
            & np.all(hand[26:] > 0)
            & (has_tanyao == 0)
        )
        yakuman[Yaku.清老頭] = (has_tanyao == 0) & (has_honor == 0)
        yakuman[Yaku.字一色] = np.all(flatten[0:27] == 0)
        yakuman[Yaku.緑一色] = (
            np.all(flatten[0:19] == 0)
            & (flatten[22] == 0)
            & (flatten[24] == 0)
            & np.all(flatten[26:32] == 0)
            & (flatten[33] == 0)
        )
        yakuman[Yaku.四暗刻] = np.any(n_concealed_pung == 4)

        if np.any(yakuman):
            return (yakuman, 0, 0)
        return yaku_best, np.dot(fan, yaku_best), fu_best

    @staticmethod
    def flatten(
        hand: np.ndarray, melds: np.ndarray, n_meld: int
    ) -> np.ndarray:
        for i in range(n_meld):
            hand = Yaku._flatten(hand, melds[i])
        return hand

    @staticmethod
    def _flatten(hand: np.ndarray, meld: int) -> np.ndarray:
        target, action = Meld.target(meld), Meld.action(meld)
        if Action.is_selfkan(action):
            return Hand.add(hand, target, 4)
        if action == Action.PON:
            return Hand.add(hand, target, 3)
        if action == Action.MINKAN:
            return Hand.add(hand, target, 4)
        if action == Action.CHI_L:
            return Hand.add(
                Hand.add(Hand.add(hand, target + 1), target + 2), target
            )
        if action == Action.CHI_M:
            return Hand.add(
                Hand.add(Hand.add(hand, target - 1), target + 1), target
            )
        if action == Action.CHI_R:
            return Hand.add(
                Hand.add(Hand.add(hand, target - 2), target - 1), target
            )
        assert False
