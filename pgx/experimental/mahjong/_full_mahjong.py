from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

import numpy as np


class Tile:
    # 123456789m123456789p123456789s1234567z0m0p0s
    @staticmethod
    def num(tile: int) -> int:
        return tile % 9 if tile < 34 else 4

    @staticmethod
    def suit(tile: int) -> int:
        return tile // 9 if tile < 34 else tile - 34

    @staticmethod
    def unred(tile: int) -> int:
        return tile if tile < 34 else 4 + 9 * (tile - 34)

    @staticmethod
    def is_red(tile: int) -> bool:
        return tile >= 34

    @staticmethod
    def is_outside(tile: int) -> bool:
        num = Tile.num(tile)
        suit = Tile.suit(tile)
        return (suit == 3) | (num == 0) | (num == 8)

    @staticmethod
    def next(tile: int) -> int:
        num = Tile.num(tile)
        suit = Tile.suit(tile)
        if suit < 3:
            return (num + 1) % 9 + 9 * suit
        elif num < 4:
            return (num + 1) % 4 + 27
        return num % 3 + 31

    @staticmethod
    def to_str(tile: int) -> str:
        num = str(tile % 9 + 1 if tile < 34 else 0)
        suit = ["m", "p", "s", "z"][Tile.suit(tile)]
        return num + suit

    @staticmethod
    def from_str(s: str) -> int:
        num = int(s[0])
        suit = ["m", "p", "s", "z"].index(s[1])
        return 34 + suit if num == 0 else num - 1 + 9 * suit


class Action:
    # 手出し: 0~36
    # 加槓: 37~73
    # 暗槓: 74~107
    TSUMOGIRI = 108
    RIICHI = 109
    TSUMO = 110

    RON = 111
    PON = 112
    PON_EXPOSE_RED = 113
    MINKAN = 114
    CHI_L = 115
    CHI_L_EXPOSE_RED = 116
    CHI_M = 117
    CHI_M_EXPOSE_RED = 118
    CHI_R = 119
    CHI_R_EXPOSE_RED = 120
    PASS = 121

    NONE = 122

    @staticmethod
    def is_discard(action: int) -> bool:
        return action < 37

    @staticmethod
    def is_kakan(action: int) -> bool:
        return (37 <= action) & (action <= 73)

    @staticmethod
    def kakan(tile: int) -> int:
        return tile + 37

    @staticmethod
    def kakan_tile(action: int) -> int:
        assert Action.is_kakan(action)
        return action - 37

    @staticmethod
    def is_ankan(action: int) -> bool:
        return (74 <= action) & (action <= 107)

    @staticmethod
    def ankan(tile: int) -> int:
        assert 0 <= tile < 34
        return tile + 74

    @staticmethod
    def ankan_tile(action: int) -> int:
        assert Action.is_ankan(action)
        return action - 74

    @staticmethod
    def is_pon(action: int) -> bool:
        return (action == Action.PON) | (action == Action.PON_EXPOSE_RED)

    @staticmethod
    def is_chi_l(action: int) -> bool:
        return (action == Action.CHI_L) | (action == Action.CHI_L_EXPOSE_RED)

    @staticmethod
    def is_chi_m(action: int) -> bool:
        return (action == Action.CHI_M) | (action == Action.CHI_M_EXPOSE_RED)

    @staticmethod
    def is_chi_r(action: int) -> bool:
        return (action == Action.CHI_R) | (action == Action.CHI_R_EXPOSE_RED)

    @staticmethod
    def is_steal(action: int) -> bool:
        return (Action.PON <= action) & (action <= Action.CHI_R_EXPOSE_RED)


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

    @staticmethod
    def load_shanten_cache():
        with open(os.path.join(CacheLoader.DIR, "shanten_cache.json")) as f:
            return np.array(json.load(f), dtype=np.int32)


class Hand:
    # 純手牌を管理するクラス

    CACHE = CacheLoader.load_hand_cache()

    @staticmethod
    def cache(code: int) -> int:
        return (Hand.CACHE[code >> 5] >> (code & 0b11111)) & 1

    @staticmethod
    def can_ron(hand: np.ndarray, tile: int) -> bool:
        assert np.sum(hand) % 3 == 1
        return Hand.can_tsumo(Hand._add(hand, Tile.unred(tile)))

    @staticmethod
    def can_riichi(hand: np.ndarray) -> bool:
        assert np.sum(hand) % 3 == 2
        for tile in range(34):
            if hand[tile] > 0 and Hand.is_tenpai(Hand._sub(hand, tile)):
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
    def can_steal(
        hand: np.ndarray, red: np.ndarray, tile: int, action: int
    ) -> bool:
        assert np.sum(hand) % 3 == 1

        tile = Tile.unred(tile)
        num = Tile.num(tile)
        suit = Tile.suit(tile)
        has_red = red[suit] if suit < 3 else False

        if action == Action.PON:
            return hand[tile] >= 2 + has_red
        if action == Action.PON_EXPOSE_RED:
            return (hand[tile] >= 2) & has_red
        if action == Action.MINKAN:
            return hand[tile] >= 3
        if action == Action.CHI_L:
            if (suit == 3) | (num >= 7):
                return False
            return (hand[tile + 1] > (num + 1 == 4) & has_red) & (
                hand[tile + 2] > (num + 2 == 4) & has_red
            )
        if action == Action.CHI_L_EXPOSE_RED:
            if (suit == 3) | (num >= 7):
                return False
            return (
                has_red
                & (hand[tile + 1] > 0)
                & (hand[tile + 2] > 0)
                & ((num + 1 == 4) | (num + 2 == 4))
            )
        if action == Action.CHI_M:
            if (suit == 3) | (num == 0) | (num == 8):
                return False
            return (hand[tile - 1] > (num - 1 == 4) & has_red) & (
                hand[tile + 1] > (num + 1 == 4) & has_red
            )
        if action == Action.CHI_M_EXPOSE_RED:
            if (suit == 3) | (num == 0) | (num == 8):
                return False
            return (
                has_red
                & (hand[tile - 1] > 0)
                & (hand[tile + 1] > 0)
                & ((num - 1 == 4) | (num + 1 == 4))
            )
        if action == Action.CHI_R:
            if (suit == 3) | (num <= 1):
                return False
            return (hand[tile - 2] > (num - 2 == 4) & has_red) & (
                hand[tile - 1] > (num - 1 == 4) & has_red
            )
        if action == Action.CHI_R_EXPOSE_RED:
            if (suit == 3) | (num <= 1):
                return False
            return (
                has_red
                & (hand[tile - 2] > 0)
                & (hand[tile - 1] > 0)
                & ((num - 2 == 4) | (num - 1 == 4))
            )

        assert False

    @staticmethod
    def can_kakan(hand: np.ndarray, red: np.ndarray, tile: int) -> bool:
        assert np.sum(hand) % 3 == 2
        if Tile.is_red(tile):
            return red[Tile.suit(tile)]
        else:
            return hand[tile] == 1

    @staticmethod
    def can_ankan(hand: np.ndarray, tile: int) -> bool:
        assert np.sum(hand) % 3 == 2
        assert 0 <= tile < 34
        return hand[tile] == 4

    @staticmethod
    def _add(hand: np.ndarray, tile: int, x: int = 1) -> np.ndarray:
        assert 0 <= tile < 34
        hand = hand.copy()
        hand[tile] += x
        return hand

    @staticmethod
    def _sub(hand: np.ndarray, tile: int, x: int = 1) -> np.ndarray:
        return Hand._add(hand, tile, -x)

    @staticmethod
    def add(
        hand: np.ndarray, red: np.ndarray, tile: int, x: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        hand = hand.copy()
        red = red.copy()

        hand[Tile.unred(tile)] += x
        if Tile.is_red(tile):
            if x == 1:
                assert not red[Tile.suit(tile)]
                red[Tile.suit(tile)] = True
            elif x == -1:
                assert red[Tile.suit(tile)]
                red[Tile.suit(tile)] = False
            else:
                assert False

        return hand, red

    @staticmethod
    def sub(
        hand: np.ndarray, red: np.ndarray, tile: int, x: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        return Hand.add(hand, red, tile, -x)

    @staticmethod
    def steal(
        hand: np.ndarray, red: np.ndarray, tile: int, action: int
    ) -> tuple[np.ndarray, np.ndarray]:
        assert Hand.can_steal(hand, red, tile, action)
        tile = Tile.unred(tile)
        if action == Action.PON:
            hand[tile] -= 2
        if action == Action.PON_EXPOSE_RED:
            hand[tile] -= 2
            red[Tile.suit(tile)] = False
        if action == Action.MINKAN:
            hand[tile] -= 3
        if action == Action.CHI_L:
            hand[tile + 1] -= 1
            hand[tile + 2] -= 1
        if action == Action.CHI_L_EXPOSE_RED:
            hand[tile + 1] -= 1
            hand[tile + 2] -= 1
            red[Tile.suit(tile)] = False
        if action == Action.CHI_M:
            hand[tile - 1] -= 1
            hand[tile + 1] -= 1
        if action == Action.CHI_M_EXPOSE_RED:
            hand[tile - 1] -= 1
            hand[tile + 1] -= 1
            red[Tile.suit(tile)] = False
        if action == Action.CHI_R:
            hand[tile - 2] -= 1
            hand[tile - 1] -= 1
        if action == Action.CHI_R_EXPOSE_RED:
            hand[tile - 2] -= 1
            hand[tile - 1] -= 1
            red[Tile.suit(tile)] = False

        return hand, red

    @staticmethod
    def kakan(
        hand: np.ndarray, red: np.ndarray, tile: int
    ) -> tuple[np.ndarray, np.ndarray]:
        assert Hand.can_kakan(hand, red, tile)
        hand, red = Hand.sub(hand, red, tile)
        suit = Tile.suit(tile)
        num = Tile.num(tile)
        if (suit < 3) & (num == 4):
            red[suit] = False
        return hand, red

    @staticmethod
    def ankan(
        hand: np.ndarray, red: np.ndarray, tile: int
    ) -> tuple[np.ndarray, np.ndarray]:
        assert Hand.can_ankan(hand, tile)
        hand, red = Hand.sub(hand, red, tile, 4)
        suit = Tile.suit(tile)
        num = Tile.num(tile)
        if (suit < 3) & (num == 4):
            red[suit] = False
        return hand, red

    @staticmethod
    def can_discard(
        hand: np.ndarray, red: np.ndarray, tile: int, last_draw: int
    ) -> bool:
        # 手出しできるかどうか
        suit = Tile.suit(tile)
        if Tile.is_red(tile):
            return red[suit] & (tile != last_draw)
        else:
            has_red = red[suit] if suit < 3 and Tile.num(tile) == 4 else False
            return hand[tile] > (tile == last_draw) + has_red

    @staticmethod
    def to_str(hand: np.ndarray, red: np.ndarray) -> str:
        s = ""
        for i in range(4):
            t = ""
            for j in range(9 if i < 3 else 7):
                if j == 4 and i < 3 and red[i]:
                    assert hand[9 * i + j] > 0
                    t += "0" + str(j + 1) * (hand[9 * i + j] - 1)
                else:
                    t += str(j + 1) * hand[9 * i + j]
            if t:
                t += ["m", "p", "s", "z"][i]
            s += t
        return s

    @staticmethod
    def from_str(s: str) -> tuple[np.ndarray, np.ndarray]:
        suit = ""
        hand = np.zeros(34, dtype=np.uint32)
        red = np.full(3, False)
        for c in reversed(s):
            if c in ["m", "p", "s", "z"]:
                suit = c
            else:
                hand, red = Hand.add(hand, red, Tile.from_str(c + suit))

        return hand, red


@dataclass
class Deck:
    # fmt: off
    DeckList = np.array([
         0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,34, 4,  # noqa
         4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8,  # noqa
         9, 9, 9, 9,10,10,10,10,11,11,11,11,12,12,12,12,35,13,  # noqa
        13,13,14,14,14,14,15,15,15,15,16,16,16,16,17,17,17,17,  # noqa
        18,18,18,18,19,19,19,19,20,20,20,20,21,21,21,21,36,22,  # noqa
        22,22,23,23,23,23,24,24,24,24,25,25,25,25,26,26,26,26,  # noqa
        27,27,27,27,28,28,28,28,29,29,29,29,30,30,30,30,31,31,  # noqa
        31,31,32,32,32,32,33,33,33,33,  # noqa
    ])
    # fmt: on

    arr: np.ndarray
    idx: int = 135
    end: int = 13
    n_dora: int = 1

    # 1   3 | 5 - 13 | 15 - 135
    # 0   2 | 4 - 12 | 14 - 134
    # -------------------------
    # 嶺上牌| ドラ   | to be used
    #       | 裏ドラ |

    def is_empty(self) -> bool:
        return self.size() == 0

    def size(self) -> int:
        return self.idx - self.end

    def dora(self, is_riichi: bool) -> np.ndarray:
        dora = np.zeros(34, dtype=np.uint32)
        for i in range(self.n_dora):
            dora[Tile.next(self.arr[5 + 2 * i])] += 1

        if is_riichi:
            for i in range(self.n_dora):
                dora[Tile.next(self.arr[4 + 2 * i])] += 1

        return dora

    @staticmethod
    def init() -> Deck:
        return Deck(np.random.permutation(Deck.DeckList))

    @staticmethod
    def deal(deck: Deck) -> tuple[Deck, np.ndarray, np.ndarray, int]:
        hand = np.zeros((4, 34), dtype=np.uint32)
        red = np.full((4, 3), False)
        for i in range(3):
            for j in range(4):
                for k in range(4):
                    hand[j], red[j] = Hand.add(
                        hand[j], red[j], deck.arr[-(16 * i + 4 * j + k + 1)]
                    )
        for j in range(4):
            hand[j], red[j] = Hand.add(
                hand[j], red[j], deck.arr[-(16 * 3 + j + 1)]
            )

        last_draw = deck.arr[-(16 * 3 + 4 + 1)]
        hand[0], red[0] = Hand.add(hand[0], red[0], last_draw)

        deck.idx -= 53

        return deck, hand, red, last_draw

    @staticmethod
    def draw(deck: Deck, is_kan: bool = False) -> tuple[Deck, int]:
        assert not deck.is_empty()
        tile = deck.arr[(deck.n_dora - 1) ^ 1 if is_kan else deck.idx]
        if is_kan:
            deck.end += 1
            deck.n_dora += 1  # NOTE: 先めくりで統一
        else:
            deck.idx -= 1
        return deck, tile


@dataclass
class Observation:
    hand: np.ndarray
    red: np.ndarray
    target: int
    last_draw: int
    riichi: np.ndarray


class Meld:
    @staticmethod
    def init(action: int, stolen: int, src: int) -> int:
        # src: 相対位置
        # - 0: 自分(暗槓の場合のみ)
        # - 1: 下家
        # - 2: 対面
        # - 3: 上家
        return (src << 13) | (stolen << 7) | action

    @staticmethod
    def src(meld: int) -> int:
        return meld >> 13 & 0b11

    @staticmethod
    def stolen(meld: int) -> int:
        return (meld >> 7) & 0b111111

    @staticmethod
    def action(meld: int) -> int:
        return meld & 0b1111111

    @staticmethod
    def suited_pung(meld: int) -> int:
        action = Meld.action(meld)

        if Action.is_ankan(action):
            tile = Action.ankan_tile(meld)
            return (tile < 27) << tile

        if (
            Action.is_kakan(action)
            | Action.is_pon(action)
            | (action == Action.MINKAN)
        ):
            stolen = Tile.unred(Meld.stolen(meld))
            return (stolen < 27) << stolen

        return 0

    @staticmethod
    def chow(meld: int) -> int:
        action = Meld.action(meld)
        stolen = Tile.unred(Meld.stolen(meld))
        if Action.is_chi_l(action):
            return 1 << stolen
        if Action.is_chi_m(action):
            return 1 << stolen - 1
        if Action.is_chi_r(action):
            return 1 << stolen - 2
        return 0

    @staticmethod
    def is_outside(meld: int) -> int:
        action = Meld.action(meld)

        if Action.is_ankan(action):
            return Tile.is_outside(Action.ankan_tile(meld))

        stolen = Tile.unred(Meld.stolen(meld))

        if Action.is_chi_l(action):
            return Tile.is_outside(stolen) | Tile.is_outside(stolen + 2)
        if Action.is_chi_m(action):
            return Tile.is_outside(stolen - 1) | Tile.is_outside(stolen + 1)
        if Action.is_chi_r(action):
            return Tile.is_outside(stolen - 2) | Tile.is_outside(stolen)

        return Tile.is_outside(stolen)

    @staticmethod
    def fu(meld: int) -> int:
        action = Meld.action(meld)

        if Action.is_ankan(action):
            tile = Action.ankan_tile(action)
            return 16 * (1 + Tile.is_outside(tile))

        stolen = Meld.stolen(meld)

        fu = (
            Action.is_pon(action) * 2
            + (action == Action.MINKAN) * 8
            + Action.is_kakan(action) * 8
        )

        return fu * (1 + (Tile.is_outside(stolen)))

    @staticmethod
    def pon_to_kakan(tile: int, pon: int) -> int:
        return Meld.init(
            Action.kakan(tile), stolen=Meld.stolen(pon), src=Meld.src(pon)
        )

    @staticmethod
    def _to_str_pon(src: int, suit: str, num: int, is_red: bool) -> str:
        stolen = 0 if is_red else num
        if src == 1:
            return f"{num}{num}[{stolen}]{suit}"
        elif src == 2:
            return f"{num}[{stolen}]{num}{suit}"
        elif src == 3:
            return f"[{stolen}]{num}{num}{suit}"
        assert False

    @staticmethod
    def _to_str_pon_expose_red(src: int, suit: str, num: int) -> str:
        if src == 1:
            return f"{0}{num}[{num}]{suit}"
        elif src == 2:
            return f"{0}[{num}]{num}{suit}"
        elif src == 3:
            return f"[{num}]{0}{num}{suit}"
        assert False

    @staticmethod
    def _to_str_minkan(src: int, suit: str, num: int, is_red: bool) -> str:
        stolen = 0 if is_red else num
        first = 0 if num == 5 and not is_red else num
        if src == 1:
            return f"{first}{num}{num}[{stolen}]{suit}"
        elif src == 2:
            return f"{first}[{stolen}]{num}{num}{suit}"
        elif src == 3:
            return f"[{stolen}]{first}{num}{num}{suit}"
        assert False

    @staticmethod
    def _to_str_chi_l(suit: str, num: int, is_red: bool) -> str:
        stolen = 0 if is_red else num
        return f"[{stolen}]{num + 1}{num + 2}{suit}"

    @staticmethod
    def _to_str_chi_l_expose_red(suit: str, num: int) -> str:
        first = 0 if num + 1 == 5 else num + 1
        second = 0 if num + 2 == 5 else num + 2
        return f"[{num}]{first}{second}{suit}"

    @staticmethod
    def _to_str_chi_m(suit: str, num: int, is_red: bool) -> str:
        stolen = 0 if is_red else num
        return f"[{stolen}]{num - 1}{num + 1}{suit}"

    @staticmethod
    def _to_str_chi_m_expose_red(suit: str, num: int) -> str:
        first = 0 if num - 1 == 5 else num - 1
        second = 0 if num + 1 == 5 else num + 1
        return f"[{num}]{first}{second}{suit}"

    @staticmethod
    def _to_str_chi_r(suit: str, num: int, is_red: bool) -> str:
        stolen = 0 if is_red else num
        return f"[{stolen}]{num - 2}{num - 1}{suit}"

    @staticmethod
    def _to_str_chi_r_expose_red(suit: str, num: int) -> str:
        first = 0 if num - 2 == 5 else num - 2
        second = 0 if num - 1 == 5 else num - 1
        return f"[{num}]{first}{second}{suit}"

    @staticmethod
    def _to_str_kakan(
        tile: int, src: int, suit: str, num: int, is_red: bool
    ) -> str:
        add_red = Tile.is_red(tile)
        first = 0 if num == 5 and suit != "z" else num
        if src == 1:
            if is_red:
                return f"{num}{num}[{0}{num}]{suit}"
            if add_red:
                return f"{num}{num}[{num}{0}]{suit}"
            return f"{first}{num}[{num}{num}]{suit}"
        if src == 2:
            if is_red:
                return f"{num}[{0}{num}]{num}{suit}"
            if add_red:
                return f"{num}[{num}{0}]{num}{suit}"
            return f"{first}[{num}{num}]{num}{suit}"
        if src == 3:
            if is_red:
                return f"[{0}{num}]{num}{num}{suit}"
            if add_red:
                return f"[{num}{0}]{num}{num}{suit}"
            return f"[{num}{num}]{first}{num}{suit}"
        assert False

    @staticmethod
    def _to_str_ankan(suit: str, num: int) -> str:
        first = 0 if num == 5 and suit != "z" else num
        return f"{first}{num}{num}{num}{suit}"

    @staticmethod
    def to_str(meld: int) -> str:
        action = Meld.action(meld)

        if Action.is_ankan(action):
            tile = Action.ankan_tile(action)
            suit = ["m", "p", "s", "z"][Tile.suit(tile)]
            num = Tile.num(tile) + 1
            return Meld._to_str_ankan(suit, num)

        stolen = Meld.stolen(meld)
        src = Meld.src(meld)
        suit = ["m", "p", "s", "z"][Tile.suit(stolen)]
        num = Tile.num(stolen) + 1
        is_red = Tile.is_red(stolen)

        if Action.is_kakan(action):
            return Meld._to_str_kakan(
                Action.kakan_tile(action), src, suit, num, is_red
            )
        elif action == Action.PON:
            return Meld._to_str_pon(src, suit, num, is_red)
        elif action == Action.PON_EXPOSE_RED:
            return Meld._to_str_pon_expose_red(src, suit, num)
        elif action == Action.MINKAN:
            return Meld._to_str_minkan(src, suit, num, is_red)
        elif action == Action.CHI_L:
            return Meld._to_str_chi_l(suit, num, is_red)
        elif action == Action.CHI_L_EXPOSE_RED:
            return Meld._to_str_chi_l_expose_red(suit, num)
        elif action == Action.CHI_M:
            return Meld._to_str_chi_m(suit, num, is_red)
        elif action == Action.CHI_M_EXPOSE_RED:
            return Meld._to_str_chi_m_expose_red(suit, num)
        elif action == Action.CHI_R:
            return Meld._to_str_chi_r(suit, num, is_red)
        elif action == Action.CHI_R_EXPOSE_RED:
            return Meld._to_str_chi_r_expose_red(suit, num)
        assert False

    @staticmethod
    def from_str(s: str) -> int:
        if g := re.match(r"^\[(\d)\](\d)(\d)([mpsz])$", s):
            stolen = Tile.from_str(g[1] + g[4])
            src = 3
            num = list(
                map(
                    Tile.num,
                    map(
                        Tile.from_str, [g[1] + g[4], g[2] + g[4], g[3] + g[4]]
                    ),
                )
            )
            expose_red = g[2] == "0" or g[3] == "0"
            if num[0] == num[1] and num[0] == num[2]:
                return Meld.init(
                    Action.PON_EXPOSE_RED if expose_red else Action.PON,
                    stolen,
                    src,
                )
            if num[0] + 1 == num[1] and num[0] + 2 == num[2]:
                return Meld.init(
                    Action.CHI_L_EXPOSE_RED if expose_red else Action.CHI_L,
                    stolen,
                    src,
                )
            if num[0] - 1 == num[1] and num[0] + 1 == num[2]:
                return Meld.init(
                    Action.CHI_M_EXPOSE_RED if expose_red else Action.CHI_M,
                    stolen,
                    src,
                )
            if num[0] - 2 == num[1] and num[0] - 1 == num[2]:
                return Meld.init(
                    Action.CHI_R_EXPOSE_RED if expose_red else Action.CHI_R,
                    stolen,
                    src,
                )

        if g := re.match(r"^(\d)\[(\d)\](\d)([mpsz])$", s):
            stolen = Tile.from_str(g[2] + g[4])
            src = 2
            expose_red = g[1] == "0" or g[3] == "0"
            action = Action.PON_EXPOSE_RED if expose_red else Action.PON
            return Meld.init(action, stolen, src)

        if g := re.match(r"^(\d)(\d)\[(\d)\]([mpsz])$", s):
            stolen = Tile.from_str(g[3] + g[4])
            src = 1
            expose_red = g[1] == "0" or g[2] == "0"
            action = Action.PON_EXPOSE_RED if expose_red else Action.PON
            return Meld.init(action, stolen, src)

        if g := re.match(r"^\[(\d)\](\d)(\d)(\d)([mpsz])$", s):
            stolen = Tile.from_str(g[1] + g[5])
            src = 3
            return Meld.init(Action.MINKAN, stolen, src)

        if g := re.match(r"^(\d)\[(\d)\](\d)(\d)([mpsz])$", s):
            stolen = Tile.from_str(g[2] + g[5])
            src = 2
            return Meld.init(Action.MINKAN, stolen, src)

        if g := re.match(r"^(\d)(\d)(\d)\[(\d)\]([mpsz])$", s):
            stolen = Tile.from_str(g[4] + g[5])
            src = 1
            return Meld.init(Action.MINKAN, stolen, src)

        if g := re.match(r"^\[(\d)(\d)\](\d)(\d)([mpsz])$", s):
            action = Action.kakan(Tile.from_str(g[2] + g[5]))
            stolen = Tile.from_str(g[1] + g[5])
            src = 3
            return Meld.init(action, stolen, src)

        if g := re.match(r"^(\d)\[(\d)(\d)\](\d)([mpsz])$", s):
            action = Action.kakan(Tile.from_str(g[3] + g[5]))
            stolen = Tile.from_str(g[2] + g[5])
            src = 2
            return Meld.init(action, stolen, src)

        if g := re.match(r"^(\d)(\d)\[(\d)(\d)\]([mpsz])$", s):
            action = Action.kakan(Tile.from_str(g[4] + g[5]))
            stolen = Tile.from_str(g[3] + g[5])
            src = 1
            return Meld.init(action, stolen, src)

        if g := re.match(r"^(\d)(\d)(\d)(\d)([mpsz])$", s):
            action = Action.ankan(Tile.unred(Tile.from_str(g[1] + g[5])))
            stolen = 0
            src = 0
            return Meld.init(action, stolen, src)

        assert False


@dataclass
class State:
    deck: Deck
    hand: np.ndarray
    red: np.ndarray
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
    pon_idx: np.ndarray
    # pon_idx[i][j]: player i がj(0<j<34) のポンを所有している場合, そのindex + 1. or 0

    def can_kakan(self, tile: int) -> bool:
        return (
            self.pon_idx[(self.turn, Tile.unred(tile))] > 0
        ) & Hand.can_kakan(self.hand[self.turn], self.red[self.turn], tile)

    def can_tsumo(self) -> bool:
        if Hand.can_tsumo(self.hand[self.turn]):
            fan = Yaku.judge(
                self.hand[self.turn],
                self.melds[self.turn],
                self.n_meld[self.turn],
                Tile.unred(self.last_draw),
                self.riichi[self.turn],
                False,
            )[1]
            return fan > 0
        return False

    def can_ron(self, player: int) -> bool:
        if Hand.can_ron(self.hand[player], self.target):
            fan = Yaku.judge(
                Hand._add(self.hand[self.turn], Tile.unred(self.target)),
                self.melds[self.turn],
                self.n_meld[self.turn],
                Tile.unred(self.last_draw),
                self.riichi[self.turn],
                True,
            )[1]
            return fan > 0
        return False

    def legal_actions(self) -> np.ndarray:
        legal_actions = np.full((4, Action.NONE), False)

        # 暗/加槓, ツモ切り, ツモ, リーチ
        if (self.last_draw != -1) & (not self.riichi_declared):
            for tile in range(37):
                legal_actions[(self.turn, Action.kakan(tile))] = (
                    self.can_kakan(tile)
                    & (self.deck.n_dora < 5)  # 5回目の槓はできない
                    & (self.deck.is_empty() == 0)
                )
            for tile in range(34):
                legal_actions[(self.turn, Action.ankan(tile))] = (
                    Hand.can_ankan(self.hand[self.turn], tile)
                    & (self.deck.n_dora < 5)  # 5回目の槓はできない
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
            for tile in range(37):
                if Hand.can_discard(
                    self.hand[self.turn],
                    self.red[self.turn],
                    tile,
                    self.last_draw,
                ):
                    legal_actions[(self.turn, tile)] = Hand.is_tenpai(
                        Hand._sub(self.hand[self.turn], Tile.unred(tile))
                    )
            legal_actions[(self.turn, Action.TSUMOGIRI)] = Hand.is_tenpai(
                Hand._sub(self.hand[self.turn], Tile.unred(self.last_draw))
            )

        # 手出し, 鳴いた後の手出し
        if (
            (self.target == -1)
            & (not self.riichi_declared)
            & (not self.riichi[self.turn])
        ):
            for tile in range(37):
                legal_actions[(self.turn, tile)] = Hand.can_discard(
                    self.hand[self.turn],
                    self.red[self.turn],
                    tile,
                    self.last_draw,
                )

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

                legal_actions[(player, Action.PON)] = Hand.can_steal(
                    self.hand[player],
                    self.red[player],
                    self.target,
                    Action.PON,
                )
                legal_actions[
                    (player, Action.PON_EXPOSE_RED)
                ] = Hand.can_steal(
                    self.hand[player],
                    self.red[player],
                    self.target,
                    Action.PON_EXPOSE_RED,
                )

                legal_actions[(player, Action.MINKAN)] = Hand.can_steal(
                    self.hand[player],
                    self.red[player],
                    self.target,
                    Action.MINKAN,
                ) & (self.deck.n_dora < 5)

                if (player - self.turn) % 4 != 1:
                    continue
                legal_actions[(player, Action.CHI_L)] = Hand.can_steal(
                    self.hand[player],
                    self.red[player],
                    self.target,
                    Action.CHI_L,
                )
                legal_actions[
                    (player, Action.CHI_L_EXPOSE_RED)
                ] = Hand.can_steal(
                    self.hand[player],
                    self.red[player],
                    self.target,
                    Action.CHI_L_EXPOSE_RED,
                )
                legal_actions[(player, Action.CHI_M)] = Hand.can_steal(
                    self.hand[player],
                    self.red[player],
                    self.target,
                    Action.CHI_M,
                )
                legal_actions[
                    (player, Action.CHI_M_EXPOSE_RED)
                ] = Hand.can_steal(
                    self.hand[player],
                    self.red[player],
                    self.target,
                    Action.CHI_M_EXPOSE_RED,
                )
                legal_actions[(player, Action.CHI_R)] = Hand.can_steal(
                    self.hand[player],
                    self.red[player],
                    self.target,
                    Action.CHI_R,
                )
                legal_actions[
                    (player, Action.CHI_R_EXPOSE_RED)
                ] = Hand.can_steal(
                    self.hand[player],
                    self.red[player],
                    self.target,
                    Action.CHI_R_EXPOSE_RED,
                )

            for player in range(4):
                if player == self.turn:
                    continue
                legal_actions[(player, Action.PASS)] = np.any(
                    legal_actions[player]
                )

        return legal_actions

    def observe(self, player: int) -> Observation:
        return Observation(
            self.hand[player],
            self.red[player],
            self.target,
            self.last_draw,
            self.riichi,
        )

    @staticmethod
    def init() -> State:
        return State.init_with_deck(Deck.init())

    @staticmethod
    def init_with_deck_arr(arr: np.ndarray) -> State:
        return State.init_with_deck(Deck(arr))

    @staticmethod
    def init_with_deck(deck: Deck) -> State:
        deck, hand, red, last_draw = Deck.deal(deck)

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
            red,
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
        if Action.is_discard(action):
            return State._discard(state, action)
        if Action.is_kakan(action):
            return State._kakan(state, action)
        if Action.is_ankan(action):
            return State._ankan(state, action)
        if action == Action.TSUMOGIRI:
            return State._tsumogiri(state)
        if action == Action.RIICHI:
            return State._riichi(state)
        if action == Action.TSUMO:
            return State._tsumo(state)
        if action == Action.RON:
            return State._ron(state, player)
        if Action.is_steal(action):
            return State._steal(state, player, action)
        if action == Action.PASS:
            return State._try_draw(state)
        assert False

    @staticmethod
    def _tsumogiri(state: State) -> tuple[State, np.ndarray, bool]:
        return State._discard(state, state.last_draw)

    @staticmethod
    def _discard(state: State, tile: int) -> tuple[State, np.ndarray, bool]:
        state.hand[state.turn], state.red[state.turn] = Hand.sub(
            state.hand[state.turn], state.red[state.turn], tile
        )
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
        state.hand[state.turn], state.red[state.turn] = Hand.add(
            state.hand[state.turn], state.red[state.turn], tile
        )
        return state, np.zeros(4, dtype=np.int32), False

    @staticmethod
    def _draw_rinshan(state: State) -> State:
        state.deck, tile = Deck.draw(state.deck, is_kan=True)
        state.last_draw = tile
        state.hand[state.turn], state.red[state.turn] = Hand.add(
            state.hand[state.turn], state.red[state.turn], tile
        )
        return state

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
        last = Tile.unred(state.target)
        is_ron = True
        score = Yaku.score(
            *Hand.add(state.hand[player], state.red[player], state.target),
            state.melds[player],
            state.n_meld[player],
            last,
            state.riichi[player],
            is_ron,
            state.deck.dora(state.riichi[player]),
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
    def _steal(
        state: State, player: int, action: int
    ) -> tuple[State, np.ndarray, bool]:
        state = State._accept_riichi(state)
        src = (state.turn - player) % 4
        meld = Meld.init(action, state.target, src)
        state = State._append_meld(state, meld, player)
        state.hand[player], state.red[player] = Hand.steal(
            state.hand[player], state.red[player], state.target, action
        )
        state.target = -1
        state.turn = player
        state.is_menzen[player] = False

        if Action.is_pon(action):
            state.pon_idx[(player, Tile.unred(state.target))] = state.n_meld[
                player
            ]

        if action == Action.MINKAN:
            state = State._draw_rinshan(state)

        return state, np.zeros(4, dtype=np.int32), False

    @staticmethod
    def _ankan(state: State, action: int) -> tuple[State, np.ndarray, bool]:
        meld = Meld.init(action, stolen=0, src=0)
        state = State._append_meld(state, meld, state.turn)
        state.hand[state.turn], state.red[state.turn] = Hand.ankan(
            state.hand[state.turn],
            state.red[state.turn],
            Action.ankan_tile(action),
        )

        # TODO: 国士無双ロンの受付

        state = State._draw_rinshan(state)
        return state, np.zeros(4, dtype=np.int32), False

    @staticmethod
    def _kakan(state: State, action: int) -> tuple[State, np.ndarray, bool]:
        tile = Action.kakan_tile(action)
        pon_idx = state.pon_idx[(state.turn, Tile.unred(tile))]
        assert pon_idx > 0

        pon = state.melds[(state.turn, pon_idx - 1)]
        kakan = Meld.pon_to_kakan(tile, pon)
        state.melds[(state.turn, pon_idx - 1)] = kakan

        state.hand[state.turn], state.red[state.turn] = Hand.kakan(
            state.hand[state.turn], state.red[state.turn], tile
        )
        state.pon_idx[(state.turn, Tile.unred(tile))] = 0

        # TODO: 槍槓の受付

        state = State._draw_rinshan(state)
        return state, np.zeros(4, dtype=np.int32), False

    @staticmethod
    def _tsumo(state: State) -> tuple[State, np.ndarray, bool]:
        last = Tile.unred(state.last_draw)
        is_ron = False
        score = Yaku.score(
            state.hand[state.turn],
            state.red[state.turn],
            state.melds[state.turn],
            state.n_meld[state.turn],
            last,
            state.riichi[state.turn],
            is_ron,
            state.deck.dora(state.riichi[state.turn]),
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
        red: np.ndarray,
        melds: np.ndarray,
        n_meld: int,
        last: int,
        riichi: bool,
        is_ron: bool,
        dora: np.ndarray,
    ) -> int:
        assert Hand.can_tsumo(hand)
        flatten, n_red = Yaku.flatten(hand, melds, n_meld)
        n_red += np.sum(red)
        yaku, fan, fu = Yaku._judge(
            hand, melds, n_meld, last, riichi, is_ron, flatten
        )
        score = fu << (fan + np.dot(flatten, dora) + n_red + 2)
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
        # Returns: yaku, fan, fu
        flatten, _ = Yaku.flatten(hand, melds, n_meld)
        return Yaku._judge(hand, melds, n_meld, last, riichi, is_ron, flatten)

    @staticmethod
    def _judge(
        hand: np.ndarray,
        melds: np.ndarray,
        n_meld: int,
        last: int,
        riichi: bool,
        is_ron: bool,
        flatten: np.ndarray,
    ) -> tuple[np.ndarray, int, int]:
        assert 0 <= last < 34

        is_menzen = True
        for i in range(n_meld):
            is_menzen &= Action.is_ankan(Meld.action(melds[i]))

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
    ) -> tuple[np.ndarray, int]:
        n_red = 0
        for i in range(n_meld):
            hand, has_red = Yaku._flatten(hand, melds[i])
            n_red += has_red

        return hand, n_red

    @staticmethod
    def _flatten(hand: np.ndarray, meld: int) -> tuple[np.ndarray, bool]:
        hand = hand.copy()

        action = Meld.action(meld)

        if Action.is_ankan(action):
            tile = Action.ankan_tile(action)
            hand = Hand._add(hand, tile, 4)

        stolen = Meld.stolen(meld)
        tile = Tile.unred(stolen)

        if Action.is_kakan(action) | (action == Action.MINKAN):
            hand = Hand._add(hand, tile, 4)
            has_red = (Tile.suit(tile) < 3) & (Tile.num(tile) == 4)
            return hand, has_red

        has_red = (
            Tile.is_red(stolen)
            | (action == Action.PON_EXPOSE_RED)
            | (action == Action.CHI_L_EXPOSE_RED)
            | (action == Action.CHI_M_EXPOSE_RED)
            | (action == Action.CHI_R_EXPOSE_RED)
        )

        if Action.is_pon(action):
            hand = Hand._add(hand, tile, 3)
        if Action.is_chi_l(action):
            hand = Hand._add(
                Hand._add(Hand._add(hand, tile), tile + 1), tile + 2
            )
        if Action.is_chi_m(action):
            hand = Hand._add(
                Hand._add(Hand._add(hand, tile - 1), tile), tile + 1
            )
        if Action.is_chi_r(action):
            hand = Hand._add(
                Hand._add(Hand._add(hand, tile - 2), tile - 1), tile
            )

        return hand, has_red


class Shanten:
    # See the link below for the algorithm details.
    # https://github.com/sotetsuk/pgx/pull/123

    CACHE = CacheLoader.load_shanten_cache()

    @staticmethod
    def discard(hand: np.ndarray) -> np.ndarray:
        s = np.full(34, 0)
        for i in range(34):
            if hand[i] == 0:
                continue
            hand[i] -= 1
            s[i] = Shanten.number(hand)
            hand[i] += 1
        return s

    @staticmethod
    def number(hand: np.ndarray) -> int:
        return min(
            [
                Shanten.normal(hand),
                Shanten.seven_pairs(hand),
                Shanten.thirteen_orphan(hand),
            ]
        )

    @staticmethod
    def seven_pairs(hand: np.ndarray) -> int:
        n_pair = int(np.sum(hand >= 2))
        n_kind = int(np.sum(hand > 0))
        return 7 - n_pair + max(7 - n_kind, 0)

    @staticmethod
    def thirteen_orphan(hand: np.ndarray) -> int:
        n_pair = (
            int(hand[0] >= 2)
            + int(hand[8] >= 2)
            + int(hand[9] >= 2)
            + int(hand[17] >= 2)
            + int(hand[18] >= 2)
            + int(np.sum(hand[26:34] >= 2))
        )
        n_kind = (
            int(hand[0] > 0)
            + int(hand[8] > 0)
            + int(hand[9] > 0)
            + int(hand[17] > 0)
            + int(hand[18] > 0)
            + int(np.sum(hand[26:34] > 0))
        )
        return 14 - n_kind - (n_pair > 0)

    @staticmethod
    def normal(hand: np.ndarray) -> int:
        code = np.full(4, 0)
        for suit in range(3):
            for i in range(9 * suit, 9 * (suit + 1)):
                code[suit] = code[suit] * 5 + hand[i]
        for i in range(27, 34):
            code[3] = code[3] * 5 + hand[i]

        n_set = int(np.sum(hand)) // 3

        return min([Shanten._normal(code, n_set, suit) for suit in range(4)])

    @staticmethod
    def _normal(code: np.ndarray, n_set: int, head_suit: int) -> int:
        cost = Shanten.CACHE[code[head_suit]][4]
        idx = np.full(4, 0)
        idx[head_suit] = 5

        for _ in range(n_set):
            i = np.argmin(Shanten.CACHE[code][[0, 1, 2, 3], idx])
            cost += Shanten.CACHE[code][i][idx[i]]
            idx[i] += 1

        return cost
