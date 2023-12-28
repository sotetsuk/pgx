import json
import os

import jax
import jax.numpy as jnp

from pgx._src.types import Array
from pgx.mahjong._action import Action
from pgx.mahjong._hand import Hand
from pgx.mahjong._meld import Meld

DIR = os.path.join(os.path.dirname(__file__), "cache")


def load_yaku_cache():
    with open(os.path.join(DIR, "yaku_cache.json")) as f:
        return jnp.array(json.load(f), dtype=jnp.uint32)


class Yaku:
    CACHE = load_yaku_cache()
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
    def score(
        hand: Array,
        melds: Array,
        n_meld: Array,
        last: Array,
        riichi: Array,
        is_ron: Array,
        dora: Array,
    ) -> int:
        """handはlast_tileを加えたもの"""
        yaku, fan, fu = Yaku.judge(hand, melds, n_meld, last, riichi, is_ron, dora)
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
    def head(code) -> Array:
        return Yaku.CACHE[code] & 0b1111

    @staticmethod
    def chow(code) -> Array:
        return Yaku.CACHE[code] >> 4 & 0b1111111

    @staticmethod
    def pung(code) -> Array:
        return Yaku.CACHE[code] >> 11 & 0b111111111

    @staticmethod
    def n_pung(code) -> Array:
        return Yaku.CACHE[code] >> 20 & 0b111

    @staticmethod
    def n_double_chow(code) -> Array:
        return Yaku.CACHE[code] >> 23 & 0b11

    @staticmethod
    def outside(code) -> Array:
        return Yaku.CACHE[code] >> 25 & 1

    @staticmethod
    def nine_gates(code) -> Array:
        return Yaku.CACHE[code] >> 26

    @staticmethod
    def is_pure_straight(chow: Array) -> Array:
        return (
            ((chow & 0b1001001) == 0b1001001)
            | ((chow >> 9 & 0b1001001) == 0b1001001)
            | ((chow >> 18 & 0b1001001) == 0b1001001)
        ) == 1

    @staticmethod
    def is_triple_chow(chow: Array) -> Array:
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
    def is_triple_pung(pung: Array) -> Array:
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
        is_pinfu: Array,
        is_outside: Array,
        n_double_chow: Array,
        all_chow: Array,
        all_pung: Array,
        n_concealed_pung: Array,
        nine_gates: Array,
        fu: Array,
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

        loss = is_ron & in_range & ((chow_range >> pos & 1) == 0) & (pung >> pos & 1)
        # ロンして明刻扱いになってしまう場合

        n_concealed_pung += n_pung - loss

        nine_gates |= Yaku.nine_gates(code) == 1

        outside_pung = pung & 0b100000001

        strong = in_range & ((1 << Yaku.head(code)) | ((chow & 1) << 2) | (chow & 0b1000000) | (chow << 1)) >> pos & 1
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
        hand: Array,
        melds: Array,
        n_meld: Array,
        last,
        riichi,
        is_ron,
        dora,
    ):
        is_menzen = jax.lax.fori_loop(
            jnp.int8(0),
            n_meld,
            lambda i, menzen: menzen & (Action.is_selfkan(Meld.action(melds[i])) & (Meld.src(melds[i]) == 0)),
            True,
        )

        is_pinfu = jnp.full(
            Yaku.MAX_PATTERNS,
            is_menzen & jnp.all(hand[28:31] < 3) & (hand[27] == 0) & jnp.all(hand[31:34] == 0),
        )
        # NOTE: 東場東家

        is_outside = jnp.full(
            Yaku.MAX_PATTERNS,
            jax.lax.fori_loop(
                jnp.int8(0),
                n_meld,
                lambda i, valid: valid & Meld.is_outside(melds[i]),
                True,
            ),
        )
        n_double_chow = jnp.full(Yaku.MAX_PATTERNS, 0)

        all_chow = jnp.full(
            Yaku.MAX_PATTERNS,
            jax.lax.fori_loop(
                jnp.int8(0),
                n_meld,
                lambda i, chow: chow | Meld.chow(melds[i]),
                0,
            ),
        )
        all_pung = jnp.full(
            Yaku.MAX_PATTERNS,
            jax.lax.fori_loop(
                jnp.int8(0),
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
            + jax.lax.fori_loop(jnp.int8(0), n_meld, lambda i, sum: sum + Meld.fu(melds[i]), 0)
            + (hand[27] == 2) * 4
            + jnp.any(hand[31:] == 2) * 2
            + (hand[27] == 3) * 4 * (2 - (is_ron & (27 == last)))
            + (hand[31] == 3) * 4 * (2 - (is_ron & (31 == last)))
            + (hand[32] == 3) * 4 * (2 - (is_ron & (32 == last)))
            + (hand[33] == 3) * 4 * (2 - (is_ron & (33 == last)))
            # NOTE: 東場東家
            + ((27 <= last) & (hand[last] == 2)),
            dtype=jnp.int32,
        )

        def _update_yaku(suit, tpl):
            code = jax.lax.fori_loop(
                9 * suit,
                9 * (suit + 1),
                lambda i, code: code * 5 + hand[i].astype(int),
                0,
            )
            return Yaku.update(
                tpl[0],
                tpl[1],
                tpl[2],
                tpl[3],
                tpl[4],
                tpl[5],
                tpl[6],
                tpl[7],
                code,
                suit,
                last,
                is_ron,
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
        ) = jax.lax.fori_loop(
            0,
            3,
            _update_yaku,
            (
                is_pinfu,
                is_outside,
                n_double_chow,
                all_chow,
                all_pung,
                n_concealed_pung,
                nine_gates,
                fu,
            ),
        )
        # for suit in range(3):
        #    code = jax.lax.fori_loop(
        #        9 * suit,
        #        9 * (suit + 1),
        #        lambda i, code: code * 5 + hand[i].astype(int),
        #        0,
        #    )
        #    (
        #        is_pinfu,
        #        is_outside,
        #        n_double_chow,
        #        all_chow,
        #        all_pung,
        #        n_concealed_pung,
        #        nine_gates,
        #        fu,
        #    ) = Yaku.update(
        #        is_pinfu,
        #        is_outside,
        #        n_double_chow,
        #        all_chow,
        #        all_pung,
        #        n_concealed_pung,
        #        nine_gates,
        #        fu,
        #        code,
        #        suit,
        #        last,
        #        is_ron,
        #    )

        n_concealed_pung += jnp.sum(hand[27:] >= 3) - (is_ron & (last >= 27) & (hand[last] >= 3))

        fu *= is_pinfu == 0
        fu += 20 + 10 * (is_menzen & is_ron)
        fu += 10 * ((is_menzen == 0) & (fu == 20))

        flatten = Yaku.flatten(hand, melds, n_meld)

        four_winds = jnp.sum(flatten[27:31] >= 3)
        three_dragons = jnp.sum(flatten[31:34] >= 3)

        has_tanyao = jnp.any(flatten[1:8]) | jnp.any(flatten[10:17]) | jnp.any(flatten[19:26])
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
            lambda: (
                yaku_best,
                jnp.dot(fan, yaku_best) + jnp.dot(flatten, dora),
                fu_best,
            ),
        )

    @staticmethod
    def flatten(hand: Array, melds: Array, n_meld) -> Array:
        return jax.lax.fori_loop(
            jnp.int8(0),
            n_meld,
            lambda i, arr: Yaku._flatten(arr, melds[i]),
            hand,
        )

    @staticmethod
    def _flatten(hand: Array, meld) -> Array:
        target, action = Meld.target(meld), Meld.action(meld)
        return jax.lax.switch(
            action - Action.PON + 1,
            [
                lambda: Hand.add(hand, target, 4),
                lambda: Hand.add(hand, target, 3),
                lambda: Hand.add(hand, target, 4),
                lambda: Hand.add(Hand.add(Hand.add(hand, target + 1), target + 2), target),
                lambda: Hand.add(Hand.add(Hand.add(hand, target - 1), target + 1), target),
                lambda: Hand.add(Hand.add(Hand.add(hand, target - 2), target - 1), target),
            ],
        )
