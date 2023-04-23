import re

import jax

from pgx._mahjong._action import Action


class Meld:
    @staticmethod
    def init(action, target, src) -> int:
        # src: 相対位置
        # - 0: 自分(暗槓の場合のみ)
        # - 1: 下家
        # - 2: 対面
        # - 3: 上家
        return (src << 13) | (target << 7) | action

    @staticmethod
    def to_str(meld) -> str:
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
    def src(meld) -> int:
        return meld >> 13 & 0b11

    @staticmethod
    def target(meld) -> int:
        return (meld >> 7) & 0b111111

    @staticmethod
    def action(meld) -> int:
        return meld & 0b1111111

    @staticmethod
    def suited_pung(meld) -> int:
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
    def chow(meld) -> int:
        action = Meld.action(meld)
        is_chi = (Action.CHI_L <= action) & (action <= Action.CHI_R)
        pos = Meld.target(meld) - (
            action - Action.CHI_L
        )  # WARNING: may be negative

        pos *= is_chi
        return is_chi << pos

    @staticmethod
    def is_outside(meld) -> int:
        action = Meld.action(meld)
        target = Meld.target(meld)
        is_chi = (Action.CHI_L <= action) & (action <= Action.CHI_R)

        return jax.lax.cond(
            is_chi,
            lambda: Meld._is_outside(target - (action - Action.CHI_L))
            | Meld._is_outside(target - (action - Action.CHI_L) + 2),
            lambda: Meld._is_outside(target),
        )

    @staticmethod
    def fu(meld) -> int:
        action = Meld.action(meld)

        fu = (
            (action == Action.PON) * 2
            + (action == Action.MINKAN) * 8
            + (Action.is_selfkan(action) * 8 * (1 + (Meld.src(meld) == 0)))
        )

        return fu * (1 + (Meld._is_outside(Meld.target(meld))))

    @staticmethod
    def _is_outside(tile) -> bool:
        num = tile % 9
        return (tile >= 27) | (num == 0) | (num == 8)
