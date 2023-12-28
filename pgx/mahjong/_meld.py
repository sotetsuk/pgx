import jax
import jax.numpy as jnp

from pgx.mahjong._action import Action


class Meld:
    @staticmethod
    def init(action, target, src):
        # src: 相対位置
        # - 0: 自分(暗槓の場合のみ)
        # - 1: 下家
        # - 2: 対面
        # - 3: 上家
        return (jnp.int32(src) << 13) | (jnp.int32(target) << 7) | jnp.int32(action)

    @staticmethod
    def to_str(meld) -> str:
        action = Meld.action(meld)
        target = Meld.target(meld)
        src = Meld.src(meld)
        suit, num = target // 9, target % 9 + 1

        if action == Action.PON:
            if src == 1:
                return "{}{}[{}]{}".format(num, num, num, ["m", "p", "s", "z"][suit])
            elif src == 2:
                return "{}[{}]{}{}".format(num, num, num, ["m", "p", "s", "z"][suit])
            elif src == 3:
                return "[{}]{}{}{}".format(num, num, num, ["m", "p", "s", "z"][suit])
        elif Action.is_selfkan(action):
            if src == 0:
                return "{}{}{}{}{}".format(num, num, num, num, ["m", "p", "s", "z"][suit])
            if src == 1:
                return "{}{}[{}{}]{}".format(num, num, num, num, ["m", "p", "s", "z"][suit])
            elif src == 2:
                return "{}[{}{}]{}{}".format(num, num, num, num, ["m", "p", "s", "z"][suit])
            elif src == 3:
                return "[{}{}]{}{}{}".format(num, num, num, num, ["m", "p", "s", "z"][suit])
        elif action == Action.MINKAN:
            if src == 1:
                return "{}{}{}[{}]{}".format(num, num, num, num, ["m", "p", "s", "z"][suit])
            elif src == 2:
                return "{}[{}]{}{}{}".format(num, num, num, num, ["m", "p", "s", "z"][suit])
            elif src == 3:
                return "[{}]{}{}{}{}".format(num, num, num, num, ["m", "p", "s", "z"][suit])
        elif Action.CHI_L <= action <= Action.CHI_R:
            assert src == 3
            pos = action - Action.CHI_L
            t = [num - pos + i for i in range(3)]
            t.insert(0, t.pop(pos))
            return "[{}]{}{}{}".format(*t, ["m", "p", "s", "z"][suit])
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
        is_pung = (action == Action.PON) | (action == Action.MINKAN) | Action.is_selfkan(action)
        is_suited_pon = is_pung & (target < 27)

        return is_suited_pon << target

    @staticmethod
    def chow(meld) -> int:
        action = Meld.action(meld)
        is_chi = (Action.CHI_L <= action) & (action <= Action.CHI_R)
        pos = Meld.target(meld) - (action - Action.CHI_L)  # WARNING: may be negative

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
