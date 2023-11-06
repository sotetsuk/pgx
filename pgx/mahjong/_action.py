from typing import NamedTuple


class Action(NamedTuple):
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
    def is_selfkan(action: int) -> bool:
        return (34 <= action) & (action < 68)


"""
discard
 0  1  2  3  4  5  6  7  8
 9 10 11 12 13 14 15 16 17
18 19 20 21 22 23 24 25 26
27 28 29 30 31 32 33

ankan/kakan
34 35 36 37 38 39 40 41 42
43 44 45 46 47 48 49 50 51
52 53 54 55 56 57 58 59 60
61 62 63 64 65 66 67
"""
