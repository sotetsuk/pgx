from pgx._flax.struct import dataclass


@dataclass
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
    def is_selfkan(action: int) -> bool:
        return (34 <= action) & (action < 68)
