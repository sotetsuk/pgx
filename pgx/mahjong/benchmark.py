import time

import jax.numpy as jnp
from full_mahjong import Hand, Meld, Tile, Yaku


def yaku():
    start = time.time()

    hand = Hand.from_str("11223388m123p")
    melds = jnp.zeros(4, dtype=jnp.int32)
    melds = melds.at[0].set(Meld.from_str("[2]13s"))
    meld_num = 1
    last = Tile.from_str("1m")
    Yaku.judge(hand, melds, meld_num, last, riichi=False, is_ron=True)

    print("compile: {:.6}[s]".format(time.time() - start))

    start = time.time()
    for i in range(100000):
        Yaku.judge(hand, melds, meld_num, last, riichi=False, is_ron=True)
    print("{:.6}[s] per 100000".format(time.time() - start))


if __name__ == "__main__":
    yaku()
