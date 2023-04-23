import sys
import time
import warnings
import jax
import jax.numpy as jnp

from pgx._mahjong._hand import Hand
from pgx._mahjong._yaku import Yaku
from pgx._mahjong._shanten import Shanten

warnings.simplefilter("ignore")


def test_hand(func):
    # fmt:off
    hand= jnp.int8([
        1, 0, 0, 0, 0, 0, 0, 0, 1,
        1, 0, 0, 0, 0, 0, 0, 0, 1,
        1, 0, 0, 0, 0, 0, 0, 0, 1,
        1, 1, 1, 1, 1, 1, 1
    ])
    # fmt:on
    time_sta = time.perf_counter()
    jax.jit(func)(hand)
    time_end = time.perf_counter()
    delta = (time_end - time_sta) * 1000
    exp = jax.make_jaxpr(func)(hand)
    n_line = len(str(exp).split("\n"))
    print(f"| `{func.__name__}` | {n_line} | {delta:.1f}ms |")


def test_yaku(func):
    # fmt:off
    hand= jnp.int8([
        1, 0, 0, 0, 0, 0, 0, 0, 1,
        1, 0, 0, 0, 0, 0, 0, 0, 1,
        1, 0, 0, 0, 0, 0, 0, 0, 1,
        1, 1, 1, 1, 1, 1, 1
    ])
    # fmt:on
    time_sta = time.perf_counter()
    jax.jit(func)(
        hand,
        jnp.zeros(4, dtype=jnp.int32),
        0,
        0,
        jnp.bool_(False),
        jnp.bool_(False),
    )
    time_end = time.perf_counter()
    delta = (time_end - time_sta) * 1000
    exp = jax.make_jaxpr(func)(
        hand,
        jnp.zeros(4, dtype=jnp.int32),
        0,
        0,
        jnp.bool_(False),
        jnp.bool_(False),
    )
    n_line = len(str(exp).split("\n"))
    print(f"| `{func.__name__}` | {n_line} | {delta:.1f}ms |")


func_name = sys.argv[1]
if func_name == "can_riichi":
    func = Hand.can_riichi
    test_hand(func=func)
elif func_name == "is_tenpai":
    func = Hand.is_tenpai
    test_hand(func=func)
elif func_name == "can_tsumo":
    func = Hand.can_tsumo
    test_hand(func=func)
elif func_name == "score":
    func = Yaku.score
    test_yaku(func=func)
elif func_name == "number":
    func = Shanten.number
    test_hand(func=func)
else:
    print(func_name)
    assert False
