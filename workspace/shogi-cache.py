#! python3
import jax.numpy as jnp
from flax.serialization import to_bytes


# 初期盤面生成
def _make_init_board() -> jnp.ndarray:
    array = jnp.zeros((29, 81), dtype=jnp.int32)
    for i in range(81):
        if i % 9 == 6:
            # 先手歩
            p = 1
        elif i == 8 or i == 80:
            # 先手香車
            p = 2
        elif i == 17 or i == 71:
            # 先手桂馬
            p = 3
        elif i == 26 or i == 62:
            # 先手銀
            p = 4
        elif i == 70:
            # 先手角
            p = 5
        elif i == 16:
            # 先手飛車
            p = 6
        elif i == 35 or i == 53:
            # 先手金
            p = 7
        elif i == 44:
            # 先手玉
            p = 8
        elif i % 9 == 2:
            # 後手歩
            p = 15
        elif i == 72 or i == 0:
            # 後手香車
            p = 16
        elif i == 63 or i == 9:
            # 後手桂馬
            p = 17
        elif i == 54 or i == 18:
            # 後手銀
            p = 18
        elif i == 10:
            # 後手角
            p = 19
        elif i == 64:
            # 後手飛車
            p = 20
        elif i == 45 or i == 27:
            # 後手金
            p = 21
        elif i == 36:
            # 後手玉
            p = 22
        else:
            # 空白
            p = 0
        array = array.at[p, i].set(1)
    return array


init_board = _make_init_board()
print(f"INIT_BOARD = {to_bytes(init_board)}")
