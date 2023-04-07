import jax.numpy as jnp


CAN_MOVE = -jnp.ones((6, 64, 28))
# usage: CAN_MOVE[piece, from_x, from_y]
# 将棋と違い、中央から点対称でないので、注意が必要。
# 視点は常に白側のイメージが良い。
# PAWN以外の動きは上下左右対称。PAWNは上下と斜めへ動ける駒と定義して、手番に応じてフィルタする。

# PAWN