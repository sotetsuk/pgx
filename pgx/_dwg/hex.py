import jax.numpy as jnp

from pgx.hex import State as HexState
from pgx.hex import _get_abs_board


def _make_hex_dwg(dwg, state: HexState, config):
    GRID_SIZE = config["GRID_SIZE"] / 2  # 六角形の1辺
    BOARD_SIZE = int(state.size)
    color_set = config["COLOR_SET"]

    # background
    dwg.add(
        dwg.rect(
            (0, 0),
            (BOARD_SIZE * GRID_SIZE, BOARD_SIZE * GRID_SIZE),
            # stroke=svgwrite.rgb(10, 10, 16, "%"),
            fill=color_set.background_color,
        )
    )

    # board
    # grid
    board_g = dwg.g()

    # stones
    board = _get_abs_board(state)
    for xy, stone in enumerate(board):
        # ndarrayのx,yと違うことに注意
        # svgではヨコがx
        _y = (xy // BOARD_SIZE) * GRID_SIZE * 3 / 2
        _x = (
            (xy % BOARD_SIZE + (xy // BOARD_SIZE) / 2)
            * GRID_SIZE
            * jnp.sqrt(3)
        )

        # hexagon
        r = GRID_SIZE
        board_g.add(
            dwg.polygon(
                points=[
                    (
                        four_dig(_x + r * jnp.sin(jnp.pi / 3 * i)),
                        four_dig(_y + r * jnp.cos(jnp.pi / 3 * i)),
                    )
                    for i in range(6)
                ],
                fill=color_set.text_color,
                stroke=color_set.grid_color,
                stroke_width="0.5px",
            )
        )
        # 太線で表示させる場合
        #
        # if xy // BOARD_SIZE == 0:
        #     board_g.add(
        #         _dwg.polyline(
        #             points=[
        #                 (
        #                     four_dig(_x + r * jnp.sin(jnp.pi / 3 * i)),
        #                     four_dig(_y + r * jnp.cos(jnp.pi / 3 * i)),
        #                 )
        #                 for i in range(2, 5)
        #             ],
        #             fill=color_set.text_color,
        #             stroke=color_set.grid_color,
        #             stroke_width="2px",
        #         )
        #     )
        # elif xy // BOARD_SIZE == 10:
        #     board_g.add(
        #         _dwg.polyline(
        #             points=[
        #                 (
        #                     four_dig(_x + r * jnp.sin(jnp.pi / 3 * i)),
        #                     four_dig(_y + r * jnp.cos(jnp.pi / 3 * i)),
        #                 )
        #                 for i in range(-1,2)
        #             ],
        #             fill=color_set.text_color,
        #             stroke=color_set.grid_color,
        #             stroke_width="2px",
        #         )
        #     )
        if stone == 0:
            continue

        color = color_set.p1_color if stone > 0 else color_set.p2_color
        outline = color_set.p1_outline if stone > 0 else color_set.p2_outline
        board_g.add(
            dwg.circle(
                center=(four_dig(_x), four_dig(_y)),
                r=GRID_SIZE / 1.5,
                stroke=outline,
                fill=color,
            )
        )

    # 周りの領域
    for i in range(BOARD_SIZE):
        # 黒三角形
        board_g.add(
            dwg.polygon(
                # fmt:off
                points=[
                    (four_dig((i - 1 / 2) * jnp.sqrt(3) * GRID_SIZE), -four_dig(GRID_SIZE / 2)),
                    (four_dig(i * jnp.sqrt(3) * GRID_SIZE), -GRID_SIZE),
                    (four_dig((i - 1) * jnp.sqrt(3) * GRID_SIZE), -GRID_SIZE),
                ],
                # fmt:on
                stroke=color_set.grid_color,
                stroke_width="0.5px",
                fill=color_set.p1_color,
            )
        )
        offset = jnp.sqrt(3) / 2 * GRID_SIZE * (BOARD_SIZE + 1)
        board_g.add(
            dwg.polygon(
                # fmt:off
                points=[
                    (four_dig((i - 1 / 2) * jnp.sqrt(3) * GRID_SIZE + offset), four_dig(GRID_SIZE * BOARD_SIZE * 3 / 2 - GRID_SIZE)),
                    (four_dig(i * jnp.sqrt(3) * GRID_SIZE + offset), four_dig(GRID_SIZE * BOARD_SIZE * 3 / 2 - GRID_SIZE / 2)),
                    (four_dig((i - 1) * jnp.sqrt(3) * GRID_SIZE + offset), four_dig(GRID_SIZE * BOARD_SIZE * 3 / 2 - GRID_SIZE / 2)),
                ],
                # fmt:on
                stroke=color_set.grid_color,
                stroke_width="0.5px",
                fill=color_set.p1_color,
            )
        )

        # 白三角形
        board_g.add(
            dwg.polygon(
                points=[
                    (
                        four_dig((i / 2 - 1) * jnp.sqrt(3) * GRID_SIZE),
                        four_dig(i * 3 / 2 * GRID_SIZE - GRID_SIZE),
                    ),
                    (
                        four_dig(((i + 1) / 2 - 1) * jnp.sqrt(3) * GRID_SIZE),
                        four_dig((i + 1) * 3 / 2 * GRID_SIZE - GRID_SIZE),
                    ),
                    (
                        four_dig(((i + 1) / 2 - 1) * jnp.sqrt(3) * GRID_SIZE),
                        four_dig((i + 1) * 3 / 2 * GRID_SIZE - 2 * GRID_SIZE),
                    ),
                ],
                fill=color_set.p2_color,
                stroke=color_set.grid_color,
                stroke_width="0.5px",
            )
        )
        offset = jnp.sqrt(3) * GRID_SIZE * (BOARD_SIZE + 1 / 2)
        board_g.add(
            dwg.polygon(
                points=[
                    (
                        four_dig(
                            (i / 2 - 1) * jnp.sqrt(3) * GRID_SIZE + offset
                        ),
                        four_dig(i * 3 / 2 * GRID_SIZE - GRID_SIZE / 2),
                    ),
                    (
                        four_dig(
                            ((i + 1) / 2 - 1) * jnp.sqrt(3) * GRID_SIZE
                            + offset
                        ),
                        four_dig((i + 1) * 3 / 2 * GRID_SIZE - GRID_SIZE / 2),
                    ),
                    (
                        four_dig(
                            (i / 2 - 1) * jnp.sqrt(3) * GRID_SIZE + offset
                        ),
                        four_dig((i + 1) * 3 / 2 * GRID_SIZE - GRID_SIZE),
                    ),
                ],
                fill=color_set.p2_color,
                stroke=color_set.grid_color,
                stroke_width="0.5px",
            )
        )
    board_g.translate(GRID_SIZE, GRID_SIZE / 2)

    return board_g


def four_dig(num):
    """
    numbers must not have more than 4 decimal digits in the fractional part of their
    decimal expansion and must be in the range -32,767.9999 to +32,767.9999, inclusive.

    see https://svgwrite.readthedocs.io/en/latest/overview.html
    """
    num_str = f"{num:.4f}"
    return float(num_str)
