from copy import deepcopy

import jax.numpy as jnp

from pgx.hex import State as HexState
from pgx.hex import _get_abs_board

r3 = jnp.sqrt(3)


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
        _x = (xy % BOARD_SIZE + (xy // BOARD_SIZE) / 2) * GRID_SIZE * r3

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

        if stone == 0:
            continue

        color = color_set.p1_color if stone > 0 else color_set.p2_color
        outline = color_set.p1_outline if stone > 0 else color_set.p2_outline
        r = GRID_SIZE / 1.3

        # 駒を円で表示
        # board_g.add(
        #    dwg.circle(
        #        center=(four_dig(_x), four_dig(_y)),
        #        r=r * 0.9,
        #        fill=color,
        #        stroke=outline,
        #        stroke_width="0.5px",
        #    )
        # )

        # 駒を六角形で表示
        board_g.add(
            dwg.polygon(
                points=[
                    (
                        four_dig(_x + r * jnp.sin(jnp.pi / 3 * i)),
                        four_dig(_y + r * jnp.cos(jnp.pi / 3 * i)),
                    )
                    for i in range(6)
                ],
                fill=color,
                stroke=outline,
                stroke_width="0.5px",
            )
        )
    # 周りの領域
    b_points = []
    w_points = []
    x1, y1 = (BOARD_SIZE - 1) * GRID_SIZE * r3, 0
    x2, y2 = ((BOARD_SIZE - 1) / 2) * GRID_SIZE * r3, (
        BOARD_SIZE - 1
    ) * GRID_SIZE * 3 / 2
    cx, cy = four_dig((x1 + x2) / 2), four_dig((y1 + y2) / 2)
    # fmt:off
    for i in range(BOARD_SIZE):
        b_points.append((four_dig((i - 1 / 2) * r3 * GRID_SIZE), -four_dig(GRID_SIZE / 2)))
        b_points.append((four_dig(i * r3 * GRID_SIZE), -GRID_SIZE))
        w_points.append((four_dig(((i - 1) / 2) * r3 * GRID_SIZE), four_dig((i + 1) * 3 / 2 * GRID_SIZE - 2 * GRID_SIZE)))
        w_points.append((four_dig(((i - 1) / 2) * r3 * GRID_SIZE), four_dig((i + 1) * 3 / 2 * GRID_SIZE - GRID_SIZE)))
    b_points.append((four_dig((BOARD_SIZE - 1) * r3 * GRID_SIZE + GRID_SIZE * r3 / 4), -GRID_SIZE + GRID_SIZE / 4))
    b_points.append((four_dig((BOARD_SIZE - 1 / 2) * r3 * GRID_SIZE), -1.5 * GRID_SIZE))
    b_points.append((four_dig(-1.5 * r3 * GRID_SIZE), -1.5 * GRID_SIZE))
    w_points.append((four_dig(((BOARD_SIZE - 2) / 2) * r3 * GRID_SIZE + GRID_SIZE * r3 / 4), four_dig((i + 1) * 3 / 2 * GRID_SIZE - GRID_SIZE * 3 / 4)))
    w_points.append((four_dig(((BOARD_SIZE - 2) / 2) * r3 * GRID_SIZE), four_dig((i + 1) * 3 / 2 * GRID_SIZE)))
    w_points.append((four_dig(-1.5 * r3 * GRID_SIZE), -1.5 * GRID_SIZE))
    # fmt:on
    edge = dwg.g()
    edge.add(
        dwg.polygon(
            points=b_points,
            fill=color_set.p1_color,
            stroke=color_set.grid_color,
            stroke_width="0.5px",
        )
    )
    edge.add(
        dwg.polygon(
            points=w_points,
            fill=color_set.p2_color,
            stroke=color_set.grid_color,
            stroke_width="0.5px",
        )
    )
    board_g.add(edge)
    edge = deepcopy(edge)
    edge.rotate(angle=180, center=(cx, cy))
    board_g.add(edge)
    # board_g.rotate(
    #    angle=-30,
    #    center=(0, 0),
    # )
    board_g.translate(3 * GRID_SIZE, 2 * GRID_SIZE)
    return board_g


def four_dig(num):
    """
    numbers must not have more than 4 decimal digits in the fractional part of their
    decimal expansion and must be in the range -32,767.9999 to +32,767.9999, inclusive.

    see https://svgwrite.readthedocs.io/en/latest/overview.html
    """
    num_str = f"{num:.4f}"
    return float(num_str)
