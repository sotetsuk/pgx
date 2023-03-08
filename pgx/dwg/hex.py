import jax.numpy as jnp

from pgx.hex import State as HexState


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
    board = state.board
    for xy, stone in enumerate(board):
        # ndarrayのx,yと違うことに注意
        # svgではヨコがx
        _y = (xy // BOARD_SIZE) * GRID_SIZE * 3 / 2
        _x = (
            (xy % BOARD_SIZE + (xy // BOARD_SIZE) / 2)
            * GRID_SIZE
            * jnp.sqrt(3)
        )
        # if stone==1:
        #    color="black"
        # elif stone==-1:
        #    color="gray"
        # else:
        #    color="white"
        board_g.add(
            dwg.polygon(
                # fmt:off
                points=[
                    (int(_x), int(_y)),
                    (int(_x + GRID_SIZE * jnp.sqrt(3) / 2), int(_y - GRID_SIZE / 2)),
                    (int(_x + GRID_SIZE * jnp.sqrt(3)), int(_y)),
                    (int(_x + GRID_SIZE * jnp.sqrt(3)), int(_y + GRID_SIZE)),
                    (int(_x + GRID_SIZE * jnp.sqrt(3) / 2), int(_y + GRID_SIZE * 3 / 2)),
                    (int(_x), int(_y + GRID_SIZE)),
                ],
                # fmt:on
                fill=color_set.background_color,
                # fill=color,
                stroke=color_set.grid_color,
            )
        )
        if stone == 0:
            continue

        color = color_set.p1_color if stone == 1 else color_set.p2_color
        outline = color_set.p1_outline if stone == 1 else color_set.p2_outline
        board_g.add(
            dwg.circle(
                center=(
                    int(_x + GRID_SIZE * jnp.sqrt(3) / 2),
                    int(_y + GRID_SIZE / 2),
                ),
                r=GRID_SIZE / 1.5,
                stroke=outline,
                fill=color,
            )
        )

    return board_g
