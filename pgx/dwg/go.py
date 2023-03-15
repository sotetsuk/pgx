import jax.numpy as jnp

from pgx.go import State as GoState


def _make_go_dwg(dwg, state: GoState, config):
    GRID_SIZE = config["GRID_SIZE"]
    BOARD_SIZE = config["BOARD_WIDTH"]
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
    hlines = board_g.add(dwg.g(id="hlines", stroke=color_set.grid_color))
    for y in range(1, BOARD_SIZE - 1):
        hlines.add(
            dwg.line(
                start=(0, GRID_SIZE * y),
                end=(
                    GRID_SIZE * (BOARD_SIZE - 1),
                    GRID_SIZE * y,
                ),
                stroke_width="0.5px",
            )
        )
    vlines = board_g.add(dwg.g(id="vline", stroke=color_set.grid_color))
    for x in range(1, BOARD_SIZE - 1):
        vlines.add(
            dwg.line(
                start=(GRID_SIZE * x, 0),
                end=(
                    GRID_SIZE * x,
                    GRID_SIZE * (BOARD_SIZE - 1),
                ),
                stroke_width="0.5px",
            )
        )
    board_g.add(
        dwg.rect(
            (0, 0),
            (
                (BOARD_SIZE - 1) * GRID_SIZE,
                (BOARD_SIZE - 1) * GRID_SIZE,
            ),
            fill="none",
            stroke=color_set.grid_color,
            stroke_width="2px",
        )
    )
    # hoshi
    hoshi_g = dwg.g()
    hosi_pos = []
    if BOARD_SIZE == 19:
        hosi_pos = [
            (4, 4),
            (4, 10),
            (4, 16),
            (10, 4),
            (10, 10),
            (10, 16),
            (16, 4),
            (16, 10),
            (16, 16),
        ]
    elif BOARD_SIZE == 5:
        hosi_pos = [(3, 3)]

    for x, y in hosi_pos:
        hoshi_g.add(
            dwg.circle(
                center=((x - 1) * GRID_SIZE, (y - 1) * GRID_SIZE),
                r=GRID_SIZE / 10,
                fill=color_set.grid_color,
            )
        )
    board_g.add(hoshi_g)

    # stones
    board = jnp.clip(state.chain_id_board, -1, 1)
    for xy, stone in enumerate(board):
        if stone == 0:
            continue
        # ndarrayのx,yと違うことに注意
        stone_y = xy // BOARD_SIZE * GRID_SIZE
        stone_x = xy % BOARD_SIZE * GRID_SIZE

        color = color_set.p1_color if stone == 1 else color_set.p2_color
        outline = color_set.p1_outline if stone == 1 else color_set.p2_outline
        board_g.add(
            dwg.circle(
                center=(stone_x, stone_y),
                r=GRID_SIZE / 2.2,
                stroke=outline,
                fill=color,
            )
        )
    board_g.translate(GRID_SIZE / 2, GRID_SIZE / 2)

    return board_g
