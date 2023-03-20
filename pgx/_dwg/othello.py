from pgx.othello import State as OthelloState
from pgx.othello import _get_abs_board


def _make_othello_dwg(dwg, state: OthelloState, config):
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
    board_g.add(
        dwg.rect(
            (0, 0),
            (
                BOARD_SIZE * GRID_SIZE,
                BOARD_SIZE * GRID_SIZE,
            ),
            fill="none",
            stroke=color_set.grid_color,
            stroke_width="20px",
            rx="3px",
            ry="3px",
        )
    )
    board_g.add(
        dwg.rect(
            (0, 0),
            (
                BOARD_SIZE * GRID_SIZE,
                BOARD_SIZE * GRID_SIZE,
            ),
            fill=color_set.background_color,
            stroke=color_set.grid_color,
        )
    )
    hlines = board_g.add(dwg.g(id="hlines", stroke=color_set.grid_color))
    for y in range(BOARD_SIZE):
        hlines.add(
            dwg.line(
                start=(0, GRID_SIZE * y),
                end=(GRID_SIZE * BOARD_SIZE, GRID_SIZE * y),
                stroke_width="0.5px",
            )
        )
    vlines = board_g.add(dwg.g(id="vline", stroke=color_set.grid_color))
    for x in range(BOARD_SIZE):
        vlines.add(
            dwg.line(
                start=(GRID_SIZE * x, 0),
                end=(GRID_SIZE * x, GRID_SIZE * BOARD_SIZE),
                stroke_width="0.5px",
            )
        )

    # hoshi
    hoshi_g = dwg.g()
    hosi_pos = [
        (2, 2),
        (2, 6),
        (6, 2),
        (6, 6),
    ]

    for x, y in hosi_pos:
        hoshi_g.add(
            dwg.circle(
                center=(x * GRID_SIZE, y * GRID_SIZE),
                r=GRID_SIZE / 10,
                fill=color_set.grid_color,
            )
        )
    board_g.add(hoshi_g)

    # stones
    board = _get_abs_board(state)
    for xy, stone in enumerate(board):
        if stone == 0:
            continue
        # ndarrayのx,yと違うことに注意
        # svgではヨコがx
        stone_y = xy // BOARD_SIZE * GRID_SIZE + GRID_SIZE / 2
        stone_x = xy % BOARD_SIZE * GRID_SIZE + GRID_SIZE / 2

        color = color_set.p1_color if stone == 1 else color_set.p2_color
        outline = color_set.p1_outline if stone == 1 else color_set.p2_outline
        board_g.add(
            dwg.circle(
                center=(stone_x, stone_y),
                r=GRID_SIZE / 2.4,
                stroke=outline,
                fill=color,
            )
        )

    return board_g
