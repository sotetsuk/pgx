from pgx.connect_four import State as ConnectFourState


def _make_connect_four_dwg(dwg, state: ConnectFourState, config):

    GRID_SIZE = config["GRID_SIZE"]
    BOARD_WIDTH = config["BOARD_WIDTH"] - 1
    BOARD_HEIGHT = config["BOARD_HEIGHT"]
    color_set = config["COLOR_SET"]

    # background
    dwg.add(
        dwg.rect(
            (0, 0),
            (BOARD_WIDTH * GRID_SIZE, BOARD_HEIGHT * GRID_SIZE),
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
                BOARD_WIDTH * GRID_SIZE,
                (BOARD_HEIGHT - 1) * GRID_SIZE,
            ),
            fill=color_set.p1_outline,
        )
    )

    _width = 6
    board_g.add(
        dwg.rect(
            (0, (BOARD_HEIGHT - 1) * GRID_SIZE),
            (
                BOARD_WIDTH * GRID_SIZE,
                _width,
            ),
            fill=color_set.grid_color,
            stroke=color_set.grid_color,
        )
    )
    board_g.add(
        dwg.rect(
            (-_width, 0),
            (
                _width,
                BOARD_HEIGHT * GRID_SIZE,
            ),
            fill=color_set.grid_color,
            stroke=color_set.grid_color,
        )
    )
    _center = -_width / 2
    board_g.add(
        dwg.polygon(
            points=[
                (_center, (BOARD_HEIGHT - 0.5) * GRID_SIZE),
                (_center - 0.5 * GRID_SIZE, BOARD_HEIGHT * GRID_SIZE),
                (_center + 0.5 * GRID_SIZE, BOARD_HEIGHT * GRID_SIZE),
            ],
            fill=color_set.grid_color,
            stroke=color_set.grid_color,
        )
    )
    board_g.add(
        dwg.rect(
            (GRID_SIZE * BOARD_WIDTH, 0),
            (
                _width,
                BOARD_HEIGHT * GRID_SIZE,
            ),
            fill=color_set.grid_color,
            stroke=color_set.grid_color,
        )
    )
    _center = GRID_SIZE * BOARD_WIDTH + _width / 2
    board_g.add(
        dwg.polygon(
            points=[
                (_center, (BOARD_HEIGHT - 0.5) * GRID_SIZE),
                (_center - 0.5 * GRID_SIZE, BOARD_HEIGHT * GRID_SIZE),
                (_center + 0.5 * GRID_SIZE, BOARD_HEIGHT * GRID_SIZE),
            ],
            fill=color_set.grid_color,
            stroke=color_set.grid_color,
        )
    )

    # stones
    board = state.board
    for xy, stone in enumerate(board):
        if stone == -1:
            continue
        # ndarrayのx,yと違うことに注意
        # svgではヨコがx
        stone_y = xy // BOARD_HEIGHT * GRID_SIZE + GRID_SIZE / 2
        stone_x = xy % BOARD_WIDTH * GRID_SIZE + GRID_SIZE / 2

        color = color_set.p1_color if stone == 0 else color_set.p2_color
        outline = color_set.p2_color if stone == 0 else color_set.p1_color
        board_g.add(
            dwg.circle(
                center=(stone_x, stone_y),
                r=GRID_SIZE / 3,
                stroke=outline,
                fill=color,
            )
        )
    board_g.translate(GRID_SIZE / 2, 0)

    return board_g
