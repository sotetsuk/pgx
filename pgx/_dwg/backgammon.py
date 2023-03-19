from pgx.backgammon import State as BackgammonState
from pgx.backgammon import _get_abs_board


def _make_backgammon_dwg(dwg, state: BackgammonState, config):
    board = _get_abs_board(state)
    BOARD_WIDTH = config["BOARD_WIDTH"]
    BOARD_HEIGHT = config["BOARD_HEIGHT"]
    GRID_SIZE = config["GRID_SIZE"]
    color_set = config["COLOR_SET"]
    # background
    dwg.add(
        dwg.rect(
            (0, 0),
            (
                (BOARD_WIDTH + 6) * GRID_SIZE,
                (BOARD_HEIGHT + 3) * GRID_SIZE,
            ),
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
                13 * GRID_SIZE,
                14 * GRID_SIZE,
            ),
            stroke=color_set.grid_color,
            fill=color_set.background_color,
        )
    )
    board_g.add(
        dwg.rect(
            (6 * GRID_SIZE, 0),
            (
                1 * GRID_SIZE,
                14 * GRID_SIZE,
            ),
            stroke=color_set.grid_color,
            fill=color_set.background_color,
        )
    )

    for i in range(24):
        p1 = (i * GRID_SIZE, 0)
        p2 = ((i + 1) * GRID_SIZE, 0)
        p3 = ((i + 0.5) * GRID_SIZE, 6 * GRID_SIZE)
        if 6 <= i < 12:
            p1 = ((i + 1) * GRID_SIZE, 0)
            p2 = ((i + 2) * GRID_SIZE, 0)
            p3 = ((i + 1.5) * GRID_SIZE, 6 * GRID_SIZE)
        elif 12 <= i < 18:
            p1 = ((i - 12) * GRID_SIZE, 14 * GRID_SIZE)
            p2 = ((i + 1 - 12) * GRID_SIZE, 14 * GRID_SIZE)
            p3 = ((i + 0.5 - 12) * GRID_SIZE, 8 * GRID_SIZE)
        elif 18 <= i:
            p1 = ((i + 1 - 12) * GRID_SIZE, 14 * GRID_SIZE)
            p2 = ((i + 2 - 12) * GRID_SIZE, 14 * GRID_SIZE)
            p3 = ((i + 1.5 - 12) * GRID_SIZE, 8 * GRID_SIZE)

        fill_color = (
            color_set.p1_outline if i % 2 == 0 else color_set.p2_outline
        )

        board_g.add(
            dwg.polygon(
                points=[p1, p2, p3],
                stroke=color_set.grid_color,
                fill=fill_color,
            )
        )

    # pieces
    for i, piece in enumerate(board[:24]):
        fill_color = color_set.p2_color
        if piece < 0:  # 白
            piece = -piece
            fill_color = color_set.p1_color

        x = (12 - i) + 0.5
        y = 13.5
        diff = -1
        if 6 <= i < 12:
            x = (12 - i) - 0.5
            y = 13.5
        elif 12 <= i < 18:
            x = i - 12 + 0.5
            y = 0.5
            diff = 1
        elif 18 <= i:
            x = i - 12 + 1.5
            y = 0.5
            diff = 1
        for n in range(piece):
            board_g.add(
                dwg.circle(
                    center=(x * GRID_SIZE, (y + n * diff) * GRID_SIZE),
                    r=0.5 * GRID_SIZE,
                    stroke=color_set.grid_color,
                    fill=fill_color,
                )
            )

    # bar
    for i, piece in enumerate(board[24:26]):  # 24:黒, 25:白
        fill_color = color_set.p2_color
        delta = 1
        offset = 0.5
        if i == 1:
            piece = -piece
            fill_color = color_set.p1_color
            delta = -1
            offset = -0.5
        for n in range(piece):
            board_g.add(
                dwg.circle(
                    center=(
                        6.5 * GRID_SIZE,
                        (7 + offset + n * delta) * GRID_SIZE,
                    ),
                    r=0.5 * GRID_SIZE,
                    stroke=color_set.grid_color,
                    fill=fill_color,
                )
            )

    # off
    board_g.add(
        dwg.rect(
            (13 * GRID_SIZE, 0),
            (
                4 * GRID_SIZE,
                2 * GRID_SIZE,
            ),
            stroke=color_set.grid_color,
            fill=color_set.background_color,
        )
    )
    board_g.add(
        dwg.circle(
            center=(14 * GRID_SIZE, 1 * GRID_SIZE),
            r=0.5 * GRID_SIZE,
            stroke=color_set.grid_color,
            fill=color_set.p2_color,
        )
    )
    board_g.add(
        dwg.text(
            text=f"×{board[26]}",  # 26:黒
            insert=(
                14.6 * GRID_SIZE,
                1.4 * GRID_SIZE,
            ),
            fill=color_set.grid_color,
            font_size="34px",
            font_family="serif",
        )
    )

    board_g.add(
        dwg.rect(
            (13 * GRID_SIZE, 12 * GRID_SIZE),
            (
                4 * GRID_SIZE,
                2 * GRID_SIZE,
            ),
            stroke=color_set.grid_color,
            fill=color_set.background_color,
        )
    )
    board_g.add(
        dwg.circle(
            center=(14 * GRID_SIZE, 13 * GRID_SIZE),
            r=0.5 * GRID_SIZE,
            stroke=color_set.grid_color,
            fill=color_set.p1_color,
        )
    )
    board_g.add(
        dwg.text(
            text=f"×{-board[27]}",  # 27:白
            insert=(
                14.6 * GRID_SIZE,
                13.4 * GRID_SIZE,
            ),
            fill=color_set.grid_color,
            font_size="34px",
            font_family="serif",
        )
    )

    return board_g
