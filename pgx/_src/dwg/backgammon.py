from pgx.backgammon import State as BackgammonState
from pgx.backgammon import _get_abs_board


def _make_backgammon_dwg(dwg, state: BackgammonState, config):  # noqa: C901
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

        if i < 12:
            if i % 2 == 0:
                fill_color = color_set.p1_color
            else:
                fill_color = color_set.text_color
        else:
            if i % 2 == 1:
                fill_color = color_set.p1_color
            else:
                fill_color = color_set.text_color

        board_g.add(
            dwg.polygon(
                points=[p1, p2, p3],
                stroke=color_set.text_color,
                fill=fill_color,
            )
        )
    board_g.add(
        dwg.rect(
            (0, 0),
            (
                13 * GRID_SIZE,
                14 * GRID_SIZE,
            ),
            stroke=color_set.grid_color,
            fill="none",
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
            fill="none",
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
    # 24:黒
    piece = board[24]
    for n in range(piece):
        board_g.add(
            dwg.circle(
                center=(
                    6.5 * GRID_SIZE,
                    (7.5 + n) * GRID_SIZE,
                ),
                r=0.5 * GRID_SIZE,
                stroke=color_set.grid_color,
                fill=color_set.p2_color,
            )
        )
    # 25:白
    piece = -board[25]
    for n in range(piece):
        board_g.add(
            dwg.circle(
                center=(
                    6.5 * GRID_SIZE,
                    (6.5 - n) * GRID_SIZE,
                ),
                r=0.5 * GRID_SIZE,
                stroke=color_set.grid_color,
                fill=color_set.p1_color,
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

    # dice
    def _add_dice():
        DICE = "⚀⚁⚂⚃⚄⚅"
        if state._playable_dice[2] != -1:
            for xy in range(4):
                x = xy // 2
                y = xy % 2

                if state._playable_dice[y + 2 * x] == -1:
                    continue

                board_g.add(
                    dwg.text(
                        text=f"{DICE[state._playable_dice[y+2*x]]}",
                        insert=(
                            (13.5 + x * 1.3) * GRID_SIZE,
                            (7.0 + y * 1.3) * GRID_SIZE,
                        ),
                        fill=color_set.grid_color,
                        font_size="44px",
                        font_family="sans serif",
                    )
                )
        else:
            x = 0
            for dice in state._playable_dice:
                if dice == -1:
                    continue
                board_g.add(
                    dwg.text(
                        text=f"{DICE[dice]}",
                        insert=(
                            (13.5 + x * 1.3) * GRID_SIZE,
                            7.5 * GRID_SIZE,
                        ),
                        fill=color_set.grid_color,
                        font_size="44px",
                        font_family="sans serif",
                    )
                )
                x += 1
        return board_g

    board_g = _add_dice()
    return board_g
