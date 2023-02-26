from pgx.tic_tac_toe import State as TictactoeState


def _make_tictactoe_dwg(dwg, state: TictactoeState, config):
    GRID_SIZE = config["GRID_SIZE"]
    BOARD_WIDTH = config["BOARD_WIDTH"]
    BOARD_HEIGHT = config["BOARD_HEIGHT"]
    color_set = config["COLOR_SET"]
    # board
    board_g = dwg.g()

    # grid
    hlines = board_g.add(
        dwg.g(
            id="hlines",
            stroke=color_set.grid_color,
            fill=color_set.grid_color,
        )
    )
    for y in range(1, BOARD_HEIGHT):
        hlines.add(
            dwg.line(
                start=(0, GRID_SIZE * y),
                end=(
                    GRID_SIZE * BOARD_WIDTH,
                    GRID_SIZE * y,
                ),
                stroke_width=GRID_SIZE * 0.05,
            )
        )
        hlines.add(
            dwg.circle(
                center=(0, GRID_SIZE * y),
                r=GRID_SIZE * 0.014,
            )
        )
        hlines.add(
            dwg.circle(
                center=(
                    GRID_SIZE * BOARD_WIDTH,
                    GRID_SIZE * y,
                ),
                r=GRID_SIZE * 0.014,
            )
        )
    vlines = board_g.add(
        dwg.g(
            id="vline",
            stroke=color_set.grid_color,
            fill=color_set.grid_color,
        )
    )
    for x in range(1, BOARD_WIDTH):
        vlines.add(
            dwg.line(
                start=(GRID_SIZE * x, 0),
                end=(
                    GRID_SIZE * x,
                    GRID_SIZE * BOARD_HEIGHT,
                ),
                stroke_width=GRID_SIZE * 0.05,
            )
        )
        vlines.add(
            dwg.circle(
                center=(GRID_SIZE * x, 0),
                r=GRID_SIZE * 0.014,
            )
        )
        vlines.add(
            dwg.circle(
                center=(
                    GRID_SIZE * x,
                    GRID_SIZE * BOARD_HEIGHT,
                ),
                r=GRID_SIZE * 0.014,
            )
        )

    for i, mark in enumerate(state.board):
        x = i % BOARD_WIDTH
        y = i // BOARD_HEIGHT
        if mark == 0:  # 先手
            board_g.add(
                dwg.line(
                    start=(
                        (x + 0.1) * GRID_SIZE,
                        (y + 0.1) * GRID_SIZE,
                    ),
                    end=(
                        (x + 0.9) * GRID_SIZE,
                        (y + 0.9) * GRID_SIZE,
                    ),
                    stroke=color_set.grid_color,
                    stroke_width=0.05 * GRID_SIZE,
                )
            )
            board_g.add(
                dwg.line(
                    start=(
                        (x + 0.1) * GRID_SIZE,
                        (y + 0.9) * GRID_SIZE,
                    ),
                    end=(
                        (x + 0.9) * GRID_SIZE,
                        (y + 0.1) * GRID_SIZE,
                    ),
                    stroke=color_set.grid_color,
                    stroke_width=0.05 * GRID_SIZE,
                )
            )
        elif mark == 1:  # 後手
            board_g.add(
                dwg.circle(
                    center=(
                        (x + 0.5) * GRID_SIZE,
                        (y + 0.5) * GRID_SIZE,
                    ),
                    r=0.4 * GRID_SIZE,
                    stroke=color_set.grid_color,
                    stroke_width=0.05 * GRID_SIZE,
                    fill="none",
                )
            )
    return board_g
