from pgx.play2048 import State as Play2048State


def _make_2048_dwg(dwg, state: Play2048State, config):
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

    for i in range(16):
        x = (i % 4) * GRID_SIZE
        y = (i // 4) * GRID_SIZE
        board_g.add(
            dwg.rect(
                (x + 2, y + 2),
                (
                    GRID_SIZE - 4,
                    GRID_SIZE - 4,
                ),
                fill=color_set.p1_color,
                # stroke=color_set.grid_color,
                rx="3px",
                ry="3px",
            )
        )

    for i, num in enumerate(state.board):
        if num == 0:
            continue
        num = 2**num
        font_size = 18  # 28 - 2 * len(str(num))
        x = (i % 4) * GRID_SIZE
        y = (i // 4) * GRID_SIZE
        board_g.add(
            dwg.text(
                text=str(num),
                insert=(
                    x + GRID_SIZE / 2 - font_size * len(str(num)) * 0.3,
                    y + GRID_SIZE / 2 + 5,
                ),
                fill=color_set.text_color,
                font_size=f"{font_size}px",
                font_family="Courier",
                font_weight="bold",
            )
        )
    return board_g
