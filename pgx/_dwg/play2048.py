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
            fill=color_set.background_color,
        )
    )

    # board
    # grid
    board_g = dwg.g()
    for i, _exp2 in enumerate(state.board):
        exp2 = int(_exp2)
        num = 2**exp2
        if exp2 > 11:
            exp2 = 11
        x = (i % 4) * GRID_SIZE
        y = (i // 4) * GRID_SIZE
        _color = (
            f"{(242-exp2*22):02x}"
            if color_set.background_color == "white"
            else f"{(35+exp2*20):02x}"
        )
        board_g.add(
            dwg.rect(
                (x + 2, y + 2),
                (
                    GRID_SIZE - 4,
                    GRID_SIZE - 4,
                ),
                fill=f"#{_color}{_color}{_color}",
                stroke=color_set.text_color,
                stroke_width="0.5px",
                rx="3px",
                ry="3px",
            )
        )

        if exp2 == 0:
            continue

        font_size = 18
        large_num_color = (
            f"#{(145+exp2*10):02x}{(145+exp2*10):02x}{(145+exp2*10):02x}"
            if color_set.background_color == "white"
            else f"#{(120-exp2*10):02x}{(120-exp2*10):02x}{(120-exp2*10):02x}"
        )
        board_g.add(
            dwg.text(
                text=str(num),
                insert=(
                    x + GRID_SIZE / 2 - font_size * len(str(num)) * 0.3,
                    y + GRID_SIZE / 2 + 5,
                ),
                fill=color_set.text_color if exp2 < 7 else large_num_color,
                font_size=f"{font_size}px",
                font_family="Courier",
                font_weight="bold",
            )
        )
    return board_g
