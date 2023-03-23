from pgx._khun_poker import State as KhunPokerState

CARD = ["J", "Q", "K"]


def _make_khunpoker_dwg(dwg, state: KhunPokerState, config):
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
    board_g.add(
        dwg.line(
            start=(2 * GRID_SIZE, 8 * GRID_SIZE),
            end=(6 * GRID_SIZE, 0),
            stroke=color_set.grid_color,
            stroke_width="2px",
        )
    )
    p0 = dwg.g()
    p0.add(
        dwg.rect(
            (0, 0),
            (2 * GRID_SIZE, 3 * GRID_SIZE),
            fill=color_set.background_color,
            stroke=color_set.grid_color,
            stroke_width="2px",
            rx="5px",
            ry="5px",
        )
    )
    p0.add(
        dwg.text(
            text="●" * state.pot[0],
            insert=(GRID_SIZE / 4, 4 * GRID_SIZE),
            fill=color_set.text_color,
            font_size="40px",
            font_family="Courier",
            # font_weight="bold",
        )
    )
    board_g.add(p0)

    p1 = dwg.g()
    p1.add(
        dwg.rect(
            (0, 0),
            (2 * GRID_SIZE, 3 * GRID_SIZE),
            fill=color_set.background_color,
            stroke=color_set.grid_color,
            stroke_width="2px",
            rx="5px",
            ry="5px",
        )
    )
    p1.add(
        dwg.text(
            text="●" * state.pot[1],
            insert=(GRID_SIZE / 4, 4 * GRID_SIZE),
            fill=color_set.text_color,
            font_size="40px",
            font_family="Courier",
            # font_weight="bold",
        )
    )
    p1.rotate(
        angle=180,
        center=(BOARD_SIZE * GRID_SIZE / 2, BOARD_SIZE * GRID_SIZE / 2),
    )
    board_g.add(p1)

    board_g.add(
        dwg.text(
            text=CARD[state.cards[0]],
            insert=(GRID_SIZE, GRID_SIZE),
            fill=color_set.text_color,
            font_size="40px",
            font_family="Courier",
            # font_weight="bold",
        )
    )
    board_g.add(
        dwg.text(
            text=CARD[state.cards[1]],
            insert=(7 * GRID_SIZE, 6 * GRID_SIZE),
            fill=color_set.text_color,
            font_size="40px",
            font_family="Courier",
            # font_weight="bold",
        )
    )
    return board_g
