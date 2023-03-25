from pgx.leduc_holdem import State as LeducHoldemState

CARD = ["J", "Q", "K"]


def _make_leducHoldem_dwg(dwg, state: LeducHoldemState, config):
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

    # card
    board_g.add(
        dwg.rect(
            (0, 4 * GRID_SIZE),
            (2 * GRID_SIZE, 3 * GRID_SIZE),
            fill=color_set.background_color,
            stroke=color_set.grid_color,
            stroke_width="2px",
            rx="5px",
            ry="5px",
        )
    )
    board_g.add(
        dwg.text(
            text=CARD[state.cards[0]],
            insert=(GRID_SIZE, 5 * GRID_SIZE),
            fill=color_set.text_color,
            font_size="40px",
            font_family="Courier",
            # font_weight="bold",
        )
    )
    # chip
    board_g.add(
        dwg.text(
            text=f"chip +{state.chips[0]}",
            insert=(0, 7.6 * GRID_SIZE),
            fill=color_set.text_color,
            font_size="18px",
            font_family="Courier",
            # font_weight="bold",
        )
    )

    # card
    board_g.add(
        dwg.rect(
            (6 * GRID_SIZE, 4 * GRID_SIZE),
            (2 * GRID_SIZE, 3 * GRID_SIZE),
            fill=color_set.background_color,
            stroke=color_set.grid_color,
            stroke_width="2px",
            rx="5px",
            ry="5px",
        )
    )
    board_g.add(
        dwg.text(
            text=CARD[state.cards[1]],
            insert=(7 * GRID_SIZE, 5 * GRID_SIZE),
            fill=color_set.text_color,
            font_size="40px",
            font_family="Courier",
            # font_weight="bold",
        )
    )
    # chip
    chip = f"chip +{state.chips[1]}"
    board_g.add(
        dwg.text(
            text=chip,
            insert=(
                8 * GRID_SIZE - 10 * len(chip),
                7.6 * GRID_SIZE,
            ),
            fill=color_set.text_color,
            font_size="18px",
            font_family="Courier",
            # font_weight="bold",
        )
    )

    # public
    board_g.add(
        dwg.line(
            start=(0, 3.5 * GRID_SIZE),
            end=(8 * GRID_SIZE, 3.5 * GRID_SIZE),
            stroke=color_set.grid_color,
            stroke_width="2px",
        )
    )
    board_g.add(
        dwg.rect(
            (3 * GRID_SIZE, 0),
            (2 * GRID_SIZE, 3 * GRID_SIZE),
            fill=color_set.background_color,
            stroke=color_set.p1_color
            if state.round == 0
            else color_set.p2_color,
            stroke_width="2px",
            rx="5px",
            ry="5px",
        )
    )
    board_g.add(
        dwg.text(
            text=CARD[state.cards[2]],
            insert=(4 * GRID_SIZE, GRID_SIZE),
            fill=color_set.p1_color
            if state.round == 0
            else color_set.p2_color,
            font_size="40px",
            font_family="Courier",
            # font_weight="bold",
        )
    )

    return board_g
