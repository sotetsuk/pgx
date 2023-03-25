from pgx._kuhn_poker import State as KuhnPokerState

CARD = ["J", "Q", "K"]


def _make_kuhnpoker_dwg(dwg, state: KuhnPokerState, config):
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

    # card
    board_g.add(
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
    # chip
    board_g.add(
        dwg.text(
            text=f"chip {state.pot[0]}",
            insert=(0, 3.6 * GRID_SIZE),
            fill=color_set.text_color,
            font_size="18px",
            font_family="Courier",
            # font_weight="bold",
        )
    )

    # card
    board_g.add(
        dwg.rect(
            (6 * GRID_SIZE, 5 * GRID_SIZE),
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
            insert=(7 * GRID_SIZE, 6 * GRID_SIZE),
            fill=color_set.text_color,
            font_size="40px",
            font_family="Courier",
            # font_weight="bold",
        )
    )
    # chip
    board_g.add(
        dwg.text(
            text=f"chip {state.pot[1]}",
            insert=(6 * GRID_SIZE, 4.6 * GRID_SIZE),
            fill=color_set.text_color,
            font_size="18px",
            font_family="Courier",
            # font_weight="bold",
        )
    )

    return board_g
