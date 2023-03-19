import base64
import os

from pgx.sparrow_mahjong import NUM_TILE_TYPES, NUM_TILES
from pgx.sparrow_mahjong import State as SparrowMahjongState
from pgx.sparrow_mahjong import _tile_type_to_str


def _make_sparrowmahjong_dwg(dwg, state: SparrowMahjongState, config):
    GRID_SIZE = config["GRID_SIZE"]
    BOARD_WIDTH = config["BOARD_WIDTH"]
    BOARD_HEIGHT = config["BOARD_HEIGHT"]
    color_set = config["COLOR_SET"]

    def _set_piece(
        _x,
        _y,
        _type,
        _is_red,
        _dwg,
        _dwg_g,
        grid_size,
    ):
        _dwg_g.add(
            _dwg.rect(
                ((_x + 9), (_y + 1)),
                (
                    32,
                    47,
                ),
                fill=color_set.p1_color,  # "#f9f7e8",
                stroke="none",
            )
        )
        type_str = _tile_type_to_str(_type)
        if _is_red and type_str != "g" and type_str != "r":
            type_str += "r"
        PATH = {
            f"{i+1}": f"images/sparrow_mahjong/{i+1}p.svg" for i in range(9)
        }
        PATH.update(
            {
                f"{i+1}r": f"images/sparrow_mahjong/{i+1}pr.svg"
                for i in range(9)
            }
        )
        PATH["0"] = "images/sparrow_mahjong/b.svg"
        PATH["g"] = "images/sparrow_mahjong/gd.svg"
        PATH["r"] = "images/sparrow_mahjong/rd.svg"

        file_path = PATH[type_str]
        with open(
            os.path.join(os.path.dirname(__file__), file_path),
            "rb",
        ) as f:
            b64_img = base64.b64encode(f.read())
        img = _dwg.image(
            "data:image/svg+xml;base64," + b64_img.decode("ascii"),
            insert=(_x, _y),
            size=(grid_size, grid_size),
        )
        _dwg_g.add(img)

        return _dwg_g

    # background
    dwg.add(
        dwg.rect(
            (0, 0),
            (
                BOARD_WIDTH * GRID_SIZE,
                BOARD_HEIGHT * GRID_SIZE,
            ),
            fill=color_set.background_color,
        )
    )

    # board
    # grid
    board_g = dwg.g()
    p1_g = dwg.g()
    p2_g = dwg.g()
    p3_g = dwg.g()

    # pieces
    for player_id, pieces_g in zip(state.shuffled_players, [p1_g, p2_g, p3_g]):
        pieces_g = dwg.g()

        # border
        pieces_g.add(
            dwg.rect(
                (245, 360),
                (
                    260,
                    70,
                ),
                rx="2px",
                ry="2px",
                fill=color_set.p2_color,
                stroke=color_set.grid_color,
                stroke_width="5px",
            )
        )

        # hands
        x = 250
        y = 370
        for type, num in zip(
            range(NUM_TILE_TYPES),
            state.hands[player_id],
        ):
            num_red = state.n_red_in_hands[player_id, type]
            for _ in range(num):
                pieces_g = _set_piece(
                    x,
                    y,
                    type,
                    num_red > 0,
                    dwg,
                    pieces_g,
                    GRID_SIZE,
                )
                x += 40
                num_red -= 1

        # river
        x = 270
        y = 220
        river_count = 0
        for type, is_red in zip(
            state.rivers[player_id], state.is_red_in_river[player_id]
        ):
            if type >= 0:
                pieces_g = _set_piece(
                    x,
                    y,
                    type,
                    is_red,
                    dwg,
                    pieces_g,
                    GRID_SIZE,
                )
                x += 40
                river_count += 1
                if river_count > 4:
                    river_count = 0
                    x = 270
                    y += 60

        if player_id == state.shuffled_players[1]:
            pieces_g.rotate(
                angle=90, center=(BOARD_WIDTH * GRID_SIZE / 2, 100)
            )

        elif player_id == state.shuffled_players[2]:
            pieces_g.rotate(
                angle=-90, center=(BOARD_WIDTH * GRID_SIZE / 2, 100)
            )

        board_g.add(pieces_g)

    # dora
    board_g.add(
        dwg.rect(
            (BOARD_WIDTH * GRID_SIZE / 2 - 40, 0),
            (
                80,
                80,
            ),
            rx="5px",
            ry="5px",
            fill=color_set.p1_outline,
        )
    )
    board_g.add(
        dwg.rect(
            (BOARD_WIDTH * GRID_SIZE / 2 - 34, 6),
            (
                68,
                68,
            ),
            rx="5px",
            ry="5px",
            fill="none",
            stroke=color_set.p2_outline,
            stroke_width="3px",
        )
    )
    board_g = _set_piece(
        BOARD_WIDTH * GRID_SIZE / 2 - 25,
        15,
        state.dora,
        False,
        dwg,
        board_g,
        GRID_SIZE,
    )

    # wall
    wall_type = -1
    board_g = _set_piece(330, 120, wall_type, False, dwg, board_g, GRID_SIZE)
    board_g.add(
        dwg.text(
            text=f"× {NUM_TILES - state.draw_ix-1}",
            insert=(380, 150),
            fill=color_set.text_color,
            font_size="20px",
            font_family="serif",
        )
    )
    board_g.translate(0, 40)

    return board_g
