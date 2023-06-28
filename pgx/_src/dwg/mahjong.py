from pgx._src.dwg.mahjong_tile import TilePath

from pgx._mahjong._mahjong2 import State as MahjongState

path_list = TilePath.str_list


def _make_mahjong_dwg(dwg, state: MahjongState, config):
    GRID_SIZE = config["GRID_SIZE"]
    BOARD_WIDTH = config["BOARD_WIDTH"]
    BOARD_HEIGHT = config["BOARD_HEIGHT"]
    color_set = config["COLOR_SET"]

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
    players_g = [
        dwg.g(
            style="stroke:#000000;stroke-width:0.01mm;fill:#000000",
            fill_rule="evenodd",
        ),
        dwg.g(
            style="stroke:#000000;stroke-width:0.01mm;fill:#000000",
            fill_rule="evenodd",
        ),
        dwg.g(
            style="stroke:#000000;stroke-width:0.01mm;fill:#000000",
            fill_rule="evenodd",
        ),
        dwg.g(
            style="stroke:#000000;stroke-width:0.01mm;fill:#000000",
            fill_rule="evenodd",
        ),
    ]
    for i in range(4):
        offset = 0
        hand = state.hand[i]
        for tile, num in enumerate(hand):
            if num > 0:
                p = dwg.path(d=path_list[tile])
                p.translate(80 + offset * 30, 550)
                players_g[i].add(p)
                offset += 1
        players_g[i].rotate(
            angle=-90 * i,
            center=(BOARD_WIDTH * GRID_SIZE / 2, BOARD_WIDTH * GRID_SIZE / 2),
        )
        board_g.add(players_g[i])
    # pieces

    return board_g
