from pgx._mahjong._action import Action
from pgx._mahjong._mahjong2 import State as MahjongState
from pgx._mahjong._meld import Meld
from pgx._src.dwg.mahjong_tile import TilePath

path_list = TilePath.str_list
tile_w = 30
tile_h = 45
hand_x = 150
hand_y = 640


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

        # hand
        offset = 0
        hand = state.hand[i]
        for tile, num in enumerate(hand):
            if num > 0:
                p = dwg.path(d=path_list[tile])
                p.translate(hand_x + offset, hand_y)
                players_g[i].add(p)
                offset += tile_w

        offset += tile_w

        # meld
        for meld in state.melds[i]:
            if meld == 0:
                continue
            if Meld.action(meld) == Action.PON:
                players_g[i], offset = _apply_pon(
                    dwg, players_g[i], meld, offset
                )
            elif (
                (Meld.action(meld) == Action.CHI_L)
                or (Meld.action(meld) == Action.CHI_M)
                or (Meld.action(meld) == Action.CHI_R)
            ):
                players_g[i], offset = _apply_chi(
                    dwg, players_g[i], meld, offset
                )

        # river
        offset = 0
        river = state.river[i]
        for tile in river:
            if tile >= 0:
                p = dwg.path(d=path_list[tile])
                p.translate(
                    BOARD_WIDTH * GRID_SIZE / 2 + (offset % 6 - 3) * tile_w,
                    450 + (offset // 6) * tile_h,
                )
                players_g[i].add(p)
                offset += 1
        players_g[i].rotate(
            angle=-90 * i,
            center=(BOARD_WIDTH * GRID_SIZE / 2, BOARD_WIDTH * GRID_SIZE / 2),
        )
        board_g.add(players_g[i])
    # pieces

    return board_g


def _apply_pon(dwg, g, meld, offset):
    tile = Meld.target(meld)
    if Meld.src(meld) == 3:
        p = dwg.path(d=path_list[tile])
        p.rotate(-90, center=(hand_x + offset, hand_y))
        p.translate(hand_x + offset - tile_h + 4, hand_y + 1)
        g.add(p)
        offset += tile_h
        p = dwg.path(d=path_list[tile])
        p.translate(hand_x + offset, hand_y)
        g.add(p)
        offset += tile_w
        p = dwg.path(d=path_list[tile])
        p.translate(hand_x + offset, hand_y)
        g.add(p)
        offset += tile_w
    elif Meld.src(meld) == 2:
        p = dwg.path(d=path_list[tile])
        p.translate(hand_x + offset, hand_y)
        g.add(p)
        offset += tile_w
        p = dwg.path(d=path_list[tile])
        p.rotate(-90, center=(hand_x + offset, hand_y))
        p.translate(hand_x + offset - tile_h + 4, hand_y + 1)
        g.add(p)
        offset += tile_h
        p = dwg.path(d=path_list[tile])
        p.translate(hand_x + offset, hand_y)
        g.add(p)
        offset += tile_w
    elif Meld.src(meld) == 1:
        p = dwg.path(d=path_list[tile])
        p.translate(hand_x + offset, hand_y)
        g.add(p)
        offset += tile_w
        p = dwg.path(d=path_list[tile])
        p.translate(hand_x + offset, hand_y)
        g.add(p)
        offset += tile_w
        p = dwg.path(d=path_list[tile])
        p.rotate(-90, center=(hand_x + offset, hand_y))
        p.translate(hand_x + offset - tile_h + 4, hand_y + 1)
        g.add(p)
        offset += tile_h
    return g, offset


def _apply_chi(dwg, g, meld, offset):
    tile = Meld.target(meld)
    if Meld.action(meld) == Action.CHI_L:
        tile1 = tile
        tile2 = tile + 1
        tile3 = tile + 2
    elif Meld.action(meld) == Action.CHI_M:
        tile1 = tile - 1
        tile2 = tile
        tile3 = tile + 1
    else:
        tile1 = tile - 2
        tile2 = tile - 1
        tile3 = tile

    if Meld.src(meld) == 3:
        p = dwg.path(d=path_list[tile1])
        p.rotate(-90, center=(hand_x + offset, hand_y))
        p.translate(hand_x + offset - tile_h + 4, hand_y + 1)
        g.add(p)
        offset += tile_h
        p = dwg.path(d=path_list[tile2])
        p.translate(hand_x + offset, hand_y)
        g.add(p)
        offset += tile_w
        p = dwg.path(d=path_list[tile3])
        p.translate(hand_x + offset, hand_y)
        g.add(p)
        offset += tile_w
    elif Meld.src(meld) == 2:
        p = dwg.path(d=path_list[tile1])
        p.translate(hand_x + offset, hand_y)
        g.add(p)
        offset += tile_w
        p = dwg.path(d=path_list[tile2])
        p.rotate(-90, center=(hand_x + offset, hand_y))
        p.translate(hand_x + offset - tile_h + 4, hand_y + 1)
        g.add(p)
        offset += tile_h
        p = dwg.path(d=path_list[tile3])
        p.translate(hand_x + offset, hand_y)
        g.add(p)
        offset += tile_w
    elif Meld.src(meld) == 1:
        p = dwg.path(d=path_list[tile1])
        p.translate(hand_x + offset, hand_y)
        g.add(p)
        offset += tile_w
        p = dwg.path(d=path_list[tile2])
        p.translate(hand_x + offset, hand_y)
        g.add(p)
        offset += tile_w
        p = dwg.path(d=path_list[tile3])
        p.rotate(-90, center=(hand_x + offset, hand_y))
        p.translate(hand_x + offset - tile_h + 4, hand_y + 1)
        g.add(p)
        offset += tile_h
    return g, offset
