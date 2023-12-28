from pgx._src.dwg.mahjong_tile import TilePath
from pgx.mahjong._action import Action
from pgx.mahjong._env import State as MahjongState
from pgx.mahjong._meld import Meld

path_list = TilePath.str_list
tile_w = 30
tile_h = 45
hand_x = 120
hand_y = 640
wind = ["東", "南", "西", "北"]


def _make_mahjong_dwg(dwg, state: MahjongState, config):
    GRID_SIZE = config["GRID_SIZE"]
    BOARD_WIDTH = config["BOARD_WIDTH"]
    BOARD_HEIGHT = config["BOARD_HEIGHT"]
    color_set = config["COLOR_SET"]
    board_g = dwg.g(
        style="stroke:#000000;stroke-width:0.01mm;fill:#000000",
        fill_rule="evenodd",
    )

    # background
    board_g.add(
        dwg.rect(
            (0, 0),
            (
                BOARD_WIDTH * GRID_SIZE,
                BOARD_HEIGHT * GRID_SIZE,
            ),
            fill=color_set.background_color,
        )
    )
    # central info
    width = 180
    board_g.add(
        dwg.rect(
            (
                (BOARD_WIDTH * GRID_SIZE - width) / 2,
                (BOARD_HEIGHT * GRID_SIZE - width) / 2,
            ),
            (width, width),
            fill=color_set.background_color,
            stroke=color_set.grid_color,
            stroke_width="2px",
            rx="3px",
            ry="3px",
        )
    )
    kanji = ["", "一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]
    ro = state._round
    round = f"{wind[ro//4]}{kanji[ro%4+1]}局"
    if state._honba > 0:
        round += f"{kanji[state._honba]}本場"

    fontsize = 20
    y = -25
    board_g.add(
        dwg.text(
            text=round,
            insert=(
                (BOARD_WIDTH * GRID_SIZE) / 2 - len(round) * fontsize / 2,
                (BOARD_HEIGHT * GRID_SIZE) / 2 + y,
            ),
            fill=color_set.text_color,
            font_size=f"{fontsize}px",
            font_family="serif",
        )
    )

    # dora
    dora_scale = 0.6
    x = (BOARD_WIDTH * GRID_SIZE) / 2 - tile_w * dora_scale * 2.5
    y = (BOARD_WIDTH * GRID_SIZE) / 2 - 15
    for _x, dora in enumerate(state._doras):
        if dora == -1:
            dora = 34
        p = dwg.path(d=path_list[dora])
        p.translate(x + _x * tile_w * dora_scale, y)
        p.scale(dora_scale)
        board_g.add(p)

    # yama
    yama_scale = 0.6
    x = (BOARD_WIDTH * GRID_SIZE) / 2 - 25
    y = (BOARD_WIDTH * GRID_SIZE) / 2 + 22
    fontsize = 20
    p = dwg.path(d=path_list[34])
    p.translate(x, y)
    p.scale(yama_scale)
    board_g.add(p)
    board_g.add(
        dwg.text(
            text=f"x {state._next_deck_ix-14+1}",
            insert=(x + tile_w * yama_scale + 5, y + tile_h * yama_scale - 5),
            fill=color_set.text_color,
            font_size=f"{fontsize}px",
            font_family="serif",
        )
    )

    # board
    for i in range(4):
        players_g = _make_players_dwg(dwg, state, i, color_set, BOARD_WIDTH, BOARD_HEIGHT, GRID_SIZE)
        players_g.rotate(
            angle=-90 * i,
            center=(BOARD_WIDTH * GRID_SIZE / 2, BOARD_WIDTH * GRID_SIZE / 2),
        )
        board_g.add(players_g)

    return board_g


def _make_players_dwg(
    dwg,
    state: MahjongState,
    i,
    color_set,
    BOARD_WIDTH,
    BOARD_HEIGHT,
    GRID_SIZE,
):
    players_g = dwg.g(
        style="stroke:#000000;stroke-width:0.01mm;fill:#000000",
        fill_rule="evenodd",
    )

    # wind
    x = 265
    y = 435
    fontsize = 22
    players_g.add(
        dwg.text(
            text=wind[(i - state._oya) % 4],
            insert=(x, y),
            fill=color_set.text_color,
            font_size=f"{fontsize}px",
            font_family="serif",
        )
    )

    # score
    fontsize = 20
    score = str(int(state._score[i]) * 100)
    y = 70
    players_g.add(
        dwg.text(
            text=score,
            insert=(
                (BOARD_WIDTH * GRID_SIZE) / 2 - len(score) * fontsize / 4,
                BOARD_HEIGHT * GRID_SIZE / 2 + y,
            ),
            fill=color_set.text_color,
            font_size=f"{fontsize}px",
            font_family="serif",
        )
    )

    # riichi bou
    width = 100
    height = 10
    y = 75
    if state._riichi[i]:
        players_g.add(
            dwg.rect(
                (
                    (BOARD_WIDTH * GRID_SIZE - width) / 2,
                    BOARD_HEIGHT * GRID_SIZE / 2 + y,
                ),
                (width, height),
                fill=color_set.background_color,
                stroke=color_set.grid_color,
                stroke_width="1px",
                rx="3px",
                ry="3px",
            )
        )
        players_g.add(
            dwg.circle(
                center=(
                    BOARD_HEIGHT * GRID_SIZE / 2,
                    BOARD_HEIGHT * GRID_SIZE / 2 + y + height / 2,
                ),
                r="3px",
                fill="red",
            )
        )

    # hand
    offset = 0
    hand = state._hand[i]
    for tile, num in enumerate(hand):
        for _ in range(num):
            p = dwg.path(d=path_list[tile])
            p.translate(hand_x + offset, hand_y)
            players_g.add(p)
            offset += tile_w

    offset += tile_w

    # meld
    for meld in state._melds[i]:
        if meld == 0:
            continue
        if Meld.action(meld) == Action.PON:
            players_g, offset = _apply_pon(dwg, players_g, meld, offset)
        elif (
            (Meld.action(meld) == Action.CHI_L)
            or (Meld.action(meld) == Action.CHI_M)
            or (Meld.action(meld) == Action.CHI_R)
        ):
            players_g, offset = _apply_chi(dwg, players_g, meld, offset)
        elif (34 <= Meld.action(meld) <= 67) and Meld.src(meld) == 0:
            players_g, offset = _apply_ankan(dwg, players_g, meld, offset)
        elif 34 <= Meld.action(meld) <= 67:
            players_g, offset = _apply_kakan(dwg, players_g, meld, offset)
        elif Meld.action(meld) == Action.MINKAN:
            players_g, offset = _apply_minkan(dwg, players_g, meld, offset)

    # river
    x = BOARD_WIDTH * GRID_SIZE / 2 - 3 * tile_w
    y = 450

    river = state._river[i]
    for river_ix, tile in enumerate(river):
        fill = "black"
        if (tile >> 7) & 0b1:
            fill = "gray"
            tile &= 0b01111111

        if (tile >> 6) & 0b1:
            # riichi
            p = dwg.path(d=path_list[tile & 0b10111111], fill=fill)
            p.rotate(angle=-90, center=(x, y))
            p.translate(x - tile_h + 4, y + 2)
            players_g.add(p)
            x += tile_h

        elif tile < 34:
            p = dwg.path(d=path_list[tile], fill=fill)
            p.translate(x, y)
            players_g.add(p)
            x += tile_w

        if river_ix % 6 == 5:
            x = BOARD_WIDTH * GRID_SIZE / 2 - 3 * tile_w
            y += tile_h
    return players_g


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
    return g, offset + tile_w


def _apply_chi(dwg, g, meld, offset):
    tile = Meld.target(meld)
    if Meld.action(meld) == Action.CHI_L:
        tile1 = tile
        tile2 = tile + 1
        tile3 = tile + 2
    elif Meld.action(meld) == Action.CHI_M:
        tile1 = tile
        tile2 = tile - 1
        tile3 = tile + 1
    else:
        tile1 = tile
        tile2 = tile - 1
        tile3 = tile - 2

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
        p = dwg.path(d=path_list[tile2])
        p.translate(hand_x + offset, hand_y)
        g.add(p)
        offset += tile_w
        p = dwg.path(d=path_list[tile1])
        p.rotate(-90, center=(hand_x + offset, hand_y))
        p.translate(hand_x + offset - tile_h + 4, hand_y + 1)
        g.add(p)
        offset += tile_h
        p = dwg.path(d=path_list[tile3])
        p.translate(hand_x + offset, hand_y)
        g.add(p)
        offset += tile_w
    elif Meld.src(meld) == 1:
        p = dwg.path(d=path_list[tile3])
        p.translate(hand_x + offset, hand_y)
        g.add(p)
        offset += tile_w
        p = dwg.path(d=path_list[tile2])
        p.translate(hand_x + offset, hand_y)
        g.add(p)
        offset += tile_w
        p = dwg.path(d=path_list[tile1])
        p.rotate(-90, center=(hand_x + offset, hand_y))
        p.translate(hand_x + offset - tile_h + 4, hand_y + 1)
        g.add(p)
        offset += tile_h
    return g, offset + tile_w


def _apply_ankan(dwg, g, meld, offset):
    tile = Meld.target(meld)
    p = dwg.path(d=path_list[tile])
    p.translate(hand_x + offset, hand_y)
    g.add(p)
    offset += tile_w
    p = dwg.path(d=TilePath.back)
    p.translate(hand_x + offset, hand_y)
    g.add(p)
    offset += tile_w
    p = dwg.path(d=TilePath.back)
    p.translate(hand_x + offset, hand_y)
    g.add(p)
    offset += tile_w
    p = dwg.path(d=path_list[tile])
    p.translate(hand_x + offset, hand_y)
    g.add(p)
    offset += tile_w

    return g, offset + tile_w


def _apply_kakan(dwg, g, meld, offset):
    tile = Meld.target(meld)
    if Meld.src(meld) == 3:
        p = dwg.path(d=path_list[tile])
        p.rotate(-90, center=(hand_x + offset, hand_y))
        p.translate(hand_x + offset - tile_h + 4, hand_y + 1)
        g.add(p)
        p = dwg.path(d=path_list[tile])
        p.rotate(-90, center=(hand_x + offset, hand_y))
        p.translate(hand_x + offset - tile_h + 4, 0)
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
        p = dwg.path(d=path_list[tile])
        p.rotate(-90, center=(hand_x + offset, hand_y))
        p.translate(hand_x + offset - tile_h + 4, 0)
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
        p = dwg.path(d=path_list[tile])
        p.rotate(-90, center=(hand_x + offset, hand_y))
        p.translate(hand_x + offset - tile_h + 4, 0)
        g.add(p)
        offset += tile_h
    return g, offset + tile_w


def _apply_minkan(dwg, g, meld, offset):
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
        p.translate(hand_x + offset, hand_y)
        g.add(p)
        offset += tile_w
        p = dwg.path(d=path_list[tile])
        p.rotate(-90, center=(hand_x + offset, hand_y))
        p.translate(hand_x + offset - tile_h + 4, hand_y + 1)
        g.add(p)
        offset += tile_h
    return g, offset + tile_w
