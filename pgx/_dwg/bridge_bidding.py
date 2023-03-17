import numpy as np

from pgx._bridge_bidding import State as BridgeBiddingState


def _make_bridge_dwg(dwg, state: BridgeBiddingState, config):
    NUM_CARD_TYPE = 13
    # fmt: off
    TO_CARD = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
    SUITS = ["\u2660", "\u2665", "\u2666", "\u2663", "N"]  # ♠♡♢♣
    ACT = ["P", "X", "XX"]
    # fmt:on
    color_set = config["COLOR_SET"]

    # board
    board_g = dwg.g()

    # hand
    x_offset = [235, 480, 235, -10]
    y_offset = [10, 190, 370, 190]
    area_width = 230
    area_height = 150
    for i in range(4):  # player0,1,2,3
        hand = sorted(state.hand[i * NUM_CARD_TYPE : (i + 1) * NUM_CARD_TYPE])
        assert len(hand) == NUM_CARD_TYPE
        # player
        pos = np.array(["North", "East", "South", "West"], dtype=object)
        pos[state.dealer] = pos[state.dealer] + "(Dealer)"
        newline_offset = 0
        over_offset = 0

        for j in range(4):  # spades,hearts,diamonds,clubs
            # h line
            board_g.add(
                dwg.line(
                    start=(
                        x_offset[i],
                        y_offset[i] + 10 + 30 * j + newline_offset,
                    ),
                    end=(
                        x_offset[i] + area_width,
                        y_offset[i] + 10 + 30 * j + newline_offset,
                    ),
                    stroke=color_set.grid_color,
                    stroke_width="2px",
                )
            )

            # suit
            board_g.add(
                dwg.text(
                    text=SUITS[j],
                    insert=(
                        x_offset[i] + 9,
                        y_offset[i] + 30 * (j + 1) + newline_offset,
                    ),
                    fill="orangered" if 0 < j < 3 else color_set.text_color,
                    font_size="26px",
                    font_family="Courier",
                    font_weight="bold",
                )
            )

            # card
            card = [
                TO_CARD[i % NUM_CARD_TYPE]
                for i in hand
                if j * NUM_CARD_TYPE <= i < (j + 1) * NUM_CARD_TYPE
            ][::-1]
            if card != [] and card[-1] == "A":
                card = card[-1:] + card[:-1]
            card_str = " ".join(card)
            card_num = len(card_str)
            if card_num > 12 and i % 2 == 1:
                board_g.add(
                    dwg.text(
                        text=card_str[:12],
                        insert=(
                            x_offset[i] + 40,
                            y_offset[i] + 30 * (j + 1) + newline_offset,
                        ),
                        fill="orangered"
                        if 0 < j < 3
                        else color_set.text_color,
                        font_size="24px",
                        font_family="Courier",
                        font_weight="bold",
                    )
                )
                newline_offset += 30
                board_g.add(
                    dwg.text(
                        text=card_str[12:],
                        insert=(
                            x_offset[i] + 40,
                            y_offset[i] + 30 * (j + 1) + newline_offset,
                        ),
                        fill="orangered"
                        if 0 < j < 3
                        else color_set.text_color,
                        font_size="24px",
                        font_family="Courier",
                        font_weight="bold",
                    )
                )
            else:
                if card_num > 12 and i % 2 == 0:
                    over_offset = (card_num - 12) * 14
                    # 延長した分横線を埋める
                    for k in range(4):
                        board_g.add(
                            dwg.line(
                                start=(
                                    x_offset[i] + area_width,
                                    y_offset[i] + 10 + 30 * k + newline_offset,
                                ),
                                end=(
                                    x_offset[i] + area_width + over_offset,
                                    y_offset[i] + 10 + 30 * k + newline_offset,
                                ),
                                stroke=color_set.grid_color,
                                stroke_width="2px",
                            )
                        )
                board_g.add(
                    dwg.text(
                        text=card_str,
                        insert=(
                            x_offset[i] + 40,
                            y_offset[i] + 30 * (j + 1) + newline_offset,
                        ),
                        fill="orangered"
                        if 0 < j < 3
                        else color_set.text_color,
                        font_size="24px",
                        font_family="Courier",
                        font_weight="bold",
                    )
                )

        # rect
        board_g.add(
            dwg.rect(
                (x_offset[i], y_offset[i] - 20),
                (area_width + over_offset, area_height + newline_offset),
                rx="5px",
                ry="5px",
                fill="none",
                stroke=color_set.grid_color,
                stroke_width="2px",
            )
        )

        # v line
        board_g.add(
            dwg.line(
                start=(x_offset[i] + 32, y_offset[i] + 10),
                end=(x_offset[i] + 32, y_offset[i] + 130 + newline_offset),
                stroke=color_set.grid_color,
                stroke_width="2px",
            )
        )

        # pos
        board_g.add(
            dwg.text(
                text=pos[i],
                insert=(x_offset[i] + 10, y_offset[i]),
                fill=color_set.grid_color,
                font_size="20px",
                font_family="Courier",
                font_weight="bold",
            )
        )

        # val
        if (state.vul_NS and i % 2 == 0) or (state.vul_EW and i % 2 == 1):
            board_g.add(
                dwg.text(
                    text="Vul.",
                    insert=(
                        x_offset[i] + area_width + over_offset - 50,
                        y_offset[i],
                    ),
                    fill="orangered",
                    font_size="20px",
                    font_family="Courier",
                    font_weight="bold",
                )
            )

    # history
    _x = 245
    _y = 155
    _w = 210
    _h = 180
    board_g.add(
        dwg.rect(
            (_x, _y),
            (_w, _h),
            rx="5px",
            ry="5px",
            fill="none",
            stroke=color_set.grid_color,
            stroke_width="5px",
        )
    )
    for i, act in enumerate(state.bidding_history):
        if act == -1:
            break
        act_str = (
            str(act // 5 + 1) + SUITS[act % 5]
            if 0 <= act < 35
            else ACT[act - 35]
        )
        color = (
            "orangered"
            if act % 5 == 1 or act % 5 == 2
            else color_set.text_color
        )
        board_g.add(
            dwg.text(
                text=act_str,
                insert=(_x + 15 + 50 * (i % 4), _y + 50 + 20 * (i // 4)),
                fill=color,
                font_size="20px",
                font_family="Courier",
            )
        )
    board_g.add(
        dwg.line(
            start=(_x, _y + 30),
            end=(_x + _w, _y + 30),
            stroke=color_set.grid_color,
            stroke_width="2px",
        )
    )
    # player
    pos = np.array(["N", "E", "S", "W"], dtype=object)
    pos = np.roll(pos, -state.dealer)
    pos[0] = pos[0] + "(D)"
    for i in range(4):
        board_g.add(
            dwg.text(
                text=pos[i],
                insert=(_x + 15 + 50 * (i % 4), _y + 20),
                fill=color_set.text_color,
                font_size="20px",
                font_family="Courier",
                font_weight="bold",
            )
        )

    return board_g
