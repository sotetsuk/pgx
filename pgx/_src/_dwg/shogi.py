import numpy as np

from pgx.shogi import State as ShogiState


def _make_shogi_dwg(dwg, state: ShogiState, config):  # noqa: C901
    if state.turn == 1:
        from pgx.shogi import _flip

        state = _flip(state)
    # fmt: off
    PIECES = ["歩", "香", "桂", "銀", "角", "飛", "金", "玉", "と", "成香", "成桂", "成銀", "馬", "龍"] * 2
    NUM_TO_CHAR = ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]
    # fmt: on

    def _sort_pieces(state, p1_hand, p2_hand):
        """
        ShogiStateのhandを飛、角、金、銀、桂、香、歩の順にする
        """
        hands = state.hand.flatten()[::-1]
        tmp = hands
        hands = hands.at[0].set(tmp[1])
        hands = hands.at[1].set(tmp[2])
        hands = hands.at[2].set(tmp[0])
        hands = hands.at[7].set(tmp[8])
        hands = hands.at[8].set(tmp[9])
        hands = hands.at[9].set(tmp[7])
        pieces = PIECES[6::-1]
        pieces[0], pieces[1], pieces[2] = pieces[1], pieces[2], pieces[0]

        for i, piece_num, piece_type in zip(range(14), hands, pieces + pieces):
            hand = p2_hand if i < 7 else p1_hand
            if piece_num == 10:
                hand.append(piece_type)
                hand.append("十")
            elif piece_num > 0:
                hand.append(piece_type)
                if piece_num > 9:
                    hand.append("十")
                if piece_num > 1:
                    hand.append(NUM_TO_CHAR[piece_num % 10 - 1])

        return p1_hand, p2_hand

    GRID_SIZE = config["GRID_SIZE"]
    BOARD_WIDTH = 9
    BOARD_HEIGHT = 9
    color_set = config["COLOR_SET"]

    # background
    dwg.add(
        dwg.rect(
            (0, 0),
            (
                (BOARD_WIDTH + 2.5) * GRID_SIZE,
                (BOARD_HEIGHT + 1) * GRID_SIZE,
            ),
            fill=color_set.background_color,
        )
    )

    # board
    # grid
    board_g = dwg.g()
    hlines = board_g.add(dwg.g(id="hlines", stroke=color_set.grid_color))
    for y in range(1, 9):
        hlines.add(
            dwg.line(
                start=(0, GRID_SIZE * y),
                end=(
                    GRID_SIZE * BOARD_WIDTH,
                    GRID_SIZE * y,
                ),
                stroke_width="2px",
            )
        )
    vlines = board_g.add(dwg.g(id="vline", stroke=color_set.grid_color))
    for x in range(1, 9):
        vlines.add(
            dwg.line(
                start=(GRID_SIZE * x, 0),
                end=(
                    GRID_SIZE * x,
                    GRID_SIZE * BOARD_HEIGHT,
                ),
                stroke_width="2px",
            )
        )
    board_g.add(
        dwg.rect(
            (0, 0),
            (
                BOARD_WIDTH * GRID_SIZE,
                BOARD_HEIGHT * GRID_SIZE,
            ),
            fill="none",
            stroke=color_set.grid_color,
            stroke_width="4px",
        )
    )

    # dan,suji
    cord = board_g.add(dwg.g(id="cord", fill=color_set.grid_color))
    for i in range(9):
        cord.add(
            dwg.text(
                text=f"{NUM_TO_CHAR[i]}",
                insert=(
                    (9.1) * GRID_SIZE,
                    (i + 0.6) * GRID_SIZE,
                ),
                font_size="20px",
                font_family="Serif",
            )
        )
        cord.add(
            dwg.text(
                text=f"{i+1}",
                insert=(
                    (8 - i + 0.4) * GRID_SIZE,
                    (-0.1) * GRID_SIZE,
                ),
                font_size="20px",
                font_family="Serif",
            )
        )

    # pieces
    p1_pieces_g = dwg.g()
    p2_pieces_g = dwg.g()
    one_hot_board = np.zeros((29, 81))
    board = state.piece_board
    for i in range(81):
        piece = board[i]
        one_hot_board[piece, i] = 1
    for i, piece_pos, piece_type in zip(
        range(28),
        one_hot_board,
        PIECES,
    ):
        for xy, is_set in enumerate(piece_pos):
            if is_set == 1:
                if i < 14:
                    pieces_g = p1_pieces_g
                    x = 8 - xy // BOARD_HEIGHT  # ShogiStateは右上原点
                    y = xy % BOARD_HEIGHT
                    stroke = color_set.p1_outline
                else:
                    pieces_g = p2_pieces_g
                    x = xy // BOARD_HEIGHT
                    y = 8 - xy % BOARD_HEIGHT
                    stroke = color_set.p2_outline

                if len(piece_type) > 1:
                    pieces_g.add(
                        dwg.text(
                            text=piece_type[0],
                            insert=(
                                (x + 0.22) * GRID_SIZE,
                                (y + 0.45) * GRID_SIZE,
                            ),
                            fill=stroke,
                            font_size="26px",
                            font_family="Serif",
                        )
                    )
                    pieces_g.add(
                        dwg.text(
                            text=piece_type[1],
                            insert=(
                                (x + 0.22) * GRID_SIZE,
                                (y + 0.95) * GRID_SIZE,
                            ),
                            fill=stroke,
                            font_size="26px",
                            font_family="Serif",
                        )
                    )
                else:
                    pieces_g.add(
                        dwg.text(
                            text=piece_type,
                            insert=(
                                (x + 0.05) * GRID_SIZE,
                                (y + 0.85) * GRID_SIZE,
                            ),
                            fill=stroke,
                            font_size="45px",
                            font_family="Serif",
                        )
                    )

    # hand
    p1_hand = ["☗", "先", "手", ""]
    p2_hand = ["☖", "後", "手", ""]

    # 成り駒をソートする処理
    p1_hand, p2_hand = _sort_pieces(state, p1_hand, p2_hand)

    for i in range(2):
        if i == 0:
            pieces_g = p1_pieces_g
            hand = p1_hand
            stroke = color_set.p1_outline
            offset = len(p1_hand)
        else:
            pieces_g = p2_pieces_g
            hand = p2_hand
            stroke = color_set.p2_outline
            offset = len(p2_hand)
        for j, txt in enumerate(hand):
            pieces_g.add(
                dwg.text(
                    text=txt,
                    insert=(
                        (9.5 - i * 0.4) * GRID_SIZE,
                        (9.5 - (offset - j) * 0.7) * GRID_SIZE,
                    ),
                    fill=stroke,
                    font_size="34px",
                    font_family="Serif",
                )
            )

    board_g.add(p1_pieces_g)
    p2_pieces_g.rotate(
        angle=180,
        center=(BOARD_WIDTH * GRID_SIZE / 2, BOARD_HEIGHT * GRID_SIZE / 2),
    )
    board_g.add(p2_pieces_g)
    board_g.translate(15, 0)

    return board_g
