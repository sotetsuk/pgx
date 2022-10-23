from dataclasses import dataclass
from typing import Optional, Union

import svgwrite  # type: ignore
from svgwrite import cm

from .shogi import ShogiState

PIECES = [
    "歩",
    "香",
    "桂",
    "銀",
    "角",
    "飛",
    "金",
    "玉",
    "と",
    "成香",
    "成桂",
    "成銀",
    "馬",
    "龍",
    "歩",
    "香",
    "桂",
    "銀",
    "角",
    "飛",
    "金",
    "玉",
    "と",
    "成香",
    "成桂",
    "成銀",
    "馬",
    "龍",
]

NUM_TO_CHAR = [
    "一",
    "二",
    "三",
    "四",
    "五",
    "六",
    "七",
    "八",
    "九",
    "十",
]


@dataclass
class VisualizerConfig:
    p1_color: str = "black"
    p2_color: str = "white"
    p1_outline: str = "black"
    p2_outline: str = "black"
    background_color: str = "white"
    grid_color: str = "black"


class Visualizer:
    def __init__(
        self,
        state: Union[None, ShogiState] = None,
        color_mode: str = "light",
    ) -> None:
        self.state = state
        self.color_mode = color_mode

    def _repr_html_(self) -> str:
        assert self.state is not None
        return self._to_svg_string()

    def save_svg(self, filename="temp.svg") -> None:
        assert self.state is not None
        assert filename.endswith(".svg")
        self._to_dwg().saveas(filename=filename)

    def show_svg(
        self,
        state: Union[None, ShogiState] = None,
        color_mode: Optional[str] = None,
    ) -> None:
        import sys

        if "ipykernel" in sys.modules:
            # Jupyter Notebook
            from IPython.display import display_svg  # type:ignore

            display_svg(
                self._to_dwg(state=state, color_mode=color_mode).tostring(),
                raw=True,
            )
        else:
            # Not Jupyter
            sys.stdout.write("This function only works in Jupyter Notebook.")

    def set_state(self, state: ShogiState) -> None:
        self.state = state

    def _to_dwg(
        self,
        *,
        state: Union[None, ShogiState] = None,
        color_mode: Optional[str] = None,
    ) -> svgwrite.Drawing:
        if state is None:
            assert self.state is not None
            state = self.state
        BOARD_WIDTH = 9
        BOARD_HEIGHT = 9
        GRID_SIZE = 1

        if (
            color_mode is None and self.color_mode == "dark"
        ) or color_mode == "dark":
            color_set = VisualizerConfig(
                "dimgray",
                "black",
                "whitesmoke",
                "whitesmoke",
                "#202020",
                "white",
            )
        else:
            color_set = VisualizerConfig(
                "white", "lightgray", "black", "black", "white", "black"
            )

        dwg = svgwrite.Drawing(
            "temp.svg",
            (
                (BOARD_WIDTH + 2.5) * GRID_SIZE * cm,
                (BOARD_HEIGHT + 1) * GRID_SIZE * cm,
            ),
        )
        # background
        dwg.add(
            dwg.rect(
                (0, 0),
                (
                    (BOARD_WIDTH + 2.5) * GRID_SIZE * cm,
                    (BOARD_HEIGHT + 1) * GRID_SIZE * cm,
                ),
                fill=color_set.background_color,
            )
        )

        # board
        # grid
        board_g = dwg.g()
        hlines = board_g.add(dwg.g(id="hlines", stroke=color_set.grid_color))
        for y in range(1, BOARD_HEIGHT):
            hlines.add(
                dwg.line(
                    start=(0 * cm, GRID_SIZE * y * cm),
                    end=(
                        GRID_SIZE * BOARD_WIDTH * cm,
                        GRID_SIZE * y * cm,
                    ),
                    stroke_width=GRID_SIZE * 2,
                )
            )
        vlines = board_g.add(dwg.g(id="vline", stroke=color_set.grid_color))
        for x in range(1, BOARD_WIDTH):
            vlines.add(
                dwg.line(
                    start=(GRID_SIZE * x * cm, 0 * cm),
                    end=(
                        GRID_SIZE * x * cm,
                        GRID_SIZE * BOARD_HEIGHT * cm,
                    ),
                    stroke_width=GRID_SIZE * 2,
                )
            )
        board_g.add(
            dwg.rect(
                (0, 0),
                (
                    BOARD_WIDTH * GRID_SIZE * cm,
                    BOARD_HEIGHT * GRID_SIZE * cm,
                ),
                fill="none",
                stroke=color_set.grid_color,
                stroke_width=GRID_SIZE * 3,
            )
        )

        # dan,suji
        cord = board_g.add(dwg.g(id="cord", stroke=color_set.grid_color))
        for i in range(9):
            cord.add(
                dwg.text(
                    text=f"{NUM_TO_CHAR[i]}",
                    insert=(
                        (9.1) * GRID_SIZE * cm,
                        (i + 0.6) * GRID_SIZE * cm,
                    ),
                    font_size=f"{GRID_SIZE*14}px",
                    font_family="Serif",
                )
            )
            cord.add(
                dwg.text(
                    text=f"{i+1}",
                    insert=(
                        (8 - i + 0.4) * GRID_SIZE * cm,
                        (-0.1) * GRID_SIZE * cm,
                    ),
                    font_size=f"{GRID_SIZE*14}px",
                    font_family="Serif",
                )
            )

        # pieces
        p1_pieces_g = dwg.g()
        p2_pieces_g = dwg.g()
        for i, piece_pos, piece_type in zip(
            range(28),
            state.board[1:29],
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
                                    (x + 0.22) * GRID_SIZE * cm,
                                    (y + 0.45) * GRID_SIZE * cm,
                                ),
                                fill=stroke,
                                font_size=f"{GRID_SIZE*21}px",
                                font_family="Serif",
                            )
                        )
                        pieces_g.add(
                            dwg.text(
                                text=piece_type[1],
                                insert=(
                                    (x + 0.22) * GRID_SIZE * cm,
                                    (y + 0.95) * GRID_SIZE * cm,
                                ),
                                fill=stroke,
                                font_size=f"{GRID_SIZE*21}px",
                                font_family="Serif",
                            )
                        )
                    else:
                        pieces_g.add(
                            dwg.text(
                                text=piece_type,
                                insert=(
                                    (x + 0.05) * GRID_SIZE * cm,
                                    (y + 0.85) * GRID_SIZE * cm,
                                ),
                                fill=stroke,
                                font_size=f"{GRID_SIZE*35}px",
                                font_family="Serif",
                            )
                        )

        # hand
        p1_hand = ["☗", "先", "手", ""]
        p2_hand = ["☖", "後", "手", ""]

        # 成り駒をソートする処理
        p1_hand, p2_hand = self._sort_pieces(state, p1_hand, p2_hand)

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
                            (9.5 - i * 0.4) * GRID_SIZE * cm,
                            (9 - (offset - j) * 0.7) * GRID_SIZE * cm,
                        ),
                        fill=stroke,
                        font_size=f"{GRID_SIZE*28}px",
                        font_family="Serif",
                    )
                )

        board_g.add(p1_pieces_g)
        p2_pieces_g.rotate(angle=180)
        p2_pieces_g.translate(
            -GRID_SIZE * 340, -GRID_SIZE * 340
        )  # no units allowed
        board_g.add(p2_pieces_g)

        board_g.translate(GRID_SIZE * 35, GRID_SIZE * 20)  # no units allowed
        dwg.add(board_g)

        return dwg

    def _to_svg_string(self) -> str:
        return self._to_dwg(color_mode=self.color_mode).tostring()

    def _sort_pieces(self, state, p1_hand, p2_hand):
        """
        ShogiStateのhandを飛、角、金、銀、桂、香、歩の順にする
        """
        hands = state.hand[::-1]
        hands[0], hands[1], hands[2], hands[7], hands[8], hands[9] = (
            hands[1],
            hands[2],
            hands[0],
            hands[8],
            hands[9],
            hands[7],
        )
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
