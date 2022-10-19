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

PIECES_FULL = [
    "歩兵",
    "香車",
    "桂馬",
    "銀将",
    "角行",
    "飛車",
    "金将",
    "玉将",
    "と金",
    "成香",
    "成桂",
    "成銀",
    "竜馬",
    "龍王",
    "歩兵",
    "香車",
    "桂馬",
    "銀将",
    "角行",
    "飛車",
    "金将",
    "玉将",
    "と金",
    "成香",
    "成桂",
    "成銀",
    "竜馬",
    "龍王",
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
        GRID_SIZE = 2

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
                (BOARD_WIDTH + 2) * GRID_SIZE * cm,
                (BOARD_HEIGHT + 1) * GRID_SIZE * cm,
            ),
        )
        # background
        dwg.add(
            dwg.rect(
                (0, 0),
                (
                    (BOARD_WIDTH + 2) * GRID_SIZE * cm,
                    (BOARD_HEIGHT + 1) * GRID_SIZE * cm,
                ),
                fill=color_set.background_color,
            )
        )

        # board
        # grid
        board_g = dwg.g()
        hlines = board_g.add(dwg.g(id="hlines", stroke=color_set.grid_color))
        for y in range(BOARD_HEIGHT + 1):
            hlines.add(
                dwg.line(
                    start=(0 * cm, GRID_SIZE * y * cm),
                    end=(
                        GRID_SIZE * BOARD_WIDTH * cm,
                        GRID_SIZE * y * cm,
                    ),
                )
            )
        vlines = board_g.add(dwg.g(id="vline", stroke=color_set.grid_color))
        for x in range(BOARD_WIDTH + 1):
            vlines.add(
                dwg.line(
                    start=(GRID_SIZE * x * cm, 0 * cm),
                    end=(
                        GRID_SIZE * x * cm,
                        GRID_SIZE * BOARD_HEIGHT * cm,
                    ),
                )
            )

        # pieces
        p1_pieces_g = dwg.g()
        p2_pieces_g = dwg.g()
        for i, piece_pos, piece_type in zip(
            range(28),
            state.board[1:29],
            PIECES_FULL,
        ):
            for xy, is_set in enumerate(piece_pos):
                if is_set == 1:
                    if i < 14:
                        pieces_g = p1_pieces_g
                        x = 8 - xy // BOARD_HEIGHT  # ShogiStateは右上原点
                        y = xy % BOARD_HEIGHT
                        fill_color = color_set.p1_color
                        stroke = color_set.p1_outline
                    else:
                        pieces_g = p2_pieces_g
                        x = xy // BOARD_HEIGHT
                        y = 8 - xy % BOARD_HEIGHT
                        fill_color = color_set.p2_color
                        stroke = color_set.p2_outline

                    # cm = 37.8
                    # pieces_g.add(
                    #    dwg.polygon(
                    #        points=[
                    #            (
                    #                (x + 0.5) * GRID_SIZE * _cm,
                    #                (y + 0.1) * GRID_SIZE * _cm,
                    #            ),
                    #            (
                    #                (x + 0.3) * GRID_SIZE * _cm,
                    #                (y + 0.2) * GRID_SIZE * _cm,
                    #            ),
                    #            (
                    #                (x + 0.1) * GRID_SIZE * _cm,
                    #                (y + 0.9) * GRID_SIZE * _cm,
                    #            ),
                    #            (
                    #                (x + 0.9) * GRID_SIZE * _cm,
                    #                (y + 0.9) * GRID_SIZE * _cm,
                    #            ),
                    #            (
                    #                (x + 0.7) * GRID_SIZE * _cm,
                    #                (y + 0.2) * GRID_SIZE * _cm,
                    #            ),
                    #        ],
                    #        stroke=stroke,
                    #        fill=fill_color,
                    #    )
                    # )
                    pieces_g.add(
                        dwg.text(
                            text="☖",
                            insert=(
                                (x + 0.05) * GRID_SIZE * cm,
                                (y + 0.8) * GRID_SIZE * cm,
                            ),
                            fill=stroke,
                            font_size="4.8em",
                            font_family="Serif",
                        )
                    )
                    pieces_g.add(
                        dwg.text(
                            text=piece_type[0],
                            insert=(
                                (x + 0.3) * GRID_SIZE * cm,
                                (y + 0.45) * GRID_SIZE * cm,
                            ),
                            fill=stroke,
                            font_size="2em",
                            font_family="Serif",
                        )
                    )
                    pieces_g.add(
                        dwg.text(
                            text=piece_type[1],
                            insert=(
                                (x + 0.3) * GRID_SIZE * cm,
                                (y + 0.78) * GRID_SIZE * cm,
                            ),
                            fill=stroke,
                            font_size="2em",
                            font_family="Serif",
                        )
                    )

        # hand
        for i, piece_num, piece_type in zip(
            range(14),
            state.hand,
            [
                "歩",
                "香",
                "桂",
                "銀",
                "角",
                "飛",
                "金",
                "歩",
                "香",
                "桂",
                "銀",
                "角",
                "飛",
                "金",
            ],
        ):
            _g = p1_pieces_g if i < 7 else p2_pieces_g
            _g.add(
                dwg.text(
                    text=f"{piece_type}:{piece_num}",
                    insert=(
                        9.1 * GRID_SIZE * cm,
                        (6 + (i % 7) * 0.5) * GRID_SIZE * cm,
                    ),
                    fill=color_set.p1_outline,
                    font_size="2em",
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
