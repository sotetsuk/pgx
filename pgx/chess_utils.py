"""
Original source:
https://commons.wikimedia.org/wiki/File:Chess_kdt45.svg
https://commons.wikimedia.org/wiki/File:Chess_klt45.svg
https://commons.wikimedia.org/wiki/File:Chess_qdt45.svg
https://commons.wikimedia.org/wiki/File:Chess_qlt45.svg
https://commons.wikimedia.org/wiki/File:Chess_rdt45.svg
https://commons.wikimedia.org/wiki/File:Chess_rlt45.svg
https://commons.wikimedia.org/wiki/File:Chess_bdt45.svg
https://commons.wikimedia.org/wiki/File:Chess_blt45.svg
https://commons.wikimedia.org/wiki/File:Chess_ndt45.svg
https://commons.wikimedia.org/wiki/File:Chess_nlt45.svg
https://commons.wikimedia.org/wiki/File:Chess_pdt45.svg
https://commons.wikimedia.org/wiki/File:Chess_plt45.svg

Cburnett, CC BY-SA 3.0 <http://creativecommons.org/licenses/by-sa/3.0/>, via Wikimedia Commons
"""
import base64
import os
from dataclasses import dataclass
from typing import Optional, Union

import svgwrite  # type: ignore
from svgwrite import cm

from .chess import ChessState

NUM_TO_CHAR = ["a", "b", "c", "d", "e", "f", "g", "h"]
PIECES = [
    "P",
    "N",
    "B",
    "R",
    "Q",
    "K",
    "wP",
    "wN",
    "wB",
    "wR",
    "wQ",
    "wK",
]  # k"N"ight


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
        state: Union[None, ChessState] = None,
        color_mode: str = "light",
    ) -> None:
        self.state = state
        self.color_mode = color_mode
        self.GRID_SIZE = 1

    def _repr_html_(self) -> str:
        assert self.state is not None
        return self._to_svg_string()

    def save_svg(self, filename="temp.svg") -> None:
        assert self.state is not None
        assert filename.endswith(".svg")
        self._to_dwg().saveas(filename=filename)

    def show_svg(
        self,
        state: Union[None, ChessState] = None,
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

    def set_state(self, state: ChessState) -> None:
        self.state = state

    def _to_dwg(
        self,
        *,
        state: Union[None, ChessState] = None,
        color_mode: Optional[str] = None,
    ) -> svgwrite.Drawing:
        if state is None:
            assert self.state is not None
            state = self.state
        BOARD_WIDTH = 8
        BOARD_HEIGHT = 8
        GRID_SIZE = self.GRID_SIZE

        if (
            color_mode is None and self.color_mode == "dark"
        ) or color_mode == "dark":
            color_set = VisualizerConfig(
                "none",
                "none",
                "#404040",
                "gray",
                "#202020",
                "silver",
            )
        else:
            color_set = VisualizerConfig(
                "none",
                "none",
                "gray",
                "white",
                "white",
                "black",
            )

        dwg = svgwrite.Drawing(
            "temp.svg",
            (
                (BOARD_WIDTH + 1.5) * GRID_SIZE * cm,
                (BOARD_HEIGHT + 1) * GRID_SIZE * cm,
            ),
        )
        # background
        dwg.add(
            dwg.rect(
                (0, 0),
                (
                    (BOARD_WIDTH + 1.5) * GRID_SIZE * cm,
                    (BOARD_HEIGHT + 1) * GRID_SIZE * cm,
                ),
                fill=color_set.background_color,
            )
        )

        # board
        # grid
        board_g = dwg.g()
        for i in range(BOARD_WIDTH * BOARD_HEIGHT):
            if (i // BOARD_HEIGHT) % 2 != i % 2:
                fill_color = color_set.p1_outline
            else:
                fill_color = color_set.p2_outline

            x = i % BOARD_WIDTH
            y = i // BOARD_HEIGHT
            board_g.add(
                dwg.rect(
                    (x * cm, y * cm),
                    (
                        GRID_SIZE * cm,
                        GRID_SIZE * cm,
                    ),
                    fill=fill_color,
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
        for i in range(BOARD_WIDTH):
            cord.add(
                dwg.text(
                    text=f"{8-i}",
                    insert=(
                        (-0.3) * GRID_SIZE * cm,
                        (i + 0.6) * GRID_SIZE * cm,
                    ),
                    font_size=f"{GRID_SIZE*14}px",
                    font_family="Serif",
                )
            )
            cord.add(
                dwg.text(
                    text=f"{NUM_TO_CHAR[i]}",
                    insert=(
                        (i + 0.4) * GRID_SIZE * cm,
                        (8.35) * GRID_SIZE * cm,
                    ),
                    font_size=f"{GRID_SIZE*14}px",
                    font_family="Serif",
                )
            )

        # pieces

        pieces_g = dwg.g()
        for i, piece_pos, piece_type in zip(
            range(12),
            state.board[1:13],
            PIECES,
        ):
            for xy, is_set in enumerate(piece_pos):
                if is_set == 1:
                    x = xy // BOARD_HEIGHT  # ChessStateは左下原点
                    y = 7 - xy % BOARD_HEIGHT
                    pieces_g = self._set_piece(
                        x,
                        y,
                        piece_type,
                        dwg,
                        pieces_g,
                    )

        board_g.add(pieces_g)

        board_g.translate(GRID_SIZE * 35, GRID_SIZE * 20)  # no units allowed
        dwg.add(board_g)

        return dwg

    def _to_svg_string(self) -> str:
        return self._to_dwg().tostring()

    def _set_piece(self, _x, _y, _type, _dwg, _dwg_g):
        PATH = {
            "P": "images/chess/bPawn.svg",
            "N": "images/chess/bKnight.svg",
            "B": "images/chess/bBishop.svg",
            "R": "images/chess/bRook.svg",
            "Q": "images/chess/bQueen.svg",
            "K": "images/chess/bKing.svg",
            "wP": "images/chess/wPawn.svg",
            "wN": "images/chess/wKnight.svg",
            "wB": "images/chess/wBishop.svg",
            "wR": "images/chess/wRook.svg",
            "wQ": "images/chess/wQueen.svg",
            "wK": "images/chess/wKing.svg",
        }
        file_path = PATH[_type]
        with open(
            os.path.join(os.path.dirname(__file__), file_path),
            "rb",
        ) as f:
            b64_img = base64.b64encode(f.read())
        img = _dwg.image(
            "data:image/svg+xml;base64," + b64_img.decode("ascii"),
            insert=((_x + 0.1) * cm, (_y + 0.1) * cm),
            size=(self.GRID_SIZE * 0.8 * cm, self.GRID_SIZE * 0.8 * cm),
        )
        _dwg_g.add(img)

        return _dwg_g
