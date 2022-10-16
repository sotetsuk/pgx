from dataclasses import dataclass
from typing import Optional

import svgwrite  # type: ignore
from svgwrite import cm

from ._animal_shogi import AnimalShogiState


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
        self, state: AnimalShogiState, color_mode: str = "light"
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

    def show_svg(self, color_mode: Optional[str] = None) -> None:
        import sys

        if "ipykernel" in sys.modules:
            # Jupyter Notebook
            from IPython.display import display_svg  # type:ignore

            display_svg(self._to_dwg(color_mode).tostring(), raw=True)
        else:
            # Not Jupyter
            sys.stdout.write("This function only works in Jupyter Notebook.")

    def _to_dwg(self, color_mode: Optional[str] = None) -> svgwrite.Drawing:
        BOARD_WIDTH = 3
        BOARD_HEIGHT = 4
        GRID_SIZE = 2
        if color_mode is None:  # selfを参照
            color_set = (
                VisualizerConfig(
                    "black", "gray", "white", "white", "#202020", "white"
                )
                if self.color_mode == "dark"
                else VisualizerConfig()
            )
        else:  # 引数を参照
            color_set = (
                VisualizerConfig(
                    "black", "gray", "white", "white", "#202020", "white"
                )
                if color_mode == "dark"
                else VisualizerConfig()
            )

        dwg = svgwrite.Drawing(
            "temp.svg",
            (
                (BOARD_WIDTH + 1) * GRID_SIZE * cm,
                (BOARD_HEIGHT + 1) * GRID_SIZE * cm,
            ),
        )
        # background
        dwg.add(
            dwg.rect(
                (0, 0),
                (
                    (BOARD_WIDTH + 1) * GRID_SIZE * cm,
                    (BOARD_HEIGHT + 1) * GRID_SIZE * cm,
                ),
                # stroke=svgwrite.rgb(10, 10, 16, "%"),
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
        board = self.state.board
        p1_pieces_g = dwg.g()
        p2_pieces_g = dwg.g()
        for i, piece_pos, piece_type in zip(
            range(10),
            board[1:11],
            ["P", "R", "B", "K", "G", "P", "R", "B", "K", "G"],
        ):
            for xy, is_set in enumerate(piece_pos):
                if is_set == 1:
                    if i < 5:
                        pieces_g = p1_pieces_g
                        x = 2 - xy // BOARD_HEIGHT  # AnimalShogiStateは右上原点
                        y = xy % BOARD_HEIGHT
                    else:
                        pieces_g = p2_pieces_g
                        x = xy // BOARD_HEIGHT
                        y = 3 - xy % BOARD_HEIGHT
                    print(x, y, piece_type)
                    pieces_g.add(
                        dwg.rect(
                            ((x + 1) * cm, (y + 1) * cm),
                            ((GRID_SIZE - 2) * cm, (GRID_SIZE - 2) * cm),
                            stroke=svgwrite.rgb(10, 10, 16, "%"),
                            fill="black",
                        )
                    )
                    pieces_g.add(
                        dwg.rect(
                            insert=(
                                (x + 0.1) * GRID_SIZE * cm,
                                (y + 0.1) * GRID_SIZE * cm,
                            ),
                            size=(
                                0.8 * GRID_SIZE * cm,
                                0.8 * GRID_SIZE * cm,
                            ),
                            rx=0.1 * cm,
                            ry=0.1 * cm,
                            stroke=svgwrite.rgb(10, 10, 16, "%"),
                            fill="none",
                        )
                    )
                    pieces_g.add(
                        dwg.text(
                            text=piece_type,
                            insert=(
                                (x + 0.27) * GRID_SIZE * cm,
                                (y + 0.72) * GRID_SIZE * cm,
                            ),
                            font_size="4em",
                            font_family="Courier",
                        )
                    )

        p2_pieces_g.add(
            dwg.text(
                text="a",
                insert=(
                    -1 * GRID_SIZE * cm,
                    -1 * GRID_SIZE * cm,
                ),
                font_size="4em",
                font_family="Courier",
            )
        )

        board_g.add(p1_pieces_g)
        p2_pieces_g.rotate(angle=180)
        p2_pieces_g.translate(
            -GRID_SIZE * 113.5, -GRID_SIZE * 151.5
        )  # no units allowed
        board_g.add(p2_pieces_g)

        """
        board = state.board
        for xy, stone in enumerate(board):
            if stone == 2:
                continue
            # ndarrayのx,yと違うことに注意
            stone_y = xy // BOARD_SIZE * GRID_SIZE
            stone_x = xy % BOARD_SIZE * GRID_SIZE

            color = (
                color_set.black_color if stone == 0 else color_set.white_color
            )
            outline = (
                color_set.black_outline
                if stone == 0
                else color_set.white_outline
            )
            board_g.add(
                dwg.circle(
                    center=(stone_x * cm, stone_y * cm),
                    r=GRID_SIZE / 2.2 * cm,
                    stroke=outline,
                    fill=color,
                )
            )
        """
        board_g.translate(GRID_SIZE * 20, GRID_SIZE * 20)  # no units allowed
        dwg.add(board_g)

        return dwg

    def _to_svg_string(self) -> str:
        return self._to_dwg(self.color_mode).tostring()
