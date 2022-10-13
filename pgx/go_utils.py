from dataclasses import dataclass
from typing import Optional

import svgwrite  # type: ignore
from svgwrite import cm

from .go import GoState, get_board


@dataclass
class VisualizerConfig:
    black_color: str = "black"
    white_color: str = "white"
    black_outline: str = "black"
    white_outline: str = "black"
    background_color: str = "white"
    grid_color: str = "black"


class Visualizer:
    def __init__(self, state: GoState, color_mode: str = "light") -> None:
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
        state = self.state
        BOARD_SIZE = state.size[0]
        GRID_SIZE = 1
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
                BOARD_SIZE * GRID_SIZE * cm,
                BOARD_SIZE * GRID_SIZE * cm,
            ),
        )
        # background
        dwg.add(
            dwg.rect(
                (0, 0),
                (BOARD_SIZE * GRID_SIZE * cm, BOARD_SIZE * GRID_SIZE * cm),
                # stroke=svgwrite.rgb(10, 10, 16, "%"),
                fill=color_set.background_color,
            )
        )

        # board
        # grid
        board_g = dwg.g()
        hlines = board_g.add(dwg.g(id="hlines", stroke=color_set.grid_color))
        for y in range(BOARD_SIZE):
            hlines.add(
                dwg.line(
                    start=(0 * cm, GRID_SIZE * y * cm),
                    end=(
                        GRID_SIZE * (BOARD_SIZE - 1) * cm,
                        GRID_SIZE * y * cm,
                    ),
                )
            )
        vlines = board_g.add(dwg.g(id="vline", stroke=color_set.grid_color))
        for x in range(BOARD_SIZE):
            vlines.add(
                dwg.line(
                    start=(GRID_SIZE * x * cm, 0 * cm),
                    end=(
                        GRID_SIZE * x * cm,
                        GRID_SIZE * (BOARD_SIZE - 1) * cm,
                    ),
                )
            )

        # hoshi
        hoshi_g = dwg.g()
        hosi_pos = []
        if BOARD_SIZE == 19:
            hosi_pos = [
                (4, 4),
                (4, 10),
                (4, 16),
                (10, 4),
                (10, 10),
                (10, 16),
                (16, 4),
                (16, 10),
                (16, 16),
            ]
        elif BOARD_SIZE == 5:
            hosi_pos = [(3, 3)]

        for x, y in hosi_pos:
            hoshi_g.add(
                dwg.circle(
                    center=((x - 1) * cm, (y - 1) * cm),
                    r=GRID_SIZE / 10 * cm,
                    fill=color_set.grid_color,
                )
            )
        board_g.add(hoshi_g)

        # stones
        board = get_board(state)
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
        board_g.translate(GRID_SIZE * 20, GRID_SIZE * 20)  # no units allowed
        dwg.add(board_g)

        return dwg

    def _to_svg_string(self) -> str:
        return self._to_dwg(self.color_mode).tostring()
