from dataclasses import dataclass
from typing import Optional, Union

import svgwrite  # type: ignore
from svgwrite import cm

from .tic_tac_toe import State


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
        state: Union[None, State] = None,
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
        state: Union[None, State] = None,
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

    def set_state(self, state: State) -> None:
        self.state = state

    def _to_dwg(
        self,
        *,
        state: Union[None, State] = None,
        color_mode: Optional[str] = None,
    ) -> svgwrite.Drawing:
        if state is None:
            assert self.state is not None
            state = self.state
        GRID_SIZE = 2
        BOARD_WIDTH = 3
        BOARD_HEIGHT = 3
        # cm = 20 * GRID_SIZE

        if (
            color_mode is None and self.color_mode == "dark"
        ) or color_mode == "dark":
            color_set = VisualizerConfig(
                "gray",
                "black",
                "black",
                "dimgray",
                "#202020",
                "gainsboro",
            )
        else:
            color_set = VisualizerConfig(
                "white", "black", "lightgray", "white", "white", "black"
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
                fill=color_set.background_color,
            )
        )

        # board
        board_g = dwg.g()

        # grid
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
            hlines.add(
                dwg.circle(
                    center=(0 * cm, GRID_SIZE * y * cm),
                    r=0.5 * GRID_SIZE,
                )
            )
            hlines.add(
                dwg.circle(
                    center=(
                        GRID_SIZE * BOARD_WIDTH * cm,
                        GRID_SIZE * y * cm,
                    ),
                    r=0.5 * GRID_SIZE,
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
            vlines.add(
                dwg.circle(
                    center=(GRID_SIZE * x * cm, 0 * cm),
                    r=0.5 * GRID_SIZE,
                )
            )
            vlines.add(
                dwg.circle(
                    center=(
                        GRID_SIZE * x * cm,
                        GRID_SIZE * BOARD_HEIGHT * cm,
                    ),
                    r=0.5 * GRID_SIZE,
                )
            )

        for i, mark in enumerate(state.board):
            x = i % BOARD_WIDTH
            y = i // BOARD_HEIGHT
            if mark == 0:  # 先手
                board_g.add(
                    dwg.circle(
                        center=(
                            (x + 0.5) * GRID_SIZE * cm,
                            (y + 0.5) * GRID_SIZE * cm,
                        ),
                        r=0.4 * GRID_SIZE * cm,
                        stroke=color_set.grid_color,
                        stroke_width=2 * GRID_SIZE,
                        fill="none",
                    )
                )
            elif mark == 1:  # 後手
                board_g.add(
                    dwg.line(
                        start=(
                            (x + 0.1) * GRID_SIZE * cm,
                            (y + 0.1) * GRID_SIZE * cm,
                        ),
                        end=(
                            (x + 0.9) * GRID_SIZE * cm,
                            (y + 0.9) * GRID_SIZE * cm,
                        ),
                        stroke=color_set.grid_color,
                        stroke_width=GRID_SIZE * 2,
                    )
                )
                board_g.add(
                    dwg.line(
                        start=(
                            (x + 0.1) * GRID_SIZE * cm,
                            (y + 0.9) * GRID_SIZE * cm,
                        ),
                        end=(
                            (x + 0.9) * GRID_SIZE * cm,
                            (y + 0.1) * GRID_SIZE * cm,
                        ),
                        stroke=color_set.grid_color,
                        stroke_width=GRID_SIZE * 2,
                    )
                )

        board_g.translate(GRID_SIZE * 19, GRID_SIZE * 19)  # no units allowed
        dwg.add(board_g)

        return dwg

    def _to_svg_string(self) -> str:
        return self._to_dwg().tostring()
