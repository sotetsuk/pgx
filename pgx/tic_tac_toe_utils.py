import math
from dataclasses import dataclass
from typing import Optional, Union

import svgwrite  # type: ignore

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
        self.GRID_SIZE = 50
        self.BOARD_WIDTH = 3
        self.BOARD_HEIGHT = 3

    def _repr_html_(self) -> str:
        assert self.state is not None
        return self._to_svg_string()

    def save_svg(self, filename="temp.svg", state=None) -> None:
        assert filename.endswith(".svg")
        if state is None:
            assert self.state is not None
            state = self.state
        self._to_dwg_from_states(states=state).saveas(filename=filename)

    def show_svg(
        self,
        states: Union[None, State] = None,
        color_mode: Optional[str] = None,
    ) -> None:
        import sys

        if "ipykernel" in sys.modules:
            # Jupyter Notebook
            from IPython.display import display_svg  # type:ignore

            display_svg(
                self._to_dwg_from_states(
                    states=states, color_mode=color_mode
                ).tostring(),
                raw=True,
            )
        else:
            # Not Jupyter
            sys.stdout.write("This function only works in Jupyter Notebook.")

    def set_state(self, state: State) -> None:
        self.state = state

    def _to_dwg_from_states(
        self,
        *,
        states,
        color_mode: Optional[str] = None,
    ):
        try:
            SIZE = len(states.curr_player)
            WIDTH = int(math.sqrt(SIZE))
            HEIGHT = WIDTH + 1
        except TypeError:
            SIZE = 1
            WIDTH = 1
            HEIGHT = 1
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
                (self.BOARD_WIDTH + 1) * self.GRID_SIZE * WIDTH,
                (self.BOARD_HEIGHT + 1) * self.GRID_SIZE * HEIGHT,
            ),
        )

        # background
        dwg.add(
            dwg.rect(
                (0, 0),
                (
                    (self.BOARD_WIDTH + 1) * self.GRID_SIZE * WIDTH,
                    (self.BOARD_HEIGHT + 1) * self.GRID_SIZE * HEIGHT,
                ),
                fill=color_set.background_color,
            )
        )

        if SIZE == 1:
            g = self._make_dwg_group(dwg, states, color_set)
            g.translate(
                self.GRID_SIZE * 1 / 2,
                self.GRID_SIZE * 1 / 2,
            )
            dwg.add(g)
            return dwg

        for i in range(SIZE):
            x = i % WIDTH
            y = i // WIDTH
            _state = State(  # type:ignore
                curr_player=states.curr_player[i],
                legal_action_mask=states.legal_action_mask[i],
                terminated=states.terminated[i],
                turn=states.turn[i],
                board=states.board[i],
            )
            g = self._make_dwg_group(dwg, _state, color_set)
            g.translate(
                self.GRID_SIZE * 1 / 2
                + (self.BOARD_WIDTH + 1) * self.GRID_SIZE * x,
                self.GRID_SIZE * 1 / 2
                + (self.BOARD_WIDTH + 1) * self.GRID_SIZE * y,
            )
            dwg.add(g)
        return dwg

    def _make_dwg_group(
        self,
        dwg,
        state: State,
        color_set: VisualizerConfig,
    ) -> svgwrite.Drawing:
        GRID_SIZE = self.GRID_SIZE
        BOARD_WIDTH = self.BOARD_WIDTH
        BOARD_HEIGHT = self.BOARD_HEIGHT
        # board
        board_g = dwg.g()

        # grid
        hlines = board_g.add(
            dwg.g(
                id="hlines",
                stroke=color_set.grid_color,
                fill=color_set.grid_color,
            )
        )
        for y in range(1, BOARD_HEIGHT):
            hlines.add(
                dwg.line(
                    start=(0, GRID_SIZE * y),
                    end=(
                        GRID_SIZE * BOARD_WIDTH,
                        GRID_SIZE * y,
                    ),
                    stroke_width=GRID_SIZE * 0.05,
                )
            )
            hlines.add(
                dwg.circle(
                    center=(0, GRID_SIZE * y),
                    r=GRID_SIZE * 0.014,
                )
            )
            hlines.add(
                dwg.circle(
                    center=(
                        GRID_SIZE * BOARD_WIDTH,
                        GRID_SIZE * y,
                    ),
                    r=GRID_SIZE * 0.014,
                )
            )
        vlines = board_g.add(
            dwg.g(
                id="vline",
                stroke=color_set.grid_color,
                fill=color_set.grid_color,
            )
        )
        for x in range(1, BOARD_WIDTH):
            vlines.add(
                dwg.line(
                    start=(GRID_SIZE * x, 0),
                    end=(
                        GRID_SIZE * x,
                        GRID_SIZE * BOARD_HEIGHT,
                    ),
                    stroke_width=GRID_SIZE * 0.05,
                )
            )
            vlines.add(
                dwg.circle(
                    center=(GRID_SIZE * x, 0),
                    r=GRID_SIZE * 0.014,
                )
            )
            vlines.add(
                dwg.circle(
                    center=(
                        GRID_SIZE * x,
                        GRID_SIZE * BOARD_HEIGHT,
                    ),
                    r=GRID_SIZE * 0.014,
                )
            )

        for i, mark in enumerate(state.board):
            x = i % BOARD_WIDTH
            y = i // BOARD_HEIGHT
            if mark == 0:  # 先手
                board_g.add(
                    dwg.circle(
                        center=(
                            (x + 0.5) * GRID_SIZE,
                            (y + 0.5) * GRID_SIZE,
                        ),
                        r=0.4 * GRID_SIZE,
                        stroke=color_set.grid_color,
                        stroke_width=0.05 * GRID_SIZE,
                        fill="none",
                    )
                )
            elif mark == 1:  # 後手
                board_g.add(
                    dwg.line(
                        start=(
                            (x + 0.1) * GRID_SIZE,
                            (y + 0.1) * GRID_SIZE,
                        ),
                        end=(
                            (x + 0.9) * GRID_SIZE,
                            (y + 0.9) * GRID_SIZE,
                        ),
                        stroke=color_set.grid_color,
                        stroke_width=0.05 * GRID_SIZE,
                    )
                )
                board_g.add(
                    dwg.line(
                        start=(
                            (x + 0.1) * GRID_SIZE,
                            (y + 0.9) * GRID_SIZE,
                        ),
                        end=(
                            (x + 0.9) * GRID_SIZE,
                            (y + 0.1) * GRID_SIZE,
                        ),
                        stroke=color_set.grid_color,
                        stroke_width=0.05 * GRID_SIZE,
                    )
                )
        return board_g

    def _to_svg_string(self) -> str:
        return self._to_dwg_from_states(states=self.state).tostring()
