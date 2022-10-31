from dataclasses import dataclass
from typing import Optional, Union

import svgwrite  # type: ignore

from .backgammon import BackgammonState


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
        state: Union[None, BackgammonState] = None,
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
        state: Union[None, BackgammonState] = None,
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

    def set_state(self, state: BackgammonState) -> None:
        self.state = state

    def _to_dwg(
        self,
        *,
        state: Union[None, BackgammonState] = None,
        color_mode: Optional[str] = None,
    ) -> svgwrite.Drawing:
        if state is None:
            assert self.state is not None
            state = self.state
        GRID_SIZE = 2
        BOARD_WIDTH = 13
        BOARD_HEIGHT = 14
        cm = 20 * GRID_SIZE

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
                (BOARD_WIDTH + 6) * cm,
                (BOARD_HEIGHT + 3) * cm,
            ),
        )

        # background
        dwg.add(
            dwg.rect(
                (0, 0),
                (
                    (BOARD_WIDTH + 6) * cm,
                    (BOARD_HEIGHT + 3) * cm,
                ),
                fill=color_set.background_color,
            )
        )

        # board
        # grid
        board_g = dwg.g()
        board_g.add(
            dwg.rect(
                (0, 0),
                (
                    13 * cm,
                    14 * cm,
                ),
                stroke=color_set.grid_color,
                fill=color_set.background_color,
            )
        )
        board_g.add(
            dwg.rect(
                (6 * cm, 0),
                (
                    1 * cm,
                    14 * cm,
                ),
                stroke=color_set.grid_color,
                fill=color_set.background_color,
            )
        )

        for i in range(24):
            p1 = (i * cm, 0)
            p2 = ((i + 1) * cm, 0)
            p3 = ((i + 0.5) * cm, 6 * cm)
            if 6 <= i < 12:
                p1 = ((i + 1) * cm, 0)
                p2 = ((i + 2) * cm, 0)
                p3 = ((i + 1.5) * cm, 6 * cm)
            elif 12 <= i < 18:
                p1 = ((i - 12) * cm, 14 * cm)
                p2 = ((i + 1 - 12) * cm, 14 * cm)
                p3 = ((i + 0.5 - 12) * cm, 8 * cm)
            elif 18 <= i:
                p1 = ((i + 1 - 12) * cm, 14 * cm)
                p2 = ((i + 2 - 12) * cm, 14 * cm)
                p3 = ((i + 1.5 - 12) * cm, 8 * cm)

            fill_color = (
                color_set.p1_outline if i % 2 == 0 else color_set.p2_outline
            )

            board_g.add(
                dwg.polygon(
                    points=[p1, p2, p3],
                    stroke=color_set.grid_color,
                    fill=fill_color,
                )
            )

        # pieces
        for i, piece in enumerate(state.board[:24]):
            fill_color = color_set.p2_color
            if piece < 0:  # 白
                piece = -piece
                fill_color = color_set.p1_color

            x = (12 - i) + 0.5
            y = 13.5
            diff = -1
            if 6 <= i < 12:
                x = (12 - i) - 0.5
                y = 13.5
            elif 12 <= i < 18:
                x = i - 12 + 0.5
                y = 0.5
                diff = 1
            elif 18 <= i:
                x = i - 12 + 1.5
                y = 0.5
                diff = 1
            for n in range(piece):
                board_g.add(
                    dwg.circle(
                        center=(x * cm, (y + n * diff) * cm),
                        r=0.5 * cm,
                        stroke=color_set.grid_color,
                        fill=fill_color,
                        # stroke_width=GRID_SIZE,
                    )
                )

        # bar
        for i, piece in enumerate(state.board[24:26]):  # 24:白,25:黒
            fill_color = color_set.p2_color
            diff = 1
            if i == 0:
                piece = -piece
                fill_color = color_set.p1_color
                diff = -1
            for n in range(piece):
                board_g.add(
                    dwg.circle(
                        center=(6.5 * cm, (6 + i * 2 + n * diff) * cm),
                        r=0.5 * cm,
                        stroke=color_set.grid_color,
                        fill=fill_color,
                        # stroke_width=GRID_SIZE,
                    )
                )

        # off
        board_g.add(
            dwg.rect(
                (13 * cm, 12 * cm),
                (
                    4 * cm,
                    2 * cm,
                ),
                stroke=color_set.grid_color,
                fill=color_set.background_color,
            )
        )
        board_g.add(
            dwg.circle(
                center=(14 * cm, 13 * cm),
                r=0.5 * cm,
                stroke=color_set.grid_color,
                fill=color_set.p1_color,
            )
        )
        board_g.add(
            dwg.text(
                text=f"× {-state.board[26]}",  # 26:白
                insert=(
                    14.8 * cm,
                    13.3 * cm,
                ),
                fill=color_set.grid_color,
                font_size=f"{GRID_SIZE*17}px",
                font_family="serif",
            )
        )

        board_g.add(
            dwg.rect(
                (13 * cm, 0),
                (
                    4 * cm,
                    2 * cm,
                ),
                stroke=color_set.grid_color,
                fill=color_set.background_color,
            )
        )
        board_g.add(
            dwg.circle(
                center=(14 * cm, 1 * cm),
                r=0.5 * cm,
                stroke=color_set.grid_color,
                fill=color_set.p2_color,
            )
        )
        board_g.add(
            dwg.text(
                text=f"× {state.board[27]}",  # 27:黒
                insert=(
                    14.8 * cm,
                    1.3 * cm,
                ),
                fill=color_set.grid_color,
                font_size=f"{GRID_SIZE*17}px",
                font_family="serif",
            )
        )

        board_g.translate(GRID_SIZE * 20, GRID_SIZE * 20)  # no units allowed
        dwg.add(board_g)

        return dwg

    def _to_svg_string(self) -> str:
        return self._to_dwg().tostring()
