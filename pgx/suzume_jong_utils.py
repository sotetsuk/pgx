import base64
import os
from dataclasses import dataclass
from typing import Optional, Union

import svgwrite  # type: ignore
from svgwrite import cm

from .suzume_jong import NUM_TILE_TYPES, State, _tile_type_to_str

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
        state: Union[None, State] = None,
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
        BOARD_WIDTH = 15
        BOARD_HEIGHT = 10
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
                "white",  # "#e4cab1",
                "black",
            )

        dwg = svgwrite.Drawing(
            "temp.svg",
            (
                BOARD_WIDTH * GRID_SIZE * cm,
                BOARD_HEIGHT * GRID_SIZE * cm,
            ),
        )
        # background
        dwg.add(
            dwg.rect(
                (0, 0),
                (
                    BOARD_WIDTH * GRID_SIZE * cm,
                    BOARD_HEIGHT * GRID_SIZE * cm,
                ),
                fill=color_set.background_color,
            )
        )

        # board
        # grid
        board_g = dwg.g()
        p1_g = dwg.g()
        p2_g = dwg.g()
        p3_g = dwg.g()

        # pieces
        for player_id, pieces_g in zip(range(3), [p1_g, p2_g, p3_g]):
            pieces_g = dwg.g()

            # border
            pieces_g.add(
                dwg.rect(
                    (4.9 * cm, 7.8 * cm),
                    (
                        5.2 * cm,
                        1.4 * cm,
                    ),
                    rx="2px",
                    ry="2px",
                    fill="white",
                    stroke="#8f3740",
                    stroke_width="5px",
                )
            )

            # hands
            x = 5
            y = 8
            for type, num in zip(
                range(NUM_TILE_TYPES),
                state.hands[player_id],
            ):
                for _ in range(num):
                    pieces_g = self._set_piece(
                        x,
                        y,
                        type,
                        dwg,
                        pieces_g,
                    )
                    x += 0.8

            # river
            x = 5
            y = 5
            river_count = 0
            for type in state.rivers[player_id]:
                if type >= 0:
                    pieces_g = self._set_piece(
                        x,
                        y,
                        type,
                        dwg,
                        pieces_g,
                    )
                    x += 0.8
                    river_count += 1
                    if river_count > 4:
                        river_count = 0
                        x = 5
                        y += 1

            if player_id == 1:
                pieces_g.rotate(
                    angle=90, center=(GRID_SIZE * 285, GRID_SIZE * 100)
                )

            elif player_id == 2:
                pieces_g.rotate(
                    angle=270, center=(GRID_SIZE * 285, GRID_SIZE * 100)
                )

            board_g.add(pieces_g)

        # dora
        board_g.add(
            dwg.rect(
                (6.3 * cm, 0.6 * cm),
                (
                    2.4 * cm,
                    1.8 * cm,
                ),
                rx="2px",
                ry="2px",
                fill="#8f3740",
            )
        )
        board_g.add(
            dwg.rect(
                (6.5 * cm, 0.8 * cm),
                (
                    2 * cm,
                    1.4 * cm,
                ),
                rx="2px",
                ry="2px",
                fill="none",
                stroke="#bfa40e",
                stroke_width="2px",
            )
        )
        board_g = self._set_piece(
            6.5,
            1,
            state.dora,
            dwg,
            board_g,
        )

        board_g.translate(0, GRID_SIZE * 20)  # no units allowed
        dwg.add(board_g)

        return dwg

    def _to_svg_string(self) -> str:
        return self._to_dwg().tostring()

    def _set_piece(self, _x, _y, _type, _dwg, _dwg_g):
        _dwg_g.add(
            _dwg.rect(
                ((_x + 0.2) * cm, _y * cm),
                (
                    0.6 * cm,
                    1 * cm,
                ),
                fill="white",  # "#f9f7e8",
                stroke="none",
            )
        )
        type_str = _tile_type_to_str(_type)
        PATH = {
            "1": "images/suzume_jong/1p.svg",
            "2": "images/suzume_jong/2p.svg",
            "3": "images/suzume_jong/3p.svg",
            "4": "images/suzume_jong/4p.svg",
            "5": "images/suzume_jong/5p.svg",
            "6": "images/suzume_jong/6p.svg",
            "7": "images/suzume_jong/7p.svg",
            "8": "images/suzume_jong/8p.svg",
            "9": "images/suzume_jong/9p.svg",
            "g": "images/suzume_jong/gd.svg",
            "r": "images/suzume_jong/rd.svg",
        }
        file_path = PATH[type_str]
        with open(
            os.path.join(os.path.dirname(__file__), file_path),
            "rb",
        ) as f:
            b64_img = base64.b64encode(f.read())
        img = _dwg.image(
            "data:image/svg+xml;base64," + b64_img.decode("ascii"),
            insert=(_x * cm, _y * cm),
            size=(self.GRID_SIZE * cm, self.GRID_SIZE * cm),
        )
        _dwg_g.add(img)

        return _dwg_g
