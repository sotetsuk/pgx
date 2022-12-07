import base64
import os
from dataclasses import dataclass
from typing import Optional, Union

import svgwrite  # type: ignore
from svgwrite import cm

from .suzume_jong import NUM_TILE_TYPES, NUM_TILES, State, _tile_type_to_str


@dataclass
class VisualizerConfig:
    p1_color: str = "black"  # 牌の裏地
    p2_color: str = "white"  # 手牌部分の背景色
    p1_outline: str = "black"  # ドラ表示部の背景色
    p2_outline: str = "black"  # ドラ表示部の囲み線
    background_color: str = "white"
    grid_color: str = "black"  # 手牌の囲み線
    text_color: str = "black"


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
                "lightgray",
                "dimgray",
                "#404040",
                "gray",
                "#202020",
                "darkgray",
                "whitesmoke",
            )
        else:
            color_set = VisualizerConfig(
                "white", "white", "gray", "white", "white", "silver", "black"
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
        for player_id, pieces_g in zip(
            state.shuffled_players, [p1_g, p2_g, p3_g]
        ):
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
                    fill=color_set.p2_color,
                    stroke=color_set.grid_color,
                    stroke_width="5px",
                )
            )

            # hands
            x = 5.0
            y = 8.0
            for type, num in zip(
                range(NUM_TILE_TYPES),
                state.hands[player_id],
            ):
                num_red = state.n_red_in_hands[player_id, type]
                for _ in range(num):
                    pieces_g = self._set_piece(
                        x, y, type, num_red > 0, dwg, pieces_g, color_set
                    )
                    x += 0.8
                    num_red -= 1

            # river
            x = 5.4
            y = 5.0
            river_count = 0
            for type, is_red in zip(
                state.rivers[player_id], state.is_red_in_river[player_id]
            ):
                if type >= 0:
                    pieces_g = self._set_piece(
                        x, y, type, is_red, dwg, pieces_g, color_set
                    )
                    x += 0.8
                    river_count += 1
                    if river_count > 4:
                        river_count = 0
                        x = 5.4
                        y += 1.1

            if player_id == state.shuffled_players[1]:
                pieces_g.rotate(
                    angle=-90, center=(GRID_SIZE * 285, GRID_SIZE * 100)
                )

            elif player_id == state.shuffled_players[2]:
                pieces_g.rotate(
                    angle=90, center=(GRID_SIZE * 285, GRID_SIZE * 100)
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
                fill=color_set.p1_outline,
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
                stroke=color_set.p2_outline,
                stroke_width="2px",
            )
        )
        board_g = self._set_piece(
            6.5, 1, state.dora, False, dwg, board_g, color_set
        )

        """
        親表示
        board_g.add(
            dwg.rect(
                (7.5 * cm, 1.1 * cm),
                (
                    0.8 * cm,
                    0.8 * cm,
                ),
                rx="2px",
                ry="2px",
                fill="whitesmoke",
            )
        )

        with open(
            os.path.join(
                os.path.dirname(__file__), "images/suzume_jong/oya.svg"
            ),
            "rb",
        ) as f:
            b64_img = base64.b64encode(f.read())
        img = dwg.image(
            "data:image/svg+xml;base64," + b64_img.decode("ascii"),
            insert=(7.65 * cm, 1.25 * cm),
            size=(0.5 * cm, 0.5 * cm),
        )
        board_g.add(img)
        """

        # wall
        board_g = self._set_piece(6.4, 3, -1, False, dwg, board_g, color_set)
        board_g.add(
            dwg.text(
                text=f"× {NUM_TILES - state.draw_ix-1}",
                insert=(
                    7.5 * cm,
                    3.65 * cm,
                ),
                fill=color_set.text_color,
                font_size=f"{GRID_SIZE*17}px",
                font_family="serif",
            )
        )
        board_g.translate(0, GRID_SIZE * 20)  # no units allowed
        dwg.add(board_g)

        return dwg

    def _to_svg_string(self) -> str:
        return self._to_dwg().tostring()

    def _set_piece(
        self,
        _x,
        _y,
        _type,
        _is_red,
        _dwg,
        _dwg_g,
        _color_set: VisualizerConfig,
    ):
        _dwg_g.add(
            _dwg.rect(
                ((_x + 0.18) * cm, (_y + 0.02) * cm),
                (
                    0.64 * cm,
                    0.96 * cm,
                ),
                fill=_color_set.p1_color,  # "#f9f7e8",
                stroke="none",
            )
        )
        type_str = _tile_type_to_str(_type)
        if _is_red and type_str != "g" and type_str != "r":
            type_str += "r"
        PATH = {f"{i+1}": f"images/suzume_jong/{i+1}p.svg" for i in range(9)}
        PATH.update(
            {f"{i+1}r": f"images/suzume_jong/{i+1}pr.svg" for i in range(9)}
        )
        PATH["0"] = "images/suzume_jong/b.svg"
        PATH["g"] = "images/suzume_jong/gd.svg"
        PATH["r"] = "images/suzume_jong/rd.svg"

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
