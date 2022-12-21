import math
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import svgwrite  # type: ignore

from .contractbridgebidding import ContractBridgeBiddingState

TO_CARD = [
    "\U0001F0A1",
    "\U0001F0A2",
    "\U0001F0A3",
    "\U0001F0A4",
    "\U0001F0A5",
    "\U0001F0A6",
    "\U0001F0A7",
    "\U0001F0A8",
    "\U0001F0A9",
    "\U0001F0AA",
    "\U0001F0AB",
    "\U0001F0AC",
    "\U0001F0AD",
    "\U0001F0AE",
    "\U0001F0B1",
    "\U0001F0B2",
    "\U0001F0B3",
    "\U0001F0B4",
    "\U0001F0B5",
    "\U0001F0B6",
    "\U0001F0B7",
    "\U0001F0B8",
    "\U0001F0B9",
    "\U0001F0BA",
    "\U0001F0BB",
    "\U0001F0BC",
    "\U0001F0BD",
    "\U0001F0BE",
    "\U0001F0C1",
    "\U0001F0C2",
    "\U0001F0C3",
    "\U0001F0C4",
    "\U0001F0C5",
    "\U0001F0C6",
    "\U0001F0C7",
    "\U0001F0C8",
    "\U0001F0C9",
    "\U0001F0CA",
    "\U0001F0CB",
    "\U0001F0CC",
    "\U0001F0CD",
    "\U0001F0CE",
    "\U0001F0D1",
    "\U0001F0D2",
    "\U0001F0D3",
    "\U0001F0D4",
    "\U0001F0D5",
    "\U0001F0D6",
    "\U0001F0D7",
    "\U0001F0D8",
    "\U0001F0D9",
    "\U0001F0DA",
    "\U0001F0DB",
    "\U0001F0DC",
    "\U0001F0DD",
    "\U0001F0DE",
]
SUITS = ["\u2660", "\u2661", "\u2662", "\u2663", "NT"]  # ♠♡♢♣
ACT = ["P", "X", "XX"]


@dataclass
class VisualizerConfig:
    p1_color: str = "black"
    p2_color: str = "white"
    p1_outline: str = "black"
    p2_outline: str = "black"
    background_color: str = "white"
    grid_color: str = "black"
    text_color: str = "black"


class Visualizer:
    def __init__(
        self,
        state: Union[None, ContractBridgeBiddingState] = None,
        color_mode: str = "light",
    ) -> None:
        self.state = state
        self.color_mode = color_mode
        self.GRID_SIZE = 50
        self.BOARD_WIDTH = 14
        self.BOARD_HEIGHT = 12

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
        states: Union[None, ContractBridgeBiddingState] = None,
        scale=1,
        color_mode: Optional[str] = None,
    ) -> None:
        import sys

        if "ipykernel" in sys.modules:
            # Jupyter Notebook
            from IPython.display import display_svg  # type:ignore

            display_svg(
                self._to_dwg_from_states(
                    states=states, scale=scale, color_mode=color_mode
                ).tostring(),
                raw=True,
            )
        else:
            # Not Jupyter
            sys.stdout.write("This function only works in Jupyter Notebook.")

    def set_state(self, state: ContractBridgeBiddingState) -> None:
        self.state = state

    def _to_dwg_from_states(
        self,
        *,
        states,
        scale=1,
        color_mode: Optional[str] = None,
    ):
        try:
            SIZE = len(states.curr_player)
            WIDTH = math.ceil(math.sqrt(SIZE - 0.1))
            if SIZE - (WIDTH - 1) ** 2 >= WIDTH:
                HEIGHT = WIDTH
            else:
                HEIGHT = WIDTH - 1
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
                "white",
            )
        else:
            color_set = VisualizerConfig(
                "white",
                "black",
                "lightgray",
                "white",
                "white",
                "black",
                "black",
            )
        dwg = svgwrite.Drawing(
            "temp.svg",
            (
                (self.BOARD_WIDTH + 1) * self.GRID_SIZE * WIDTH * scale,
                (self.BOARD_HEIGHT + 1) * self.GRID_SIZE * HEIGHT * scale,
            ),
        )

        group = dwg.g()

        # background
        group.add(
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
            group.add(g)
            group.scale(scale)
            dwg.add(group)
            return dwg

        for i in range(SIZE):
            x = i % WIDTH
            y = i // WIDTH
            _state = ContractBridgeBiddingState(  # type:ignore
                turn=states.turn[i],
                curr_player=states.curr_player[i],
                hand=states.hand[i],
                bidding_history=states.bidding_history[i],
                vul_NS=states.vul_NS[i],
                vul_EW=states.vul_EW[i],
            )
            g = self._make_dwg_group(dwg, _state, color_set)
            g.translate(
                self.GRID_SIZE * 1 / 2
                + (self.BOARD_WIDTH + 1) * self.GRID_SIZE * x,
                self.GRID_SIZE * 1 / 2
                + (self.BOARD_WIDTH + 1) * self.GRID_SIZE * y,
            )
            group.add(g)
        group.scale(scale)
        dwg.add(group)
        return dwg

    def _make_dwg_group(
        self,
        dwg,
        state: ContractBridgeBiddingState,
        color_set: VisualizerConfig,
    ) -> svgwrite.Drawing:
        GRID_SIZE = self.GRID_SIZE
        BOARD_WIDTH = self.BOARD_WIDTH
        BOARD_HEIGHT = self.BOARD_HEIGHT
        # board
        board_g = dwg.g()

        # hand
        x_offset = [250, 500, 250, 0]
        y_offset = [0, 200, 400, 200]
        for i in range(4):  # player0,1,2,3
            hand = sorted(state.hand[i * 13 : (i + 1) * 13])
            assert len(hand) == 13
            # suit
            for j in range(4):  # spades,hearts,diamonds,clubs
                suit = "".join(
                    [TO_CARD[i] for i in hand if j * 13 <= i < (j + 1) * 13]
                )
                board_g.add(
                    dwg.text(
                        text=suit,
                        insert=(x_offset[i], y_offset[i] + 50 * (j + 1)),
                        fill="red" if 0 < j < 3 else color_set.text_color,
                        font_size="50px",
                        font_family="monospace",
                    )
                )

        board_g.add(
            dwg.rect(
                (250, 220),
                (210, 180),
                rx="5px",
                ry="5px",
                fill="none",
                stroke="gray",
                stroke_width="2px",
            )
        )
        print(state.bidding_history)

        # history
        history = [
            str(i // 5 + 1) + SUITS[i % 5] if 0 <= i < 35 else ACT[i - 35]
            for i in state.bidding_history
            if 0 <= i
        ]
        print(history)
        for i, act in enumerate(state.bidding_history):
            if act == -1:
                break
            act_str = (
                str(act // 5 + 1) + SUITS[act % 5]
                if 0 <= act < 35
                else ACT[act - 35]
            )
            color = "red" if act % 5 == 1 or act % 5 == 2 else "black"
            board_g.add(
                dwg.text(
                    text=act_str,
                    insert=(265 + 50 * (i % 4), 270 + 20 * (i // 4)),
                    fill=color,
                    font_size="20px",
                    font_family="monospace",
                )
            )

        # player
        pos = np.array(["N", "E", "S", "W"])
        pos = np.roll(pos, -state.dealer)
        for i in range(4):
            board_g.add(
                dwg.text(
                    text=pos[i],
                    insert=(265 + 50 * (i % 4), 240),
                    fill="black",
                    font_size="20px",
                    font_family="monospace",
                )
            )

        return board_g

    def _to_svg_string(self) -> str:
        return self._to_dwg_from_states(states=self.state).tostring()
