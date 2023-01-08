import math
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import svgwrite  # type: ignore

from .contractbridgebidding import ContractBridgeBiddingState
from .suzume_jong import State as SuzumeJongState
from .tic_tac_toe import State as TictactoeState

TO_CARD = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
SUITS = ["\u2660", "\u2665", "\u2666", "\u2663", "N"]  # ♠♡♢♣
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
        state: Union[None, ContractBridgeBiddingState, SuzumeJongState] = None,
        color_mode: str = "light",
    ) -> None:
        self.state = state
        self.color_mode = color_mode
        self.color_set = VisualizerConfig()
        self.GRID_SIZE = 50
        self.BOARD_WIDTH = 14
        self.BOARD_HEIGHT = 12
        self._make_dwg_group = None

    def _repr_html_(self) -> str:
        assert self.state is not None
        return self._to_dwg_from_states(states=self.state).tostring()

    def save_svg(self, filename="temp.svg", state=None) -> None:
        assert filename.endswith(".svg")
        if state is None:
            assert self.state is not None
            state = self.state
        self._to_dwg_from_states(states=state).saveas(filename=filename)

    def show_svg(
        self,
        states: Union[
            None, ContractBridgeBiddingState, SuzumeJongState
        ] = None,
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

    def set_state(
        self, state: Union[ContractBridgeBiddingState, SuzumeJongState]
    ) -> None:
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

        self._set_config_by_state(states, color_mode)
        assert self._make_dwg_group is not None

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
                fill=self.color_set.background_color,
            )
        )

        if SIZE == 1:
            g = self._make_dwg_group(dwg, states, self.color_set)
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
            _state = self._get_nth_state(states, i)
            g = self._make_dwg_group(dwg, _state, self.color_set)
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

    def _set_config_by_state(self, _state, _color_mode):
        if isinstance(_state, ContractBridgeBiddingState):
            self.GRID_SIZE = 50
            self.BOARD_WIDTH = 14
            self.BOARD_HEIGHT = 12
            self._make_dwg_group = self._make_bridge_dwg
            if (
                _color_mode is None and self.color_mode == "dark"
            ) or _color_mode == "dark":
                self.color_set = VisualizerConfig(
                    "gray",
                    "black",
                    "black",
                    "dimgray",
                    "#202020",
                    "gainsboro",
                    "white",
                )
            else:
                self.color_set = VisualizerConfig(
                    "white",
                    "black",
                    "lightgray",
                    "white",
                    "white",
                    "gray",
                    "black",
                )
        elif isinstance(_state, TictactoeState):
            self.GRID_SIZE = 50
            self.BOARD_WIDTH = 3
            self.BOARD_HEIGHT = 3
            self._make_dwg_group = self._make_tictactoe_dwg
            if (
                _color_mode is None and self.color_mode == "dark"
            ) or _color_mode == "dark":
                self.color_set = VisualizerConfig(
                    "gray",
                    "black",
                    "black",
                    "dimgray",
                    "#202020",
                    "gainsboro",
                )
            else:
                self.color_set = VisualizerConfig(
                    "white", "black", "lightgray", "white", "white", "black"
                )

    def _get_nth_state(self, _states, _i):
        if isinstance(_states, ContractBridgeBiddingState):
            return ContractBridgeBiddingState(  # type:ignore
                turn=_states.turn[_i],
                curr_player=_states.curr_player[_i],
                hand=_states.hand[_i],
                bidding_history=_states.bidding_history[_i],
                vul_NS=_states.vul_NS[_i],
                vul_EW=_states.vul_EW[_i],
            )
        elif isinstance(_states, TictactoeState):
            return TictactoeState(
                curr_player=_states.curr_player[_i],
                legal_action_mask=_states.legal_action_mask[_i],
                terminated=_states.terminated[_i],
                turn=_states.turn[_i],
                board=_states.board[_i],
            )
        else:
            assert False

    def _make_bridge_dwg(
        self,
        dwg,
        state: ContractBridgeBiddingState,
        color_set: VisualizerConfig,
    ) -> svgwrite.Drawing:
        NUM_CARD_TYPE = 13
        # board
        board_g = dwg.g()

        # hand
        x_offset = [240, 490, 240, -10]
        y_offset = [50, 250, 450, 250]
        for i in range(4):  # player0,1,2,3
            hand = sorted(
                state.hand[i * NUM_CARD_TYPE : (i + 1) * NUM_CARD_TYPE]
            )
            assert len(hand) == NUM_CARD_TYPE
            # player
            pos = np.array(["North", "East", "South", "West"], dtype=object)
            pos[state.dealer] = pos[state.dealer] + "(Dealer)"
            # suit
            for j in range(4):  # spades,hearts,diamonds,clubs
                area_width = 230
                area_height = 150
                board_g.add(
                    dwg.rect(
                        (x_offset[i], y_offset[i] - 20),
                        (area_width, area_height),
                        rx="5px",
                        ry="5px",
                        fill="none",
                        stroke=color_set.grid_color,
                        stroke_width="2px",
                    )
                )
                board_g.add(
                    dwg.line(
                        start=(x_offset[i], y_offset[i] + 10),
                        end=(x_offset[i] + area_width, y_offset[i] + 10),
                        stroke=color_set.grid_color,
                        stroke_width="2px",
                    )
                )
                board_g.add(
                    dwg.line(
                        start=(x_offset[i], y_offset[i] + 38),
                        end=(x_offset[i] + area_width, y_offset[i] + 38),
                        stroke=color_set.grid_color,
                        stroke_width="2px",
                    )
                )
                board_g.add(
                    dwg.line(
                        start=(x_offset[i], y_offset[i] + 68),
                        end=(x_offset[i] + area_width, y_offset[i] + 68),
                        stroke=color_set.grid_color,
                        stroke_width="2px",
                    )
                )
                board_g.add(
                    dwg.line(
                        start=(x_offset[i], y_offset[i] + 98),
                        end=(x_offset[i] + area_width, y_offset[i] + 98),
                        stroke=color_set.grid_color,
                        stroke_width="2px",
                    )
                )
                board_g.add(
                    dwg.line(
                        start=(x_offset[i] + 32, y_offset[i] + 10),
                        end=(x_offset[i] + 32, y_offset[i] + 130),
                        stroke=color_set.grid_color,
                        stroke_width="2px",
                    )
                )
                board_g.add(
                    dwg.text(
                        text=pos[i],
                        insert=(x_offset[i] + 10, y_offset[i]),
                        fill=color_set.grid_color,
                        font_size="20px",
                        font_family="Courier",
                        font_weight="bold",
                    )
                )
                if (state.vul_NS and i % 2 == 0) or (
                    state.vul_EW and i % 2 == 1
                ):
                    board_g.add(
                        dwg.text(
                            text="Vul.",
                            insert=(x_offset[i] + 180, y_offset[i]),
                            fill="orangered",
                            font_size="20px",
                            font_family="Courier",
                            font_weight="bold",
                        )
                    )

                card = [
                    TO_CARD[i % NUM_CARD_TYPE]
                    for i in hand
                    if j * NUM_CARD_TYPE <= i < (j + 1) * NUM_CARD_TYPE
                ][::-1]

                if card != [] and card[-1] == "A":
                    card = card[-1:] + card[:-1]

                suit = SUITS[j] + " " + " ".join(card)
                board_g.add(
                    dwg.text(
                        text=suit,
                        insert=(x_offset[i] + 10, y_offset[i] + 30 * (j + 1)),
                        fill="orangered"
                        if 0 < j < 3
                        else color_set.text_color,
                        font_size="24px",
                        font_family="Courier",
                        font_weight="bold",
                    )
                )

        board_g.add(
            dwg.rect(
                (250, 220),
                (210, 180),
                rx="5px",
                ry="5px",
                fill="none",
                stroke=color_set.grid_color,
                stroke_width="5px",
            )
        )

        # history
        for i, act in enumerate(state.bidding_history):
            if act == -1:
                break
            act_str = (
                str(act // 5 + 1) + SUITS[act % 5]
                if 0 <= act < 35
                else ACT[act - 35]
            )
            color = (
                "orangered"
                if act % 5 == 1 or act % 5 == 2
                else color_set.text_color
            )
            board_g.add(
                dwg.text(
                    text=act_str,
                    insert=(265 + 50 * (i % 4), 270 + 20 * (i // 4)),
                    fill=color,
                    font_size="20px",
                    font_family="Courier",
                )
            )
        board_g.add(
            dwg.line(
                start=(250, 250),
                end=(460, 250),
                stroke=color_set.grid_color,
                stroke_width="2px",
            )
        )

        # player
        pos = np.array(["N", "E", "S", "W"], dtype=object)
        pos = np.roll(pos, -state.dealer)
        pos[0] = pos[0] + "(D)"
        for i in range(4):
            board_g.add(
                dwg.text(
                    text=pos[i],
                    insert=(265 + 50 * (i % 4), 240),
                    fill=color_set.text_color,
                    font_size="20px",
                    font_family="Courier",
                )
            )

        return board_g

    def _make_tictactoe_dwg(
        self,
        dwg,
        state: TictactoeState,
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
