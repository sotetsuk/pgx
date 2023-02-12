import base64
import math
import os
from dataclasses import dataclass
from typing import List, Literal, Union

import numpy as np
import svgwrite  # type: ignore

from .animal_shogi import JaxAnimalShogiState as AnimalShogiState
from .backgammon import BackgammonState
from .chess import ChessState
from .contractbridgebidding import ContractBridgeBiddingState
from .go import GoState
from .shogi import State as ShogiState
from .suzume_jong import State as SuzumeJongState
from .tic_tac_toe import State as TictactoeState


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
    """PGXのVisualizer.

    color_mode: Literal["light", "dark"]
        light(ライトモードで表示)/dark(ダークモードで表示)
    scale: float
        表示する画像のサイズを拡大率で指定, デフォルトは1.0
    """

    def __init__(
        self,
        color_mode: Literal["light", "dark"] = "light",
        scale: float = 1.0,
    ) -> None:
        self.color_mode = color_mode
        self.color_set = VisualizerConfig()
        self.scale = scale
        self.GRID_SIZE = -1
        self.BOARD_WIDTH = -1
        self.BOARD_HEIGHT = -1
        self._make_dwg_group = None

    """
    notebook で可視化する際に、変数名のみで表示させる場合
    def _repr_html_(self) -> str:
        assert self.state is not None
        return self._to_dwg_from_states(states=self.state).tostring()
    """

    def save_svg(
        self,
        state: Union[
            AnimalShogiState,
            BackgammonState,
            ChessState,
            ContractBridgeBiddingState,
            GoState,
            ShogiState,
            SuzumeJongState,
            TictactoeState,
        ],
        filename="temp.svg",
    ) -> None:
        assert filename.endswith(".svg")
        self._to_dwg_from_states(states=state).saveas(filename=filename)

    def show_svg(
        self,
        states: Union[
            list,
            AnimalShogiState,
            BackgammonState,
            ChessState,
            ContractBridgeBiddingState,
            GoState,
            ShogiState,
            SuzumeJongState,
            TictactoeState,
        ],
    ) -> None:
        """Pgxのstate,batch処理されたstate,stateのリストをnotebook上で可視化する.

        states: Union[list, AnimalShogiState, BackgammonState, ChessState, ContractBridgeBiddingState, GoState, ShogiState, SuzumeJongState, TictactoeState]
            表示させるstate
        """
        import sys

        if "ipykernel" in sys.modules:
            # Jupyter Notebook

            if isinstance(states, list):
                self._show_states_in_widgets(
                    states=states,
                )

            else:
                from IPython.display import display_svg  # type:ignore

                display_svg(
                    self._to_dwg_from_states(
                        states=states,
                    ).tostring(),
                    raw=True,
                )
        else:
            # Not Jupyter
            sys.stdout.write("This function only works in Jupyter Notebook.")

    def _show_states_in_widgets(
        self,
        states: List[
            Union[
                AnimalShogiState,
                BackgammonState,
                ChessState,
                ContractBridgeBiddingState,
                GoState,
                ShogiState,
                SuzumeJongState,
                TictactoeState,
            ]
        ],
    ):
        import ipywidgets as widgets  # type:ignore
        from IPython.display import display, display_svg

        svg_strings = [
            self._to_dwg_from_states(
                states=_state,
            ).tostring()
            for _state in states
        ]
        N = len(svg_strings)
        self.i = -1

        def _on_click(button: widgets.Button):
            output.clear_output(True)
            with output:
                if button.description == "next":
                    self.i = (self.i + 1) % N
                else:
                    self.i = (self.i - 1) % N
                print(self.i)
                display_svg(
                    svg_strings[self.i],
                    raw=True,
                )

        button1 = widgets.Button(description="next")
        button1.on_click(_on_click)

        button2 = widgets.Button(description="back")
        button2.on_click(_on_click)

        output = widgets.Output()
        box = widgets.Box([button2, button1])

        display(box, output)
        button1.click()

    def _to_dwg_from_states(
        self,
        states,
    ):
        try:
            SIZE = len(states.turn)
            WIDTH = math.ceil(math.sqrt(SIZE - 0.1))
            if SIZE - (WIDTH - 1) ** 2 >= WIDTH:
                HEIGHT = WIDTH
            else:
                HEIGHT = WIDTH - 1
        except TypeError:
            SIZE = 1
            WIDTH = 1
            HEIGHT = 1

        self._set_config_by_state(
            states,
        )
        assert self._make_dwg_group is not None
        dwg = svgwrite.Drawing(
            "temp.svg",
            (
                (self.BOARD_WIDTH + 1) * self.GRID_SIZE * WIDTH * self.scale,
                (self.BOARD_HEIGHT + 1) * self.GRID_SIZE * HEIGHT * self.scale,
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
            g = self._make_dwg_group(dwg, states)
            g.translate(
                self.GRID_SIZE * 1 / 2,
                self.GRID_SIZE * 1 / 2,
            )
            group.add(g)
            group.scale(self.scale)
            dwg.add(group)
            return dwg

        for i in range(SIZE):
            x = i % WIDTH
            y = i // WIDTH
            _state = self._get_nth_state(states, i)
            g = self._make_dwg_group(
                dwg,
                _state,  # type:ignore
            )

            g.translate(
                self.GRID_SIZE * 1 / 2
                + (self.BOARD_WIDTH + 1) * self.GRID_SIZE * x,
                self.GRID_SIZE * 1 / 2
                + (self.BOARD_HEIGHT + 1) * self.GRID_SIZE * y,
            )
            group.add(g)
            group.add(
                dwg.rect(
                    (
                        (self.BOARD_WIDTH + 1) * self.GRID_SIZE * x,
                        (self.BOARD_HEIGHT + 1) * self.GRID_SIZE * y,
                    ),
                    (
                        (self.BOARD_WIDTH + 1) * self.GRID_SIZE,
                        (self.BOARD_HEIGHT + 1) * self.GRID_SIZE,
                    ),
                    fill="none",
                    stroke="gray",
                )
            )
        group.scale(self.scale)
        dwg.add(group)
        return dwg

    def _set_config_by_state(self, _state):  # noqa: C901
        if isinstance(_state, AnimalShogiState):
            self.GRID_SIZE = 60
            self.BOARD_WIDTH = 4
            self.BOARD_HEIGHT = 4
            self._make_dwg_group = self._make_animalshogi_dwg
            if (
                self.color_mode is None and self.color_mode == "dark"
            ) or self.color_mode == "dark":
                self.color_set = VisualizerConfig(
                    "dimgray",
                    "black",
                    "whitesmoke",
                    "whitesmoke",
                    "#202020",
                    "white",
                    "",
                )
            else:
                self.color_set = VisualizerConfig(
                    "white",
                    "lightgray",
                    "black",
                    "black",
                    "white",
                    "black",
                    "",
                )
        elif isinstance(_state, BackgammonState):
            self.GRID_SIZE = 50
            self.BOARD_WIDTH = 17
            self.BOARD_HEIGHT = 14
            self._make_dwg_group = self._make_backgammon_dwg
            if (
                self.color_mode is None and self.color_mode == "dark"
            ) or self.color_mode == "dark":
                self.color_set = VisualizerConfig(
                    "gray",
                    "black",
                    "black",
                    "dimgray",
                    "#202020",
                    "gainsboro",
                    "",
                )
            else:
                self.color_set = VisualizerConfig(
                    "white",
                    "black",
                    "lightgray",
                    "white",
                    "white",
                    "black",
                    "",
                )
        elif isinstance(_state, ChessState):
            self.GRID_SIZE = 50
            self.BOARD_WIDTH = 8
            self.BOARD_HEIGHT = 8
            self._make_dwg_group = self._make_chess_dwg
            if (
                self.color_mode is None and self.color_mode == "dark"
            ) or self.color_mode == "dark":
                self.color_set = VisualizerConfig(
                    "none",
                    "none",
                    "#404040",
                    "gray",
                    "#202020",
                    "silver",
                    "",
                )
            else:
                self.color_set = VisualizerConfig(
                    "none",
                    "none",
                    "gray",
                    "white",
                    "white",
                    "black",
                    "",
                )
        elif isinstance(_state, ContractBridgeBiddingState):
            self.GRID_SIZE = 50
            self.BOARD_WIDTH = 14
            self.BOARD_HEIGHT = 12
            self._make_dwg_group = self._make_bridge_dwg
            if (
                self.color_mode is None and self.color_mode == "dark"
            ) or self.color_mode == "dark":
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
        elif isinstance(_state, GoState):
            self.GRID_SIZE = 50
            try:
                self.BOARD_WIDTH = int(_state.size[0])
                self.BOARD_HEIGHT = int(_state.size[0])
            except IndexError:
                self.BOARD_WIDTH = int(_state.size)
                self.BOARD_HEIGHT = int(_state.size)
            self._make_dwg_group = self._make_go_dwg
            if (
                self.color_mode is None and self.color_mode == "dark"
            ) or self.color_mode == "dark":
                self.color_set = VisualizerConfig(
                    "black", "gray", "white", "white", "#202020", "white", ""
                )
            else:
                self.color_set = VisualizerConfig(
                    "black",
                    "white",
                    "black",
                    "black",
                    "white",
                    "black",
                    "",
                )
        elif isinstance(_state, ShogiState):
            self.GRID_SIZE = 50
            self.BOARD_WIDTH = 10
            self.BOARD_HEIGHT = 9
            self._make_dwg_group = self._make_shogi_dwg
            if (
                self.color_mode is None and self.color_mode == "dark"
            ) or self.color_mode == "dark":
                self.color_set = VisualizerConfig(
                    "gray", "black", "gray", "gray", "#202020", "gray", ""
                )
            else:
                self.color_set = VisualizerConfig(
                    "white",
                    "lightgray",
                    "black",
                    "black",
                    "white",
                    "black",
                    "",
                )
        elif isinstance(_state, SuzumeJongState):
            self.GRID_SIZE = 50
            self.BOARD_WIDTH = 15
            self.BOARD_HEIGHT = 10
            self._make_dwg_group = self._make_suzumejong_dwg
            if (
                self.color_mode is None and self.color_mode == "dark"
            ) or self.color_mode == "dark":
                self.color_set = VisualizerConfig(
                    "lightgray",
                    "dimgray",
                    "#404040",
                    "gray",
                    "#202020",
                    "darkgray",
                    "whitesmoke",
                )
            else:
                self.color_set = VisualizerConfig(
                    "white",
                    "white",
                    "gray",
                    "white",
                    "white",
                    "silver",
                    "black",
                )
        elif isinstance(_state, TictactoeState):
            self.GRID_SIZE = 50
            self.BOARD_WIDTH = 3
            self.BOARD_HEIGHT = 3
            self._make_dwg_group = self._make_tictactoe_dwg
            if (
                self.color_mode is None and self.color_mode == "dark"
            ) or self.color_mode == "dark":
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
        else:
            assert False

    def _get_nth_state(self, _states, _i):
        if isinstance(_states, AnimalShogiState):
            return AnimalShogiState(
                turn=_states.turn[_i],  # type:ignore
                board=_states.board[_i],
                hand=_states.hand[_i],
            )
        elif isinstance(_states, BackgammonState):
            return BackgammonState(
                turn=_states.turn[_i],  # type:ignore
                board=_states.board[_i],
            )
        elif isinstance(_states, ChessState):
            return ChessState(
                turn=_states.turn[_i],  # type:ignore
                board=_states.board[_i],
            )
        elif isinstance(_states, ContractBridgeBiddingState):
            return ContractBridgeBiddingState(  # type:ignore
                turn=_states.turn[_i],
                curr_player=_states.curr_player[_i],
                hand=_states.hand[_i],
                bidding_history=_states.bidding_history[_i],
                vul_NS=_states.vul_NS[_i],
                vul_EW=_states.vul_EW[_i],
            )
        elif isinstance(_states, GoState):
            return GoState(
                size=_states.size[_i],  # type:ignore
                ren_id_board=_states.ren_id_board[_i],
                turn=_states.turn[_i],
            )
        elif isinstance(_states, ShogiState):
            return ShogiState(
                turn=_states.turn[_i],  # type:ignore
                piece_board=_states.piece_board[_i],
                hand=_states.hand[_i],
            )
        elif isinstance(_states, SuzumeJongState):
            return SuzumeJongState(
                curr_player=_states.curr_player[_i],
                turn=_states.turn[_i],
                rivers=_states.rivers[_i],
                hands=_states.hands[_i],
                n_red_in_hands=_states.n_red_in_hands[_i],
                is_red_in_river=_states.is_red_in_river[_i],
                wall=_states.wall[_i],
                draw_ix=_states.draw_ix[_i],
                shuffled_players=_states.shuffled_players[_i],
                dora=_states.dora[_i],
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

    def _make_animalshogi_dwg(
        self,
        dwg,
        state: AnimalShogiState,
    ) -> svgwrite.Drawing:
        BOARD_WIDTH = 3
        BOARD_HEIGHT = 4
        GRID_SIZE = self.GRID_SIZE
        MOVE = {
            "P": [(0, -1)],
            "R": [(1, 0), (0, 1), (-1, 0), (0, -1)],
            "B": [(1, 1), (-1, 1), (-1, -1), (1, -1)],
            "K": [
                (1, 0),
                (0, 1),
                (-1, 0),
                (0, -1),
                (1, 1),
                (-1, 1),
                (-1, -1),
                (1, -1),
            ],
            "G": [(1, 0), (0, 1), (-1, 0), (0, -1), (-1, -1), (1, -1)],
        }

        # background
        dwg.add(
            dwg.rect(
                (0, 0),
                (
                    (BOARD_WIDTH + 2) * GRID_SIZE,
                    (BOARD_HEIGHT + 1) * GRID_SIZE,
                ),
                fill=self.color_set.background_color,
            )
        )

        # board
        # grid
        board_g = dwg.g()
        hlines = board_g.add(
            dwg.g(id="hlines", stroke=self.color_set.grid_color)
        )
        for y in range(1, BOARD_HEIGHT):
            hlines.add(
                dwg.line(
                    start=(0, GRID_SIZE * y),
                    end=(
                        GRID_SIZE * BOARD_WIDTH,
                        GRID_SIZE * y,
                    ),
                )
            )
        vlines = board_g.add(
            dwg.g(id="vline", stroke=self.color_set.grid_color)
        )
        for x in range(1, BOARD_WIDTH):
            vlines.add(
                dwg.line(
                    start=(GRID_SIZE * x, 0),
                    end=(
                        GRID_SIZE * x,
                        GRID_SIZE * BOARD_HEIGHT,
                    ),
                )
            )
        board_g.add(
            dwg.rect(
                (0, 0),
                (
                    BOARD_WIDTH * GRID_SIZE,
                    BOARD_HEIGHT * GRID_SIZE,
                ),
                fill="none",
                stroke=self.color_set.grid_color,
            )
        )

        # pieces
        p1_pieces_g = dwg.g()
        p2_pieces_g = dwg.g()
        for i, piece_pos, piece_type in zip(
            range(10),
            state.board[1:11],
            ["P", "R", "B", "K", "G", "P", "R", "B", "K", "G"],
        ):
            for xy, is_set in enumerate(piece_pos):
                if is_set == 1:
                    if i < 5:
                        pieces_g = p1_pieces_g
                        x = 2 - xy // BOARD_HEIGHT  # AnimalShogiStateは右上原点
                        y = xy % BOARD_HEIGHT
                        fill_color = self.color_set.p1_color
                        stroke = self.color_set.p1_outline
                    else:
                        pieces_g = p2_pieces_g
                        x = xy // BOARD_HEIGHT
                        y = 3 - xy % BOARD_HEIGHT
                        fill_color = self.color_set.p2_color
                        stroke = self.color_set.p2_outline

                    pieces_g.add(
                        dwg.rect(
                            insert=(
                                (x + 0.1) * GRID_SIZE,
                                (y + 0.1) * GRID_SIZE,
                            ),
                            size=(
                                0.8 * GRID_SIZE,
                                0.8 * GRID_SIZE,
                            ),
                            rx="3px",
                            ry="3px",
                            stroke=stroke,
                            fill=fill_color,
                        )
                    )
                    pieces_g.add(
                        dwg.text(
                            text=piece_type,
                            insert=(
                                (x + 0.27) * GRID_SIZE,
                                (y + 0.72) * GRID_SIZE,
                            ),
                            fill=stroke,
                            font_size="46px",
                            font_family="Courier",
                        )
                    )
                    # 移動可能方向
                    for _x, _y in MOVE[piece_type]:
                        pieces_g.add(
                            dwg.circle(
                                center=(
                                    (x + 0.5 + _x * 0.35) * GRID_SIZE,
                                    (y + 0.5 + _y * 0.35) * GRID_SIZE,
                                ),
                                r=GRID_SIZE * 0.01,
                                stroke=stroke,
                                fill=stroke,
                            )
                        )
        # hand
        for i, piece_num, piece_type in zip(
            range(6), state.hand, ["P", "R", "B", "P", "R", "B"]
        ):
            _g = p1_pieces_g if i < 3 else p2_pieces_g
            _g.add(
                dwg.text(
                    text=f"{piece_type}:{piece_num}",
                    insert=(
                        3.1 * GRID_SIZE,
                        (3.3 + (i % 3) * 0.3) * GRID_SIZE,
                    ),
                    fill=self.color_set.p1_outline,
                    font_size="20px",
                    font_family="Courier",
                )
            )

        board_g.add(p1_pieces_g)
        p2_pieces_g.rotate(
            angle=180,
            center=(GRID_SIZE * BOARD_WIDTH / 2, GRID_SIZE * BOARD_HEIGHT / 2),
        )
        board_g.add(p2_pieces_g)
        board_g.translate(GRID_SIZE / 2, 0)

        return board_g

    def _make_backgammon_dwg(
        self,
        dwg,
        state: BackgammonState,
    ) -> svgwrite.Drawing:
        BOARD_WIDTH = self.BOARD_WIDTH
        BOARD_HEIGHT = self.BOARD_HEIGHT
        GRID_SIZE = self.GRID_SIZE
        # background
        dwg.add(
            dwg.rect(
                (0, 0),
                (
                    (BOARD_WIDTH + 6) * GRID_SIZE,
                    (BOARD_HEIGHT + 3) * GRID_SIZE,
                ),
                fill=self.color_set.background_color,
            )
        )

        # board
        # grid
        board_g = dwg.g()
        board_g.add(
            dwg.rect(
                (0, 0),
                (
                    13 * GRID_SIZE,
                    14 * GRID_SIZE,
                ),
                stroke=self.color_set.grid_color,
                fill=self.color_set.background_color,
            )
        )
        board_g.add(
            dwg.rect(
                (6 * GRID_SIZE, 0),
                (
                    1 * GRID_SIZE,
                    14 * GRID_SIZE,
                ),
                stroke=self.color_set.grid_color,
                fill=self.color_set.background_color,
            )
        )

        for i in range(24):
            p1 = (i * GRID_SIZE, 0)
            p2 = ((i + 1) * GRID_SIZE, 0)
            p3 = ((i + 0.5) * GRID_SIZE, 6 * GRID_SIZE)
            if 6 <= i < 12:
                p1 = ((i + 1) * GRID_SIZE, 0)
                p2 = ((i + 2) * GRID_SIZE, 0)
                p3 = ((i + 1.5) * GRID_SIZE, 6 * GRID_SIZE)
            elif 12 <= i < 18:
                p1 = ((i - 12) * GRID_SIZE, 14 * GRID_SIZE)
                p2 = ((i + 1 - 12) * GRID_SIZE, 14 * GRID_SIZE)
                p3 = ((i + 0.5 - 12) * GRID_SIZE, 8 * GRID_SIZE)
            elif 18 <= i:
                p1 = ((i + 1 - 12) * GRID_SIZE, 14 * GRID_SIZE)
                p2 = ((i + 2 - 12) * GRID_SIZE, 14 * GRID_SIZE)
                p3 = ((i + 1.5 - 12) * GRID_SIZE, 8 * GRID_SIZE)

            fill_color = (
                self.color_set.p1_outline
                if i % 2 == 0
                else self.color_set.p2_outline
            )

            board_g.add(
                dwg.polygon(
                    points=[p1, p2, p3],
                    stroke=self.color_set.grid_color,
                    fill=fill_color,
                )
            )

        # pieces
        for i, piece in enumerate(state.board[:24]):
            fill_color = self.color_set.p2_color
            if piece < 0:  # 白
                piece = -piece
                fill_color = self.color_set.p1_color

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
                        center=(x * GRID_SIZE, (y + n * diff) * GRID_SIZE),
                        r=0.5 * GRID_SIZE,
                        stroke=self.color_set.grid_color,
                        fill=fill_color,
                        # stroke_width=GRID_SIZE,
                    )
                )

        # bar
        for i, piece in enumerate(state.board[24:26]):  # 24:白,25:黒
            fill_color = self.color_set.p2_color
            diff = 1
            if i == 0:
                piece = -piece
                fill_color = self.color_set.p1_color
                diff = -1
            for n in range(piece):
                board_g.add(
                    dwg.circle(
                        center=(
                            6.5 * GRID_SIZE,
                            (6 + i * 2 + n * diff) * GRID_SIZE,
                        ),
                        r=0.5 * GRID_SIZE,
                        stroke=self.color_set.grid_color,
                        fill=fill_color,
                        # stroke_width=GRID_SIZE,
                    )
                )

        # off
        board_g.add(
            dwg.rect(
                (13 * GRID_SIZE, 12 * GRID_SIZE),
                (
                    4 * GRID_SIZE,
                    2 * GRID_SIZE,
                ),
                stroke=self.color_set.grid_color,
                fill=self.color_set.background_color,
            )
        )
        board_g.add(
            dwg.circle(
                center=(14 * GRID_SIZE, 13 * GRID_SIZE),
                r=0.5 * GRID_SIZE,
                stroke=self.color_set.grid_color,
                fill=self.color_set.p1_color,
            )
        )
        board_g.add(
            dwg.text(
                text=f"× {-state.board[26]}",  # 26:白
                insert=(
                    14.8 * GRID_SIZE,
                    13.3 * GRID_SIZE,
                ),
                fill=self.color_set.grid_color,
                font_size="34px",
                font_family="serif",
            )
        )

        board_g.add(
            dwg.rect(
                (13 * GRID_SIZE, 0),
                (
                    4 * GRID_SIZE,
                    2 * GRID_SIZE,
                ),
                stroke=self.color_set.grid_color,
                fill=self.color_set.background_color,
            )
        )
        board_g.add(
            dwg.circle(
                center=(14 * GRID_SIZE, 1 * GRID_SIZE),
                r=0.5 * GRID_SIZE,
                stroke=self.color_set.grid_color,
                fill=self.color_set.p2_color,
            )
        )
        board_g.add(
            dwg.text(
                text=f"× {state.board[27]}",  # 27:黒
                insert=(
                    14.8 * GRID_SIZE,
                    1.3 * GRID_SIZE,
                ),
                fill=self.color_set.grid_color,
                font_size="34px",
                font_family="serif",
            )
        )

        return board_g

    def _make_chess_dwg(
        self,
        dwg,
        state: ChessState,
    ) -> svgwrite.Drawing:
        def _set_piece(_x, _y, _type, _dwg, _dwg_g, grid_size):
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
                insert=((_x + 5), (_y + 5)),
                size=(grid_size * 0.8, grid_size * 0.8),
            )
            _dwg_g.add(img)

            return _dwg_g

        GRID_SIZE = self.GRID_SIZE
        BOARD_WIDTH = 8
        BOARD_HEIGHT = 8
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

        # background
        dwg.add(
            dwg.rect(
                (0, 0),
                (
                    (BOARD_WIDTH + 1.5) * GRID_SIZE,
                    (BOARD_HEIGHT + 1) * GRID_SIZE,
                ),
                fill=self.color_set.background_color,
            )
        )

        # board
        # grid
        board_g = dwg.g()
        for i in range(BOARD_WIDTH * BOARD_HEIGHT):
            if (i // BOARD_HEIGHT) % 2 != i % 2:
                fill_color = self.color_set.p1_outline
            else:
                fill_color = self.color_set.p2_outline

            x = i % BOARD_WIDTH
            y = i // BOARD_HEIGHT
            board_g.add(
                dwg.rect(
                    (x * GRID_SIZE, y * GRID_SIZE),
                    (
                        GRID_SIZE,
                        GRID_SIZE,
                    ),
                    fill=fill_color,
                )
            )

        board_g.add(
            dwg.rect(
                (0, 0),
                (
                    BOARD_WIDTH * GRID_SIZE,
                    BOARD_HEIGHT * GRID_SIZE,
                ),
                fill="none",
                stroke=self.color_set.grid_color,
                stroke_width="3px",
            )
        )

        # dan,suji
        cord = board_g.add(dwg.g(id="cord", fill=self.color_set.grid_color))
        for i in range(BOARD_WIDTH):
            cord.add(
                dwg.text(
                    text=f"{8-i}",
                    insert=(
                        (-0.3) * GRID_SIZE,
                        (i + 0.6) * GRID_SIZE,
                    ),
                    font_size="20px",
                    font_family="Serif",
                )
            )
            cord.add(
                dwg.text(
                    text=f"{NUM_TO_CHAR[i]}",
                    insert=(
                        (i + 0.4) * GRID_SIZE,
                        (8.35) * GRID_SIZE,
                    ),
                    font_size="20px",
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
                    pieces_g = _set_piece(
                        x * GRID_SIZE,
                        y * GRID_SIZE,
                        piece_type,
                        dwg,
                        pieces_g,
                        GRID_SIZE,
                    )

        board_g.add(pieces_g)
        board_g.translate(10, 0)

        return board_g

    def _make_bridge_dwg(
        self,
        dwg,
        state: ContractBridgeBiddingState,
    ) -> svgwrite.Drawing:
        NUM_CARD_TYPE = 13
        TO_CARD = [
            "A",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "J",
            "Q",
            "K",
        ]
        SUITS = ["\u2660", "\u2665", "\u2666", "\u2663", "N"]  # ♠♡♢♣
        ACT = ["P", "X", "XX"]

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
                        stroke=self.color_set.grid_color,
                        stroke_width="2px",
                    )
                )
                board_g.add(
                    dwg.line(
                        start=(x_offset[i], y_offset[i] + 10),
                        end=(x_offset[i] + area_width, y_offset[i] + 10),
                        stroke=self.color_set.grid_color,
                        stroke_width="2px",
                    )
                )
                board_g.add(
                    dwg.line(
                        start=(x_offset[i], y_offset[i] + 38),
                        end=(x_offset[i] + area_width, y_offset[i] + 38),
                        stroke=self.color_set.grid_color,
                        stroke_width="2px",
                    )
                )
                board_g.add(
                    dwg.line(
                        start=(x_offset[i], y_offset[i] + 68),
                        end=(x_offset[i] + area_width, y_offset[i] + 68),
                        stroke=self.color_set.grid_color,
                        stroke_width="2px",
                    )
                )
                board_g.add(
                    dwg.line(
                        start=(x_offset[i], y_offset[i] + 98),
                        end=(x_offset[i] + area_width, y_offset[i] + 98),
                        stroke=self.color_set.grid_color,
                        stroke_width="2px",
                    )
                )
                board_g.add(
                    dwg.line(
                        start=(x_offset[i] + 32, y_offset[i] + 10),
                        end=(x_offset[i] + 32, y_offset[i] + 130),
                        stroke=self.color_set.grid_color,
                        stroke_width="2px",
                    )
                )
                board_g.add(
                    dwg.text(
                        text=pos[i],
                        insert=(x_offset[i] + 10, y_offset[i]),
                        fill=self.color_set.grid_color,
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
                        else self.color_set.text_color,
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
                stroke=self.color_set.grid_color,
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
                else self.color_set.text_color
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
                stroke=self.color_set.grid_color,
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
                    fill=self.color_set.text_color,
                    font_size="20px",
                    font_family="Courier",
                )
            )

        return board_g

    def _make_go_dwg(
        self,
        dwg,
        state: GoState,
    ) -> svgwrite.Drawing:
        from pgx.go import get_board

        BOARD_SIZE = self.BOARD_WIDTH
        GRID_SIZE = 50

        # background
        dwg.add(
            dwg.rect(
                (0, 0),
                (BOARD_SIZE * GRID_SIZE, BOARD_SIZE * GRID_SIZE),
                # stroke=svgwrite.rgb(10, 10, 16, "%"),
                fill=self.color_set.background_color,
            )
        )

        # board
        # grid
        board_g = dwg.g()
        hlines = board_g.add(
            dwg.g(id="hlines", stroke=self.color_set.grid_color)
        )
        for y in range(BOARD_SIZE):
            hlines.add(
                dwg.line(
                    start=(0, GRID_SIZE * y),
                    end=(
                        GRID_SIZE * (BOARD_SIZE - 1),
                        GRID_SIZE * y,
                    ),
                )
            )
        vlines = board_g.add(
            dwg.g(id="vline", stroke=self.color_set.grid_color)
        )
        for x in range(BOARD_SIZE):
            vlines.add(
                dwg.line(
                    start=(GRID_SIZE * x, 0),
                    end=(
                        GRID_SIZE * x,
                        GRID_SIZE * (BOARD_SIZE - 1),
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
                    center=((x - 1) * GRID_SIZE, (y - 1) * GRID_SIZE),
                    r=GRID_SIZE / 10,
                    fill=self.color_set.grid_color,
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
                self.color_set.p1_color
                if stone == 0
                else self.color_set.p2_color
            )
            outline = (
                self.color_set.p1_outline
                if stone == 0
                else self.color_set.p2_outline
            )
            board_g.add(
                dwg.circle(
                    center=(stone_x, stone_y),
                    r=GRID_SIZE / 2.2,
                    stroke=outline,
                    fill=color,
                )
            )
        board_g.translate(GRID_SIZE / 2, GRID_SIZE / 2)

        return board_g

    def _make_shogi_dwg(  # noqa: C901
        self,
        dwg,
        state: ShogiState,
    ) -> svgwrite.Drawing:
        if state.turn == 1:
            from pgx.shogi import _flip

            state = _flip(state)
        PIECES = [
            "歩",
            "香",
            "桂",
            "銀",
            "角",
            "飛",
            "金",
            "玉",
            "と",
            "成香",
            "成桂",
            "成銀",
            "馬",
            "龍",
            "歩",
            "香",
            "桂",
            "銀",
            "角",
            "飛",
            "金",
            "玉",
            "と",
            "成香",
            "成桂",
            "成銀",
            "馬",
            "龍",
        ]

        NUM_TO_CHAR = [
            "一",
            "二",
            "三",
            "四",
            "五",
            "六",
            "七",
            "八",
            "九",
            "十",
        ]

        def _sort_pieces(state, p1_hand, p2_hand):
            """
            ShogiStateのhandを飛、角、金、銀、桂、香、歩の順にする
            """
            hands = state.hand.flatten()[::-1]
            tmp = hands
            hands = hands.at[0].set(tmp[1])
            hands = hands.at[1].set(tmp[2])
            hands = hands.at[2].set(tmp[0])
            hands = hands.at[7].set(tmp[8])
            hands = hands.at[8].set(tmp[9])
            hands = hands.at[9].set(tmp[7])
            pieces = PIECES[6::-1]
            pieces[0], pieces[1], pieces[2] = pieces[1], pieces[2], pieces[0]

            for i, piece_num, piece_type in zip(
                range(14), hands, pieces + pieces
            ):
                hand = p2_hand if i < 7 else p1_hand
                if piece_num == 10:
                    hand.append(piece_type)
                    hand.append("十")
                elif piece_num > 0:
                    hand.append(piece_type)
                    if piece_num > 9:
                        hand.append("十")
                    if piece_num > 1:
                        hand.append(NUM_TO_CHAR[piece_num % 10 - 1])

            return p1_hand, p2_hand

        GRID_SIZE = self.GRID_SIZE
        BOARD_WIDTH = 9
        BOARD_HEIGHT = 9

        # background
        dwg.add(
            dwg.rect(
                (0, 0),
                (
                    (BOARD_WIDTH + 2.5) * GRID_SIZE,
                    (BOARD_HEIGHT + 1) * GRID_SIZE,
                ),
                fill=self.color_set.background_color,
            )
        )

        # board
        # grid
        board_g = dwg.g()
        hlines = board_g.add(
            dwg.g(id="hlines", stroke=self.color_set.grid_color)
        )
        for y in range(1, 9):
            hlines.add(
                dwg.line(
                    start=(0, GRID_SIZE * y),
                    end=(
                        GRID_SIZE * BOARD_WIDTH,
                        GRID_SIZE * y,
                    ),
                    stroke_width="2px",
                )
            )
        vlines = board_g.add(
            dwg.g(id="vline", stroke=self.color_set.grid_color)
        )
        for x in range(1, 9):
            vlines.add(
                dwg.line(
                    start=(GRID_SIZE * x, 0),
                    end=(
                        GRID_SIZE * x,
                        GRID_SIZE * BOARD_HEIGHT,
                    ),
                    stroke_width="2px",
                )
            )
        board_g.add(
            dwg.rect(
                (0, 0),
                (
                    BOARD_WIDTH * GRID_SIZE,
                    BOARD_HEIGHT * GRID_SIZE,
                ),
                fill="none",
                stroke=self.color_set.grid_color,
                stroke_width="4px",
            )
        )

        # dan,suji
        cord = board_g.add(dwg.g(id="cord", fill=self.color_set.grid_color))
        for i in range(9):
            cord.add(
                dwg.text(
                    text=f"{NUM_TO_CHAR[i]}",
                    insert=(
                        (9.1) * GRID_SIZE,
                        (i + 0.6) * GRID_SIZE,
                    ),
                    font_size="20px",
                    font_family="Serif",
                )
            )
            cord.add(
                dwg.text(
                    text=f"{i+1}",
                    insert=(
                        (8 - i + 0.4) * GRID_SIZE,
                        (-0.1) * GRID_SIZE,
                    ),
                    font_size="20px",
                    font_family="Serif",
                )
            )

        # pieces
        p1_pieces_g = dwg.g()
        p2_pieces_g = dwg.g()
        one_hot_board = np.zeros((29, 81))
        board = state.piece_board
        for i in range(81):
            piece = board[i]
            one_hot_board[piece, i] = 1
        for i, piece_pos, piece_type in zip(
            range(28),
            one_hot_board,
            PIECES,
        ):
            for xy, is_set in enumerate(piece_pos):
                if is_set == 1:
                    if i < 14:
                        pieces_g = p1_pieces_g
                        x = 8 - xy // BOARD_HEIGHT  # ShogiStateは右上原点
                        y = xy % BOARD_HEIGHT
                        stroke = self.color_set.p1_outline
                    else:
                        pieces_g = p2_pieces_g
                        x = xy // BOARD_HEIGHT
                        y = 8 - xy % BOARD_HEIGHT
                        stroke = self.color_set.p2_outline

                    if len(piece_type) > 1:
                        pieces_g.add(
                            dwg.text(
                                text=piece_type[0],
                                insert=(
                                    (x + 0.22) * GRID_SIZE,
                                    (y + 0.45) * GRID_SIZE,
                                ),
                                fill=stroke,
                                font_size="26px",
                                font_family="Serif",
                            )
                        )
                        pieces_g.add(
                            dwg.text(
                                text=piece_type[1],
                                insert=(
                                    (x + 0.22) * GRID_SIZE,
                                    (y + 0.95) * GRID_SIZE,
                                ),
                                fill=stroke,
                                font_size="26px",
                                font_family="Serif",
                            )
                        )
                    else:
                        pieces_g.add(
                            dwg.text(
                                text=piece_type,
                                insert=(
                                    (x + 0.05) * GRID_SIZE,
                                    (y + 0.85) * GRID_SIZE,
                                ),
                                fill=stroke,
                                font_size="45px",
                                font_family="Serif",
                            )
                        )

        # hand
        p1_hand = ["☗", "先", "手", ""]
        p2_hand = ["☖", "後", "手", ""]

        # 成り駒をソートする処理
        p1_hand, p2_hand = _sort_pieces(state, p1_hand, p2_hand)

        for i in range(2):
            if i == 0:
                pieces_g = p1_pieces_g
                hand = p1_hand
                stroke = self.color_set.p1_outline
                offset = len(p1_hand)
            else:
                pieces_g = p2_pieces_g
                hand = p2_hand
                stroke = self.color_set.p2_outline
                offset = len(p2_hand)
            for j, txt in enumerate(hand):
                pieces_g.add(
                    dwg.text(
                        text=txt,
                        insert=(
                            (9.5 - i * 0.4) * GRID_SIZE,
                            (9.5 - (offset - j) * 0.7) * GRID_SIZE,
                        ),
                        fill=stroke,
                        font_size="34px",
                        font_family="Serif",
                    )
                )

        board_g.add(p1_pieces_g)
        p2_pieces_g.rotate(
            angle=180,
            center=(BOARD_WIDTH * GRID_SIZE / 2, BOARD_HEIGHT * GRID_SIZE / 2),
        )
        board_g.add(p2_pieces_g)
        board_g.translate(15, 0)

        return board_g

    def _make_suzumejong_dwg(
        self,
        dwg,
        state: SuzumeJongState,
    ) -> svgwrite.Drawing:
        from .suzume_jong import NUM_TILE_TYPES, NUM_TILES, _tile_type_to_str

        BOARD_WIDTH = 15
        BOARD_HEIGHT = 10
        GRID_SIZE = 50

        def _set_piece(
            _x,
            _y,
            _type,
            _is_red,
            _dwg,
            _dwg_g,
            grid_size,
        ):
            _dwg_g.add(
                _dwg.rect(
                    ((_x + 9), (_y + 1)),
                    (
                        32,
                        47,
                    ),
                    fill=self.color_set.p1_color,  # "#f9f7e8",
                    stroke="none",
                )
            )
            type_str = _tile_type_to_str(_type)
            if _is_red and type_str != "g" and type_str != "r":
                type_str += "r"
            PATH = {
                f"{i+1}": f"images/suzume_jong/{i+1}p.svg" for i in range(9)
            }
            PATH.update(
                {
                    f"{i+1}r": f"images/suzume_jong/{i+1}pr.svg"
                    for i in range(9)
                }
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
                insert=(_x, _y),
                size=(grid_size, grid_size),
            )
            _dwg_g.add(img)

            return _dwg_g

        # background
        dwg.add(
            dwg.rect(
                (0, 0),
                (
                    BOARD_WIDTH * GRID_SIZE,
                    BOARD_HEIGHT * GRID_SIZE,
                ),
                fill=self.color_set.background_color,
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
                    (245, 360),
                    (
                        260,
                        70,
                    ),
                    rx="2px",
                    ry="2px",
                    fill=self.color_set.p2_color,
                    stroke=self.color_set.grid_color,
                    stroke_width="5px",
                )
            )

            # hands
            x = 250
            y = 370
            for type, num in zip(
                range(NUM_TILE_TYPES),
                state.hands[player_id],
            ):
                num_red = state.n_red_in_hands[player_id, type]
                for _ in range(num):
                    pieces_g = _set_piece(
                        x,
                        y,
                        type,
                        num_red > 0,
                        dwg,
                        pieces_g,
                        GRID_SIZE,
                    )
                    x += 40
                    num_red -= 1

            # river
            x = 270
            y = 220
            river_count = 0
            for type, is_red in zip(
                state.rivers[player_id], state.is_red_in_river[player_id]
            ):
                if type >= 0:
                    pieces_g = _set_piece(
                        x,
                        y,
                        type,
                        is_red,
                        dwg,
                        pieces_g,
                        GRID_SIZE,
                    )
                    x += 40
                    river_count += 1
                    if river_count > 4:
                        river_count = 0
                        x = 270
                        y += 60

            if player_id == state.shuffled_players[1]:
                pieces_g.rotate(
                    angle=90, center=(BOARD_WIDTH * GRID_SIZE / 2, 100)
                )

            elif player_id == state.shuffled_players[2]:
                pieces_g.rotate(
                    angle=-90, center=(BOARD_WIDTH * GRID_SIZE / 2, 100)
                )

            board_g.add(pieces_g)

        # dora
        board_g.add(
            dwg.rect(
                (BOARD_WIDTH * GRID_SIZE / 2 - 40, 0),
                (
                    80,
                    80,
                ),
                rx="5px",
                ry="5px",
                fill=self.color_set.p1_outline,
            )
        )
        board_g.add(
            dwg.rect(
                (BOARD_WIDTH * GRID_SIZE / 2 - 34, 6),
                (
                    68,
                    68,
                ),
                rx="5px",
                ry="5px",
                fill="none",
                stroke=self.color_set.p2_outline,
                stroke_width="3px",
            )
        )
        board_g = _set_piece(
            BOARD_WIDTH * GRID_SIZE / 2 - 25,
            15,
            state.dora,
            False,
            dwg,
            board_g,
            GRID_SIZE,
        )

        # wall
        wall_type = -1
        board_g = _set_piece(
            330, 120, wall_type, False, dwg, board_g, GRID_SIZE
        )
        board_g.add(
            dwg.text(
                text=f"× {NUM_TILES - state.draw_ix-1}",
                insert=(380, 150),
                fill=self.color_set.text_color,
                font_size="20px",
                font_family="serif",
            )
        )
        board_g.translate(0, 40)

        return board_g

    def _make_tictactoe_dwg(
        self,
        dwg,
        state: TictactoeState,
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
                stroke=self.color_set.grid_color,
                fill=self.color_set.grid_color,
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
                stroke=self.color_set.grid_color,
                fill=self.color_set.grid_color,
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
                        stroke=self.color_set.grid_color,
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
                        stroke=self.color_set.grid_color,
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
                        stroke=self.color_set.grid_color,
                        stroke_width=0.05 * GRID_SIZE,
                    )
                )
        return board_g
