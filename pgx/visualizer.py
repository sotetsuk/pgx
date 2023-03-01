import math
from dataclasses import dataclass
from typing import Literal

import svgwrite  # type: ignore

from .dwg.animalshogi import AnimalShogiState, _make_animalshogi_dwg
from .dwg.backgammon import BackgammonState, _make_backgammon_dwg
from .dwg.bridge_bidding import BridgeBiddingState, _make_bridge_dwg
from .dwg.chess import ChessState, _make_chess_dwg
from .dwg.connect_four import ConnectFourState, _make_connect_four_dwg
from .dwg.go import GoState, _make_go_dwg
from .dwg.othello import OthelloState, _make_othello_dwg
from .dwg.shogi import ShogiState, _make_shogi_dwg
from .dwg.sparrowmahjong import SparrowMahjongState, _make_sparrowmahjong_dwg
from .dwg.tictactoe import TictactoeState, _make_tictactoe_dwg


@dataclass
class ColorSet:
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
        self.config = {
            "GRID_SIZE": -1,
            "BOARD_WIDTH": -1,
            "BOARD_HEIGHT": -1,
            "COLOR_MODE": color_mode,
            "COLOR_SET": ColorSet(),
            "SCALE": scale,
        }
        self._make_dwg_group = None

    """
    notebook で可視化する際に、変数名のみで表示させる場合
    def _repr_html_(self) -> str:
        assert self.state is not None
        return self._to_dwg_from_states(states=self.state).tostring()
    """

    def save_svg(
        self,
        state,
        filename="temp.svg",
    ) -> None:
        assert filename.endswith(".svg")
        self._to_dwg_from_states(states=state).saveas(filename=filename)

    def show_svg(
        self,
        states,
    ) -> None:
        """Pgxのstate,batch処理されたstate,stateのリストをnotebook上で可視化する.

        states: Union[list, AnimalShogiState, BackgammonState, ChessState, BridgeBiddingState, GoState, OthelloState,ShogiState, SparrowMahjongState, TictactoeState]
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
        states,
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

        self._set_config_by_state(states)
        assert self._make_dwg_group is not None

        GRID_SIZE = self.config["GRID_SIZE"]
        BOARD_WIDTH = self.config["BOARD_WIDTH"]
        BOARD_HEIGHT = self.config["BOARD_HEIGHT"]
        SCALE = self.config["SCALE"]

        dwg = svgwrite.Drawing(
            "temp.svg",
            (
                (BOARD_WIDTH + 1) * GRID_SIZE * WIDTH * SCALE,
                (BOARD_HEIGHT + 1) * GRID_SIZE * HEIGHT * SCALE,
            ),
        )
        group = dwg.g()

        # background
        group.add(
            dwg.rect(
                (0, 0),
                (
                    (BOARD_WIDTH + 1) * GRID_SIZE * WIDTH,
                    (BOARD_HEIGHT + 1) * GRID_SIZE * HEIGHT,
                ),
                fill=self.config["COLOR_SET"].background_color,
            )
        )

        if SIZE == 1:
            g = self._make_dwg_group(dwg, states, self.config)
            g.translate(
                GRID_SIZE * 1 / 2,
                GRID_SIZE * 1 / 2,
            )
            group.add(g)
            group.scale(SCALE)
            dwg.add(group)
            return dwg

        for i in range(SIZE):
            x = i % WIDTH
            y = i // WIDTH
            _state = self._get_nth_state(states, i)
            g = self._make_dwg_group(
                dwg,
                _state,  # type:ignore
                self.config,
            )

            g.translate(
                GRID_SIZE * 1 / 2 + (BOARD_WIDTH + 1) * GRID_SIZE * x,
                GRID_SIZE * 1 / 2 + (BOARD_HEIGHT + 1) * GRID_SIZE * y,
            )
            group.add(g)
            group.add(
                dwg.rect(
                    (
                        (BOARD_WIDTH + 1) * GRID_SIZE * x,
                        (BOARD_HEIGHT + 1) * GRID_SIZE * y,
                    ),
                    (
                        (BOARD_WIDTH + 1) * GRID_SIZE,
                        (BOARD_HEIGHT + 1) * GRID_SIZE,
                    ),
                    fill="none",
                    stroke="gray",
                )
            )
        group.scale(SCALE)
        dwg.add(group)
        return dwg

    def _set_config_by_state(self, _state):  # noqa: C901
        if isinstance(_state, AnimalShogiState):
            self.config["GRID_SIZE"] = 60
            self.config["BOARD_WIDTH"] = 4
            self.config["BOARD_HEIGHT"] = 4
            self._make_dwg_group = _make_animalshogi_dwg
            if (
                self.config["COLOR_MODE"] is None
                and self.config["COLOR_MODE"] == "dark"
            ) or self.config["COLOR_MODE"] == "dark":
                self.config["COLOR_SET"] = ColorSet(
                    "dimgray",
                    "black",
                    "whitesmoke",
                    "whitesmoke",
                    "#202020",
                    "white",
                    "",
                )
            else:
                self.config["COLOR_SET"] = ColorSet(
                    "white",
                    "lightgray",
                    "black",
                    "black",
                    "white",
                    "black",
                    "",
                )
        elif isinstance(_state, BackgammonState):
            self.config["GRID_SIZE"] = 25
            self.config["BOARD_WIDTH"] = 17
            self.config["BOARD_HEIGHT"] = 14
            self._make_dwg_group = _make_backgammon_dwg
            if (
                self.config["COLOR_MODE"] is None
                and self.config["COLOR_MODE"] == "dark"
            ) or self.config["COLOR_MODE"] == "dark":
                self.config["COLOR_SET"] = ColorSet(
                    "gray",
                    "black",
                    "black",
                    "dimgray",
                    "#202020",
                    "gainsboro",
                    "",
                )
            else:
                self.config["COLOR_SET"] = ColorSet(
                    "white",
                    "black",
                    "lightgray",
                    "white",
                    "white",
                    "black",
                    "",
                )
        elif isinstance(_state, BridgeBiddingState):
            self.config["GRID_SIZE"] = 50
            self.config["BOARD_WIDTH"] = 13
            self.config["BOARD_HEIGHT"] = 10
            self._make_dwg_group = _make_bridge_dwg
            if (
                self.config["COLOR_MODE"] is None
                and self.config["COLOR_MODE"] == "dark"
            ) or self.config["COLOR_MODE"] == "dark":
                self.config["COLOR_SET"] = ColorSet(
                    "gray",
                    "black",
                    "black",
                    "dimgray",
                    "#202020",
                    "gainsboro",
                    "white",
                )
            else:
                self.config["COLOR_SET"] = ColorSet(
                    "white",
                    "black",
                    "lightgray",
                    "white",
                    "white",
                    "gray",
                    "black",
                )
        elif isinstance(_state, ChessState):
            self.config["GRID_SIZE"] = 50
            self.config["BOARD_WIDTH"] = 8
            self.config["BOARD_HEIGHT"] = 8
            self._make_dwg_group = _make_chess_dwg
            if (
                self.config["COLOR_MODE"] is None
                and self.config["COLOR_MODE"] == "dark"
            ) or self.config["COLOR_MODE"] == "dark":
                self.config["COLOR_SET"] = ColorSet(
                    "none",
                    "none",
                    "#404040",
                    "gray",
                    "#202020",
                    "silver",
                    "",
                )
            else:
                self.config["COLOR_SET"] = ColorSet(
                    "none",
                    "none",
                    "gray",
                    "white",
                    "white",
                    "black",
                    "",
                )
        elif isinstance(_state, ConnectFourState):
            self.config["GRID_SIZE"] = 35
            self.config["BOARD_WIDTH"] = 7
            self.config["BOARD_HEIGHT"] = 7
            self._make_dwg_group = _make_connect_four_dwg
            if (
                self.config["COLOR_MODE"] is None
                and self.config["COLOR_MODE"] == "dark"
            ) or self.config["COLOR_MODE"] == "dark":
                self.config["COLOR_SET"] = ColorSet(
                    "black",
                    "lightgray",
                    "white",
                    "lightgray",
                    "#202020",
                    "lightgray",
                    "",
                )
            else:
                self.config["COLOR_SET"] = ColorSet(
                    "black",
                    "white",
                    "black",
                    "black",
                    "white",
                    "black",
                    "",
                )
        elif isinstance(_state, GoState):
            self.config["GRID_SIZE"] = 25
            try:
                self.config["BOARD_WIDTH"] = int(_state.size[0])
                self.config["BOARD_HEIGHT"] = int(_state.size[0])
            except IndexError:
                self.config["BOARD_WIDTH"] = int(_state.size)
                self.config["BOARD_HEIGHT"] = int(_state.size)
            self._make_dwg_group = _make_go_dwg
            if (
                self.config["COLOR_MODE"] is None
                and self.config["COLOR_MODE"] == "dark"
            ) or self.config["COLOR_MODE"] == "dark":
                self.config["COLOR_SET"] = ColorSet(
                    "black", "gray", "white", "white", "#202020", "white", ""
                )
            else:
                self.config["COLOR_SET"] = ColorSet(
                    "black",
                    "white",
                    "black",
                    "black",
                    "white",
                    "black",
                    "",
                )
        elif isinstance(_state, OthelloState):
            self.config["GRID_SIZE"] = 30
            self.config["BOARD_WIDTH"] = 8
            self.config["BOARD_HEIGHT"] = 8
            self._make_dwg_group = _make_othello_dwg
            if (
                self.config["COLOR_MODE"] is None
                and self.config["COLOR_MODE"] == "dark"
            ) or self.config["COLOR_MODE"] == "dark":
                self.config["COLOR_SET"] = ColorSet(
                    "black",
                    "lightgray",
                    "white",
                    "lightgray",
                    "#202020",
                    "lightgray",
                    "",
                )
            else:
                self.config["COLOR_SET"] = ColorSet(
                    "black",
                    "white",
                    "black",
                    "black",
                    "white",
                    "black",
                    "",
                )
        elif isinstance(_state, ShogiState):
            self.config["GRID_SIZE"] = 50
            self.config["BOARD_WIDTH"] = 10
            self.config["BOARD_HEIGHT"] = 9
            self._make_dwg_group = _make_shogi_dwg
            if (
                self.config["COLOR_MODE"] is None
                and self.config["COLOR_MODE"] == "dark"
            ) or self.config["COLOR_MODE"] == "dark":
                self.config["COLOR_SET"] = ColorSet(
                    "gray", "black", "gray", "gray", "#202020", "gray", ""
                )
            else:
                self.config["COLOR_SET"] = ColorSet(
                    "white",
                    "lightgray",
                    "black",
                    "black",
                    "white",
                    "black",
                    "",
                )
        elif isinstance(_state, SparrowMahjongState):
            self.config["GRID_SIZE"] = 50
            self.config["BOARD_WIDTH"] = 15
            self.config["BOARD_HEIGHT"] = 10
            self._make_dwg_group = _make_sparrowmahjong_dwg
            if (
                self.config["COLOR_MODE"] is None
                and self.config["COLOR_MODE"] == "dark"
            ) or self.config["COLOR_MODE"] == "dark":
                self.config["COLOR_SET"] = ColorSet(
                    "lightgray",
                    "dimgray",
                    "#404040",
                    "gray",
                    "#202020",
                    "darkgray",
                    "whitesmoke",
                )
            else:
                self.config["COLOR_SET"] = ColorSet(
                    "white",
                    "white",
                    "gray",
                    "white",
                    "white",
                    "silver",
                    "black",
                )
        elif isinstance(_state, TictactoeState):
            self.config["GRID_SIZE"] = 60
            self.config["BOARD_WIDTH"] = 3
            self.config["BOARD_HEIGHT"] = 3
            self._make_dwg_group = _make_tictactoe_dwg
            if (
                self.config["COLOR_MODE"] is None
                and self.config["COLOR_MODE"] == "dark"
            ) or self.config["COLOR_MODE"] == "dark":
                self.config["COLOR_SET"] = ColorSet(
                    "gray",
                    "black",
                    "black",
                    "dimgray",
                    "#202020",
                    "gainsboro",
                )
            else:
                self.config["COLOR_SET"] = ColorSet(
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
        elif isinstance(_states, BridgeBiddingState):
            return BridgeBiddingState(  # type:ignore
                turn=_states.turn[_i],
                curr_player=_states.curr_player[_i],
                hand=_states.hand[_i],
                bidding_history=_states.bidding_history[_i],
                vul_NS=_states.vul_NS[_i],
                vul_EW=_states.vul_EW[_i],
            )
        elif isinstance(_states, GoState):
            return GoState(  # type:ignore
                size=_states.size[_i],
                ren_id_board=_states.ren_id_board[_i],
                turn=_states.turn[_i],
            )
        elif isinstance(_states, OthelloState):
            return OthelloState(
                board=_states.board[_i],
            )
        elif isinstance(_states, ShogiState):
            return ShogiState(  # type:ignore
                turn=_states.turn[_i],
                piece_board=_states.piece_board[_i],
                hand=_states.hand[_i],
            )
        elif isinstance(_states, SparrowMahjongState):
            return SparrowMahjongState(
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
