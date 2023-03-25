# Copyright 2023 The Pgx Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from dataclasses import dataclass
from typing import Literal, Optional

import svgwrite  # type: ignore

from ._dwg.animalshogi import AnimalShogiState, _make_animalshogi_dwg
from ._dwg.backgammon import BackgammonState, _make_backgammon_dwg
from ._dwg.bridge_bidding import BridgeBiddingState, _make_bridge_dwg
from ._dwg.chess import ChessState, _make_chess_dwg
from ._dwg.connect_four import ConnectFourState, _make_connect_four_dwg
from ._dwg.go import GoState, _make_go_dwg
from ._dwg.hex import HexState, _make_hex_dwg
from ._dwg.kuhn_poker import KuhnPokerState, _make_kuhnpoker_dwg
from ._dwg.othello import OthelloState, _make_othello_dwg
from ._dwg.play2048 import Play2048State, _make_2048_dwg
from ._dwg.shogi import ShogiState, _make_shogi_dwg
from ._dwg.sparrow_mahjong import SparrowMahjongState, _make_sparrowmahjong_dwg
from ._dwg.tictactoe import TictactoeState, _make_tictactoe_dwg

ColorTheme = Literal["light", "dark"]


@dataclass
class Config:
    color_theme: ColorTheme = "light"
    scale: float = 1.0


global_config = Config()


def set_visualization_config(
    *, color_theme: ColorTheme = "light", scale: float = 1.0
):
    global_config.color_theme = color_theme
    global_config.scale = scale


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
    """The Pgx Visualizer

    color_theme: Default(None) is "light"
    scale: change image size. Default(None) is 1.0
    """

    def __init__(
        self,
        *,
        color_theme: Optional[ColorTheme] = None,
        scale: Optional[float] = None,
    ) -> None:
        color_theme = (
            color_theme
            if color_theme is not None
            else global_config.color_theme
        )
        scale = scale if scale is not None else global_config.scale

        self.config = {
            "GRID_SIZE": -1,
            "BOARD_WIDTH": -1,
            "BOARD_HEIGHT": -1,
            "COLOR_THEME": color_theme,
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
        self.get_dwg(states=state).saveas(filename=filename)

    def get_dwg(
        self,
        states,
    ):
        try:
            SIZE = len(states.current_player)
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
                self.config["COLOR_THEME"] is None
                and self.config["COLOR_THEME"] == "dark"
            ) or self.config["COLOR_THEME"] == "dark":
                self.config["COLOR_SET"] = ColorSet(
                    "dimgray",
                    "black",
                    "whitesmoke",
                    "whitesmoke",
                    "#1e1e1e",
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
                self.config["COLOR_THEME"] is None
                and self.config["COLOR_THEME"] == "dark"
            ) or self.config["COLOR_THEME"] == "dark":
                self.config["COLOR_SET"] = ColorSet(
                    "gray",
                    "black",
                    "black",
                    "dimgray",
                    "#1e1e1e",
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
            self.config["BOARD_WIDTH"] = 14
            self.config["BOARD_HEIGHT"] = 10
            self._make_dwg_group = _make_bridge_dwg
            if (
                self.config["COLOR_THEME"] is None
                and self.config["COLOR_THEME"] == "dark"
            ) or self.config["COLOR_THEME"] == "dark":
                self.config["COLOR_SET"] = ColorSet(
                    "gray",
                    "black",
                    "black",
                    "dimgray",
                    "#1e1e1e",
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
                    "black",
                    "black",
                )
        elif isinstance(_state, ChessState):
            self.config["GRID_SIZE"] = 50
            self.config["BOARD_WIDTH"] = 8
            self.config["BOARD_HEIGHT"] = 8
            self._make_dwg_group = _make_chess_dwg
            if (
                self.config["COLOR_THEME"] is None
                and self.config["COLOR_THEME"] == "dark"
            ) or self.config["COLOR_THEME"] == "dark":
                self.config["COLOR_SET"] = ColorSet(
                    "none",
                    "none",
                    "#404040",
                    "gray",
                    "#1e1e1e",
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
                self.config["COLOR_THEME"] is None
                and self.config["COLOR_THEME"] == "dark"
            ) or self.config["COLOR_THEME"] == "dark":
                self.config["COLOR_SET"] = ColorSet(
                    "black",
                    "lightgray",
                    "white",
                    "lightgray",
                    "#1e1e1e",
                    "lightgray",
                    "gray",
                )
            else:
                self.config["COLOR_SET"] = ColorSet(
                    "black",
                    "white",
                    "black",
                    "black",
                    "white",
                    "black",
                    "gray",
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
                self.config["COLOR_THEME"] is None
                and self.config["COLOR_THEME"] == "dark"
            ) or self.config["COLOR_THEME"] == "dark":
                self.config["COLOR_SET"] = ColorSet(
                    "black", "gray", "white", "white", "#1e1e1e", "white", ""
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
        elif isinstance(_state, HexState):
            self.config["GRID_SIZE"] = 30
            try:
                self.config["BOARD_WIDTH"] = int(_state.size[0] * 1.3)
                self.config["BOARD_HEIGHT"] = int(_state.size[0] * 0.8)
            except IndexError:
                self.config["BOARD_WIDTH"] = int(_state.size * 1.3)
                self.config["BOARD_HEIGHT"] = int(_state.size * 0.8)
            self._make_dwg_group = _make_hex_dwg
            if (
                self.config["COLOR_THEME"] is None
                and self.config["COLOR_THEME"] == "dark"
            ) or self.config["COLOR_THEME"] == "dark":
                self.config["COLOR_SET"] = ColorSet(
                    "black",
                    "white",
                    "white",
                    "black",
                    "#1e1e1e",
                    "white",
                    "dimgray",
                )
            else:
                self.config["COLOR_SET"] = ColorSet(
                    "black",
                    "white",
                    "black",
                    "black",
                    "white",
                    "black",
                    "lightgray",
                )
        elif isinstance(_state, KuhnPokerState):
            self.config["GRID_SIZE"] = 30
            self.config["BOARD_WIDTH"] = 8
            self.config["BOARD_HEIGHT"] = 8
            self._make_dwg_group = _make_kuhnpoker_dwg
            if (
                self.config["COLOR_THEME"] is None
                and self.config["COLOR_THEME"] == "dark"
            ) or self.config["COLOR_THEME"] == "dark":
                self.config["COLOR_SET"] = ColorSet(
                    "black",
                    "lightgray",
                    "white",
                    "lightgray",
                    "#1e1e1e",
                    "lightgray",
                    "lightgray",
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
                self.config["COLOR_THEME"] is None
                and self.config["COLOR_THEME"] == "dark"
            ) or self.config["COLOR_THEME"] == "dark":
                self.config["COLOR_SET"] = ColorSet(
                    "black",
                    "lightgray",
                    "white",
                    "lightgray",
                    "#1e1e1e",
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
        elif isinstance(_state, Play2048State):
            self.config["GRID_SIZE"] = 50
            self.config["BOARD_WIDTH"] = 4
            self.config["BOARD_HEIGHT"] = 4
            self._make_dwg_group = _make_2048_dwg
            if (
                self.config["COLOR_THEME"] is None
                and self.config["COLOR_THEME"] == "dark"
            ) or self.config["COLOR_THEME"] == "dark":
                self.config["COLOR_SET"] = ColorSet(
                    "lightgray",
                    "",
                    "",
                    "",
                    "#1e1e1e",
                    "black",
                    "white",
                )
            else:
                self.config["COLOR_SET"] = ColorSet(
                    "black",
                    "#f0f0f0",
                    "",
                    "",
                    "white",
                    "black",
                    "black",
                )
        elif isinstance(_state, ShogiState):
            self.config["GRID_SIZE"] = 50
            self.config["BOARD_WIDTH"] = 10
            self.config["BOARD_HEIGHT"] = 9
            self._make_dwg_group = _make_shogi_dwg
            if (
                self.config["COLOR_THEME"] is None
                and self.config["COLOR_THEME"] == "dark"
            ) or self.config["COLOR_THEME"] == "dark":
                self.config["COLOR_SET"] = ColorSet(
                    "gray", "black", "gray", "gray", "#1e1e1e", "gray", ""
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
                self.config["COLOR_THEME"] is None
                and self.config["COLOR_THEME"] == "dark"
            ) or self.config["COLOR_THEME"] == "dark":
                self.config["COLOR_SET"] = ColorSet(
                    "lightgray",
                    "dimgray",
                    "#404040",
                    "gray",
                    "#1e1e1e",
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
                self.config["COLOR_THEME"] is None
                and self.config["COLOR_THEME"] == "dark"
            ) or self.config["COLOR_THEME"] == "dark":
                self.config["COLOR_SET"] = ColorSet(
                    "gray",
                    "black",
                    "black",
                    "dimgray",
                    "#1e1e1e",
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
        elif isinstance(_states, ConnectFourState):
            return ConnectFourState(  # type:ignore
                turn=_states.turn[_i],
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
                current_player=_states.current_player[_i],
                hand=_states.hand[_i],
                bidding_history=_states.bidding_history[_i],
                vul_NS=_states.vul_NS[_i],
                vul_EW=_states.vul_EW[_i],
            )
        elif isinstance(_states, GoState):
            return GoState(  # type:ignore
                size=_states.size[_i],
                chain_id_board=_states.chain_id_board[_i],
                turn=_states.turn[_i],
            )
        elif isinstance(_states, HexState):
            return HexState(
                size=_states.size[_i],
                turn=_states.turn[_i],
                board=_states.board[_i],
            )
        elif isinstance(_states, KuhnPokerState):
            return KuhnPokerState(
                cards=_states.cards[_i],
            )
        elif isinstance(_states, OthelloState):
            return OthelloState(
                turn=_states.turn[_i],
                board=_states.board[_i],
            )
        elif isinstance(_states, Play2048State):
            return Play2048State(
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
                current_player=_states.current_player[_i],
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
                current_player=_states.current_player[_i],
                legal_action_mask=_states.legal_action_mask[_i],
                terminated=_states.terminated[_i],
                turn=_states.turn[_i],
                board=_states.board[_i],
            )
        else:
            assert False
