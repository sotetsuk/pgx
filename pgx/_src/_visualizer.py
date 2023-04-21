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
from pathlib import Path
from typing import Literal, Optional, Sequence, Union

import svgwrite  # type: ignore

from pgx.v1 import State

ColorTheme = Literal["light", "dark"]


@dataclass
class Config:
    color_theme: ColorTheme = "light"
    scale: float = 1.0
    frame_duration_seconds: float = 0.2


global_config = Config()


def set_visualization_config(
    *,
    color_theme: ColorTheme = "light",
    scale: float = 1.0,
    frame_duration_seconds: float = 0.2,
):
    global_config.color_theme = color_theme
    global_config.scale = scale
    global_config.frame_duration_seconds = frame_duration_seconds


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
            if SIZE == 1:
                states = self._get_nth_state(states, 0)
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

    def _set_config_by_state(self, _state: State):  # noqa: C901
        if _state.env_id == "animal_shogi":
            from pgx._src.dwg.animalshogi import _make_animalshogi_dwg

            self.config["GRID_SIZE"] = 60
            self.config["BOARD_WIDTH"] = 4
            self.config["BOARD_HEIGHT"] = 4
            self._make_dwg_group = _make_animalshogi_dwg  # type:ignore
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
        elif _state.env_id == "backgammon":
            from pgx._src.dwg.backgammon import _make_backgammon_dwg

            self.config["GRID_SIZE"] = 25
            self.config["BOARD_WIDTH"] = 17
            self.config["BOARD_HEIGHT"] = 14
            self._make_dwg_group = _make_backgammon_dwg  # type:ignore
            if (
                self.config["COLOR_THEME"] is None
                and self.config["COLOR_THEME"] == "dark"
            ) or self.config["COLOR_THEME"] == "dark":
                self.config["COLOR_SET"] = ColorSet(
                    "black",
                    "darkgray",
                    "white",
                    "white",
                    "#1e1e1e",
                    "silver",
                    "dimgray",
                )
            else:
                self.config["COLOR_SET"] = ColorSet(
                    "white",
                    "black",
                    "lightgray",
                    "white",
                    "white",
                    "black",
                    "gray",
                )
        elif _state.env_id == "bridge_bidding":
            from pgx._src.dwg.bridge_bidding import _make_bridge_dwg

            self.config["GRID_SIZE"] = 50
            self.config["BOARD_WIDTH"] = 14
            self.config["BOARD_HEIGHT"] = 10
            self._make_dwg_group = _make_bridge_dwg  # type:ignore
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
        elif _state.env_id == "chess":
            from pgx._src.dwg.chess import _make_chess_dwg

            self.config["GRID_SIZE"] = 50
            self.config["BOARD_WIDTH"] = 8
            self.config["BOARD_HEIGHT"] = 8
            self._make_dwg_group = _make_chess_dwg  # type:ignore
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
        elif _state.env_id == "connect_four":
            from pgx._src.dwg.connect_four import _make_connect_four_dwg

            self.config["GRID_SIZE"] = 35
            self.config["BOARD_WIDTH"] = 7
            self.config["BOARD_HEIGHT"] = 7
            self._make_dwg_group = _make_connect_four_dwg  # type:ignore
            if (
                self.config["COLOR_THEME"] is None
                and self.config["COLOR_THEME"] == "dark"
            ) or self.config["COLOR_THEME"] == "dark":
                self.config["COLOR_SET"] = ColorSet(
                    "black",
                    "darkgray",
                    "white",
                    "white",
                    "#1e1e1e",
                    "silver",
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
        elif _state.env_id in ("go-9x9", "go-19x19"):
            from pgx._src.dwg.go import _make_go_dwg

            self.config["GRID_SIZE"] = 25
            try:
                self.config["BOARD_WIDTH"] = int(_state.size[0])  # type:ignore
                self.config["BOARD_HEIGHT"] = int(
                    _state.size[0]  # type:ignore
                )
            except IndexError:
                self.config["BOARD_WIDTH"] = int(_state.size)  # type:ignore
                self.config["BOARD_HEIGHT"] = int(_state.size)  # type:ignore
            self._make_dwg_group = _make_go_dwg  # type:ignore
            if (
                self.config["COLOR_THEME"] is None
                and self.config["COLOR_THEME"] == "dark"
            ) or self.config["COLOR_THEME"] == "dark":
                self.config["COLOR_SET"] = ColorSet(
                    "black",
                    "darkgray",
                    "white",
                    "white",
                    "#1e1e1e",
                    "silver",
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
        elif _state.env_id == "hex":
            import jax.numpy as jnp

            from pgx._src.dwg.hex import _make_hex_dwg, four_dig

            self.config["GRID_SIZE"] = 30
            try:
                self.config["BOARD_WIDTH"] = four_dig(
                    _state.size[0] * 1.5  # type:ignore
                )
                self.config["BOARD_HEIGHT"] = four_dig(
                    _state.size[0] * jnp.sqrt(3) / 2  # type:ignore
                )
            except IndexError:
                self.config["BOARD_WIDTH"] = four_dig(
                    _state.size * 1.5  # type:ignore
                )
                self.config["BOARD_HEIGHT"] = four_dig(
                    _state.size * jnp.sqrt(3) / 2  # type:ignore
                )
            self._make_dwg_group = _make_hex_dwg  # type:ignore
            if (
                self.config["COLOR_THEME"] is None
                and self.config["COLOR_THEME"] == "dark"
            ) or self.config["COLOR_THEME"] == "dark":
                self.config["COLOR_SET"] = ColorSet(
                    "black",
                    "darkgray",
                    "white",
                    "white",
                    "#1e1e1e",
                    "gray",
                    "#333333",
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
        elif _state.env_id == "kuhn_poker":
            from pgx._src.dwg.kuhn_poker import _make_kuhnpoker_dwg

            self.config["GRID_SIZE"] = 30
            self.config["BOARD_WIDTH"] = 8
            self.config["BOARD_HEIGHT"] = 8
            self._make_dwg_group = _make_kuhnpoker_dwg  # type:ignore
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
        elif _state.env_id == "leduc_holdem":
            from pgx._src.dwg.leduc_holdem import _make_leducHoldem_dwg

            self.config["GRID_SIZE"] = 30
            self.config["BOARD_WIDTH"] = 8
            self.config["BOARD_HEIGHT"] = 8
            self._make_dwg_group = _make_leducHoldem_dwg  # type:ignore
            if (
                self.config["COLOR_THEME"] is None
                and self.config["COLOR_THEME"] == "dark"
            ) or self.config["COLOR_THEME"] == "dark":
                self.config["COLOR_SET"] = ColorSet(
                    "gray",
                    "lightgray",
                    "",
                    "",
                    "#1e1e1e",
                    "lightgray",
                    "lightgray",
                )
            else:
                self.config["COLOR_SET"] = ColorSet(
                    "gray",
                    "black",
                    "",
                    "",
                    "white",
                    "black",
                    "",
                )
        elif _state.env_id == "othello":
            from pgx._src.dwg.othello import _make_othello_dwg

            self.config["GRID_SIZE"] = 30
            self.config["BOARD_WIDTH"] = 8
            self.config["BOARD_HEIGHT"] = 8
            self._make_dwg_group = _make_othello_dwg  # type:ignore
            if (
                self.config["COLOR_THEME"] is None
                and self.config["COLOR_THEME"] == "dark"
            ) or self.config["COLOR_THEME"] == "dark":
                self.config["COLOR_SET"] = ColorSet(
                    "black",
                    "darkgray",
                    "white",
                    "white",
                    "#1e1e1e",
                    "silver",
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
        elif _state.env_id == "2048":
            from pgx._src.dwg.play2048 import _make_2048_dwg

            self.config["GRID_SIZE"] = 50
            self.config["BOARD_WIDTH"] = 4
            self.config["BOARD_HEIGHT"] = 4
            self._make_dwg_group = _make_2048_dwg  # type:ignore
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
        elif _state.env_id == "shogi":
            from pgx._src.dwg.shogi import _make_shogi_dwg

            self.config["GRID_SIZE"] = 50
            self.config["BOARD_WIDTH"] = 10
            self.config["BOARD_HEIGHT"] = 9
            self._make_dwg_group = _make_shogi_dwg  # type:ignore
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
        elif _state.env_id == "sparrow_mahjong":
            from pgx._src.dwg.sparrow_mahjong import _make_sparrowmahjong_dwg

            self.config["GRID_SIZE"] = 50
            self.config["BOARD_WIDTH"] = 15
            self.config["BOARD_HEIGHT"] = 10
            self._make_dwg_group = _make_sparrowmahjong_dwg  # type:ignore
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
        elif _state.env_id == "tic_tac_toe":
            from pgx._src.dwg.tictactoe import _make_tictactoe_dwg

            self.config["GRID_SIZE"] = 60
            self.config["BOARD_WIDTH"] = 3
            self.config["BOARD_HEIGHT"] = 3
            self._make_dwg_group = _make_tictactoe_dwg  # type:ignore
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

    # TODO: simplify me
    def _get_nth_state(self, _states: State, _i):
        if _states.env_id == "animal_shogi":
            from pgx._src.dwg.animalshogi import AnimalShogiState

            return AnimalShogiState(
                turn=_states.turn[_i],  # type:ignore
                board=_states.board[_i],  # type:ignore
                hand=_states.hand[_i],  # type:ignore
            )
        elif _states.env_id == "backgammon":
            from pgx._src.dwg.backgammon import BackgammonState

            return BackgammonState(
                turn=_states.turn[_i],  # type:ignore
                board=_states.board[_i],  # type:ignore
            )
        elif _states.env_id == "connect_four":
            from pgx._src.dwg.connect_four import ConnectFourState

            return ConnectFourState(  # type:ignore
                turn=_states.turn[_i],  # type:ignore
                board=_states.board[_i],  # type:ignore
            )
        elif _states.env_id == "chess":
            from pgx._src.dwg.chess import ChessState

            return ChessState(
                turn=_states.turn[_i],  # type:ignore
                board=_states.board[_i],  # type:ignore
            )
        elif _states.env_id == "bridge_bidding":
            from pgx._src.dwg.bridge_bidding import BridgeBiddingState

            return BridgeBiddingState(  # type:ignore
                turn=_states.turn[_i],  # type:ignore
                dealer=_states.dealer[_i],  # type:ignore
                current_player=_states.current_player[_i],  # type:ignore
                hand=_states.hand[_i],  # type:ignore
                bidding_history=_states.bidding_history[_i],  # type:ignore
                vul_NS=_states.vul_NS[_i],  # type:ignore
                vul_EW=_states.vul_EW[_i],  # type:ignore
            )
        elif _states.env_id in ("go-9x9", "go-19x19"):
            from pgx._src.dwg.go import GoState

            return GoState(  # type:ignore
                size=_states.size[_i],  # type:ignore
                chain_id_board=_states.chain_id_board[_i],  # type:ignore
                turn=_states.turn[_i],  # type:ignore
            )
        elif _states.env_id == "hex":
            from pgx._src.dwg.hex import HexState

            return HexState(
                size=_states.size[_i],  # type:ignore
                turn=_states.turn[_i],  # type:ignore
                board=_states.board[_i],  # type:ignore
            )
        elif _states.env_id == "kuhn_poker":
            from pgx._src.dwg.kuhn_poker import KuhnPokerState

            return KuhnPokerState(
                cards=_states.cards[_i], pot=_states.pot[_i]  # type:ignore
            )
        elif _states.env_id == "leduc_holdem":
            from pgx._src.dwg.leduc_holdem import LeducHoldemState

            return LeducHoldemState(
                cards=_states.cards[_i],  # type:ignore
                chips=_states.chips[_i],  # type:ignore
                round=_states.round[_i],  # type:ignore
            )
        elif _states.env_id == "othello":
            from pgx._src.dwg.othello import OthelloState

            return OthelloState(
                turn=_states.turn[_i],  # type:ignore
                board=_states.board[_i],  # type:ignore
            )
        elif _states.env_id == "2048":
            from pgx._src.dwg.play2048 import Play2048State

            return Play2048State(
                board=_states.board[_i],  # type:ignore
            )
        elif _states.env_id == "shogi":
            from pgx._src.dwg.shogi import ShogiState

            return ShogiState(  # type:ignore
                turn=_states.turn[_i],  # type:ignore
                piece_board=_states.piece_board[_i],  # type:ignore
                hand=_states.hand[_i],  # type:ignore
            )
        elif _states.env_id == "sparrow_mahjong":
            from pgx._src.dwg.sparrow_mahjong import SparrowMahjongState

            return SparrowMahjongState(
                current_player=_states.current_player[_i],  # type:ignore
                turn=_states.turn[_i],  # type:ignore
                rivers=_states.rivers[_i],  # type:ignore
                hands=_states.hands[_i],  # type:ignore
                n_red_in_hands=_states.n_red_in_hands[_i],  # type:ignore
                is_red_in_river=_states.is_red_in_river[_i],  # type:ignore
                wall=_states.wall[_i],  # type:ignore
                draw_ix=_states.draw_ix[_i],  # type:ignore
                shuffled_players=_states.shuffled_players[_i],  # type:ignore
                dora=_states.dora[_i],  # type:ignore
            )
        elif _states.env_id == "tic_tac_toe":
            from pgx._src.dwg.tictactoe import TictactoeState

            return TictactoeState(
                current_player=_states.current_player[_i],  # type:ignore
                legal_action_mask=_states.legal_action_mask[_i],  # type:ignore
                terminated=_states.terminated[_i],  # type:ignore
                turn=_states.turn[_i],  # type:ignore
                board=_states.board[_i],  # type:ignore
            )
        else:
            assert False


def save_svg(
    states: State,
    filename: Union[str, Path],
    *,
    color_theme: Optional[Literal["light", "dark"]] = None,
    scale: Optional[float] = None,
) -> None:
    assert str(filename).endswith(".svg")
    v = Visualizer(color_theme=color_theme, scale=scale)
    v.get_dwg(states=states).saveas(filename)


def save_svg_animation(
    states: Sequence[State],
    filename: Union[str, Path],
    *,
    color_theme: Optional[Literal["light", "dark"]] = None,
    scale: Optional[float] = None,
    frame_duration_seconds: Optional[float] = None,
) -> None:
    assert str(filename).endswith(".svg")
    v = Visualizer(color_theme=color_theme, scale=scale)

    if frame_duration_seconds is None:
        frame_duration_seconds = global_config.frame_duration_seconds

    frame_groups = []
    dwg = None
    for i, state in enumerate(states):
        dwg = v.get_dwg(states=state)
        assert (
            len(
                [
                    e
                    for e in dwg.elements
                    if type(e) == svgwrite.container.Group
                ]
            )
            == 1
        ), "Drawing must contain only one group"
        group: svgwrite.container.Group = dwg.elements[-1]
        group["id"] = f"_fr{i:x}"  # hex frame number
        group["class"] = "frame"
        frame_groups.append(group)

    assert dwg is not None
    del dwg.elements[-1]
    total_seconds = frame_duration_seconds * len(frame_groups)

    style = f".frame{{visibility:hidden; animation:{total_seconds}s linear _k infinite;}}"
    style += f"@keyframes _k{{0%,{100/len(frame_groups)}%{{visibility:visible}}{100/len(frame_groups) * 1.000001}%,100%{{visibility:hidden}}}}"

    for i, group in enumerate(frame_groups):
        dwg.add(group)
        style += (
            f"#{group['id']}{{animation-delay:{i * frame_duration_seconds}s}}"
        )
    dwg.defs.add(svgwrite.container.Style(content=style))
    dwg.saveas(filename)
