import svgwrite  # type: ignore
from svgwrite import cm

from .go import GoState, get_board


class Visualizer:
    def __init__(self, state: GoState) -> None:
        self.state = state

    def _repr_html_(self) -> None:
        assert self.state is not None
        return self._to_svg_string()

    def _to_dwg(self):
        state = self.state
        BOARD_SIZE = state.size[0]
        GRID_SIZE = 1

        # grid
        dwg = svgwrite.Drawing(
            "temp.svg",
            (
                BOARD_SIZE * GRID_SIZE * cm,
                BOARD_SIZE * GRID_SIZE * cm,
            ),
        )
        # background
        dwg.add(
            dwg.rect(
                (0, 0),
                (BOARD_SIZE * GRID_SIZE * cm, BOARD_SIZE * GRID_SIZE * cm),
                # stroke=svgwrite.rgb(10, 10, 16, "%"),
                fill="white",
            )
        )

        board_g = dwg.g()
        hlines = board_g.add(dwg.g(id="hlines", stroke="black"))
        for y in range(BOARD_SIZE):
            hlines.add(
                dwg.line(
                    start=(0 * cm, GRID_SIZE * y * cm),
                    end=(
                        GRID_SIZE * (BOARD_SIZE - 1) * cm,
                        GRID_SIZE * y * cm,
                    ),
                )
            )
        vlines = board_g.add(dwg.g(id="vline", stroke="black"))
        for x in range(BOARD_SIZE):
            vlines.add(
                dwg.line(
                    start=(GRID_SIZE * x * cm, 0 * cm),
                    end=(
                        GRID_SIZE * x * cm,
                        GRID_SIZE * (BOARD_SIZE - 1) * cm,
                    ),
                )
            )

        # stones
        board = get_board(state)
        for xy, stone in enumerate(board):
            if stone == 2:
                continue
            # ndarrayのx,yと違うことに注意
            y = xy // BOARD_SIZE * GRID_SIZE
            x = xy % BOARD_SIZE * GRID_SIZE

            color = "black" if stone == 0 else "white"
            board_g.add(
                dwg.circle(
                    center=(x * cm, y * cm),
                    r=GRID_SIZE / 2.2 * cm,
                    stroke=svgwrite.rgb(10, 10, 16, "%"),
                    fill=color,
                )
            )
        board_g.translate(GRID_SIZE * 20, GRID_SIZE * 20)  # no units allowed
        dwg.add(board_g)

        return dwg

    def _to_svg_string(self):
        return self._to_dwg().tostring()

    def save_svg(self, filename="temp.svg"):
        assert self.state is not None
        assert filename.endswith(".svg")
        self._to_dwg().saveas(filename=filename)
