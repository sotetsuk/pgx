import svgwrite  # type: ignore
from svgwrite import cm

from .go import GoState, get_board

BOARD_SIZE = 5


class Visualizer:
    def __init__(self, state: GoState) -> None:
        self.state = state

    def _repr_html_(self) -> None:
        assert self.state is not None
        return self._to_svg_string()

    def _to_dwg(self):
        state = self.state

        # grid
        grid_size = 1
        dwg = svgwrite.Drawing(
            "temp.svg",
            (
                BOARD_SIZE * grid_size * cm,
                BOARD_SIZE * grid_size * cm,
            ),
        )
        # background
        dwg.add(
            dwg.rect(
                (0, 0),
                (BOARD_SIZE * grid_size * cm, BOARD_SIZE * grid_size * cm),
                # stroke=svgwrite.rgb(10, 10, 16, "%"),
                fill="white",
            )
        )

        board_g = dwg.g()
        hlines = board_g.add(dwg.g(id="hlines", stroke="black"))
        for y in range(BOARD_SIZE):
            hlines.add(
                dwg.line(
                    start=(0 * cm, grid_size * y * cm),
                    end=(
                        grid_size * (BOARD_SIZE - 1) * cm,
                        grid_size * y * cm,
                    ),
                )
            )
        vlines = board_g.add(dwg.g(id="vline", stroke="black"))
        for x in range(BOARD_SIZE):
            vlines.add(
                dwg.line(
                    start=(grid_size * x * cm, 0 * cm),
                    end=(
                        grid_size * x * cm,
                        grid_size * (BOARD_SIZE - 1) * cm,
                    ),
                )
            )

        # stones
        board = get_board(state)
        for xy, stone in enumerate(board):
            if stone == 2:
                continue
            # ndarrayのx,yと違うことに注意
            y = xy // BOARD_SIZE * grid_size
            x = xy % BOARD_SIZE * grid_size

            color = "black" if stone == 0 else "white"
            board_g.add(
                dwg.circle(
                    center=(x * cm, y * cm),
                    r=grid_size / 2.2 * cm,
                    stroke=svgwrite.rgb(10, 10, 16, "%"),
                    fill=color,
                )
            )
        board_g.translate(grid_size * 20, grid_size * 20)  # no units allowed
        dwg.add(board_g)

        return dwg

    def _to_svg_string(self):
        return self._to_dwg().tostring()

    def save_svg(self, filename="temp.svg"):
        assert self.state is not None
        assert filename.endswith(".svg")
        self._to_dwg().saveas(filename=filename)
