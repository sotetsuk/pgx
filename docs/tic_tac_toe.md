# Tic-tac-toe

=== "dark"

    <p align="center">
    <img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/tic_tac_toe_dark.gif" width="30%">
    </p>

=== "light"

    <p align="center">
    <img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/tic_tac_toe_light.gif" width="30%">
    </p>


## Usage

```py
import pgx

env = pgx.make("tic_tac_toe")
```

or you can directly load `TicTacToe` class

```py
from pgx.tic_tac_toe import TicTacToe

env = TicTacToe("tic_tac_toe")
```

| Name | Value |
|:---|:----:|
| Version | `v0` |
| Number of players | `2` |
| Number of actions | `9` |
| Observation shape | `(3, 3, 2)` |
| Rewards | `{-1, 0, 1}` |

