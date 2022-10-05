[![ci](https://github.com/sotetsuk/pgx/actions/workflows/ci.yml/badge.svg)](https://github.com/sotetsuk/pgx/actions/workflows/ci.yml)

# Pgx

Parallel game simulator for reinforcement learning.

## API

```py

@dataclass
class State:
  i: np.ndarray = jnp.zeros(1)
  board: np.ndarray = jnp.array((10, 10))


@jax.jit
def init(rng: jnp.ndarray, **kwargs) -> State:
  return State()

@jax.jit
def step(state: State, action: jnp.ndarray, rng: jnp.ndarray, **kwargs) -> Tuple[State, float, bool]:
  return State(), r, terminated

@jax.jit
def observe(state: State) -> jnp.ndarray:
  return jnp.ones(...)

```

## Roadmap

|Game|Logic| Jit                                                                                |Baseline|Visualization|Gym/PettingZoo|
|:---|:---|:-----------------------------------------------------------------------------------|:---|:---|:---|
|TicTacToe||||||
|AnimalShogi| :white_check_mark: | :white_check_mark:                                                                 ||||
|MiniMahjong| :white_check_mark: | :white_check_mark:                                                                 ||||
|MinAtar <br>[kenjyoung/MinAtar](https://github.com/kenjyoung/MinAtar)|-| :white_check_mark: Asterix<br> Breakdown<br> :white_check_mark: Freeway<br>Seaquest<br>SpaceInvaders ||||
|Chess||||||
|Shogi| :construction: |||||
|Go| :white_check_mark: | :white_check_mark:                                                                 ||||
|ContractBridgeBidding||||||
|Backgammon||||||
|Mahjong| :construction: |||||

# LICENSE

TDOO

* MinAtar is GPL-3.0 License
