[![ci](https://github.com/sotetsuk/pgx/actions/workflows/ci.yml/badge.svg)](https://github.com/sotetsuk/pgx/actions/workflows/ci.yml)

# Pgx

Highly parallel game simulator for reinforcement learning.

## Basic API
Pgx's basic API consists of *pure functions* following the JAX's design principle.

```py

@dataclass
class State:
  i: np.ndarray = jnp.zeros(1)
  board: np.ndarray = jnp.array((10, 10))


@jax.jit
def init(rng: jnp.ndarray, **kwargs) -> State:
  return State()

@jax.jit
def step(state: State, action: jnp.ndarray, rng: jnp.ndarray, **kwargs) -> Tuple[State, jnp.ndarray, jnp.ndarray]:
  return State(), r, terminated

@jax.jit
def observe(state: State) -> jnp.ndarray:
  return jnp.ones(...)

```

## Roadmap

|Game|Logic| Jit                                                                                                                      |Baseline|Visualization|Gym/PettingZoo|
|:---|:---|:-------------------------------------------------------------------------------------------------------------------------|:---|:---|:---|
|TicTacToe||||||
|AnimalShogi| :white_check_mark: | :white_check_mark:                                                                                                       ||||
|MiniMahjong| :white_check_mark: | :white_check_mark:                                                                                                       ||||
|MinAtar <br>[kenjyoung/MinAtar](https://github.com/kenjyoung/MinAtar)|-| :white_check_mark: Asterix<br> :white_check_mark: Breakdown<br> :white_check_mark: Freeway<br> :white_check_mark: Seaquest<br> :white_check_mark: SpaceInvaders ||||
|Chess||||||
|Shogi| :construction: |||||
|Go| :white_check_mark: | :white_check_mark:                                                                                                       ||||
|ContractBridgeBidding||||||
|Backgammon||||||
|Mahjong| :construction: |||||

# LICENSE

TDOO

* MinAtar is GPL-3.0 License
