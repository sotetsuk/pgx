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

* [ ] Tic-tac-toe
* [ ] AnimalShogi
* [ ] Go (5x5)
* [ ] MinAtar
  * [ ] Breakout
  * [ ] Asterix
  * [ ] SpaceInvaders
  * [ ] Seaquest
  * [ ] Freeway
* [ ] Shogi
* [ ] Go 19x19
* [ ] Chess


# LICENSE

TDOO

* MinAtar is GPL-3.0 License