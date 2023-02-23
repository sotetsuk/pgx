[![ci](https://github.com/sotetsuk/pgx/actions/workflows/ci.yml/badge.svg)](https://github.com/sotetsuk/pgx/actions/workflows/ci.yml)

# Pgx

A collection of GPU/TPU-accelerated game simulators for reinforcement learning.

## APIs
Pgx's basic API consists of *pure functions* following the JAX's design principle.
This is to explicitly let users know that state transition is determined ONLY from `state` and `action` and to make it easy to use `jax.jit`.
Pgx defines the games as AEC games (see PettingZoo paper), in which only one agent acts and then turn changes.


### Design goal
1. Be explicit
2. Be simple than be universal


### Usage

```py
import jax
import pgx

num_batch = 100

init, step, observe, info = pgx.make(env_id="Go-5x5",)
init = jax.jit(jax.vmap(init))
step = jax.jit(jax.vmap(step))
observe = jax.jit(jax.vmap(observe))

models = {0: ..., 1: ...}

rng = jax.random.PRGNKey(999)
keys = jax.random.split(rng, num_batch)

state = init(keys)
total_reward = jnp.zeros(batch_size, dtype=jnp.float32)
while not (state.terminated).all():
    observations = [observe(state, player_id) for player_id in (0, 1)]
    action = jnp.where(
        state.curr_player == 0,
        models[0](observations[0]),
        models[1](observations[1]),
    )
    state = step(obs, action)
    total_reward += reward
```

### API Description

```py
# N: num agents
# A: action space size
# M: observation dim
@dataclass
class State:
    rng: jax.random.KeyArray  # necessary for autoreset
    curr_player: jnp.ndarray
    # 0 ~ N-1. Different from turn (e.g., white/black in Chess) 
    # Behavior is undefined when terminated (set -1 is inconvenient in batch situation)
    reward: jnp.ndarray
    terminated: jnp.ndarray
    legal_action_mask: jnp.ndarray
  

def init(rng: jnp.ndarray) -> State:
  return state 

# step is deterministic by default
# if state.terminated is True, state.reward is set to zero and the other fields are unchanged
def step(state: State, 
         action: jnp.ndarray)
    -> State:
  return state  # rewards: (N,) 

def observe(state: State, 
            player_id: jnp.ndarray) 
    -> jnp.ndarray:
  # Zero array if state.curr_player is -1
  return obs 

# replace state.rng or shuffle hidden states (e.g., unopened public cards)
def shuffle(state: State, rng: Optional[jnp.ndarray]) 
    -> State:
   return state
```

### Limitations (for the simplicity)
* Does **NOT** support agent death and creation, which dynmically changes the array size. It does not well suit to GPU-accelerated computation.
* Does **NOT** support Chance player (Nature player) with action selection.
* Does **NOT** support OpenAI Gym API.
    * OpenAI Gym is for single-agent environment. Most of Pgx environments are multi-player games. Just defining opponents is not enough for converting multi-agent environemnts to OpenAI Gym environment. E.g., in the game of go, the next state s' is defined as the state just after placing a stone in AlhaGo paper. However, s' becomes the state after the opponents' play. This changes the definition of V(s').
* Does **NOT** support PettingZoo API.
    * PettingZoo is *Gym for multi-agent RL*. As far as we know, PettingZoo does not support vectorized environments (like VectorEnv in OpenAI Gym). As Pgx's main feature is highly vectorized environment via GPU/TPU support, We do not currently support PettingZoo API. 



### `skip_chance`
* We prepare skip_chance=True option for some environments. This makes it possible to consider value function for "post-decision states" (See AlgoRL book). However, we do not allow chance agent to choose action like OpenSpiel. This is because the action space of chance agent and usual agent are different. Thus, when the chance player is chosen (`current_player=-1`), `action=-1` must be returned to step function. Use `shuffle` to make `step` stochastic.

### truncatation and auto_reset
* supported by `make(env_id="...", auto_reset=True, max_episode_length=64)`
* `auto_reset` will replace the terminal state by initial state (but `is_terminal=True` is set)
* `is_truncated=True` is also set to state

### Concerns
* For efficient computation, current_player must be synchronized? but it seems difficult (or impossible?). It is impossible to synchronize the terminations.

## Roadmap

|Game|Logic| Jit                                                                                                                      |Visualization|Speed benchmark|Baseline|
|:---|:---|:-------------------------------------------------------------------------------------------------------------------------|:---|:---|:---|
| Tic-tac-toe | :white_check_mark: | :white_check_mark: ||||
| [Animal Shogi](https://en.wikipedia.org/wiki/D%C5%8Dbutsu_sh%C5%8Dgi) | :white_check_mark: | :white_check_mark:                                                                                                       | :white_check_mark: |||
| [Sparrow Mahjong](https://sugorokuya.jp/p/suzume-jong) |  |                                                                                                        ||||
| [MinAtar](https://github.com/kenjyoung/MinAtar)|-| :white_check_mark: Asterix<br> :white_check_mark: Breakdown<br> :white_check_mark: Freeway<br> :white_check_mark: Seaquest<br> :white_check_mark: SpaceInvaders ||||
|Chess| :white_check_mark: ||:construction:|||
|Shogi| :white_check_mark: || :white_check_mark: |||
|Go| :white_check_mark: | :white_check_mark:                                                                                                       |:white_check_mark: |||
|Backgammon| :construction: ||:construction:|||
|Bridge Bidding| :construction: |||||
|Mahjong| :construction: |||||


## LICENSE

TDOO

* MinAtar is GPL-3.0 License
