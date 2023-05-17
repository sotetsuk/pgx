[![ci](https://github.com/sotetsuk/pgx/actions/workflows/ci.yml/badge.svg)](https://github.com/sotetsuk/pgx/actions/workflows/ci.yml)

<div align="center">
<img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/logo.svg" width="40%">
</div>

A collection of GPU/TPU-accelerated parallel game simulators for reinforcement learning (RL)

<div align="center">
<img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/go_dark.gif#gh-dark-mode-only" width="30%"><img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/go_dark.gif#gh-dark-mode-only" width="30%" style="transform:rotate(270deg);"><img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/go_dark.gif#gh-dark-mode-only" width="30%" style="transform:rotate(90deg);">
<img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/go_light.gif#gh-light-mode-only" width="30%"><img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/go_light.gif#gh-light-mode-only" width="30%" style="transform:rotate(270deg);"><img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/go_light.gif#gh-light-mode-only" width="30%" style="transform:rotate(90deg);">
</div>

## Why Pgx?

<!--- 
throughput: https://colab.research.google.com/drive/1gIWHYLKBxE2XKDhAlEYKVecz3WG4czdz#scrollTo=V1QZhRXoGL8K
--->

[Brax](https://github.com/google/brax), a [JAX](https://github.com/google/jax)-native physics engine, provides extremely high-speed parallel simulation for RL in *continuous* state space.
Then, what about RL in *discrete* state spaces like Chess, Shogi, and Go? **Pgx** provides a wide variety of JAX-native game simulators! Highlighted features include:

- ⚡ **Super fast** in parallel execution on accelerators
- 🎲 **Various game support** including **Backgammon**, **Chess**, **Shogi**, and **Go**
- 🖼️ **Beautiful visualization** in SVG format


## Installation

```sh
pip install pgx
```

## Usage


Note that all `step` functions in Pgx environments are **JAX-native.**, i.e., they are all *JIT-able*.

<a href="https://colab.research.google.com/github/sotetsuk/pgx/blob/main/colab/pgx_hello_world.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

```py
import jax
import pgx

env = pgx.make("go_19x19")
init = jax.jit(jax.vmap(env.init))  # vectorize and JIT-compile
step = jax.jit(jax.vmap(env.step))

batch_size = 1024
keys = jax.random.split(jax.random.PRNGKey(42), batch_size)
state = init(keys)  # vectorized states
while not (state.terminated | state.terminated).all():
    action = model(state.current_player, state.observation, state.legal_action_mask)
    state = step(state, action)  # state.reward (2,)
```

> ⚠️ Pgx is currently in the beta version. Therefore, API is subject to change without notice. We aim to release v1.0.0 in May 2023. Opinions and comments are more than welcome!

<!---
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
--->

## Supported games

| Backgammon | Chess | Shogi | Go |
|:---:|:---:|:---:|:---:|
|<img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/backgammon_dark.gif#gh-dark-mode-only" width="170px"><img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/backgammon_light.gif#gh-light-mode-only" width="170px">|<img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/chess_dark.gif#gh-dark-mode-only" width="158px"><img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/chess_light.gif#gh-light-mode-only" width="158px">|<img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/shogi_dark.gif#gh-dark-mode-only" width="170px"><img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/shogi_light.gif#gh-light-mode-only" width="170px">|<img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/go-19x19_dark.gif#gh-dark-mode-only" width="160px"><img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/go-19x19_light.gif#gh-light-mode-only" width="160px">|


Use `pgx.available_envs() -> Tuple[EnvId]` to see the list of currently available games. Given an `<EnvId>`, you can create the environment via

```py
>>> env = pgx.make(<EnvId>)
```

You can check the current version of each environment by


```py
>>> env.version
```

| Game/EnvId | Visualization | Version | Five-word description |
|:---:|:---:|:---:|:---:|
|<a href="https://en.wikipedia.org/wiki/2048_(video_game)">2048</a> <br> `"2048"` |<img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/2048_dark.gif" width="60px"><img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/2048_light.gif" width="60px">| `beta` | *Merge tiles to create 2048.* |
|<a href="https://en.wikipedia.org/wiki/D%C5%8Dbutsu_sh%C5%8Dgi">Animal Shogi</a><br>`"animal_shogi"` |<img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/animal_shogi_dark.gif" width="60px"><img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/animal_shogi_light.gif" width="60px">|  `beta` | *Animal-themed child-friendly shogi.* |
|<a href="https://en.wikipedia.org/wiki/Backgammon">Backgammon</a><br>`"backgammon"` |<img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/backgammon_dark.gif" width="60px"><img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/backgammon_light.gif" width="60px">| `beta` | *Luck aids bearing off checkers.* |
|<a href="https://en.wikipedia.org/wiki/Chess">Chess</a><br>`"chess"` |<img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/chess_dark.gif" width="60px"><img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/chess_light.gif" width="60px">| `beta` | *Checkmate opponent's king to win.* |
|<a href="https://en.wikipedia.org/wiki/Connect_Four">Connect Four</a><br>`"connect_four"` |<img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/connect_four_dark.gif" width="60px"><img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/connect_four_light.gif" width="60px">| `beta` | *Connect discs, win with four.* |
|<a href="https://en.wikipedia.org/wiki/Go_(game)">Go</a><br>`"go_9x9"` `"go_19x19"` |<img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/go-19x19_dark.gif" width="60px"><img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/go-19x19_light.gif" width="60px">| `beta` | *Strategically place stones, claim territory.* |
|<a href="https://en.wikipedia.org/wiki/Hex_(board_game)">Hex</a><br>`"hex"` |<img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/hex_dark.gif" width="60px"><img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/hex_light.gif" width="60px">| `beta` | *Connect opposite sides, block opponent.* |
|<a href="https://en.wikipedia.org/wiki/Kuhn_poker">Kuhn Poker</a><br>`"kuhn_poker"` |<img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/kuhn_poker_dark.gif" width="60px"><img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/kuhn_poker_light.gif" width="60px">| `beta` | *Three-card betting and bluffing game.* |
|<a href="https://arxiv.org/abs/1207.1411">Leduc hold'em</a><br>`"leduc_holdem"` |<img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/leduc_holdem_dark.gif" width="60px"><img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/leduc_holdem_light.gif" width="60px">| `beta` | *Two-suit, limited deck poker.* |
|<a href="https://github.com/kenjyoung/MinAtar">MinAtar/Asterix</a><br>`"minatar-asterix"` |<img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/minatar-asterix.gif" width="50px">| `beta` | *Avoid enemies, collect treasure, survive.* |
|<a href="https://github.com/kenjyoung/MinAtar">MinAtar/Breakout</a><br>`"minatar-breakout"` |<img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/minatar-breakout.gif" width="50px">| `beta` | *Paddle, ball, bricks, bounce, clear.* |
|<a href="https://github.com/kenjyoung/MinAtar">MinAtar/Freeway</a><br>`"minatar-freeway"` |<img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/minatar-freeway.gif" width="50px">| `beta` | *Dodging cars, climbing up freeway.* |
|<a href="https://github.com/kenjyoung/MinAtar">MinAtar/Seaquest</a><br>`"minatar-seaquest"` |<img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/minatar-seaquest.gif" width="50px">| `beta` | *Underwater submarine rescue and combat.* |
|<a href="https://github.com/kenjyoung/MinAtar">MinAtar/SpaceInvaders</a><br>`"minatar-space_invaders"` |<img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/minatar-space_invaders.gif" width="50px">| `beta` | *Alien shooter game, dodge bullets.* |
|<a href="https://en.wikipedia.org/wiki/Reversi">Othello</a><br>`"othello"` |<img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/othello_dark.gif" width="60px"><img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/othello_light.gif" width="60px">| `beta` | *Flip and conquer opponent's pieces.* |
|<a href="https://en.wikipedia.org/wiki/Shogi">Shogi</a><br>`"shogi"` |<img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/shogi_dark.gif" width="60px"><img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/shogi_light.gif" width="60px"> | `beta` | *Japanese chess with captured pieces.* |
|<a href="https://sugorokuya.jp/p/suzume-jong">Sparrow Mahjong</a><br>`"sparrow_mahjong"` |<img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/sparrow_mahjong_dark.svg" width="60px"><img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/sparrow_mahjong_light.svg" width="60px">|  `beta` | *A simplified, children-friendly Mahjong.* |
|<a href="https://en.wikipedia.org/wiki/Tic-tac-toe">Tic-tac-toe</a><br>`"tic_tac_toe"` |<img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/tic_tac_toe_dark.gif" width="60px"><img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/tic_tac_toe_light.gif" width="60px">| `beta` | *Three in a row wins.* |

- <a href="https://en.wikipedia.org/wiki/Contract_bridge">Bridge Bidding</a> and <a href="https://en.wikipedia.org/wiki/Japanese_mahjong">Mahjong</a> environments are under development 🚧
- Five-word descriptions were generated by [ChatGPT](https://chat.openai.com/) 🤖


## See also

Pgx is intended to complement these **JAX-native environments** with (classic) board game suits:

- [RobertTLange/gymnax](https://github.com/RobertTLange/gymnax): JAX implementation of popular RL environments ([classic control](https://gymnasium.farama.org/environments/classic_control), [bsuite](https://github.com/deepmind/bsuite), MinAtar, etc) and meta RL tasks
- [google/brax](https://github.com/google/brax): Rigidbody physics simulation in JAX and continuous-space RL tasks (ant, fetch, humanoid, etc)
- [instadeepai/jumanji](https://github.com/instadeepai/jumanji): A suite of diverse and challenging
    RL environments in JAX (bin-packing, routing problems, etc)

Combining Pgx with these **JAX-native algorithms/implementations** might be an interesting direction:

- [Anakin framework](https://arxiv.org/abs/2104.06272): Highly efficient RL framework that works with JAX-native environments on TPUs
- [deepmind/mctx](https://github.com/deepmind/mctx): JAX-native MCTS implementations, including AlphaZero and MuZero
- [deepmind/rlax](https://github.com/deepmind/rlax): JAX-native RL components
- [google/evojax](https://github.com/google/evojax): Hardware-Accelerated neuroevolution
- [RobertTLange/evosax](https://github.com/RobertTLange/evosax): JAX-native evolution strategy (ES) implementations
- [adaptive-intelligent-robotics/QDax](https://github.com/adaptive-intelligent-robotics/QDax): JAX-native Quality-Diversity (QD) algorithms

## Citation

```
@article{koyamada2023pgx,
  title={Pgx: Hardware-accelerated parallel game simulation for reinforcement learning},
  author={Koyamada, Sotetsu and Okano, Shinri and Nishimori, Soichiro and Murata, Yu and Habara, Keigo and Kita, Haruka and Ishii, Shin},
  journal={arXiv preprint arXiv:2303.17503},
  year={2023}
}
```

## LICENSE

Apache-2.0
