[![ci](https://github.com/sotetsuk/pgx/actions/workflows/ci.yml/badge.svg)](https://github.com/sotetsuk/pgx/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue)](https://www.python.org/)
[![PyPI version](https://badge.fury.io/py/pgx.svg)](https://pypi.org/project/pgx/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![codecov](https://codecov.io/github/sotetsuk/pgx/graph/badge.svg?token=JNJIQ83JYG)](https://codecov.io/github/sotetsuk/pgx)


<div align="center">
<img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/logo.svg" width="40%">
</div>

A collection of GPU/TPU-accelerated parallel game simulators for reinforcement learning (RL)

<div align="center">
<img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/go_dark.gif#gh-dark-mode-only" width="30%"><img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/go_dark.gif#gh-dark-mode-only" width="30%" style="transform:rotate(270deg);"><img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/go_dark.gif#gh-dark-mode-only" width="30%" style="transform:rotate(90deg);">
<img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/go_light.gif#gh-light-mode-only" width="30%"><img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/go_light.gif#gh-light-mode-only" width="30%" style="transform:rotate(270deg);"><img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/go_light.gif#gh-light-mode-only" width="30%" style="transform:rotate(90deg);">
</div>

🎉 **`v1.0.0`** is released! (2023.6.20)

## Why Pgx?

<!--- 
throughput: https://colab.research.google.com/drive/1gIWHYLKBxE2XKDhAlEYKVecz3WG4czdz#scrollTo=V1QZhRXoGL8K
--->

[Brax](https://github.com/google/brax), a [JAX](https://github.com/google/jax)-native physics engine, provides extremely high-speed parallel simulation for RL in *continuous* state space.
Then, what about RL in *discrete* state spaces like Chess, Shogi, and Go? **Pgx** provides a wide variety of JAX-native game simulators! Highlighted features include:

- ⚡ **Super fast** in parallel execution on accelerators
- 🎲 **Various game support** including **Backgammon**, **Chess**, **Shogi**, and **Go**
- 🖼️ **Beautiful visualization** in SVG format


## Quick start

- [Getting started](https://colab.research.google.com/github/sotetsuk/pgx/blob/main/colab/pgx_hello_world.ipynb)
- [Pgx baseline models](https://colab.research.google.com/github/sotetsuk/pgx/blob/main/colab/baselines.ipynb)
- [PPO example](https://colab.research.google.com/github/sotetsuk/pgx/blob/main/colab/ppo.ipynb)
- [Export to PettingZoo API](https://colab.research.google.com/github/sotetsuk/pgx/blob/main/colab/pgx2pettingzoo.ipynb)

## Usage

The following code snippet shows a simple example of using Pgx.
You can try it out in [this Colab](https://colab.research.google.com/github/sotetsuk/pgx/blob/main/colab/pgx_hello_world.ipynb).
Note that all `step` functions in Pgx environments are **JAX-native.**, i.e., they are all *JIT-able*.
Please refer to the [documentation](https://sotetsuk.github.io/pgx) for more details.

```py
import jax
import pgx

env = pgx.make("go_19x19")
init = jax.jit(jax.vmap(env.init))
step = jax.jit(jax.vmap(env.step))

batch_size = 1024
keys = jax.random.split(jax.random.PRNGKey(42), batch_size)
state = init(keys)  # vectorized states
while not (state.terminated | state.truncated).all():
    action = model(state.current_player, state.observation, state.legal_action_mask)
    state = step(state, action)  # state.reward (2,)
```

Pgx is a library that focuses on faster implementations rather than just the API itself. 
However, the API itself is also sufficiently general. For example, all environments in Pgx can be converted to the AEC API of [PettingZoo](https://github.com/Farama-Foundation/PettingZoo), and you can run Pgx environments through the PettingZoo API.
You can see the demonstration in [this Colab](https://colab.research.google.com/github/sotetsuk/pgx/blob/main/colab/pgx2pettingzoo.ipynb).


## Installation

```sh
pip install pgx
```

Note that the [MinAtar](https://github.com/kenjyoung/MinAtar) suite is provided as a separate extension for Pgx ([`pgx-minatar`](https://github.com/sotetsuk/pgx-minatar)). Therefore, please run the following command additionaly to use the MinAtar suite in Pgx:

```sh
pip install pgx-minatar
```

Pgx is provided under the Apache 2.0 License, but the original MinAtar suite follows the GPL 3.0 License. Therefore, please note that the separated MinAtar extension for Pgx also adheres to the GPL 3.0 License.

## Supported games

| Backgammon | Chess | Shogi | Go |
|:---:|:---:|:---:|:---:|
|<img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/backgammon_dark.gif#gh-dark-mode-only" width="170px"><img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/backgammon_light.gif#gh-light-mode-only" width="170px">|<img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/chess_dark.gif#gh-dark-mode-only" width="158px"><img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/chess_light.gif#gh-light-mode-only" width="158px">|<img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/shogi_dark.gif#gh-dark-mode-only" width="170px"><img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/shogi_light.gif#gh-light-mode-only" width="170px">|<img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/go-19x19_dark.gif#gh-dark-mode-only" width="160px"><img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/go-19x19_light.gif#gh-light-mode-only" width="160px">|


Use `pgx.available_envs() -> Tuple[EnvId]` to see the list of currently available games. Given an `<EnvId>`, you can create the environment via

```py
>>> env = pgx.make(<EnvId>)
```

| Game/EnvId | Visualization | Version | Five-word description |
|:---:|:---:|:---:|:---:|
|<a href="https://en.wikipedia.org/wiki/2048_(video_game)">2048</a> <br> `"2048"` |<img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/2048_dark.gif" width="60px"><img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/2048_light.gif" width="60px">| `v0` | *Merge tiles to create 2048.* |
|<a href="https://en.wikipedia.org/wiki/D%C5%8Dbutsu_sh%C5%8Dgi">Animal Shogi</a><br>`"animal_shogi"` |<img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/animal_shogi_dark.gif" width="60px"><img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/animal_shogi_light.gif" width="60px">|  `v0` | *Animal-themed child-friendly shogi.* |
|<a href="https://en.wikipedia.org/wiki/Backgammon">Backgammon</a><br>`"backgammon"` |<img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/backgammon_dark.gif" width="60px"><img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/backgammon_light.gif" width="60px">| `v1` | *Luck aids bearing off checkers.* |
|<a href="https://en.wikipedia.org/wiki/Contract_bridge">Bridge bidding</a><br>`"bridge_bidding"` |<img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/bridge_bidding_dark.gif" width="60px"><img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/bridge_bidding_light.gif" width="60px">| `v0` | *Partners exchange information via bids.* |
|<a href="https://en.wikipedia.org/wiki/Chess">Chess</a><br>`"chess"` |<img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/chess_dark.gif" width="60px"><img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/chess_light.gif" width="60px">| `v1` | *Checkmate opponent's king to win.* |
|<a href="https://en.wikipedia.org/wiki/Connect_Four">Connect Four</a><br>`"connect_four"` |<img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/connect_four_dark.gif" width="60px"><img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/connect_four_light.gif" width="60px">| `v0` | *Connect discs, win with four.* |
|<a href="https://en.wikipedia.org/wiki/Minichess">Gardner Chess</a><br>`"gardner_chess"`|<img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/gardner_chess_dark.gif" width="60px"><img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/gardner_chess_light.gif" width="60px">| `v0` | *5x5 chess variant, excluding castling.* |
|<a href="https://en.wikipedia.org/wiki/Go_(game)">Go</a><br>`"go_9x9"` `"go_19x19"` |<img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/go-19x19_dark.gif" width="60px"><img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/go-19x19_light.gif" width="60px">| `v0` | *Strategically place stones, claim territory.* |
|<a href="https://en.wikipedia.org/wiki/Hex_(board_game)">Hex</a><br>`"hex"` |<img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/hex_dark.gif" width="60px"><img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/hex_light.gif" width="60px">| `v0` | *Connect opposite sides, block opponent.* |
|<a href="https://en.wikipedia.org/wiki/Kuhn_poker">Kuhn Poker</a><br>`"kuhn_poker"` |<img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/kuhn_poker_dark.gif" width="60px"><img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/kuhn_poker_light.gif" width="60px">| `v0` | *Three-card betting and bluffing game.* |
|<a href="https://arxiv.org/abs/1207.1411">Leduc hold'em</a><br>`"leduc_holdem"` |<img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/leduc_holdem_dark.gif" width="60px"><img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/leduc_holdem_light.gif" width="60px">| `v0` | *Two-suit, limited deck poker.* |
|<a href="https://github.com/kenjyoung/MinAtar">MinAtar/Asterix</a><br>`"minatar-asterix"` |<img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/minatar-asterix.gif" width="50px">| `v0` | *Avoid enemies, collect treasure, survive.* |
|<a href="https://github.com/kenjyoung/MinAtar">MinAtar/Breakout</a><br>`"minatar-breakout"` |<img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/minatar-breakout.gif" width="50px">| `v0` | *Paddle, ball, bricks, bounce, clear.* |
|<a href="https://github.com/kenjyoung/MinAtar">MinAtar/Freeway</a><br>`"minatar-freeway"` |<img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/minatar-freeway.gif" width="50px">| `v0` | *Dodging cars, climbing up freeway.* |
|<a href="https://github.com/kenjyoung/MinAtar">MinAtar/Seaquest</a><br>`"minatar-seaquest"` |<img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/minatar-seaquest.gif" width="50px">| `v0` | *Underwater submarine rescue and combat.* |
|<a href="https://github.com/kenjyoung/MinAtar">MinAtar/SpaceInvaders</a><br>`"minatar-space_invaders"` |<img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/minatar-space_invaders.gif" width="50px">| `v0` | *Alien shooter game, dodge bullets.* |
|<a href="https://en.wikipedia.org/wiki/Reversi">Othello</a><br>`"othello"` |<img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/othello_dark.gif" width="60px"><img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/othello_light.gif" width="60px">| `v0` | *Flip and conquer opponent's pieces.* |
|<a href="https://en.wikipedia.org/wiki/Shogi">Shogi</a><br>`"shogi"` |<img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/shogi_dark.gif" width="60px"><img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/shogi_light.gif" width="60px"> | `v0` | *Japanese chess with captured pieces.* |
|<a href="https://sugorokuya.jp/p/suzume-jong">Sparrow Mahjong</a><br>`"sparrow_mahjong"` |<img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/sparrow_mahjong_dark.svg" width="60px"><img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/sparrow_mahjong_light.svg" width="60px">|  `v1` | *A simplified, children-friendly Mahjong.* |
|<a href="https://en.wikipedia.org/wiki/Tic-tac-toe">Tic-tac-toe</a><br>`"tic_tac_toe"` |<img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/tic_tac_toe_dark.gif" width="60px"><img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/tic_tac_toe_light.gif" width="60px">| `v0` | *Three in a row wins.* |

- <a href="https://en.wikipedia.org/wiki/Japanese_mahjong">Mahjong</a> environments are under development 🚧 If you have any requests for new environments, please let us know by [opening an issue](https://github.com/sotetsuk/pgx/issues/new)
- Five-word descriptions were generated by [ChatGPT](https://chat.openai.com/) 🤖

### Versioning policy

Each environment is versioned, and the version is incremented when there are changes that affect the performance of agents or when there are changes that are not backward compatible with the API.
If you want to pursue complete reproducibility, we recommend that you check the version of Pgx and each environment as follows:

```py
>>> pgx.__version__
'1.0.0'
>>> env.version
'v0'
```

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
- [luchris429/purejaxrl](https://github.com/luchris429/purejaxrl): Jax-native RL implementations

## Citation

If you use Pgx in your work, please cite the following paper:

```
@article{koyamada2023pgx,
  title={Pgx: Hardware-accelerated Parallel Game Simulators for Reinforcement Learning},
  author={Koyamada, Sotetsu and Okano, Shinri and Nishimori, Soichiro and Murata, Yu and Habara, Keigo and Kita, Haruka and Ishii, Shin},
  journal={arXiv preprint arXiv:2303.17503},
  year={2023}
}
```

## LICENSE

Apache-2.0
