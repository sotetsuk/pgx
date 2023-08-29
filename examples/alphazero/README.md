# AlphaZero example

A simple (Gumbel) AlphaZero example using [Mctx](https://github.com/deepmind/mctx) library.
For more details, please see [our paper](https://arxiv.org/abs/2303.17503).

## Usage

Note that you need to install `jax` and `jaxlib` in addition to the packages written in `requirements.txt` according to your execution environment.

```sh
$ pip install -r requirements.txt
$ python3 train.py env_id=go_9x9 selfplay_batch_size=1024 seed=0
```%

## Reference

- [Silver+18]() "A general reinforcement learning algorithm that masters
chess, shogi, and go through self-play"
- [Danihelka+22](https://openreview.net/forum?id=bERaNdoegnO) "Policy improvement by planning with Gumbel"
