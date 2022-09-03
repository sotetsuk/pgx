"""
TDOO:

* [ ] MinAtar v1 => v2

"""


import json
import os
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime as dt
from typing import Dict, List, Optional, Tuple, Union

import argdcls
import gym
import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from jax._src import dlpack as jax_dlpack
from jax.interpreters.xla import DeviceArray
from torch.distributions import Categorical
from torch.utils import dlpack as torch_dlpack
from tqdm import tqdm

from pgx.minatar import asterix, breakout

Device = Union[str, torch.device]


def load(game, sticky_action_prob: float = 0.1):
    from functools import partial

    if game == "breakout":
        init = jax.vmap(breakout.reset)
        step = jax.vmap(
            partial(breakout.step, sticky_action_prob=sticky_action_prob)
        )
        observe = jax.vmap(breakout.to_obs)
    elif game == "asterix":
        init = jax.vmap(asterix.reset)
        step = jax.vmap(
            partial(asterix.step, sticky_action_prob=sticky_action_prob)
        )
        observe = jax.vmap(asterix.to_obs)
    else:
        raise NotImplementedError("This game is not implemented.")

    return init, step, observe


class MinAtar(gym.Env):
    def __init__(
        self,
        game: str,
        num_envs: int = 8,
        auto_reset=True,
        sticky_action_prob: float = 0.1,
    ):
        self.game = game
        self.auto_reset = auto_reset
        self.num_envs = num_envs
        self.sticky_action_prob: jnp.ndarray = (
            jnp.ones(self.num_envs) * sticky_action_prob
        )
        if self.game == "breakout":
            self._reset = jax.vmap(breakout.reset)
            self._step = jax.vmap(breakout.step)
            self._to_obs = jax.vmap(breakout.to_obs)
        elif self.game == "asterix":
            self._reset = jax.vmap(asterix.reset)
            self._step = jax.vmap(asterix.step)
            self._to_obs = jax.vmap(asterix.to_obs)
        else:
            raise NotImplementedError("This game is not implemented.")

        self.rng = jax.random.PRNGKey(0)
        self.rng, _rngs = self._split_keys(self.rng)
        self.state = self._reset(_rngs)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> jnp.ndarray:
        assert seed is not None
        self.rng = jax.random.PRNGKey(seed)
        self.rng, _rngs = self._split_keys(self.rng)
        self.state = self._reset(_rngs)
        assert not return_info  # TODO: fix
        return self._to_obs(self.state)

    def step(
        self, action: jnp.ndarray
    ) -> Tuple[jnp.ndarray, float, bool, dict]:
        self.rng, _rngs = self._split_keys(self.rng)
        self.state, r, done = self._step(
            state=self.state,
            action=action,
            rng=_rngs,
            sticky_action_prob=self.sticky_action_prob,
        )
        if self.auto_reset:

            @jax.vmap
            def where(c, x, y):
                return jax.lax.cond(c, lambda _: x, lambda _: y, 0)

            self.rng, _rngs = self._split_keys(self.rng)
            init_state = self._reset(_rngs)
            self.state = where(done, init_state, self.state)
        return self._to_obs(self.state), r, done, {}

    def _split_keys(self, rng):
        rngs = jax.random.split(rng, self.num_envs + 1)
        rng = rngs[0]
        subrngs = rngs[1:]
        return rng, subrngs


def torch_to_jax(value: torch.Tensor) -> DeviceArray:
    tensor = torch_dlpack.to_dlpack(value)
    tensor = jax_dlpack.from_dlpack(tensor)
    tensor = tensor.astype(jnp.int32)
    return tensor


def jax_to_torch(value: DeviceArray, device: Device = None) -> torch.Tensor:
    dpack = jax_dlpack.to_dlpack(value.astype("float32"))
    # dpack = jax_dlpack.to_dlpack(value)
    tensor = torch_dlpack.from_dlpack(dpack)
    if device:
        return tensor.to(device=device)
    return tensor


class JaxToTorchWrapper(gym.Wrapper):
    """Wrapper that converts Jax tensors to PyTorch tensors."""

    def __init__(self, env, device: Optional[Device] = None):
        """Creates a Wrapper around a `GymWrapper` or `VectorGymWrapper` that outputs PyTorch tensors."""
        super().__init__(env)
        self.device: Optional[Device] = device

    def observation(self, observation):
        return jax_to_torch(observation, device=self.device)

    def action(self, action):
        return torch_to_jax(action)

    def reward(self, reward):
        return jax_to_torch(reward, device=self.device)

    def done(self, done):
        return jax_to_torch(done, device=self.device)

    def info(self, info):
        return info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,  # TODO: fix
        options: Optional[dict] = None,
    ):
        obs = self.env.reset(
            seed=seed, return_info=return_info, options=options
        )
        return self.observation(obs)

    def step(self, action):
        action = self.action(action)
        obs, reward, done, info = self.env.step(action)
        obs = self.observation(obs)
        reward = self.reward(reward)
        done = self.done(done)
        info = self.info(info)
        return obs, reward, done, info


class MinAtarNetwork(nn.Module):
    def __init__(self, in_channels, num_actions, device):
        super(MinAtarNetwork, self).__init__()
        self.in_channels = in_channels
        self.device = device
        self.conv = nn.Conv2d(
            in_channels, 16, kernel_size=3, stride=1, device=self.device
        )

        def size_linear_unit(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        num_linear_units = size_linear_unit(10) * size_linear_unit(10) * 16
        self.fc_hidden = nn.Linear(
            in_features=num_linear_units, out_features=128, device=self.device
        )
        self.policy = nn.Linear(
            in_features=128, out_features=num_actions, device=self.device
        )
        self.value = nn.Linear(
            in_features=128, out_features=1, device=self.device
        )
        self.dSiLU = lambda x: torch.sigmoid(x) * (
            1 + x * (1 - torch.sigmoid(x))
        )
        self.SiLU = lambda x: x * torch.sigmoid(x)

    def forward(self, x):
        # print(x)
        x = torch.permute(x, (0, 3, 1, 2))
        x = self.SiLU(self.conv(x))
        # print(x)
        x = x.reshape(x.size(0), -1)
        # print(x)
        x = self.dSiLU(self.fc_hidden(x))
        return torch.softmax(self.policy(x), dim=1), self.value(x).squeeze()


def eval_rollout(
    env,
    model: nn.Module,
) -> float:
    model.eval()
    import random

    obs = env.reset(seed=random.randint(0, int(2e5)))  # TODO: fix
    num_envs = obs.size(0)
    R = 0
    while True:
        probs, val = model(obs)
        actions = probs.argmax(dim=1)
        obs, r, done, info = env.step(actions)
        R += r.sum()
        if done.all():
            break
    return float(R / num_envs)


def split_rng(rng, num_envs):
    rng, *_rngs = jax.random.split(rng, num_envs + 1)
    return rng, jnp.array(_rngs)


def push(data: Dict[str, List], **kwargs) -> None:
    for k, v in kwargs.items():
        if k not in data:
            data[k] = []
        data[k].append(v)


def train_rollout(
    model: nn.Module,
    init_state,
    init,
    step,
    observe,
    seed: int,
    num_envs: int,
    unroll_length: int,
):
    model.train()

    train_data = {}

    rng = jax.random.PRNGKey(seed)
    state = init_state
    obs = jax_to_torch(observe(state))
    push(train_data, obs=obs)
    R = 0
    for length in range(unroll_length):
        # agent step
        probs, value = model(obs)
        dist = Categorical(probs=probs)
        action = dist.sample()

        # environment step
        rng, _rngs = split_rng(rng, num_envs)
        state, r, terminated = step(
            state,
            torch_to_jax(action),
            _rngs,
        )
        obs = jax_to_torch(observe(state))
        R += r.sum()
        truncated = jnp.zeros_like(terminated)
        if length == unroll_length - 1:
            truncated = 1 - terminated
        push(
            train_data,
            obs=obs,
            probs=probs,
            action=action,
            value=value,
            reward=jax_to_torch(r),
            terminated=jax_to_torch(terminated),
            truncated=jax_to_torch(truncated),
        )

        # auto reset
        @jax.vmap
        def where(c, x, y):
            return jax.lax.cond(c, lambda: x, lambda: y)

        rng, _rngs = split_rng(rng, num_envs)
        init_state = init(_rngs)
        state = where(terminated, init_state, state)

    train_data = {k: torch.stack(v) for k, v in train_data.items()}
    return train_data


@dataclass
class Config:
    game: str = "breakout"
    num_envs: int = 128
    device: str = "cpu"
    seed: int = 0
    sticky_action_prob: float = 0.1
    unroll_length: int = 19
    lr: float = 0.1


args = argdcls.load(Config)

in_channels = {
    "breakout": 4,
    "asterix": 4,
    "seaquest": 10,
    "space_invaders": 6,
    "freeway": 7,
}[
    args.game
]  # TODO: fix
num_actions = 6  # TODO: fix


env = JaxToTorchWrapper(
    MinAtar(game=args.game, num_envs=args.num_envs, auto_reset=False)
)
model = MinAtarNetwork(
    in_channels=in_channels, num_actions=num_actions, device=args.device
)

print(eval_rollout(env, model))

optim = optim.Adam(model.parameters(), lr=args.lr)


init, step, observe = load(
    args.game, sticky_action_prob=args.sticky_action_prob
)
rng = jax.random.PRNGKey(args.seed)
rng, *_rngs = jax.random.split(rng, args.num_envs + 1)
init_state = init(rng=jnp.array(_rngs))

td = train_rollout(
    model,
    init_state=init_state,
    init=init,
    step=step,
    observe=observe,
    seed=args.seed,
    num_envs=args.num_envs,
    unroll_length=args.unroll_length,
)

for k, v in td.items():
    print(k, v.size(), v.type(), v.grad)
