"""
TDOO:

* [x] batching
* [x] gamma
* [x] entropy
* [ ] logging
    * [x] eval/R
    * [x] train/env_steps
    * [x] train/grad_steps
    * [x] train/num_unrolls
    * [ ] train/reward_per_step
    * [ ] train/prob
    * [ ] loss
      * [ ] policy_loss
      * [ ] value_loss
      * [ ] ent_loss
* [ ] search num_envs & batch_size (unroll_length=20)
* [ ] wandb
* [ ] search hyper parameters
* [ ] remove env
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

from pgx.envs import MinAtar
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
        return self.policy(x), self.value(x).squeeze()


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
        logits, val = model(obs)
        actions = torch.softmax(logits, dim=1).argmax(dim=1)
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
    model.eval()

    train_data = {}

    rng = jax.random.PRNGKey(seed)
    state = init_state
    obs = jax_to_torch(observe(state))
    push(train_data, obs=obs)
    for length in range(unroll_length):
        # agent step
        logits, _ = model(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()

        # environment step
        rng, _rngs = split_rng(rng, num_envs)
        state, r, terminated = step(
            state,
            torch_to_jax(action),
            _rngs,
        )
        obs = jax_to_torch(observe(state))
        truncated = jnp.zeros_like(terminated)
        if length == unroll_length - 1:
            truncated = 1 - terminated
        push(
            train_data,
            obs=obs,
            action=action,
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
    # obs: (unroll_length+1, num_envs,  10, 10, 4)
    # others: (unroll_length, num_envs)
    return train_data


def loss_fn(model, batch, gamma, policy_coef, value_coef, ent_coef):
    # assumes td has
    # obs: (unroll_length+1, batch_size, 10, 10, 4)
    # others: (unroll_length, batch_size)
    model.train()
    obs_shape = batch["obs"].size()
    # (batch_size, num_envs, 10, 10, channels) => (-1, 10, 10, channels)
    logits, value = model(batch["obs"][:-1].reshape(-1, 10, 10, obs_shape[-1]))
    with torch.no_grad():
        _, next_value = model(
            batch["obs"][1:].reshape(-1, 10, 10, obs_shape[-1])
        )
    dist = Categorical(logits=logits)
    log_probs = dist.log_prob(batch["action"].reshape((-1,)))
    reward = batch["reward"].reshape((-1,))
    next_value[batch["terminated"].bool().reshape((-1,))] = 0.0
    V_tgt = next_value.detach() * gamma + reward
    A = V_tgt - value.detach()
    # all losses are 1d (unroll_length * batch_size) array
    policy_loss = -A * log_probs
    value_loss = ((V_tgt - value) ** 2).sqrt()
    ent_loss = -dist.entropy()
    return (
        policy_coef * policy_loss
        + value_coef * value_loss
        + ent_coef * ent_loss
    ).mean()


@dataclass
class Config:
    game: str = "breakout"
    num_envs: int = 256
    device: str = "cpu"
    seed: int = 0
    sticky_action_prob: float = 0.1
    unroll_length: int = 30
    lr: float = 0.003
    batch_size: int = 32
    gamma: float = 0.99
    policy_coef: float = 1.0
    value_coef: float = 1.0
    ent_coef: float = 0.1


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
    MinAtar(game=args.game, batch_size=args.num_envs, auto_reset=False)
)
model = MinAtarNetwork(
    in_channels=in_channels, num_actions=num_actions, device=args.device
)


optim = optim.Adam(model.parameters(), lr=args.lr)


init, step, observe = load(
    args.game, sticky_action_prob=args.sticky_action_prob
)
rng = jax.random.PRNGKey(args.seed)
rng, *_rngs = jax.random.split(rng, args.num_envs + 1)
init_state = init(rng=jnp.array(_rngs))


train_env_steps = 0
train_num_unrolls = 0
train_opt_steps = 0
train_r_per_step = 0.0
assert args.num_envs % args.batch_size == 0
num_minibatches = args.num_envs // args.batch_size
# for i in tqdm(range(1000)):
for i in range(1000):
    if i % 5 == 0:
        eval_R = eval_rollout(env, model)
        log = {
            "train/env_steps": train_env_steps,
            "train/opt_steps": train_opt_steps,
            "train/r_per_step": train_r_per_step,
            "eval/R": eval_R,
        }
        print(log, flush=True)

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
    # td
    # obs: (unroll_length+1, num_envs,  10, 10, 4)
    # others: (unroll_length, num_envs)
    td_batch = {
        k: torch.reshape(
            v,
            [v.shape[0], num_minibatches, args.batch_size] + list(v.shape[2:]),
        ).transpose(0, 1)
        for k, v in td.items()
    }
    # td batch
    # obs: (num_minibatches, unroll_length+1, batch_size, 10, 10, 4)
    # others: (num_minibatches, unroll_length, batch_size)

    optim.zero_grad()
    for i_batch in range(num_minibatches):
        loss = loss_fn(
            model,
            {k: v[i_batch] for k, v in td_batch.items()},
            gamma=args.gamma,
            policy_coef=args.policy_coef,
            value_coef=args.value_coef,
            ent_coef=args.ent_coef,
        )
        loss.backward()
    optim.step()

    # update stats
    train_env_steps += args.unroll_length * args.num_envs
    train_num_unrolls += args.num_envs
    train_opt_steps += 1
    train_r_per_step = 0.99 * train_r_per_step + 0.01 * float(
        td["reward"].mean()
    )


# # Brax PPO メモ
# # num_envs: int = 2048,  # rolloutしたデータ（unroll_length * num_envs * batch_size）がメモリに載る限り大きくする
# # episode_length: int = 1000,
# # num_timesteps: int = 30_000_000,
# # eval_frequency: int = 10,
# # unroll_length: int = 5, 性能に依存するので最初に決める
# # batch_size: int = 1024, 勾配更新に使うunroll_length * batch_sizeのデータについて、メモリの載る範囲で大きくきめる
# # num_minibatches: int = 32,  # これもおそらくbatch_size決めたあとにメモリに載る範囲で大きめに決める。batch_size,num_minibatchesでnum_rolloutsが決まる。
# # num_update_epochs: int = 4  # PPOのハイパーパラメータ
# for eval_i in range(eval_frequency+1):  # eval_frequency=10
#     for _ in range(num_unrolls):  # num_unrolls(16) = batch_size(1024) * num_minibatches(32) // env.num_env(2048)
#         for _ in range(unroll_length):
#             one_unroll.observation += ...
#         one_unroll = ...
#         # one_unroll.observation: (unroll_length+1=6, num_envs=2048, feature_size=87)
#         # one_unroll.reward: (unroll_length=5, num_envs=2048)
#     td = ...
#     # td.observation (num_unrolls=16, 6, 2048, 87)
#     # td.reward (num_unrolls=16, 5, 2048)
#
#     # num_steps(163840) = batch_size(1024) * num_minibatches(32) * unroll_length(5)
#     for _ in range(num_epochs):  # num_epochs = num_timesteps(=30_000_000) // (num_steps(=163840) * eval_frequency(=10))
#         observation, td = train_unroll(agent, env, observation, num_unrolls, unroll_length)
#         td = sd_map(unroll_first, td)
#         # td.observation (6, 32768=2048*16=1024*32, 87)
#         # td.reward (5, 32768=2048*16=1024*32)
#
#         for _ in range(num_update_epochs):  # num_update_epochs=4
#         # shuffle and batch the data
#           with torch.no_grad():
#               epoch_td = sd_map(shuffle_batch, td)
#               # epoch_td.observation (num_minibatches=32, 6, batch_size=1024, 87)
#               # epoch_td.reward (num_minibatches=32, 5, batch_zie=1024)
#           for minibatch_i in range(num_minibatches):
#               td_minibatch = sd_map(lambda d: d[minibatch_i], epoch_td)
#               # td_minibatch.observation (6, 1024, 87)
#               # td_minibatch.reward (5, 1024)
#               loss = agent.loss(td_minibatch._asdict())
#               optimizer.zero_grad()
#               loss.backward()
#               optimizer.step()
#               total_loss += loss.detach()
