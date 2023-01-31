import json
import random
from typing import Dict, List, Literal, Union

import jax.numpy as jnp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from jax import dlpack as jax_dlpack
from omegaconf import OmegaConf
from pydantic import BaseModel
from torch.distributions import Categorical
from torch.utils import dlpack as torch_dlpack

import pgx.gym as gym


def to_torch(value):
    dpack = jax_dlpack.to_dlpack(value.astype("float32"))
    tensor = torch_dlpack.from_dlpack(dpack)
    return tensor


def to_jax(value):
    tensor = torch_dlpack.to_dlpack(value)
    tensor = jax_dlpack.from_dlpack(tensor)
    return tensor


class Config(BaseModel):
    steps: int = int(5e6)
    eval_interval: int = int(1e5)
    eval_deterministic: bool = False
    seed: int = 1234
    num_envs: int = 64
    lr: float = 0.003
    ent_coef: float = 0.0
    gamma: float = 0.99
    value_coef: float = 1.0
    unroll_length: int = 5
    debug: bool = False


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.torso = nn.Sequential(
            nn.Linear(27, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU()
        )

        self.policy = nn.Linear(in_features=128, out_features=9)
        self.value = nn.Linear(in_features=128, out_features=1)
        nn.init.constant_(self.value.bias, 0.0)
        nn.init.constant_(self.value.weight, 0.0)

    def forward(self, x):
        x = to_torch(x)
        x = self.torso(x)
        return self.policy(x), self.value(x)


class A2C:
    def __init__(self, config: Config):
        self.config = config

        self.n_steps: int = 0
        self.n_episodes: int = 0
        self.data: Dict[str, List[torch.Tensor]] = {}
        self.env = None
        self.model = None
        self.opt = None

        # stats
        self.n_stats_update = 0
        self.avg_R = 0.0
        self.avg_ent = 0.0
        self.avg_seq_len = 0.0
        self.avg_prob = 0.0
        self.value = 0.0

        self.observations = None
        self.info = None

    def train(
        self,
        env,
        model: nn.Module,
        opt,
        n_steps_lim: int = 100_000,
    ) -> Dict[str, float]:
        self.env, self.model, self.opt = env, model, opt

        if self.observations is None:
            self.observations, self.info = self.env.reset(
                0
            )  # (num_envs, obs_dim)

        while self.n_steps < n_steps_lim:
            # rollout data
            self.rollout()

            # compute loss and update gradient
            self.opt.zero_grad()
            loss = self.loss()
            loss.backward()
            self.opt.step()

            self.log()

        return {
            "steps": self.n_steps,
            "n_episodes": self.n_episodes,
            "ent": self.avg_ent,
            "prob": self.avg_prob,
            "value": self.value,
            "train_R": self.avg_R,
        }

    def rollout(self) -> None:
        assert self.env is not None and self.model is not None
        self.data = {}
        self.model.train()
        for unroll_ix in range(self.config.unroll_length):
            # print(unroll_ix)
            action, log_prob, entropy, value = self.act(
                self.observations, self.info
            )  # agent step
            # print(action)
            (
                self.observations,
                rewards,
                terminated,
                _,
                self.info,
            ) = self.env.step(
                to_jax(action).astype(jnp.int32)
            )  # env step
            # print(rewards, terminated)
            rewards = to_torch(rewards)
            terminated = to_torch(terminated).bool()
            self.n_steps += self.env.num_envs
            truncated = (
                int(unroll_ix == self.config.unroll_length - 1)
                * (1 - terminated.int())
            ).bool()
            with torch.no_grad():
                _, _, _, next_value = self.act(self.observations, self.info)
            self.push_data(
                terminated=terminated,
                truncated=truncated,
                log_prob=log_prob,
                entropy=entropy,
                value=value,
                next_value=next_value,
                rewards=rewards,
            )

    def loss(self, reduce=True) -> torch.Tensor:
        v = torch.stack(self.data["value"]).t()  # (num_envs, max_seq_len + 1)
        with torch.no_grad():
            v_tgt = self.compute_return()
        # pg loss
        log_prob = torch.stack(self.data["log_prob"]).t()  # (n_env, seq_len)
        b = v.detach()
        loss = -(v_tgt - b) * log_prob
        # value loss
        value_loss = (v_tgt - v) ** 2
        # ent loss
        ent = torch.stack(self.data["entropy"]).t()  # (num_env, max_seq_len)
        ent_loss = -ent

        loss += self.config.ent_coef * ent_loss
        loss += self.config.value_coef * value_loss
        return loss.sum(dim=1).mean(dim=0) if reduce else loss

    def compute_return(self):
        """compute n-step return following A3C paper"""
        rewards = torch.stack(self.data["rewards"]).t()
        next_values = torch.stack(self.data["next_value"]).t()
        truncated = torch.stack(self.data["truncated"]).t()
        terminated = torch.stack(self.data["terminated"]).t()
        done = truncated | terminated
        R = rewards + self.config.gamma * next_values * truncated.float()
        seq_len = R.size(1)
        for i in reversed(range(seq_len - 1)):
            R[:, i] += (
                self.config.gamma * R[:, i + 1] * (1 - done[:, i].float())
            )
        return R

    def act(self, observations, info):
        assert self.model is not None
        legal_action_mask = info["legal_action_mask"].astype(jnp.float16)
        logits, value = self.model(observations)  # (num_envs, action_dim)
        logits += to_torch(jnp.log(legal_action_mask + 1e-5))
        dist = Categorical(logits=logits)
        actions = dist.sample()  # (num_envs)
        log_prob = dist.log_prob(actions)  # (num_envs)
        entropy = dist.entropy()  # (num_envs)
        return actions, log_prob, entropy, value.squeeze()

    def push_data(self, **kwargs) -> None:
        for k, v in kwargs.items():
            assert isinstance(v, torch.Tensor)
            if k not in self.data:
                self.data[k] = []
            self.data[k].append(v)

    def log(self):
        self.n_stats_update += 1

        # logging
        prob = float(torch.exp(torch.stack(self.data["log_prob"])).mean())
        R = float(torch.stack(self.data["rewards"]).sum(dim=0).mean())
        ent = float(torch.stack(self.data["entropy"]).mean())
        v = float(torch.stack(self.data["value"]).mean())

        _avg = lambda x, y, n: (x * n + y * 1) / (n + 1)
        self.avg_R = _avg(self.avg_R, R, self.n_stats_update)
        self.avg_ent = _avg(self.avg_ent, ent, self.n_stats_update)
        self.avg_prob = _avg(self.avg_prob, prob, self.n_stats_update)
        self.value = _avg(self.value, v, self.n_stats_update)


def evaluate(
    env,
    model: nn.Module,
    deterministic: bool = False,
) -> float:
    model.eval()
    obs, info = env.reset(1)
    R = jnp.zeros(env.num_envs)
    while True:
        legal_action_mask = info["legal_action_mask"].astype(jnp.float16)
        logits, _ = model(obs)
        logits += to_torch(jnp.log(legal_action_mask + 1e-5))

        dist = Categorical(logits=logits)
        actions = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
        actions = to_jax(actions).astype(jnp.int32)
        obs, r, terminated, truncated, info = env.step(actions)
        done = terminated | truncated
        R += r  # If some episode is terminated, all r is zero afterwards.
        if all(done):
            break
    return float(R.mean())


args = Config(**OmegaConf.to_object(OmegaConf.from_cli()))
print(args)


# fix seeds
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)


algo = A2C(config=args)
envs = gym.RandomOpponentEnv("tic_tac_toe/v0", args.num_envs, True, False)
model = Network()
opt = optim.Adam(model.parameters(), lr=args.lr)

n_train = 0
log = {"steps": 0, "prob": 1.0 / 9}
while True:
    log["eval_R"] = evaluate(
        gym.RandomOpponentEnv(
            "tic_tac_toe/v0", num_envs=args.num_envs, auto_reset=False
        ),
        model,
        deterministic=args.eval_deterministic,
    )
    print(json.dumps(log))
    if algo.n_steps >= args.steps:
        break
    log = algo.train(
        envs, model, opt, n_steps_lim=(n_train + 1) * args.eval_interval
    )
    n_train += 1
