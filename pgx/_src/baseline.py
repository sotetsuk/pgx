import os
import pickle
from typing import Literal

import jax
import jax.numpy as jnp

from pgx._src.utils import _download

BaselineModelId = Literal[
    "animal_shogi_v0",
    "gardner_chess_v0",
    "go_9x9_v0",
    "hex_v0",
    "othello_v0",
    "minatar-asterix_v0",
    "minatar-breakout_v0",
    "minatar-freeway_v0",
    "minatar-seaquest_v0",
    "minatar-space_invaders_v0",
]


def make_baseline_model(model_id: BaselineModelId, download_dir: str = "baselines"):
    if model_id in (
        "animal_shogi_v0",
        "gardner_chess_v0",
        "go_9x9_v0",
        "hex_v0",
        "othello_v0",
    ):
        return _make_az_baseline_model(model_id, download_dir)
    elif model_id in (
        "minatar-asterix_v0",
        "minatar-breakout_v0",
        "minatar-freeway_v0",
        "minatar-seaquest_v0",
        "minatar-space_invaders_v0",
    ):
        return _make_minatar_baseline_model(model_id, download_dir)
    else:
        assert False


def _make_az_baseline_model(model_id: BaselineModelId, download_dir: str = "baselines"):
    import haiku as hk

    model_args, model_params, model_state = _load_baseline_model(model_id, download_dir)

    def forward_fn(x, is_eval=False):
        net = _create_az_model_v0(**model_args)
        policy_out, value_out = net(x, is_training=not is_eval, test_local_stats=False)
        return policy_out, value_out

    forward = hk.without_apply_rng(hk.transform_with_state(forward_fn))

    def apply(obs):
        (logits, value), _ = forward.apply(model_params, model_state, obs, is_eval=True)
        return logits, value

    return apply


def _make_minatar_baseline_model(model_id: BaselineModelId, download_dir: str = "baselines"):
    import haiku as hk

    model_args, model_params, model_state = _load_baseline_model(model_id, download_dir)
    del model_state

    class ActorCritic(hk.Module):
        def __init__(self, num_actions, activation="tanh"):
            super().__init__()
            self.num_actions = num_actions
            self.activation = activation
            assert activation in ["relu", "tanh"]

        def __call__(self, x):
            x = x.astype(jnp.float32)
            if self.activation == "relu":
                activation = jax.nn.relu
            else:
                activation = jax.nn.tanh
            x = hk.Conv2D(32, kernel_shape=2)(x)
            x = jax.nn.relu(x)
            x = hk.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding="VALID")
            x = x.reshape((x.shape[0], -1))  # flatten
            x = hk.Linear(64)(x)
            x = jax.nn.relu(x)
            actor_mean = hk.Linear(64)(x)
            actor_mean = activation(actor_mean)
            actor_mean = hk.Linear(64)(actor_mean)
            actor_mean = activation(actor_mean)
            actor_mean = hk.Linear(self.num_actions)(actor_mean)

            critic = hk.Linear(64)(x)
            critic = activation(critic)
            critic = hk.Linear(64)(critic)
            critic = activation(critic)
            critic = hk.Linear(1)(critic)

            return actor_mean, jnp.squeeze(critic, axis=-1)

    def forward_fn(x):
        net = ActorCritic(**model_args)
        logits, value = net(x)
        return logits, value

    forward = hk.without_apply_rng(hk.transform(forward_fn))

    def apply(obs):
        logits, value = forward.apply(model_params, obs)
        return logits, value

    return apply


def _load_baseline_model(baseline_model: BaselineModelId, basedir: str = "baselines"):
    os.makedirs(basedir, exist_ok=True)

    # download baseline model if not exists
    filename = os.path.join(basedir, baseline_model + ".ckpt")
    if not os.path.exists(filename):
        url = _get_download_url(baseline_model)
        _download(url, filename)

    with open(filename, "rb") as f:
        d = pickle.load(f)

    return d["args"], d["params"], d["state"]


def _get_download_url(baseline_model: BaselineModelId) -> str:
    urls = {
        "animal_shogi_v0": "https://drive.google.com/uc?id=1HpP5GLf9b6zkJL8FKUFfKS8Zycs-gzZg",
        "gardner_chess_v0": "https://drive.google.com/uc?id=1RUdrxhYseG-FliskVdemNYYM5YYmfwU7",
        "go_9x9_v0": "https://drive.google.com/uc?id=1hXMicBALW3WU43NquDoX4zthY4-KjiVu",
        "hex_v0": "https://drive.google.com/uc?id=11qpLAT4_0NgPrKRcJCPE7RdN92VP8Ws3",
        "othello_v0": "https://drive.google.com/uc?id=1mY40mWoPuYCOrlfMQk_6DPGEFaQcvNAM",
        "minatar-asterix_v0": "https://drive.google.com/uc?id=1ohUxhZTYQCwyH-WJRH_Ma9BV3M1WoY0N",
        "minatar-breakout_v0": "https://drive.google.com/uc?id=1ED1-p3Gmi4PZEH3hF-9NZzPyNkiPCnvT",
        "minatar-freeway_v0": "https://drive.google.com/uc?id=1rbnJGxlzAWkt5DtF7tiYwoNkSqk0l2kD",
        "minatar-seaquest_v0": "https://drive.google.com/uc?id=1740nIi00Z8fQWRbA-52GiSGkW7rcqM8o",
        "minatar-space_invaders_v0": "https://drive.google.com/uc?id=1I7kJ8GEhY9K3rAFnbnYtlI5KQusFReq9",
    }
    assert baseline_model in urls
    return urls[baseline_model]


def _create_az_model_v0(
    num_actions,
    num_channels: int = 128,
    num_layers: int = 6,
    resnet_v2: bool = True,
):
    # We referred to Haiku's ResNet implementation:
    # https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/nets/resnet.py
    import haiku as hk

    class BlockV1(hk.Module):
        def __init__(self, num_channels, name="BlockV1"):
            super(BlockV1, self).__init__(name=name)
            self.num_channels = num_channels

        def __call__(self, x, is_training, test_local_stats):
            i = x
            x = hk.Conv2D(self.num_channels, kernel_shape=3)(x)
            x = hk.BatchNorm(True, True, 0.9)(x, is_training, test_local_stats)
            x = jax.nn.relu(x)
            x = hk.Conv2D(self.num_channels, kernel_shape=3)(x)
            x = hk.BatchNorm(True, True, 0.9)(x, is_training, test_local_stats)
            return jax.nn.relu(x + i)

    class BlockV2(hk.Module):
        def __init__(self, num_channels, name="BlockV2"):
            super(BlockV2, self).__init__(name=name)
            self.num_channels = num_channels

        def __call__(self, x, is_training, test_local_stats):
            i = x
            x = hk.BatchNorm(True, True, 0.9)(x, is_training, test_local_stats)
            x = jax.nn.relu(x)
            x = hk.Conv2D(self.num_channels, kernel_shape=3)(x)
            x = hk.BatchNorm(True, True, 0.9)(x, is_training, test_local_stats)
            x = jax.nn.relu(x)
            x = hk.Conv2D(self.num_channels, kernel_shape=3)(x)
            return x + i

    class AZNet(hk.Module):
        """AlphaZero NN architecture."""

        def __init__(
            self,
            num_actions,
            num_channels: int,
            num_layers: int,
            resnet_v2: bool,
            name="az_net",
        ):
            super().__init__(name=name)
            self.num_actions = num_actions
            self.num_channels = num_channels
            self.num_layers = num_layers
            self.resnet_v2 = resnet_v2
            self.resnet_cls = BlockV2 if resnet_v2 else BlockV1

        def __call__(self, x, is_training, test_local_stats):
            x = x.astype(jnp.float32)
            x = hk.Conv2D(self.num_channels, kernel_shape=3)(x)

            if not self.resnet_v2:
                x = hk.BatchNorm(True, True, 0.9)(x, is_training, test_local_stats)
                x = jax.nn.relu(x)

            for i in range(self.num_layers):
                x = self.resnet_cls(self.num_channels, name=f"block_{i}")(x, is_training, test_local_stats)

            if self.resnet_v2:
                x = hk.BatchNorm(True, True, 0.9)(x, is_training, test_local_stats)
                x = jax.nn.relu(x)

            # policy head
            logits = hk.Conv2D(output_channels=2, kernel_shape=1)(x)
            logits = hk.BatchNorm(True, True, 0.9)(logits, is_training, test_local_stats)
            logits = jax.nn.relu(logits)
            logits = hk.Flatten()(logits)
            logits = hk.Linear(self.num_actions)(logits)

            # value head
            value = hk.Conv2D(output_channels=1, kernel_shape=1)(x)
            value = hk.BatchNorm(True, True, 0.9)(value, is_training, test_local_stats)
            value = jax.nn.relu(value)
            value = hk.Flatten()(value)
            value = hk.Linear(self.num_channels)(value)
            value = jax.nn.relu(value)
            value = hk.Linear(1)(value)
            value = jnp.tanh(value)
            value = value.reshape((-1,))

            return logits, value

    return AZNet(num_actions, num_channels, num_layers, resnet_v2)
