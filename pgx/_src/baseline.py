import os
import pickle
from typing import Literal

import jax
import jax.numpy as jnp

from pgx._src.utils import download

BaselineModelId = Literal[
    "animal_shogi_v0",
    "gardner_chess_v0",
    "go_9x9_v0",
    "hex_v0",
    "othello_v0",
]


def make_baseline_model(model_id: BaselineModelId):
    import haiku as hk

    create_model_fn = _make_create_model_fn(model_id)
    model_args, model_params, model_state = _load_baseline_model(model_id)

    def forward_fn(x, is_eval=False):
        net = create_model_fn(**model_args)
        policy_out, value_out = net(
            x, is_training=not is_eval, test_local_stats=False
        )
        return policy_out, value_out

    forward = hk.without_apply_rng(hk.transform_with_state(forward_fn))

    def apply(obs):
        return forward.apply(model_params, model_state, obs, is_eval=True)

    return apply


def _make_create_model_fn(baseline_model: BaselineModelId):
    if baseline_model in (
        "animal_shogi_v0",
        "gardner_chess_v0",
        "go_9x9_v0",
        "hex_v0",
        "othello_v0",
    ):
        return _create_az_model_v0
    else:
        assert False


def _load_baseline_model(
    baseline_model: BaselineModelId, basedir: str = "baselines"
):
    os.makedirs(basedir, exist_ok=True)

    # download baseline model if not exists
    filename = os.path.join(basedir, baseline_model + ".ckpt")
    if not os.path.exists(filename):
        url = _get_download_url(baseline_model)
        download(url, filename)

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
                x = hk.BatchNorm(True, True, 0.9)(
                    x, is_training, test_local_stats
                )
                x = jax.nn.relu(x)

            for i in range(self.num_layers):
                x = self.resnet_cls(self.num_channels, name=f"block_{i}")(
                    x, is_training, test_local_stats
                )

            if self.resnet_v2:
                x = hk.BatchNorm(True, True, 0.9)(
                    x, is_training, test_local_stats
                )
                x = jax.nn.relu(x)

            # policy head
            logits = hk.Conv2D(output_channels=2, kernel_shape=1)(x)
            logits = hk.BatchNorm(True, True, 0.9)(
                logits, is_training, test_local_stats
            )
            logits = jax.nn.relu(logits)
            logits = hk.Flatten()(logits)
            logits = hk.Linear(self.num_actions)(logits)

            # value head
            value = hk.Conv2D(output_channels=1, kernel_shape=1)(x)
            value = hk.BatchNorm(True, True, 0.9)(
                value, is_training, test_local_stats
            )
            value = jax.nn.relu(value)
            value = hk.Flatten()(value)
            value = hk.Linear(self.num_channels)(value)
            value = jax.nn.relu(value)
            value = hk.Linear(1)(value)
            value = jnp.tanh(value)
            value = value.reshape((-1,))

            return logits, value

    return AZNet(num_actions, num_channels, num_layers, resnet_v2)
