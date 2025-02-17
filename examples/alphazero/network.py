# We referred to Haiku's ResNet implementation:
# https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/nets/resnet.py

import equinox as eqx
import jax
import jax.numpy as jnp


class BlockV1(eqx.Module):
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    norm1: eqx.nn.BatchNorm
    norm2: eqx.nn.BatchNorm

    def __init__(self, in_channels, out_channels, key):
        keys = jax.random.split(key, 2)
        self.conv1 = eqx.nn.Conv2d(in_channels, out_channels, padding="SAME", kernel_size=3, key=keys[0])
        self.conv2 = eqx.nn.Conv2d(out_channels, out_channels, padding="SAME", kernel_size=3, key=keys[1])
        self.norm1 = eqx.nn.BatchNorm(out_channels, "batch", momentum=0.9, mode="batch")
        self.norm2 = eqx.nn.BatchNorm(out_channels, "batch", momentum=0.9, mode="batch")

    def __call__(self, x, state):
        i = x
        x = self.conv1(x)
        x, state = self.norm1(x, state)
        x = jax.nn.relu(x)
        x = self.conv2(x)
        x, state = self.norm2(x, state)
        return jax.nn.relu(x + i), state


class BlockV2(eqx.Module):
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    norm1: eqx.nn.BatchNorm
    norm2: eqx.nn.BatchNorm

    def __init__(self, in_channels, out_channels, key):
        keys = jax.random.split(key, 2)
        self.conv1 = eqx.nn.Conv2d(in_channels, out_channels, padding="SAME", kernel_size=3, key=keys[0])
        self.conv2 = eqx.nn.Conv2d(out_channels, out_channels, padding="SAME", kernel_size=3, key=keys[1])
        self.norm1 = eqx.nn.BatchNorm(in_channels, "batch", momentum=0.9, mode="batch")
        self.norm2 = eqx.nn.BatchNorm(out_channels, "batch", momentum=0.9, mode="batch")

    def __call__(self, x, state):
        i = x
        x, state = self.norm1(x, state)
        x = jax.nn.relu(x)
        x = self.conv1(x)
        x, state = self.norm2(x, state)
        x = jax.nn.relu(x)
        x = self.conv2(x)
        return x + i, state


class AZNet(eqx.Module):

    init_layers: list
    resnet: list
    post_resnet: list
    policy_head: list
    value_head: list

    def __init__(
        self,
        num_actions,
        input_channels,
        key,
        output_channels: int = 64,
        num_blocks: int = 5,
        resnet_v2: bool = True,
    ):
        resnet_cls = BlockV2 if resnet_v2 else BlockV1

        keys = jax.random.split(key, num_blocks + 5)
        self.init_layers = [eqx.nn.Conv2d(input_channels, output_channels, kernel_size=3, padding="SAME", key=keys[0])]
        if not resnet_v2:
            self.init_layers += [eqx.nn.BatchNorm(output_channels, "batch", momentum=0.9, mode="batch"), jax.nn.relu]
        self.resnet = [resnet_cls(output_channels, output_channels, keys[i + 1]) for i in range(num_blocks)]
        self.post_resnet = []
        if resnet_v2:
            self.post_resnet += [eqx.nn.BatchNorm(output_channels, "batch", momentum=0.9, mode="batch"), jax.nn.relu]
        self.policy_head = [
            eqx.nn.Conv2d(output_channels, 2, kernel_size=1, padding="SAME", key=keys[num_blocks + 1]),
            eqx.nn.BatchNorm(2, "batch", momentum=0.9, mode="batch"),
            jax.nn.relu,
            lambda x: x.flatten(),
            # TODO: infer from inputs
            eqx.nn.Linear(162, num_actions, key=keys[num_blocks + 2]),
        ]

        self.value_head = [
            eqx.nn.Conv2d(output_channels, 1, kernel_size=1, padding="SAME", key=keys[num_blocks + 3]),
            eqx.nn.BatchNorm(1, "batch", momentum=0.9, mode="batch"),
            jax.nn.relu,
            lambda x: x.flatten(),
            eqx.nn.Linear(81, output_channels, key=keys[num_blocks + 2]),
            jax.nn.relu,
            eqx.nn.Linear(output_channels, 1, key=keys[num_blocks + 2]),
            jnp.tanh,
            jnp.squeeze,
        ]

    def __call__(self, x, state):
        x = x.astype(jnp.float32)
        x = jnp.moveaxis(x, -1, 0)

        for layer in self.init_layers:
            if isinstance(layer, eqx.nn.StatefulLayer):
                x, state = layer(x, state)
            else:
                x = layer(x)

        for layer in self.resnet:
            x, state = layer(x, state)

        for layer in self.post_resnet:
            if isinstance(layer, eqx.nn.StatefulLayer):
                x, state = layer(x, state)
            else:
                x = layer(x)

        logits = x.copy()
        for layer in self.policy_head:
            if isinstance(layer, eqx.nn.StatefulLayer):
                logits, state = layer(logits, state)
            else:
                logits = layer(logits)

        v = x.copy()
        for layer in self.value_head:
            if isinstance(layer, eqx.nn.StatefulLayer):
                v, state = layer(v, state)
            else:
                v = layer(v)

        return (logits, v), state
