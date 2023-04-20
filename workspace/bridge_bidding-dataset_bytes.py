from tqdm import tqdm
import jax
import jax.numpy as jnp
from flax.serialization import to_bytes, from_bytes
from typing import Tuple
import csv
import numpy as np
import time


def to_value(sample: list) -> jnp.ndarray:
    """Convert sample to value
    >>> sample = ['0', '1', '0', '4', '0', '13', '12', '13', '9', '13', '0', '1', '0', '4', '0', '13', '12', '13', '9', '13']
    >>> to_value(sample)
    Array([  4160, 904605,   4160, 904605], dtype=int32)
    """
    jnp_sample = jnp.array([int(s) for s in sample], dtype=np.int8).reshape(
        4, 5
    )
    return to_binary(jnp_sample)


def to_binary(x) -> jnp.ndarray:
    """Convert dds information to value
    >>> jnp.array([16**i for i in range(5)], dtype=jnp.int32)[::-1]
    Array([65536,  4096,   256,    16,     1], dtype=int32)
    >>> x = jnp.arange(20, dtype=jnp.int32).reshape(4, 5) % 14
    >>> to_binary(x)
    Array([  4660, 354185, 703696,  74565], dtype=int32)
    """
    bases = jnp.array([16**i for i in range(5)], dtype=jnp.int32)[::-1]
    return (x * bases).sum(axis=1)  # shape = (4, )


def make_hash_table(
    csv_path: str,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """make key and value of hash from samples
    [start, end)
    """
    keys = []
    values = []
    with open(csv_path, "r") as f:
        reader = csv.reader(f, delimiter=",")
        count = 0
        for i in tqdm(reader):
            if count == 50000:
                break
            keys.append(_pbn_to_key(i[0]))
            values.append(to_value(i[1:]))
            count += 1
    return jnp.array(keys, dtype=jnp.int32), jnp.array(values, dtype=jnp.int32)


def _pbn_to_key(pbn: str) -> jnp.ndarray:
    """Convert pbn to key of dds table"""
    key = jnp.zeros(52, dtype=jnp.int8)
    hands = pbn[2:]
    for player, hand in enumerate(list(hands.split())):  # for each player
        for suit, cards in enumerate(list(hand.split("."))):  # for each suit
            for card in cards:  # for each card
                card_num = _card_str_to_int(card) + suit * 13
                key = key.at[card_num].set(player)
    key = key.reshape(4, 13)
    return _to_binary(key)


@jax.jit
def _to_binary(x: jnp.ndarray) -> jnp.ndarray:
    bases = jnp.array([4**i for i in range(13)], dtype=jnp.int32)[::-1]
    return (x * bases).sum(axis=1)  # shape = (4, )


def _card_str_to_int(card: str) -> int:
    if card == "K":
        return 12
    elif card == "Q":
        return 11
    elif card == "J":
        return 10
    elif card == "T":
        return 9
    elif card == "A":
        return 0
    else:
        return int(card) - 1


time1 = time.time()
keys, values = make_hash_table(
    "/Users/kitayuu/workspace/pgx/tests/assets/bridge_bidding_dds_results_2500000.csv"
)
time2 = time.time()
print(f"make hash table time: {time2-time1}")
print(f"keys BYTES = \n{to_bytes(keys)}")
print(f"values BYTES = \n{to_bytes(values)}")
time3 = time.time()
print(f"make byte time: {time3-time2}")
with open("keys_bytes.txt", "wb") as f:
    f.write(to_bytes(keys))
with open("values_bytes.txt", "wb") as f:
    f.write(to_bytes(values))
time4 = time.time()
print(f"make byte files time: {time4 - time3}")
with open("keys_bytes.txt", "rb") as f:
    keys_byte = f.read()
    from_bytes(keys, keys_byte)
    print(keys)
with open("values_bytes.txt", "rb") as f:
    values_byte = f.read()
    from_bytes(values, values_byte)
    print(values)
