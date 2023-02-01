import csv
from typing import Tuple

import numpy as np

from pgx.contractbridgebidding import _pbn_to_key


def to_value(sample: list) -> np.ndarray:
    """Convert sample to value
    >>> sample = ['0', '1', '0', '4', '0', '13', '12', '13', '9', '13', '0', '1', '0', '4', '0', '13', '12', '13', '9', '13']
    >>> to_value(sample)
    array([  4160, 904605,   4160, 904605])
    """
    np_sample = np.array(sample)
    np_sample = np_sample.astype(np.int8).reshape(4, 5)
    return to_binary(np_sample)


def to_binary(x) -> np.ndarray:
    """Convert dds information to value
    >>> x = np.arange(20, dtype=np.int32).reshape(4, 5) % 14
    >>> to_binary(x)
    array([  4660, 354185, 703696,  74565])
    """
    bases = np.array([16**i for i in range(5)], dtype=np.int32)[::-1]
    return (x * bases).sum(axis=1)  # shape = (4, )


def make_hash_table(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """make key and value of hash from samples"""
    samples = []
    with open(csv_path, "r") as f:
        reader = csv.reader(f, delimiter=",")
        for i in reader:
            samples.append(i)
    keys = []
    values = []
    for sample in samples:
        keys.append(_pbn_to_key(sample[0]))
        values.append(to_value(sample[1:]))
    return np.array(keys), np.array(values)


keys, values = make_hash_table(
    "workspace/contractbridge-ddstable-sample100.csv"
)
print(keys.tolist())
print(values.tolist())
