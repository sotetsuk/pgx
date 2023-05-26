import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
import json


data: List[Dict] = []
with open("results.json") as f:
    for line in f:
        d = (json.loads(line))
        data.append(d)


def to_numpy(data):
    # group by batch size
    max_bs = max([int(d["batch_size"]) for d in data])
    bs = min([int(d["batch_size"]) for d in data]) 
    i, bs2ix= 0, {}
    while bs <=  max_bs:
        bs2ix[bs] = i
        i += 1
        bs *= 2
    n = np.zeros(len(bs2ix), dtype=np.float32)
    steps = np.zeros(len(bs2ix), dtype=np.float32)
    for d in data:
        bs = d["batch_size"]
        n[bs2ix[bs]] += 1
        steps[bs2ix[bs]] += float(d["steps/sec"])
    return np.array(list(bs2ix.keys())), steps / n


def filter_data(data, game, library):
    data = filter(lambda d: d["game"] == game, data)
    data = filter(lambda d: d["library"] == library, data)
    data = list(data)
    return data


def get_all_field(data, field):
    fields = set()
    for d in data:
        fields.add(d[field])
    fields = list(fields)
    fields.sort()
    return fields


cmap = plt.get_cmap("tab10")
colors = {}
max_color = [0]


def plot_ax(ax, game):
    libraries = get_all_field(data, "library")
    for lib in libraries:
        filtered = filter_data(data, game, lib)
        if len(filtered) == 0:
            continue
        bs, val = to_numpy(filtered)
        if lib not in colors:
            colors[lib] = max_color[0]
            max_color[0] += 1
        c = cmap(colors[lib])
        ax.plot(bs, val, label=f"{lib}", marker=".", color=c)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_title(game)
    ax.set_xlabel("# Environments")


games = get_all_field(data, "game")
fig, axes = plt.subplots(1, len(games), figsize=(12, 3))
for i, game in enumerate(games):
    plot_ax(axes[i], game)
    if game == "tic_tac_toe":
        axes[i].legend()
axes[0].set_ylabel("# Steps per second")
plt.tight_layout()
plt.savefig("results.pdf")
