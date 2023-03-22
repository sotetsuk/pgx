import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from itertools import product
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



def plot_ax(ax, game):
    libraries = get_all_field(data, "library")
    for lib in libraries:
        filtered = filter_data(data, game, lib)
        if len(filtered) == 0:
            continue
        bs, val = to_numpy(filtered)
        ax.plot(bs, val, label=f"{lib}", marker=".")
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_title(game)
    ax.set_xlabel("# Environments")


# plt.figure()
# filtered = filter_data(data, "go", "petting_zoo")
# for l in filtered:
#     print(l)
# bs, val = to_numpy(filtered)
# plt.plot(val)
# plt.show()

games = get_all_field(data, "game")
fig, axes = plt.subplots(1, len(games), figsize=(12, 3))
for i, game in enumerate(games):
    plot_ax(axes[i], game)
    axes[i].legend()
axes[0].set_ylabel("# Steps per second")
plt.tight_layout()
plt.show()


    

# bs = np.int32([2 ** (i + 1) for i in range(10)])
# fig, axes = plt.subplots(1, 4, figsize=(12, 3))
# methods = [("pgx-v0.1.6", None), ("pgx", None), ("open_spiel", "for-loop"), ("open_spiel", "subproc"), ("petting_zoo", "for-loop"), ("cshogi", "for-loop"), ("cshogi", "subproc"), ("petting_zoo", "subproc")]
# titles = {"go": "Go 19x19", "shogi": "Shogi", "tic_tac_toe": "Tic-Tac-Toe", "backgammon": "Backgammon"}
# library_titles = {"pgx-v0.1.6": "Pgx-v0.1.6", "pgx": "Pgx", "open_spiel": "OpenSpiel", "petting_zoo": "PettingZoo", "cshogi": "cshogi"}
# venv_titles = {"subproc": " (Subprocess)", "for-loop": " (For-loop)", None: ""}
# cmap = plt.get_cmap("tab10")
# colors = {"petting_zoo": cmap(1), "open_spiel": cmap(2), "cshogi": cmap(0), "pgx": cmap(3), "pgx-v0.1.6": cmap(4)}
# marker = "."
# 
# def plot_ax(ax, game, methods):
#     for library, venv in methods:
#         ls = "--" if venv == "for-loop" else "-"
#         filtered = filter_data(data, game, library, venv)
#         if len(filtered) == 0:
#             continue
#         filtered = to_numpy(filtered)
#         if game == "backgammon":
#             filtered /= 2
#         ax.plot(bs, filtered, label=f"{library_titles[library]}{venv_titles[venv]}", color=colors[library], linestyle=ls, marker=marker)
#         # ax.set_ylim((1e2, 2e6))
#         ax.set_yscale('log')
#         ax.set_xscale('log')
#         ax.set_title(titles[game])
#         ax.set_xlabel("# Environments")
# 
# 
# game = "tic_tac_toe"
# plot_ax(axes[0], game, methods)
# 
# game = "backgammon"
# plot_ax(axes[1], game, methods)
# 
# game = "shogi"
# plot_ax(axes[2], game, methods)
# 
# game = "go"
# plot_ax(axes[3], game, methods)
# 
# axes[0].set_ylabel("# Steps per second")
# plt.tight_layout()
# plt.subplots_adjust(bottom=0.3)
# plt.savefig("random_play_speed_benchmark.pdf")
# 
# 
# # legend
# plt.figure(figsize=(10, 5))
# a = np.arange(3)
# plt.plot(a, a, label="cshogi (Subprocess)", c=colors["cshogi"], marker=marker)
# plt.plot(a, a, label="cshogi (For-loop)", linestyle="--", c=colors["cshogi"], marker=marker)
# plt.plot(a, a, label="PettingZoo (Subprocess)", c=colors["petting_zoo"], marker=marker)
# plt.plot(a, a, label="PettingZoo (For-loop)", linestyle="--", c=colors["petting_zoo"], marker=marker)
# plt.plot(a, a, label="OpenSpiel (Subprocess)", c=colors["open_spiel"], marker=marker)
# plt.plot(a, a, label="OpenSpiel (For-loop)", linestyle="--", c=colors["open_spiel"], marker=marker)
# plt.plot(a, a, label="Pgx (Ours)", c=colors["pgx"], marker=marker)
# plt.legend(ncol=4, frameon=False)
# plt.tight_layout()
# plt.savefig("legend.pdf")
