import math

import jax.numpy as jnp


def get_sizes(state):
    try:
        size = len(state.current_player)
        width = math.ceil(math.sqrt(size - 0.1))
        if size - (width - 1) ** 2 >= width:
            height = width
        else:
            height = width - 1
    except TypeError:
        size = 1
        width = 1
        height = 1
    return size, width, height


def visualize_minatar(state, savefile=None):
    # Modified from https://github.com/kenjyoung/MinAtar
    import matplotlib.colors as colors  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore
    import seaborn as sns  # type: ignore

    obs = state.observation
    n_channels = obs.shape[-1]
    cmap = sns.color_palette("cubehelix", n_channels)
    cmap.insert(0, (0, 0, 0))
    # cmap = sns.cubehelix_palette(n_channels)
    # cmap.insert(0, (10, 10, 10))
    cmap = colors.ListedColormap(cmap)
    bounds = [i for i in range(n_channels + 2)]
    norm = colors.BoundaryNorm(bounds, n_channels + 1)
    size, w, h = get_sizes(state)
    fig, ax = plt.subplots(h, w)
    n_channels = obs.shape[-1]
    if size == 1:
        numerical_state = (
            jnp.amax(
                obs * jnp.reshape(jnp.arange(n_channels) + 1, (1, 1, -1)), 2
            )
            + 0.5
        )
        ax.imshow(numerical_state, cmap=cmap, norm=norm, interpolation="none")
        ax.set_axis_off()
    else:
        for j in range(size):
            numerical_state = (
                jnp.amax(
                    obs[j]
                    * jnp.reshape(jnp.arange(n_channels) + 1, (1, 1, -1)),
                    2,
                )
                + 0.5
            )
            if h == 1:
                ax[j].imshow(
                    numerical_state, cmap=cmap, norm=norm, interpolation="none"
                )
                ax[j].set_axis_off()
            else:
                ax[j // w, j % w].imshow(
                    numerical_state, cmap=cmap, norm=norm, interpolation="none"
                )
                ax[j // w, j % w].set_axis_off()

    if savefile is None:
        from io import StringIO

        sio = StringIO()
        plt.savefig(sio, format="svg", bbox_inches="tight")
        plt.close(fig)
        return sio.getvalue()
    else:
        plt.savefig(savefile, format="svg", bbox_inches="tight")
        plt.close(fig)
        return None
