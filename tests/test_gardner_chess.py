import jax
import jax.numpy as jnp
import pgx
from pgx._gardner_chess import State, Action

pgx.set_visualization_config(color_theme="dark")


def p(s: str, b=False):
    """
    >>> p("e3")
    20
    >>> p("e3", b=True)
    24
    """
    x = "abcde".index(s[0])
    offset = int(s[1]) - 1 if not b else 5 - int(s[1])
    return x * 8 + offset


def test_action():
    ...
