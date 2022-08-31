from typing import Optional, Tuple

import gym
import jax.numpy as jnp
import jax.random


class MinAtar(gym.Env):
    def __init__(
        self,
        game: str,
        batch_size: int = 8,
        auto_reset=True,
        sticky_action_prob: float = 0.1,
    ):
        self.game = game
        self.auto_reset = auto_reset
        self.batch_size = batch_size
        self.sticky_action_prob: jnp.ndarray = (
            jnp.ones(self.batch_size) * sticky_action_prob
        )
        if self.game == "breakout":
            from pgx.minatar.breakout import reset, step, to_obs

            self._reset = jax.vmap(reset)
            self._step = jax.vmap(step)
            self._to_obs = jax.vmap(to_obs)
        else:
            raise NotImplementedError("This game is not implemented.")

        self.rng = jax.random.PRNGKey(0)
        self.rng, _rngs = self._split_keys(self.rng)
        self.state = self._reset(_rngs)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Tuple[jnp.ndarray, dict]:
        assert seed is not None
        self.rng = jax.random.PRNGKey(seed)
        self.rng, _rngs = self._split_keys(self.rng)
        self.state = self._reset(_rngs)
        return self._to_obs(self.state), {}

    def step(
        self, action: jnp.ndarray
    ) -> Tuple[jnp.ndarray, float, bool, dict]:
        self.rng, _rngs = self._split_keys(self.rng)
        self.state, r, done = self._step(
            state=self.state,
            action=action,
            rng=_rngs,
            sticky_action_prob=self.sticky_action_prob,
        )
        if self.auto_reset:

            @jax.vmap
            def where(c, x, y):
                return jax.lax.cond(c, lambda _: x, lambda _: y, 0)

            self.rng, _rngs = self._split_keys(self.rng)
            init_state = self._reset(_rngs)
            self.state = where(done, init_state, self.state)
        return self._to_obs(self.state), r, done, {}

    def _split_keys(self, rng):
        rngs = jax.random.split(rng, self.batch_size + 1)
        rng = rngs[0]
        subrngs = rngs[1:]
        return rng, subrngs
