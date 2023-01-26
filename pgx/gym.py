from typing import List

import jax
import jax.numpy as jnp

import pgx


class RandomOpponentEnv:
    def __init__(
        self,
        env_id: pgx.EnvId,
        num_envs: int,
        auto_reset=False,
        store_states=False,
    ):
        self.num_envs = num_envs
        _init, _step, observe = pgx.make(env_id)

        def _random_opponent_step(rng, state: pgx.State):
            logits = jnp.log(state.legal_action_mask.astype(jnp.float16))
            action = jax.random.categorical(rng, logits=logits)
            _, state, reward = _step(state, action)
            return state, reward

        def init(rng):
            _, state = _init(rng)
            return jax.lax.cond(
                state.curr_player != 0,
                lambda: _random_opponent_step(rng, state)[0],
                lambda: state,
            )

        def env_step_if_not_terminated(rng, state, action):
            _, state, reward = _step(state, action)
            state, opp_reward = jax.lax.cond(
                (state.curr_player == 1)
                & ~state.terminated,  # TODO: support >=3 players
                lambda: _random_opponent_step(rng, state),
                lambda: (state, jnp.zeros_like(reward)),
            )
            return state, (reward + opp_reward)[0]

        def env_step(rng, state, action):
            terminated = state.terminated
            next_state, reward = env_step_if_not_terminated(rng, state, action)
            return jax.lax.cond(
                terminated,
                lambda: (state, terminated, jnp.zeros_like(reward)),
                lambda: (next_state, next_state.terminated, reward),
            )

        def env_step_autoreset(rng, state, action):
            rng, subkey1, subkey2 = jax.random.split(rng, 3)
            state, terminated, reward = env_step(subkey1, state, action)
            state = jax.lax.cond(
                terminated, lambda: init(subkey2), lambda: state
            )
            return state, terminated, reward

        step = env_step_autoreset if auto_reset else env_step

        self.init_fn = jax.vmap(init)
        self.step_fn = jax.vmap(step)
        self.observe_fn = jax.vmap(observe)
        self.rng, self.state = self._init(0)
        self.store_states = store_states
        self.states: List[pgx.State] = []

    def _init(self, seed: int):
        rng = jax.random.PRNGKey(seed)
        rng, subkey = jax.random.split(rng)
        keys = jax.random.split(subkey, self.num_envs)
        state = self.init_fn(keys)
        return rng, state

    def reset(self, seed: int):
        self.rng, self.state = self._init(seed)
        obs = self.observe_fn(self.state)
        legal_action_mask = self.state.legal_action_mask
        if self.store_states:
            self.states.append(self.state)
        return obs, {"legal_action_mask": legal_action_mask}

    def step(self, action):
        self.rng, subkey = jax.random.split(self.rng)
        keys = jax.random.split(subkey, self.num_envs)
        self.state, terminated, reward = self.step_fn(keys, self.state, action)
        obs = self.observe_fn(self.state)
        terminated = self.state.terminated
        truncated = jnp.zeros_like(terminated)  # TODO: fix
        legal_action_mask = self.state.legal_action_mask
        info = {"legal_action_mask": legal_action_mask}
        if self.store_states:
            self.states.append(self.state)
        return obs, reward, terminated, truncated, info
