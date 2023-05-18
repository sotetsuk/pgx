import jax
import jax.numpy as jnp
import pgx
import time

TRUE = jnp.bool_(True)
FALSE = jnp.bool_(False)

def auto_reset(step_fn, init_fn):
    """Auto reset wrapper.

    There are several concerns before staging this wrapper:

    1. Final state (observation)
    When auto restting happened, the termianl (or truncated) state/observation is replaced by initial state/observation,
    This is not problematic if it's termination.
    However, when truncation happened, value of truncated state/observation might be used by agent.
    So we have to preserve the final state (observation) somewhere.
    For example, in Gymnasium,

    https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/wrappers/autoreset.py#L59

    However, currently, truncation does *NOT* actually happens because
    all of Pgx environments (games) are finite-horizon and terminates in reasonable # of steps.
    (NOTE: Chess, Shogi, and Go have `max_termination_steps` option following AlphaZero approach)
    So, curren implementation is enough (so far), and I feel implementing `final_state/observation` is useless and not necessary.

    2. Performance:
    Might harm the performance as it always generates new state.
    Memory usage might be doubled. Need to check.
    """

    def wrapped_step_fn(state, action):
        state = jax.lax.cond(
            (state.terminated | state.truncated),
            lambda: state.replace(  # type: ignore
                _step_count=jnp.int32(0),
                terminated=FALSE,
                truncated=FALSE,
                rewards=jnp.zeros_like(state.rewards),
            ),
            lambda: state,
        )
        state = step_fn(state, action)
        state = jax.lax.cond(
            (state.terminated | state.truncated),
            # state is replaced by initial state,
            # but preserve (terminated, truncated, reward)
            lambda: init_fn(state._rng_key).replace(  # type: ignore
                terminated=state.terminated,
                truncated=state.truncated,
                rewards=state.rewards,
            ),
            lambda: state,
        )
        return state
    return wrapped_step_fn


def single_play_step(step_fn, network, train_state):
    """
    assume backgammon
    """
    def wrapped_step_fn(state, action, rng):
        state = step_fn(state, action)
        rewards1 = state.rewards
        terminated1 = state.terminated
        rng, _rng = jax.random.split(rng)
        logits = network.apply(train_state.params, state.observation)
        action = jax.random.categorical(_rng, logits)
        state = step_fn(state, action)  # step by right
        rewards2 = state.rewards
        terminated2 = state.terminated

        rewards = rewards1 + rewards2
        terminated = terminated1 | terminated2
        return state.replace(rewards=rewards, terminated=terminated)
    return wrapped_step_fn


def visualize(network, params, env_name, rng_key, num_envs):
    print("evaluate is called")
    env = pgx.make(env_name)
    rng_key, sub_key = jax.random.split(rng_key)
    subkeys = jax.random.split(sub_key, num_envs)
    state = jax.vmap(env.init)(subkeys)
    states = []
    states.append(state)
    cum_return = jnp.zeros(num_envs)
    i = 0
    step = jax.jit(jax.vmap(env.step))
    while not state.terminated.all():
        i += 1
        pi, value = network.apply(params, state.observation)
        action = pi.sample(seed=rng_key)
        state = step(state, action)
        states.append(state)
        cum_return += state.rewards[:, 0]
    fname = f"{'_'.join((env.id).lower().split())}.svg"
    print(f"avarage cumulative return over{num_envs}", cum_return.mean())
    if env.id == "2048":
        pgx.save_svg_animation(states, fname, frame_duration_seconds=0.5)
