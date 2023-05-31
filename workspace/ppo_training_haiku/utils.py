import jax
import jax.numpy as jnp
import pgx
import time
import distrax

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


def single_play_step_vs_policy_in_backgammon(step_fn, forward_pass, model):
    """
    assume backgammon
    """
    model_params, model_state = model
    def act_based_on_policy(state, rng):
        (logits, value), _  = forward_pass.apply(model_params, model_state, state.observation, is_eval=True)
        logits = logits + jnp.finfo(jnp.float64).min * (~state.legal_action_mask)
        pi = distrax.Categorical(logits=logits)
        action = pi.sample(seed=rng)
        return action
    
    act_fn = act_based_on_policy
    def act_till_turn_end(state, rng, current_player, actor):
        def cond_fn(tup):
            state, _, _ = tup
            return (state.current_player != actor) & (state.current_player == current_player) & ~state.terminated
        
        def loop_fn(tup):
            state, rng, rewards = tup
            rng, _rng = jax.random.split(rng)
            action = act_fn(state, _rng)
            state = step_fn(state, action)
            rewards = rewards + state.rewards
            return state, rng, rewards
        tup = (state, rng, state.rewards)
        state, _, rewards = jax.lax.while_loop(cond_fn, loop_fn, tup)
        return state, rewards

    def enemy_step_fn(state, rng, actor):
        return jax.lax.cond(
            actor != state.current_player & ~state.terminated,
            lambda: act_till_turn_end(state, rng, state.current_player, actor),
            lambda: (state, state.rewards),
            ) 

    def wrapped_step_fn(state, action, rng):
        actor = state.current_player
        state = jax.vmap(step_fn)(state, action)
        state, rewards = jax.vmap(enemy_step_fn, in_axes=(0, None, 0))(state, rng, actor)
        return state.replace(rewards=rewards)
    return wrapped_step_fn


def single_play_step_vs_policy_in_two(step_fn, forward_pass, model):
    """
    assume backgammon
    """
    model_params, model_state = model
    def act_based_on_policy(state, rng):
        (logits, value), _  = forward_pass.apply(model_params, model_state, state.observation, is_eval=True)
        logits = logits + jnp.finfo(jnp.float64).min * (~state.legal_action_mask)
        pi = distrax.Categorical(logits=logits)
        action = pi.sample(seed=rng)
        return action
    
    act_fn = act_based_on_policy

    def wrapped_step_fn(state, action, rng):
        state = jax.vmap(step_fn)(state, action)
        terminated = state.terminated
        rewards = state.rewards
        action = jax.vmap(act_fn, in_axes=(0, None))(state, rng)
        state = jax.vmap(step_fn)(state, action)
        rewards = rewards + state.rewards
        terminated = terminated | state.terminated
        return state.replace(rewards=rewards, terminated=terminated)
    return wrapped_step_fn


def normal_step(step_fn):
    def wrapped_step_fn(state, action, rng):
        state = jax.vmap(step_fn)(state, action)
        return state
    return wrapped_step_fn


def single_play_step_vs_policy_in_sparrow_mahjong(step_fn, forward_pass, model):
    """
    assume sparrow mahjong
    """
    model_params, model_state = model
    def wrapped_step_fn(state, action, rng):
        state = step_fn(state, action)
        rewards1 = state.rewards
        terminated1 = state.terminated
        rng, _rng = jax.random.split(rng)
        (logits, _), _  = forward_pass.apply(model_params, model_state, state.observation, is_eval=True)
        logits = logits + jnp.finfo(jnp.float64).min * (~state.legal_action_mask)
        pi = distrax.Categorical(logits=logits)
        action = pi.sample(seed=_rng)
        state = step_fn(state, action)  # step by right
        rewards2 = state.rewards
        terminated2 = state.terminated

        rng, _rng = jax.random.split(rng)
        (logits, _), _  = forward_pass.apply(model_params, model_state, state.observation, is_eval=True)
        logits = logits + jnp.finfo(jnp.float64).min * (~state.legal_action_mask)
        pi = distrax.Categorical(logits=logits)
        action = pi.sample(seed=_rng)
        state = step_fn(state, action)   # step by left
        rewards3 = state.rewards
        terminated3 = state.terminated
        rewards = rewards1 + rewards2 + rewards3
        terminated = terminated1 | terminated2 | terminated3
        return state.replace(rewards=rewards, terminated=terminated)
    return wrapped_step_fn


def normal_step(step_fn):
    def wrapped_step_fn(state, action, rng):
        state = jax.vmap(step_fn)(state, action)
        return state
    return wrapped_step_fn