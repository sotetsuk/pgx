"""
Monte Carlo tree search.
"""

from functools import partial

import chex
import jax
import jax.numpy as jnp
import mctx
from pgx.utils import act_randomly

from pgx.core import Env, State
from pgx.tic_tac_toe import step, observe, init
from pgx.visualizer import Visualizer
v = Visualizer()

batched_init = jax.jit(jax.vmap(init))
batched_step = jax.jit(jax.vmap(step))
batched_observe = jax.jit(jax.vmap(observe))



def random_play_till_end(state, rng_key) -> jnp.ndarray:

    def cond_fn(tup):
        state, _ = tup
        return  ~state.terminated

    def body_fn(tup):
        state, rng_key = tup
        rng_key, subkey = jax.random.split(rng_key)
        logits = jnp.log(state.legal_action_mask.astype(jnp.float16))
        a = jax.random.categorical(subkey, logits)
        state = step(state, a)
        return (state, rng_key)
    return jax.lax.while_loop(cond_fn, body_fn, (state, rng_key))


def random_play_return(state, rng_key):
    return_state, _ = random_play_till_end(state, rng_key)
    return return_state.reward[state.curr_player]


def _get(x, idx):
    return x[idx]


def recurrent_fn(params, rng_key: chex.Array, action: chex.Array, embedding: State):
    """One simulation step in MCTS."""
    rng_key, subkey = jax.random.split(rng_key)
    subkeys = jax.random.split(subkey, N)
    state = embedding
    state = batched_step(state, action) 
    reward = jax.vmap(_get)(state.reward, state.curr_player)
    value = jax.vmap(random_play_return)(state, subkeys)  # 終局までrandom play
    prior_logits = jnp.ones(state.legal_action_mask.shape)
    discount = -1.0 * jnp.ones_like(reward)  # zero sum gameでは-1
    terminated = state.terminated
    assert value.shape == terminated.shape
    value = jnp.where(terminated, 0.0, value)
    assert discount.shape == terminated.shape
    discount = jnp.where(terminated, 0.0, discount)
    recurrent_fn_output = mctx.RecurrentFnOutput(
        reward=reward,
        discount=discount,
        prior_logits=prior_logits,
        value=value,
    )
    return recurrent_fn_output, state


def mcts(
    state: State,
    rng_key: chex.Array,
    rec_fn,
    num_simulations: int,
) -> int:
    """Improve agent policy using MCTS.
    Returns:
        An improved policy.
    """
    rng_key, subkey = jax.random.split(rng_key)
    subkeys = jax.random.split(subkey, N)
    value = jax.vmap(random_play_return)(state, subkeys) # 終局までrandom play
    prior_logits = jnp.ones(state.legal_action_mask.shape)
    root = mctx.RootFnOutput(prior_logits=prior_logits, value=value, embedding=state)
    policy_output = mctx.muzero_policy(
        params=None,
        rng_key=subkey,
        root=root,
        recurrent_fn=rec_fn,
        num_simulations=num_simulations,
        invalid_actions=~state.legal_action_mask,
    )
    action = jnp.argmax(policy_output.action_weights, axis=1)
    #print(policy_output.search_tree.node_values, policy_output.search_tree.children_values)
    return action

def set_curr_player(state, player):
    return state.replace(curr_player=player)

if __name__ == "__main__":
    N = 100
    NUMSIMULATIONS = 100
    rng = jax.random.PRNGKey(3)
    rng, subkey = jax.random.split(rng)
    subkeys = jax.random.split(subkey, N)
    # warmup
    print("warmup starts ...")
    s = batched_init(subkeys)
    rng, subkey = jax.random.split(rng)
    a = act_randomly(subkey, s)
    s = batched_step(s, a)
    print("warmup ends")
    rng = jax.random.PRNGKey(0)
    subkeys = jax.random.split(subkey, N)
    state = batched_init(subkeys)
    state = jax.vmap(partial(set_curr_player, player=0))(state)
    i = 0
    while not state.terminated.all():
        if i % 2 == 1:
            rng, subkey = jax.random.split(rng)
            action = act_randomly(subkey, state)
        else:
            rng, subkey = jax.random.split(rng)
            action = mcts(state, subkey, recurrent_fn, NUMSIMULATIONS)
        state = batched_step(state, action)
        print(i)
        i += 1
    
    print(f"average return of mcts agent {jax.vmap(partial(_get, idx=0))(state.reward).sum()/N} , average return of random agent {jax.vmap(partial(_get, idx=1))(state.reward).sum()/N}")
    
