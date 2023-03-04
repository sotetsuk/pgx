"""
Monte Carlo tree search.
"""

from functools import partial
import chex
import jax
import jax.numpy as jnp
import mctx
from pgx.utils import act_randomly
import pgx
from pgx.visualizer import Visualizer
import pygraphviz
from uct_mcts import uct_mcts_selection, uct_mcts_policy
from search import search
v = Visualizer()

env = pgx.make("tic_tac_toe/v0")
assert env is not None

# warmup start

batched_init = jax.jit(jax.vmap(env.init))
batched_step = jax.jit(jax.vmap(env.step))


def convert_tree_to_graph(
    tree: mctx.Tree,
    action_labels = None,
    batch_index: int = 0
):
  """Converts a search tree into a Graphviz graph.
  Args:
    tree: A `Tree` containing a batch of search data.
    action_labels: Optional labels for edges, defaults to the action index.
    batch_index: Index of the batch element to plot.
  Returns:
    A Graphviz graph representation of `tree`.
  """
  chex.assert_rank(tree.node_values, 2)
  batch_size = tree.node_values.shape[0]
  if action_labels is None:
    action_labels = range(tree.num_actions)
  elif len(action_labels) != tree.num_actions:
    raise ValueError(
        f"action_labels {action_labels} has the wrong number of actions "
        f"({len(action_labels)}). "
        f"Expecting {tree.num_actions}.")

  def node_to_str(node_i, reward=0, discount=1):
    return (f"{node_i}\n"
            f"Reward: {reward:.2f}\n"
            f"Discount: {discount:.2f}\n"
            f"Value: {tree.node_values[batch_index, node_i]:.2f}\n"
            f"Visits: {tree.node_visits[batch_index, node_i]}\n")

  def edge_to_str(node_i, a_i):
    node_index = jnp.full([batch_size], node_i)
    probs = jax.nn.softmax(tree.children_prior_logits[batch_index, node_i])
    return (f"{action_labels[a_i]}\n"
            f"Q: {tree.qvalues(node_index)[batch_index, a_i]:.2f}\n"
            f"p: {probs[a_i]:.2f}\n")

  graph = pygraphviz.AGraph(directed=True)

  # Add root
  graph.add_node(0, label=node_to_str(node_i=0), color="green")
  # Add all other nodes and connect them up.
  for node_i in range(tree.num_simulations):
    for a_i in range(tree.num_actions):
      # Index of children, or -1 if not expanded
      children_i = tree.children_index[batch_index, node_i, a_i]
      if children_i >= 0:
        graph.add_node(
            children_i,
            label=node_to_str(
                node_i=children_i,
                reward=tree.children_rewards[batch_index, node_i, a_i],
                discount=tree.children_discounts[batch_index, node_i, a_i]),
            color="red")
        graph.add_edge(node_i, children_i, label=edge_to_str(node_i, a_i))

  return graph


def random_play_till_end(state, rng_key) -> jnp.ndarray:

    def cond_fn(tup):
        state, _ = tup
        return  ~state.terminated

    def body_fn(tup):
        state, rng_key = tup
        rng_key, subkey = jax.random.split(rng_key)
        logits = jnp.log(state.legal_action_mask.astype(jnp.float16))
        a = jax.random.categorical(subkey, logits)
        state = env.step(state, a)
        return (state, rng_key)
    return jax.lax.while_loop(cond_fn, body_fn, (state, rng_key))


def random_play_return(state, rng_key):
    return_state, _ = random_play_till_end(state, rng_key)
    return return_state.reward[state.curr_player]


def _get(x, idx):
    return x[idx]


def recurrent_fn(params, rng_key: chex.Array, action: chex.Array, embedding):
    """One simulation step in MCTS."""
    rng_key, subkey = jax.random.split(rng_key)
    subkeys = jax.random.split(subkey, N)
    state = embedding
    state = batched_step(state, action) 
    reward = 1 - jnp.clip(jax.vmap(_get)(state.reward, state.curr_player), a_min=0, a_max=1)
    value = 1- jnp.clip(jax.vmap(random_play_return)(state, subkeys), a_min=0, a_max=1)  # 終局までrandom play
    prior_logits = jnp.ones(state.legal_action_mask.shape)
    discount = jnp.ones_like(reward)
    terminated = state.terminated
    assert value.shape == terminated.shape
    value = jnp.where(terminated, 1- value, value)  # 終端状態の場合は0
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
    state,
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
    value = 1 - jnp.clip(jax.vmap(random_play_return)(state, subkeys), a_min=0, a_max=1) # 終局までrandom play 
    prior_logits = jnp.ones(state.legal_action_mask.shape)
    root = mctx.RootFnOutput(prior_logits=prior_logits, value=value, embedding=state)
    policy_output = uct_mcts_policy(
        params=None,
        rng_key=subkey,
        root=root,
        recurrent_fn=rec_fn,
        num_simulations=num_simulations,
        invalid_actions=~state.legal_action_mask,
    )
    #print(policy_output.action_weights)
    return policy_output.action, policy_output

def set_curr_player(state, player):
    return state.replace(curr_player=player)

if __name__ == "__main__":
    N = 10
    NUMSIMULATIONS = 5000
    mctx_id = 0
    random_id = 1
    rng = jax.random.PRNGKey(0)
    rng, subkey = jax.random.split(rng)
    subkeys = jax.random.split(subkey, N)
    # warmup
    print("warmup starts ...")
    s = batched_init(subkeys)
    rng, subkey = jax.random.split(rng)
    a = act_randomly(subkey, s)
    s = batched_step(s, a)
    print("warmup ends")
    rng = jax.random.PRNGKey(5)
    subkeys = jax.random.split(subkey, N)
    state = batched_init(subkeys)
    state = jax.vmap(partial(set_curr_player, player=0))(state)  # 初期agentのplayeridを0に固定
    i = 0
    reward = jnp.zeros((N, 2))
    while not state.terminated.all():
        if i % 2 == mctx_id:  # player_id0がmcts agent
            rng, subkey = jax.random.split(rng)
            action, policy_output = mcts(state, subkey, recurrent_fn, NUMSIMULATIONS)
            #v.save_svg(policy_output.search_tree.embeddings, f"vis/tree_{i:03d}.svg")
            #graph = convert_tree_to_graph(policy_output.search_tree)
            #graph.draw(f"tree_graph/graph{str(i)}.png", prog="dot")
        else:
            rng, subkey = jax.random.split(rng)
            action = act_randomly(subkey, state)
        state = batched_step(state, action)
        reward = reward + state.reward
        print(i)
        v.save_svg(state, f"vis/{i:03d}.svg")
        i += 1
    print(reward)
    print(f"average return of mcts agent {jax.vmap(partial(_get, idx=mctx_id))(reward).sum()/N} , average return of random agent {jax.vmap(partial(_get, idx=random_id))(reward).sum()/N}")
    
