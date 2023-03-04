import mctx
import jax
import jax.numpy as jnp
import chex
from typing import Optional
import functools
from mctx import qtransform_by_parent_and_siblings
from search import search


def masked_argmax(
    to_argmax: chex.Array,
    invalid_actions: Optional[chex.Array]) -> chex.Array:
  """Returns a valid action with the highest `to_argmax`."""
  if invalid_actions is not None:
    chex.assert_equal_shape([to_argmax, invalid_actions])
    # The usage of the -inf inside the argmax does not lead to NaN.
    # Do not use -inf inside softmax, logsoftmax or cross-entropy.
    to_argmax = jnp.where(invalid_actions, -jnp.inf, to_argmax)
  # If all actions are invalid, the argmax returns action 0.
  return jnp.argmax(to_argmax, axis=-1)


def uct_mcts_selection(rng_key, tree, node_index, depth, pb_c_init: float = 1.25, qtransform=qtransform_by_parent_and_siblings):
    """
    mctxはmuzero以降のmctsしか想定していない.
    ここではvanila-puct-MCTSのaction selectionを実装したい.
    選択基準は
    win_rate_of_action_a + UCB
    """
    visit_counts = tree.children_visits[node_index]
    node_visit = tree.node_visits.sum()
    pb_c = pb_c_init
    policy_score = jnp.sqrt(2 * jnp.log(node_visit) / (visit_counts + 1))

    value_score = tree.children_values[node_index] / visit_counts  # 勝率


    to_argmax = value_score + policy_score
    legal_actions = tree.embeddings.legal_action_mask[node_index]

    # Masking the invalid actions at the root.
    return masked_argmax(to_argmax, ~legal_actions)


def uct_mcts_policy(
    params, 
    rng_key,
    root,
    recurrent_fn,
    num_simulations: int,
    invalid_actions: Optional[chex.Array] = None,
    max_depth: Optional[int] = None,
    loop_fn = jax.lax.fori_loop):
  """Runs MuZero search and returns the `PolicyOutput`.
  In the shape descriptions, `B` denotes the batch dimension.
  Args:
    params: params to be forwarded to root and recurrent functions.
    rng_key: random number generator state, the key is consumed.
    root: a `(prior_logits, value, embedding)` `RootFnOutput`. The
      `prior_logits` are from a policy network. The shapes are
      `([B, num_actions], [B], [B, ...])`, respectively.
    recurrent_fn: a callable to be called on the leaf nodes and unvisited
      actions retrieved by the simulation step, which takes as args
      `(params, rng_key, action, embedding)` and returns a `RecurrentFnOutput`
      and the new state embedding. The `rng_key` argument is consumed.
    num_simulations: the number of simulations.
    invalid_actions: a mask with invalid actions. Invalid actions
      have ones, valid actions have zeros in the mask. Shape `[B, num_actions]`.
    max_depth: maximum search tree depth allowed during simulation.
    loop_fn: Function used to run the simulations. It may be required to pass
      hk.fori_loop if using this function inside a Haiku module.
    qtransform: function to obtain completed Q-values for a node.
    dirichlet_fraction: float from 0 to 1 interpolating between using only the
      prior policy or just the Dirichlet noise.
    dirichlet_alpha: concentration parameter to parametrize the Dirichlet
      distribution.
    pb_c_init: constant c_1 in the PUCT formula.
    pb_c_base: constant c_2 in the PUCT formula.
    temperature: temperature for acting proportionally to
      `visit_counts**(1 / temperature)`.
  Returns:
    `PolicyOutput` containing the proposed action, action_weights and the used
    search tree.
  """
  rng_key, dirichlet_rng_key, search_rng_key = jax.random.split(rng_key, 3)

  # Running the search.
  interior_action_selection_fn = uct_mcts_selection
      
  root_action_selection_fn = functools.partial(
      interior_action_selection_fn,
      depth=0)
  search_tree = search(
      params=params,
      rng_key=search_rng_key,
      root=root,
      recurrent_fn=recurrent_fn,
      root_action_selection_fn=root_action_selection_fn,
      interior_action_selection_fn=interior_action_selection_fn,
      num_simulations=num_simulations,
      max_depth=max_depth,
      invalid_actions=invalid_actions,
      loop_fn=loop_fn)

  # Sampling the proposed action proportionally to the visit counts.
  summary = search_tree.summary()
  action_weights = summary.visit_probs
  visit_counts = summary.visit_counts
  q_values = summary.qvalues
  print(q_values)
  print(visit_counts)
  action = jnp.argmax(visit_counts, axis=1)
  return mctx.PolicyOutput(
      action=action,
      action_weights=action_weights,
      search_tree=search_tree)