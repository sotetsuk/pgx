import mctx
import jax
import jax.numpy as jnp
import chex
from typing import Optional
import functools
from mctx import qtransform_by_parent_and_siblings


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


def _mask_invalid_actions(logits, invalid_actions):
  """Returns logits with zero mass to invalid actions."""
  if invalid_actions is None:
    return logits
  chex.assert_equal_shape([logits, invalid_actions])
  logits = logits - jnp.max(logits, axis=-1, keepdims=True)
  # At the end of an episode, all actions can be invalid. A softmax would then
  # produce NaNs, if using -inf for the logits. We avoid the NaNs by using
  # a finite `min_logit` for the invalid actions.
  min_logit = jnp.finfo(logits.dtype).min
  return jnp.where(invalid_actions, min_logit, logits)


def _get_logits_from_probs(probs):
  tiny = jnp.finfo(probs).tiny
  return jnp.log(jnp.maximum(probs, tiny))


def _add_dirichlet_noise(rng_key, probs, *, dirichlet_alpha,
                         dirichlet_fraction):
  """Mixes the probs with Dirichlet noise."""
  chex.assert_rank(probs, 2)
  chex.assert_type([dirichlet_alpha, dirichlet_fraction], float)

  batch_size, num_actions = probs.shape
  noise = jax.random.dirichlet(
      rng_key,
      alpha=jnp.full([num_actions], fill_value=dirichlet_alpha),
      shape=(batch_size,))
  noisy_probs = (1 - dirichlet_fraction) * probs + dirichlet_fraction * noise
  return noisy_probs


def _apply_temperature(logits, temperature):
  """Returns `logits / temperature`, supporting also temperature=0."""
  # The max subtraction prevents +inf after dividing by a small temperature.
  logits = logits - jnp.max(logits, keepdims=True, axis=-1)
  tiny = jnp.finfo(logits.dtype).tiny
  return logits / jnp.maximum(tiny, temperature)


def uct_mcts_selection(rng_key, tree, node_index, depth, pb_c_init: float = 1.25, qtransform=qtransform_by_parent_and_siblings):
    """
    mctxはmuzero以降のmctsしか想定していない.
    ここではvanila-puct-MCTSのaction selectionを実装したい.
    選択基準は
    win_rate_of_action_a + UCB
    """
    visit_counts = tree.children_visits[node_index]
    node_visit = tree.node_visits[node_index]
    pb_c = pb_c_init
    # prior_logits = tree.children_prior_logits[node_index]
    # prior_probs = jax.nn.softmax(prior_logits)
    policy_score = pb_c * jnp.sqrt(node_visit) / (visit_counts + 1)
    chex.assert_shape([node_index, node_visit], ())
    # chex.assert_equal_shape([prior_probs, visit_counts, policy_score])
    value_score = tree.children_values[node_index] / visit_counts  # 勝率

    # Add tiny bit of randomness for tie break
    node_noise_score = 1e-7 * jax.random.uniform(
        rng_key, (tree.num_actions,))
    to_argmax = value_score + policy_score + node_noise_score

    # Masking the invalid actions at the root.
    return masked_argmax(to_argmax, tree.root_invalid_actions * (depth == 0))


def uct_mcts_policy(
    params, 
    rng_key,
    root,
    recurrent_fn,
    num_simulations: int,
    invalid_actions: Optional[chex.Array] = None,
    max_depth: Optional[int] = None,
    loop_fn = jax.lax.fori_loop,
    *,
    qtransform = qtransform_by_parent_and_siblings,
    dirichlet_fraction: chex.Numeric = 0.25,
    dirichlet_alpha: chex.Numeric = 0.3,
    pb_c_init: chex.Numeric = 1.25,
    pb_c_base: chex.Numeric = 19652,
    temperature: chex.Numeric = 1.0):
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
  search_tree = mctx.search(
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
  action = jax.random.categorical(rng_key, action_weights)
  return mctx.PolicyOutput(
      action=action,
      action_weights=action_weights,
      search_tree=search_tree)