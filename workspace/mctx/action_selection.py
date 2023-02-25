import mctx
import jax
import jax.numpy as jnp
import chex

def puct_mcts_selection(rng_key, tree, node_index, depth, pb_c_init):
    """
    mctxはmuzero以降のmctsしか想定していない.
    ここではvanila-puct-MCTSのaction selectionを実装したい.
    選択基準は
    win_rate_of_action_a + UCB
    """
    visit_counts = tree.children_visits[node_index]
    node_visit = tree.node_visits[node_index]
    pb_c = pb_c_init
    prior_logits = tree.children_prior_logits[node_index]
    prior_probs = jax.nn.softmax(prior_logits)
    policy_score = jnp.sqrt(node_visit) * pb_c / (visit_counts + 1)
    chex.assert_shape([node_index, node_visit], ())
    chex.assert_equal_shape([prior_probs, visit_counts, policy_score])
    value_score = qtransform(tree, node_index)

    # Add tiny bit of randomness for tie break
    node_noise_score = 1e-7 * jax.random.uniform(
        rng_key, (tree.num_actions,))
    to_argmax = value_score + policy_score + node_noise_score

    # Masking the invalid actions at the root.
    return masked_argmax(to_argmax, tree.root_invalid_actions * (depth == 0))