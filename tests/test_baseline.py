import jax
import pgx
from pgx.baseline import make_create_model_fn, load_baseline_model, BaselineModel

import haiku as hk


def test_az_basline():
    batch_size = 2
    test_cases = (
        ("animal_shogi", "animal_shogi_v0"),
        ("gardner_chess", "gardner_chess_v0"),
        ("go_9x9", "go_9x9_v0"),
        ("hex", "hex_v0"),
        ("othello", "othello_v0")
    )

    for env_id, baseline_model in test_cases:
        create_model_fn = make_create_model_fn(baseline_model)
        model_args, model_params, model_state = load_baseline_model(baseline_model)
        print(model_args)

        def forward_fn(x, is_eval=False):
            net = create_model_fn(**model_args)
            policy_out, value_out = net(
                x, is_training=not is_eval, test_local_stats=False)
            return policy_out, value_out

        forward = hk.without_apply_rng(hk.transform_with_state(forward_fn))

        env = pgx.make(env_id)
        state = jax.jit(jax.vmap(env.init))(
            jax.random.split(jax.random.PRNGKey(0), batch_size)
        )

        (logits, value), model_state = forward.apply(
            model_params, model_state, state.observation, is_eval=True
        )
        assert logits.shape == (batch_size, env.num_actions)
        assert value.shape == (batch_size,)
