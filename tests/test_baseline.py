import jax
import pgx
import pgx

import haiku as hk


# def test_az_basline():
#     batch_size = 2
#     test_cases = (
#         ("animal_shogi", "animal_shogi_v0"),
#         ("gardner_chess", "gardner_chess_v0"),
#         ("go_9x9", "go_9x9_v0"),
#         ("hex", "hex_v0"),
#         ("othello", "othello_v0"),
#         ("minatar-asterix", "minatar-asterix_v0"),
#         ("minatar-breakout", "minatar-breakout_v0"),
#         ("minatar-freeway", "minatar-freeway_v0"),
#         ("minatar-seaquest", "minatar-seaquest_v0"),
#         ("minatar-space_invaders", "minatar-space_invaders_v0")
#     )
# 
#     for env_id, model_id in test_cases:
#         env = pgx.make(env_id)
#         model = pgx.make_baseline_model(model_id)
#         state = jax.jit(jax.vmap(env.init))(
#             jax.random.split(jax.random.PRNGKey(0), batch_size)
#         )
# 
#         logits, value = model(state.observation)
#         assert logits.shape == (batch_size, env.num_actions)
#         assert value.shape == (batch_size,)
