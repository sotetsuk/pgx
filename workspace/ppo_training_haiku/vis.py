import jax
import jax.numpy as jnp
import haiku as hk
import pgx
import time
import pickle
from ppo import ActorCritic
import distrax
import argparse
TRUE = jnp.bool_(True)
FALSE = jnp.bool_(False)


def visualize(forward_pass, model, env,  rng_key):
    model_params, model_state = model
    subkeys = jax.random.split(rng_key, 5)
    state = jax.vmap(env.init)(subkeys)
    states = []
    states.append(state)
    step_fn = jax.jit(jax.vmap(env.step))
    while not state.terminated.all():
        (logits, value), _ = forward_pass.apply(model_params,model_state, state.observation, is_eval=True)
        logits = logits +  jnp.finfo(jnp.float64).min * (~state.legal_action_mask)
        pi = distrax.Categorical(logits=logits)
        rng_key, _rng = jax.random.split(rng_key)
        action = pi.sample(seed=_rng)
        rng_key, _rng = jax.random.split(rng_key)
        state = step_fn(state, action)
        states.append(state)
    fname = f"vis/{'_'.join((env.id).lower().split())}.svg"
    pgx.save_svg_animation(states, fname, frame_duration_seconds=0.7)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="backgammon")
    args = parser.parse_args()

    env = pgx.make(args.env_name)
    def forward_pass(x, is_eval=False):
        net = ActorCritic(env.num_actions, activation="tanh", env_name=env.id)
        logits, value = net(x, is_training=not is_eval, test_local_stats=False)
        return logits, value
    forward_pass = hk.without_apply_rng(hk.transform_with_state(forward_pass))
    ckpt_filename = f'checkpoints/{args.env_name}/model.ckpt'
    with open(ckpt_filename, "rb") as f:
        model = pickle.load(f)["model"]
    env = pgx.make(args.env_name)
    rng_key = jax.random.PRNGKey(3)
    visualize(forward_pass, model, env, rng_key)
