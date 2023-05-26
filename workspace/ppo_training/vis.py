import jax
import jax.numpy as jnp
import pgx
import time
import pickle
from ppo import ActorCritic
import distrax
import argparse
TRUE = jnp.bool_(True)
FALSE = jnp.bool_(False)


def visualize(network, params, env,  rng_key):
    state = env.init(rng_key)
    i = 0
    states = []
    states.append(state)
    step_fn = jax.jit(env.step)
    while not state.terminated.all():
        logits, value = network.apply(params, state.observation)
        logits = logits +  jnp.finfo(jnp.float64).min * (~state.legal_action_mask)
        pi = distrax.Categorical(logits=logits)
        actor = state.current_player
        rng_key, _rng = jax.random.split(rng_key)
        action = pi.sample(seed=_rng)
        rng_key, _rng = jax.random.split(rng_key)
        state = step_fn(state, action, _rng)
        states.append(state)
    fname = f"vis/{'_'.join((env.id).lower().split())}.svg"
    pgx.save_svg_animation(states, fname, frame_duration_seconds=0.7)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="backgammon")
    args = parser.parse_args()

    env = pgx.make(args.env_name)
    network = ActorCritic(env.num_actions, activation="tanh")
    ckpt_filename = f'checkpoints/{args.env_name}/model.ckpt'
    with open(ckpt_filename, "rb") as f:
        params = pickle.load(f)["params"]

    env = pgx.make(args.env_name)
    network = ActorCritic(env.num_actions, activation="tanh")
    rng_key = jax.random.PRNGKey(3)
    visualize(network, params, env, rng_key)
