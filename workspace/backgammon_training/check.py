import jax
import jax.numpy as jnp
import pgx
import time
import pickle
from ppo import ActorCritic, evaluate
from utils import single_play_step_vs_policy, single_play__step_vs_random
import distrax
TRUE = jnp.bool_(True)
FALSE = jnp.bool_(False)

def _get(rewards, actor):
    return rewards[actor]
def visualize(network, params, env,  rng_key, num_envs, save_svg=False):
    _single_play_step = jax.jit(single_play__step_vs_random(env.step))
    rng_key, sub_key = jax.random.split(rng_key)
    subkeys = jax.random.split(sub_key, num_envs)
    state = jax.vmap(env.init)(subkeys)
    state = state.replace(_turn=jnp.zeros(num_envs, dtype=jnp.int8))  # starts by black
    state = state.replace(current_player=jnp.zeros(num_envs, dtype=jnp.int8))  # starts by black
    cum_return = jnp.zeros(num_envs)
    R = jnp.zeros((num_envs, 2))
    i = 0
    states = []
    states.append(state)
    while not state.terminated.all():
        logits, value = network.apply(params, state.observation)
        logits = logits +  jnp.finfo(jnp.float64).min * (~state.legal_action_mask)
        pi = distrax.Categorical(logits=logits)
        actor = state.current_player
        rng_key, _rng = jax.random.split(rng_key)
        action = pi.sample(seed=_rng)
        #assert not (actor != jnp.zeros(num_envs, dtype=jnp.int8) & ~state.terminated).any()  # 終了するまでactorは変わらない
        rng_key, _rng = jax.random.split(rng_key)
        state = _single_play_step(state, action, _rng)
        cum_return = cum_return + jax.vmap(_get)(state.rewards, actor)
        R = R + state.rewards
        states.append(state)
    if save_svg:
        fname = f"{'_'.join((env.id).lower().split())}.svg"
        pgx.save_svg_animation(states, fname, frame_duration_seconds=0.5)
    print(f"avarage cumulative return over{num_envs}", cum_return.mean())
    print(f"R of {num_envs}", R.mean(axis=0))


if __name__ == "__main__":
    env = pgx.make("backgammon")
    network = ActorCritic(env.num_actions, activation="tanh")
    ckpt_filename = f'params/{"backgammon"}_vs_{"random"}_steps_{1015808}.ckpt'
    with open(ckpt_filename, "rb") as f:
        params = pickle.load(f)["params"]
    rng_key = jax.random.PRNGKey(3)
    _single_play_step = jax.jit(single_play__step_vs_random(env.step))
    print(evaluate(params, network, _single_play_step, env, rng_key, 64))
    visualize(network, params, env, rng_key, 7, save_svg=True)
