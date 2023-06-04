import jax
import jax.numpy as jnp
import haiku as hk
import pgx
import time
import pickle
from ppo_multi import ActorCritic
import distrax
import argparse
from utils import single_play_step_vs_policy_in_sparrow_mahjong, single_play_step_vs_policy_in_backgammon
TRUE = jnp.bool_(True)
FALSE = jnp.bool_(False)

def _make_step(env_name, model):
    env = pgx.make(env_name)
    step_fn = env.step
    if env_name == "backgammon":
        return single_play_step_vs_policy_in_backgammon(step_fn, forward_pass, model)
    elif env_name == "sparrow_mahjong":
        return single_play_step_vs_policy_in_sparrow_mahjong(step_fn, forward_pass, model)  # to make baseline model, random is preferred
    else:
        raise NotImplementedError

def _get(x, i):
    return x[i]

def vs_two_policies(forward_pass, model1, model2, env,  rng_key, env_num):
    model1_params, model1_state = model1
    model2_params, model2_state = model2

    # model 1 vs model 2
    step_fn_vs1 = jax.jit(_make_step(env.id, model1))
    step_fn_vs2 = jax.jit(_make_step(env.id, model2))
    subkeys = jax.random.split(rng_key, env_num)
    state = jax.vmap(env.init)(subkeys)
    rewards1 = jnp.zeros(env_num)
    while not state.terminated.all():
        actor = state.current_player
        (logits, value), _ = forward_pass.apply(model1_params,model1_state, state.observation.astype(jnp.float32), is_eval=True)
        logits = logits +  jnp.finfo(jnp.float64).min * (~state.legal_action_mask)
        pi = distrax.Categorical(logits=logits)
        rng_key, _rng = jax.random.split(rng_key)
        action = pi.sample(seed=_rng)
        rng_key, _rng = jax.random.split(rng_key)
        state = step_fn_vs2(state, action, _rng)
        rewards1 = rewards1 + jax.vmap(_get)(state.rewards, actor)

    # model 2 vs model 1
    subkeys = jax.random.split(rng_key, env_num)
    state = jax.vmap(env.init)(subkeys)
    rewards2 = jnp.zeros(env_num)
    while not state.terminated.all():
        (logits, value), _ = forward_pass.apply(model2_params,model2_state, state.observation.astype(jnp.float32), is_eval=True)
        logits = logits +  jnp.finfo(jnp.float64).min * (~state.legal_action_mask)
        pi = distrax.Categorical(logits=logits)
        rng_key, _rng = jax.random.split(rng_key)
        action = pi.sample(seed=_rng)
        rng_key, _rng = jax.random.split(rng_key)
        state = step_fn_vs1(state, action, _rng)
        rewards2 = rewards2 + jax.vmap(_get)(state.rewards, actor)
    print(rewards1.mean(), rewards2.mean())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="backgammon")
    parser.add_argument("--env_num", type=int, default=512)
    args = parser.parse_args()

    env = pgx.make(args.env_name)
    def forward_pass(x, is_eval=False):
        net = ActorCritic(env.num_actions, activation="tanh", env_name=env.id)
        logits, value = net(x, is_training=not is_eval, test_local_stats=False)
        return logits, value
    forward_pass = hk.without_apply_rng(hk.transform_with_state(forward_pass))
    agent1_filename = f'checkpoints/{args.env_name}/model_random.ckpt'  # specify the agent_1_filename
    agent2_filename = f'checkpoints/{args.env_name}/model.ckpt'  # specify the agent_2_filename
    with open(agent1_filename, "rb") as f:
        model1 = pickle.load(f)["model"]
    with open(agent2_filename, "rb") as f:
        model2 = pickle.load(f)["model"]
    env = pgx.make(args.env_name)
    rng_key = jax.random.PRNGKey(3)
    vs_two_policies(forward_pass, model1, model2, env, rng_key, args.env_num)
    