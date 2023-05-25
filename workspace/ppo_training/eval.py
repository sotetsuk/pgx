import jax
import jax.numpy as jnp
import pgx
from ppo import ActorCritic
import os
import pickle
from utils import single_play_step_vs_policy, single_play__step_vs_random
import distrax
from functools import partial
import matplotlib.pyplot as plt

def _get(rewards, actor):
    return rewards[actor]

def vs_two_policy(params1, params2, network, env, rng_key, num_envs):
    _step_fn_vs_policy2 = single_play_step_vs_policy(env.step, network, params2)
    rng_key, sub_key = jax.random.split(rng_key)
    subkeys = jax.random.split(sub_key, num_envs)
    state = jax.vmap(env.init)(subkeys)
    state = state.replace(_turn=jnp.zeros(num_envs, dtype=jnp.int8))  # starts by black
    cum_return_of_policy1 = jnp.zeros(num_envs)
    i = 0
    states = []
    def cond_fn(tup):
        state, _, _ = tup
        return ~state.terminated.all()
    def loop_fn(tup):
        state, cum_return_of_policy1, rng_key = tup
        actor = state.current_player
        logits, value = network.apply(params1, state.observation)
        logits = logits + jnp.finfo(jnp.float64).min * (~state.legal_action_mask)
        pi = distrax.Categorical(logits=logits)
        rng_key, _rng = jax.random.split(rng_key)
        action = pi.sample(seed=_rng)
        rng_key, _rng = jax.random.split(rng_key)
        state = _step_fn_vs_policy2(state, action, _rng)
        cum_return_of_policy1 = cum_return_of_policy1 + jax.vmap(_get)(state.rewards, actor)
        return state, cum_return_of_policy1 ,rng_key
    state, cum_return_of_policy1, _ = jax.lax.while_loop(cond_fn, loop_fn, (state, cum_return_of_policy1, rng_key))
    return cum_return_of_policy1.mean()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="backgammon")
    parser.add_argument("--num_envs", type=int, default=64)
    args = parser.parse_args()
    param_dir = "params"
    env = pgx.make(args.env_name)
    network = ActorCritic(env.num_actions, activation="tanh")
    cand_policy_filename = f'params/{args.env_name}/anchor.ckpt'
    with open(cand_policy_filename, "rb") as f:
        params = pickle.load(f)["params"]
    vs_fn = jax.jit(partial(vs_two_policy, params2=params, network=network, env=env, num_envs=64, rng_key=jax.random.PRNGKey(3)))
    step_seq = []
    eval_R_seq = []
    for filename in os.listdir(param_dir):
        if filename.endswith(".ckpt") and filename.startswith("backgammon_vs_prev_policy"):
            param_filename = os.path.join(param_dir, filename)
            _, _, _, _, _, steps2 = filename.split("_")
            steps2 = int(steps2.split(".")[0])
            with open(param_filename, "rb") as f:
                params2 = pickle.load(f)["params"]
            eval_R = vs_fn(params1=params2)
            print(f"{steps} vs {steps2}", eval_R)
            step_seq.append(steps2)
            eval_R_seq.append(eval_R)
    # sort by steps
    step_seq, eval_R_seq = zip(*sorted(zip(step_seq, eval_R_seq)))
    plt.plot(step_seq, eval_R_seq)
    plt.title(f"R vs policy after {steps} steps")
    plt.xlabel("steps")
    plt.ylabel(f"R")
    plt.savefig(f"eval_R.png")
    

