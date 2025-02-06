"""This PPO implementation is modified from PureJaxRL:

  https://github.com/luchris429/purejaxrl

Please refer to their work if you use this example in your research."""

import sys
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from typing import NamedTuple, Literal
from distreqx import distributions
import distrax
import pgx
from pgx.experimental import auto_reset
import time

import pickle
from omegaconf import OmegaConf
from pydantic import BaseModel
import wandb


class PPOConfig(BaseModel):
    env_name: Literal[
        "minatar-breakout",
        "minatar-freeway",
        "minatar-space_invaders",
        "minatar-asterix",
        "minatar-seaquest",
    ] = "minatar-breakout"
    seed: int = 0
    lr: float = 0.0003
    num_envs: int = 4096
    num_eval_envs: int = 100
    num_steps: int = 128
    total_timesteps: int = 20000000
    update_epochs: int = 3
    minibatch_size: int = 4096
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    wandb_project: str = "pgx-minatar-ppo"
    save_model: bool = False

    class Config:
        extra = "forbid"


args = PPOConfig(**OmegaConf.to_object(OmegaConf.from_cli()))
print(args, file=sys.stderr)
env = pgx.make(str(args.env_name))


num_updates = args.total_timesteps // args.num_envs // args.num_steps
num_minibatches = args.num_envs * args.num_steps // args.minibatch_size


class ActorCritic(eqx.Module):
    features: list
    actor: list
    critic: list

    def __init__(self, num_actions, key, activation="tanh"):
        assert activation in ["relu", "tanh"]
        if activation == "relu":
            act_fn = jax.nn.relu
        else:
            act_fn = jax.nn.tanh

        keys = jax.random.split(key, 8)

        self.features = [
            eqx.nn.Conv2d(env.observation_shape[2], 32, 2, padding="SAME", key=keys[0]),
            # (4, 10, 10) -> (32, 10, 10)
            jax.nn.relu,
            lambda x: jnp.moveaxis(x, 0, -1),
            eqx.nn.AvgPool2d(2, 2),
            # (10, 10, 32) -> (10, 5, 16)
            lambda x: x.flatten(),
            eqx.nn.Linear(10 * 5 * 16, 64, key=keys[1]),
            jax.nn.relu,
        ]

        self.actor = [
            eqx.nn.Linear(64, 64, key=keys[2]),
            act_fn,
            eqx.nn.Linear(64, 64, key=keys[3]),
            act_fn,
            eqx.nn.Linear(64, num_actions, key=keys[4]),
        ]

        self.critic = [
            eqx.nn.Linear(64, 64, key=keys[5]),
            act_fn,
            eqx.nn.Linear(64, 64, key=keys[6]),
            act_fn,
            eqx.nn.Linear(64, 1, key=keys[7]),
        ]

    def __call__(self, x):
        x = x.astype(jnp.float32)
        # make channels first
        x = jnp.moveaxis(x, -1, 0)
        for layer in self.features:
            x = layer(x)
        actor_mean = jnp.copy(x)
        for layer in self.actor:
            actor_mean = layer(actor_mean)
        critic = jnp.copy(x)
        for layer in self.critic:
            critic = layer(critic)
        return actor_mean, jnp.squeeze(critic, axis=-1)


optimizer = optax.chain(optax.clip_by_global_norm(args.max_grad_norm), optax.adam(args.lr, eps=1e-5))


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray


def make_update_fn():
    # TRAIN LOOP
    def _update_step(runner_state):
        # COLLECT TRAJECTORIES
        step_fn = jax.vmap(auto_reset(env.step, env.init))

        arrs, static = eqx.partition(runner_state[0], eqx.is_array)
        runner_state = eqx.tree_at(lambda x: x[0], runner_state, arrs)

        def _env_step(runner_state, unused):
            arr_params, opt_state, env_state, last_obs, rng = runner_state
            params = eqx.combine(arr_params, static)
            # SELECT ACTION
            rng, _rng = jax.random.split(rng)
            __rng = jax.random.split(_rng, last_obs.shape[0])
            # logits, value = eqx.filter_vmap(params)(last_obs)
            # pi = distrax.Categorical(logits=logits)
            # action = pi.sample(seed=_rng)
            # log_prob = pi.log_prob(action)
            pi = eqx.filter_vmap(distributions.Categorical)(logits)
            action = eqx.filter_vmap(lambda x, y: x.sample(y))(pi, __rng)
            action = action.astype('int32')
            log_prob = eqx.filter_vmap(lambda x, y: x.log_prob(y))(pi, action)

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            keys = jax.random.split(_rng, env_state.observation.shape[0])
            env_state = step_fn(env_state, action, keys)
            transition = Transition(
                env_state.terminated, action, value, jnp.squeeze(env_state.rewards), log_prob, last_obs
            )
            runner_state = (arr_params, opt_state, env_state, env_state.observation, rng)
            return runner_state, transition

        runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, args.num_steps)
        runner_state = eqx.tree_at(lambda x: x[0], runner_state, eqx.combine(runner_state[0], static))

        # CALCULATE ADVANTAGE
        params, opt_state, env_state, last_obs, rng = runner_state
        _, last_val = eqx.filter_vmap(params)(last_obs)

        def _calculate_gae(traj_batch, last_val):
            def _get_advantages(gae_and_next_value, transition):
                gae, next_value = gae_and_next_value
                done, value, reward = (
                    transition.done,
                    transition.value,
                    transition.reward,
                )
                delta = reward + args.gamma * next_value * (1 - done) - value
                gae = delta + args.gamma * args.gae_lambda * (1 - done) * gae
                return (gae, value), gae

            _, advantages = jax.lax.scan(
                _get_advantages,
                (jnp.zeros_like(last_val), last_val),
                traj_batch,
                reverse=True,
                unroll=16,
            )
            return advantages, advantages + traj_batch.value

        advantages, targets = _calculate_gae(traj_batch, last_val)

        params_arr, static = eqx.partition(params, eqx.is_array)

        # UPDATE NETWORK
        def _update_epoch(update_state, unused):
            def _update_minbatch(tup, batch_info):
                params, opt_state = tup
                traj_batch, advantages, targets = batch_info

                def _loss_fn(params, traj_batch, gae, targets):
                    # RERUN NETWORK
                    logits, value = eqx.filter_vmap(params)(traj_batch.obs)
                    # pi = distrax.Categorical(logits=logits)
                    # log_prob = pi.log_prob(traj_batch.action)
                    pi = eqx.filter_vmap(distributions.Categorical)(logits)
                    log_prob = eqx.filter_vmap(lambda x, y: x.log_prob(y))(pi, traj_batch.action)

                    # CALCULATE VALUE LOSS
                    value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                        -args.clip_eps, args.clip_eps
                    )
                    value_losses = jnp.square(value - targets)
                    value_losses_clipped = jnp.square(value_pred_clipped - targets)
                    value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                    # CALCULATE ACTOR LOSS
                    ratio = jnp.exp(log_prob - traj_batch.log_prob)
                    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                    loss_actor1 = ratio * gae
                    loss_actor2 = (
                        jnp.clip(
                            ratio,
                            1.0 - args.clip_eps,
                            1.0 + args.clip_eps,
                        )
                        * gae
                    )
                    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                    loss_actor = loss_actor.mean()
                    entropy = pi.entropy().mean()

                    total_loss = loss_actor + args.vf_coef * value_loss - args.ent_coef * entropy
                    return total_loss, (value_loss, loss_actor, entropy)

                grad_fn = eqx.filter_value_and_grad(_loss_fn, has_aux=True)
                total_loss, grads = grad_fn(eqx.combine(params, static), traj_batch, advantages, targets)
                updates, opt_state = optimizer.update(grads, opt_state)
                params = eqx.apply_updates(params, updates)
                return (params, opt_state), total_loss

            params, opt_state, traj_batch, advantages, targets, rng = update_state
            rng, _rng = jax.random.split(rng)
            batch_size = args.minibatch_size * num_minibatches
            assert (
                batch_size == args.num_steps * args.num_envs
            ), "batch size must be equal to number of steps * number of envs"
            permutation = jax.random.permutation(_rng, batch_size)
            batch = (traj_batch, advantages, targets)
            batch = jax.tree_util.tree_map(lambda x: x.reshape((batch_size,) + x.shape[2:]), batch)
            shuffled_batch = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=0), batch)
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, [num_minibatches, -1] + list(x.shape[1:])),
                shuffled_batch,
            )
            (params, opt_state), total_loss = jax.lax.scan(_update_minbatch, (params, opt_state), minibatches)
            update_state = (params, opt_state, traj_batch, advantages, targets, rng)
            return update_state, total_loss

        update_state = (params_arr, opt_state, traj_batch, advantages, targets, rng)
        update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, args.update_epochs)
        params, opt_state, _, _, _, rng = update_state

        runner_state = (eqx.combine(params, static), opt_state, env_state, last_obs, rng)
        return runner_state, loss_info

    return _update_step


@eqx.filter_jit
def evaluate(params, rng_key):
    step_fn = jax.vmap(env.step)
    rng_key, sub_key = jax.random.split(rng_key)
    subkeys = jax.random.split(sub_key, args.num_eval_envs)
    state = jax.vmap(env.init)(subkeys)
    R = jnp.zeros_like(state.rewards)

    def cond_fn(tup):
        state, _, _ = tup
        return ~state.terminated.all()

    def loop_fn(tup):
        state, R, rng_key = tup
        logits, value = eqx.filter_vmap(params)(state.observation)
        # action = logits.argmax(axis=-1)
        # pi = distrax.Categorical(logits=logits)
        # rng_key, _rng = jax.random.split(rng_key)
        # action = pi.sample(seed=_rng)
        pi = eqx.filter_vmap(distributions.Categorical)(logits)
        rng_key, _rng = jax.random.split(rng_key)
        __rng = jax.random.split(_rng, state.observation.shape[0])
        action = eqx.filter_vmap(lambda x, y: x.sample(y))(pi, __rng)
        action = action.astype('int32')
        rng_key, _rng = jax.random.split(rng_key)
        keys = jax.random.split(_rng, state.observation.shape[0])
        state = step_fn(state, action, keys)
        return state, R + state.rewards, rng_key

    state, R, _ = jax.lax.while_loop(cond_fn, loop_fn, (state, R, rng_key))
    return R.mean()


def train(rng):
    tt = 0
    st = time.time()
    # INIT NETWORK
    rng, _rng = jax.random.split(rng)
    model = ActorCritic(env.num_actions, _rng, "tanh")
    opt_state = optimizer.init(params=eqx.filter(model, eqx.is_inexact_array))

    # INIT UPDATE FUNCTION
    _update_step = make_update_fn()
    jitted_update_step = eqx.filter_jit(_update_step)

    # INIT ENV
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, args.num_envs)
    env_state = jax.jit(jax.vmap(env.init))(reset_rng)

    rng, _rng = jax.random.split(rng)
    runner_state = (model, opt_state, env_state, env_state.observation, _rng)

    # warm up
    _, _ = jitted_update_step(runner_state)

    steps = 0

    # initial evaluation
    et = time.time()  # exclude evaluation time
    tt += et - st
    rng, _rng = jax.random.split(rng)
    eval_R = evaluate(runner_state[0], _rng)
    log = {"sec": tt, f"{args.env_name}/eval_R": float(eval_R), "steps": steps}
    print(log)
    wandb.log(log)
    st = time.time()

    for i in range(num_updates):
        runner_state, loss_info = jitted_update_step(runner_state)
        steps += args.num_envs * args.num_steps

        # evaluation
        et = time.time()  # exclude evaluation time
        tt += et - st
        rng, _rng = jax.random.split(rng)
        eval_R = evaluate(runner_state[0], _rng)
        log = {"sec": tt, f"{args.env_name}/eval_R": float(eval_R), "steps": steps}
        print(log)
        wandb.log(log)
        st = time.time()

    return runner_state


if __name__ == "__main__":
    wandb.init(project=args.wandb_project, config=args.dict())
    rng = jax.random.PRNGKey(args.seed)
    out = train(rng)
    if args.save_model:
        with open(f"{args.env_name}-seed={args.seed}.ckpt", "wb") as f:
            pickle.dump(out[0], f)
