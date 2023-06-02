"""
This code is based on https://github.com/luchris429/purejaxrl
"""

import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Literal
import distrax
import pgx
from utils import auto_reset, single_play_step_vs_policy_in_backgammon, single_play_step_vs_policy_in_two, normal_step, single_play_step_vs_policy_in_sparrow_mahjong
import time
import os

import pickle
from omegaconf import OmegaConf
from pydantic import BaseModel
import wandb


class PPOConfig(BaseModel):
    ENV_NAME: Literal[ 
        "minatar-breakout", 
        "minatar-freeway", 
        "minatar-space_invaders", 
        "minatar-asterix", 
        "minatar-seaquest", 
        "play2048",
        ] = "minatar-breakout"
    SEED: int = 0
    LR: float = 2.5e-4
    NUM_ENVS: int = 64
    NUM_STEPS: int = 256
    TOTAL_TIMESTEPS: int = 5000000
    NUM_UPDATES: int = 5000000 // 64 // 256  # TOTAL_TIMESTEPS // NUM_ENVS // NUM_STEPS
    MINIBATCH_SIZE: int = 64 * 256 // 8  # NUM_ENVS * NUM_STEPS // NUM_MINIBATCHES
    UPDATE_EPOCHS: int = 30
    NUM_MINIBATCHES: int = 8
    GAMMA: float = 0.99
    GAE_LAMBDA: float = 0.95
    CLIP_EPS: float = 0.2
    ENT_COEF: float = 0.01
    VF_COEF: float = 0.5
    MAX_GRAD_NORM: float = 0.5
    ACTIVATION: str = "tanh"
    ANNEAL_LR: bool = True


args = PPOConfig(**OmegaConf.to_object(OmegaConf.from_cli()))
if args.ENV_NAME == "play2048":
    args.ENV_NAME = "2048"
env = pgx.make(args.ENV_NAME)


class ActorCritic(hk.Module):
    def __init__(self, num_actions, activation="tanh"):
        super().__init__()
        self.num_actions = num_actions
        self.activation = activation
        assert activation in ["relu", "tanh"]

    def __call__(self, x, is_training, test_local_stats):
        if self.activation == "relu":
            activation = jax.nn.relu
        else:
            activation = jax.nn.tanh
        x = hk.Conv2D(32, kernel_shape=2)(x)
        x = jax.nn.relu(x)
        x = hk.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        x = x.reshape((x.shape[0], -1))  # flatten
        x = hk.Linear(64)(x)
        x = jax.nn.relu(x)
        actor_mean = hk.Linear(
            64
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = hk.Linear(
            64
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = hk.Linear(
            self.num_actions
        )(actor_mean)

        critic = hk.Linear(
            64
        )(x)
        critic = activation(critic)
        critic = hk.Linear(
            64
        )(critic)
        critic = activation(critic)
        critic = hk.Linear(1)(
            critic
        )

        return actor_mean, jnp.squeeze(critic, axis=-1)

def forward_pass(x, is_eval=False):
    net = ActorCritic(env.num_actions, activation="tanh")
    logits, value = net(x, is_training=not is_eval, test_local_stats=False)
    return logits, value
forward_pass = hk.without_apply_rng(hk.transform_with_state(forward_pass))

def linear_schedule(count):
    frac = 1.0 - (count // (args.NUM_MINIBATCHES * args.UPDATE_EPOCHS)) / args.NUM_UPDATES
    return args.LR * frac
if args.ANNEAL_LR:
    optimizer = optax.chain(
        optax.clip_by_global_norm(args.MAX_GRAD_NORM),
        optax.adam(learning_rate=linear_schedule, eps=1e-5),
    )
else:
    optimizer = optax.chain(optax.clip_by_global_norm(args.MAX_GRAD_NORM), optax.adam(args.LR, eps=1e-5))


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    legal_action_mask: jnp.ndarray


def make_update_fn():
     # TRAIN LOOP
    def _update_step(runner_state):
        # COLLECT TRAJECTORIES
        step_fn = jax.vmap(auto_reset(env.step, env.init))
        def _env_step(runner_state, unused):
            model, opt_state, env_state, last_obs, rng = runner_state  # DONE
            model_params, model_state = model
            actor = env_state.current_player
            # SELECT ACTION
            rng, _rng = jax.random.split(rng)
            (logits, value), _  = forward_pass.apply(model_params, model_state, last_obs.astype(jnp.float32), is_eval=True)  # DONE
            mask = env_state.legal_action_mask
            logits = logits + jnp.finfo(np.float64).min * (~mask)
            pi = distrax.Categorical(logits=logits)
            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            env_state = step_fn(
                env_state, action
            )
            transition = Transition(
                env_state.terminated, action, value, env_state.rewards[:, 0], log_prob, last_obs, mask
            )
            runner_state = (model, opt_state, env_state, env_state.observation, rng)  # DONE
            return runner_state, transition

        runner_state, traj_batch = jax.lax.scan(
            _env_step, runner_state, None, args.NUM_STEPS
        )

        # CALCULATE ADVANTAGE
        model, opt_state, env_state, last_obs, rng = runner_state  # DONE
        model_params, model_state = model
        (_, last_val), _ = forward_pass.apply(model_params, model_state, last_obs.astype(jnp.float32), is_eval=False)  # DONE

        def _calculate_gae(traj_batch, last_val):
            def _get_advantages(gae_and_next_value, transition):
                gae, next_value = gae_and_next_value
                done, value, reward = (
                    transition.done,
                    transition.value,
                    transition.reward,
                )
                delta = reward + args.GAMMA * next_value * (1 - done) - value
                gae = (
                    delta
                    + args.GAMMA * args.GAE_LAMBDA * (1 - done) * gae
                )
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

        # UPDATE NETWORK
        def _update_epoch(update_state, unused):
            def _update_minbatch(tup, batch_info):
                model, opt_state = tup
                traj_batch, advantages, targets = batch_info
                model_params, model_state = model

                def _loss_fn(model_params, model_state,  traj_batch, gae, targets):
                    # RERUN NETWORK
                    (logits, value), model_state = forward_pass.apply(model_params, model_state, traj_batch.obs.astype(jnp.float32), is_eval=False)  # DONE
                    mask = traj_batch.legal_action_mask
                    logits = logits + jnp.finfo(np.float64).min * (~mask)
                    pi = distrax.Categorical(logits=logits)
                    log_prob = pi.log_prob(traj_batch.action)

                    # CALCULATE VALUE LOSS
                    value_pred_clipped = traj_batch.value + (
                        value - traj_batch.value
                    ).clip(-args.CLIP_EPS, args.CLIP_EPS)
                    value_losses = jnp.square(value - targets)
                    value_losses_clipped = jnp.square(value_pred_clipped - targets)
                    value_loss = (
                        0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                    )

                    # CALCULATE ACTOR LOSS
                    ratio = jnp.exp(log_prob - traj_batch.log_prob)
                    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                    loss_actor1 = ratio * gae
                    loss_actor2 = (
                        jnp.clip(
                            ratio,
                            1.0 - args.CLIP_EPS,
                            1.0 + args.CLIP_EPS,
                        )
                        * gae
                    )
                    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                    loss_actor = loss_actor.mean()
                    entropy = pi.entropy().mean()

                    total_loss = (
                        loss_actor
                        + args.VF_COEF * value_loss
                        - args.ENT_COEF * entropy
                    )
                    return total_loss, (value_loss, loss_actor, entropy)

                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                total_loss, grads = grad_fn(
                    model_params, model_state, traj_batch, advantages, targets
                )  # DONE
                updates, opt_state = optimizer.update(grads, opt_state)
                model_params = optax.apply_updates(model_params, updates)  # DONE
                return ((model_params, model_state), opt_state), total_loss  # DONE

            model, opt_state, traj_batch, advantages, targets, rng = update_state  # DONE
            rng, _rng = jax.random.split(rng)
            batch_size = args.MINIBATCH_SIZE * args.NUM_MINIBATCHES
            assert (
                batch_size == args.NUM_STEPS * args.NUM_ENVS
            ), "batch size must be equal to number of steps * number of envs"
            permutation = jax.random.permutation(_rng, batch_size)
            batch = (traj_batch, advantages, targets)
            batch = jax.tree_util.tree_map(
                lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
            )
            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0), batch
            )
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.reshape(
                    x, [args.NUM_MINIBATCHES, -1] + list(x.shape[1:])
                ),
                shuffled_batch,
            )
            (model, opt_state),  total_loss = jax.lax.scan(
                _update_minbatch, (model, opt_state), minibatches
            )  # DONE
            update_state = (model, opt_state, traj_batch, advantages, targets, rng)  # DONE
            return update_state, total_loss

        update_state = (model, opt_state, traj_batch, advantages, targets, rng)  # DONE
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, args.UPDATE_EPOCHS
        )
        model, opt_state , _, _, _, rng = update_state  # DONE

        runner_state = (model, opt_state, env_state, last_obs, rng)  # DONE
        return runner_state, loss_info
    return _update_step


def _get(x, i):
    return x[i]


def _get_zero(x, i):
    return x[0]


def evaluate(model,  env, rng_key):
    model_params, model_state = model
    step_fn = jax.vmap(env.step)
    rng_key, sub_key = jax.random.split(rng_key)
    subkeys = jax.random.split(sub_key, args.NUM_ENVS)
    state = jax.vmap(env.init)(subkeys)
    cum_return = jnp.zeros(args.NUM_ENVS)
    i = 0
    states = []
    def cond_fn(tup):
        state, _, _ = tup
        return ~state.terminated.all()
    def loop_fn(tup):
        state, cum_return, rng_key = tup
        actor = state.current_player
        (logits, value), _  = forward_pass.apply(model_params, model_state, state.observation.astype(jnp.float32), is_eval=True)  # DONE
        logits = logits + jnp.finfo(np.float64).min * (~state.legal_action_mask)
        pi = distrax.Categorical(logits=logits)
        rng_key, _rng = jax.random.split(rng_key)
        action = pi.sample(seed=_rng)
        rng_key, _rng = jax.random.split(rng_key)
        state = step_fn(state, action)
        cum_return = cum_return + state.rewards[:, 0]
        return state, cum_return ,rng_key
    state, cum_return, _ = jax.lax.while_loop(cond_fn, loop_fn, (state, cum_return, rng_key))
    return cum_return.mean()


def train(rng):
    # INIT NETWORK
    rng, _rng = jax.random.split(rng)
    init_x = jnp.zeros((1, ) + env.observation_shape)
    model = forward_pass.init(_rng, init_x)  # (params, state)  # DONE
    opt_state = optimizer.init(params=model[0])  # DONE

    # INIT UPDATE FUNCTION
    _update_step = make_update_fn()  # DONE
    jitted_update_step = jax.jit(_update_step)

    # INIT ENV
    rng, _rng = jax.random.split(rng)
    reset_rng= jax.random.split(_rng, args.NUM_ENVS)
    env_state = jax.vmap(env.init)(reset_rng)

    rng, _rng = jax.random.split(rng)
    runner_state = (model, opt_state, env_state, env_state.observation, _rng)  # DONE

    ckpt_filename = f'checkpoints/{args.ENV_NAME}/model.ckpt'
    steps = 0
    for i in range(args.NUM_UPDATES):
        eval_R = evaluate(runner_state[0], env, rng) # DONE
        log = {
            f"eval_R_{args.ENV_NAME}": float(eval_R),
            "steps": steps,
        }
        print(log)
        wandb.log(log)
        runner_state, loss_info = jitted_update_step(runner_state)  # DONE
        steps += args.NUM_ENVS * args.NUM_STEPS
        if i % 10 == 0:
            with open(f"checkpoints/{args.ENV_NAME}/model.ckpt", "wb") as writer:
                pickle.dump({"model": runner_state[0], "opt_state": runner_state[1]}, writer)
        _, (value_loss, loss_actor, entropy) = loss_info
    return runner_state


if __name__ == "__main__":
    key = "" # please specify your wandb key
    wandb.login(key=key)
    wandb.init(project=f"ppo-haiku", config=args.dict())
    print("training of", args.ENV_NAME)
    rng = jax.random.PRNGKey(args.SEED)
    sta = time.time()
    out = train(rng)
    end = time.time()
    print("training: time", end - sta)
