import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Literal
from flax.training.train_state import TrainState
import distrax
import pgx
from utils import auto_reset, single_play_step
import time
import pickle
import argparse
from omegaconf import OmegaConf
from pydantic import BaseModel
import wandb


class PPOConfig(BaseModel):
    ENV_NAME: str = "backgammon"
    LR: float = 2.5e-4
    NUM_ENVS: int = 64
    NUM_STEPS: int = 128
    TOTAL_TIMESTEPS: int = 5000000
    UPDATE_EPOCHS: int = 30
    NUM_MINIBATCHES: int = 8
    GAMMA: float = 0.99
    GAE_LAMBDA: float = 0.95
    CLIP_EPS: float = 0.2
    ENT_COEF: float = 0.01
    VF_COEF: float = 0.5
    MAX_GRAD_NORM: float = 0.5
    ACTIVATION: str = "tanh"
    NUM_UPDATES: int = 10000
    MINIBATCH_SIZE: int = 32
    ANNEAL_LR: bool = True


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        x = nn.Conv(features=32, kernel_size=(2, 2))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


def _get(x, i):
    return x[i]

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray


def make_update_fn(config, env, network):
     # TRAIN LOOP
    def _update_step(runner_state, unused):
        # COLLECT TRAJECTORIES
        auto_rese_step = auto_reset(env.step, env.init)
        _single_play_step = single_play_step(auto_rese_step, network, runner_state[0])
        def _env_step(runner_state, unused):
            train_state, env_state, last_obs, rng = runner_state
            actor = env_state.current_player
            # SELECT ACTION
            rng, _rng = jax.random.split(rng)
            pi, value = network.apply(train_state.params, last_obs)
            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            env_state = jax.vmap(_single_play_step)(
                env_state, action
            )
            transition = Transition(
                env_state.terminated, action, value, jax.vmap(_get)(env_state.rewards, actor), log_prob, last_obs
            )
            runner_state = (train_state, env_state, env_state.observation, rng)
            return runner_state, transition

        runner_state, traj_batch = jax.lax.scan(
            _env_step, runner_state, None, config["NUM_STEPS"]
        )

        # CALCULATE ADVANTAGE
        train_state, env_state, last_obs, rng = runner_state
        _, last_val = network.apply(train_state.params, last_obs)

        def _calculate_gae(traj_batch, last_val):
            def _get_advantages(gae_and_next_value, transition):
                gae, next_value = gae_and_next_value
                done, value, reward = (
                    transition.done,
                    transition.value,
                    transition.reward,
                )
                delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                gae = (
                    delta
                    + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
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
            def _update_minbatch(train_state, batch_info):
                traj_batch, advantages, targets = batch_info

                def _loss_fn(params, traj_batch, gae, targets):
                    # RERUN NETWORK
                    pi, value = network.apply(params, traj_batch.obs)
                    log_prob = pi.log_prob(traj_batch.action)

                    # CALCULATE VALUE LOSS
                    value_pred_clipped = traj_batch.value + (
                        value - traj_batch.value
                    ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
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
                            1.0 - config["CLIP_EPS"],
                            1.0 + config["CLIP_EPS"],
                        )
                        * gae
                    )
                    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                    loss_actor = loss_actor.mean()
                    entropy = pi.entropy().mean()

                    total_loss = (
                        loss_actor
                        + config["VF_COEF"] * value_loss
                        - config["ENT_COEF"] * entropy
                    )
                    return total_loss, (value_loss, loss_actor, entropy)

                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                total_loss, grads = grad_fn(
                    train_state.params, traj_batch, advantages, targets
                )
                train_state = train_state.apply_gradients(grads=grads)
                return train_state, total_loss

            train_state, traj_batch, advantages, targets, rng = update_state
            rng, _rng = jax.random.split(rng)
            batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
            assert (
                batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
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
                    x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                ),
                shuffled_batch,
            )
            train_state, total_loss = jax.lax.scan(
                _update_minbatch, train_state, minibatches
            )
            update_state = (train_state, traj_batch, advantages, targets, rng)
            return update_state, total_loss

        update_state = (train_state, traj_batch, advantages, targets, rng)
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
        )
        train_state = update_state[0]
        rng = update_state[-1]

        runner_state = (train_state, env_state, last_obs, rng)
        return runner_state, loss_info
    return _update_step

def evaluate(new_state, old_state, network, env, rng_key, num_envs):
    auto_rese_step = auto_reset(env.step, env.init)
    _single_play_step = single_play_step(auto_rese_step, network, old_state)
    rng_key, sub_key = jax.random.split(rng_key)
    subkeys = jax.random.split(sub_key, num_envs)
    state = jax.vmap(env.init)(subkeys)
    cum_return = jnp.zeros(num_envs)
    i = 0
    step = jax.jit(jax.vmap(env.step))
    while not state.terminated.all():
        pi, value = network.apply(new_state.params, state.observation)
        actor = state.current_player
        action = pi.sample(seed=rng_key)
        state = step(state, action)
        cum_return += jax.vmap(_get)(state.rewards, actor)
    return cum_return.mean()


def train(config, rng):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    env = pgx.make(config["ENV_NAME"])

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    # INIT NETWORK
    network = ActorCritic(env.num_actions, activation=config["ACTIVATION"])
    rng, _rng = jax.random.split(rng)
    init_x = jnp.zeros((1, ) + env.observation_shape)
    network_params = network.init(_rng, init_x)
    if config["ANNEAL_LR"]:
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=linear_schedule, eps=1e-5),
        )
    else:
        tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(config["LR"], eps=1e-5))
    train_state = TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx,
    )

    # INIT UPDATE FUNCTION
    _update_step = make_update_fn(config, env, network)
    jitted_update_step = jax.jit(_update_step)

    # INIT ENV
    rng, _rng = jax.random.split(rng)
    reset_rng= jax.random.split(_rng, config["NUM_ENVS"])
    env_state = jax.vmap(env.init)(reset_rng)

    rng, _rng = jax.random.split(rng)
    runner_state = (train_state, env_state, env_state.observation, _rng)
    steps = 0
    for i in range(config["NUM_UPDATES"]):
        old_train_state = runner_state[0]
        runner_state, loss_info = jitted_update_step(runner_state, None)
        steps += config["NUM_ENVS"] * config["NUM_STEPS"]
        value_loss, loss_actor, entropy = loss_info
        eval_R = evaluate(runner_state[0], old_train_state, network, env, rng, config["NUM_ENVS"])
        log = {
            "value_loss": value_loss.mean(),
            "loss_actor": loss_actor.mean(),
            "entropy": entropy.mean(),
            "eval_vs_previous_policy": eval_R,
            "steps": steps,
        }
        print(log)
        wandb.log(log)
    return runner_state


if __name__ == "__main__":
    args = PPOConfig(**OmegaConf.to_object(OmegaConf.from_cli()))
    wandb.login(key="483ca3866ab4eaa8f523bacae3cb603d27d69c3d")
    wandb.init(project=f"a2c-minatar-v1", config=args.dict())
    config = {
        "LR": args.LR,
        "NUM_ENVS": args.NUM_ENVS,
        "NUM_STEPS": args.NUM_STEPS,
        "TOTAL_TIMESTEPS": args.TOTAL_TIMESTEPS,
        "UPDATE_EPOCHS": args.UPDATE_EPOCHS,
        "NUM_MINIBATCHES": args.NUM_MINIBATCHES,
        "GAMMA": args.GAMMA,
        "GAE_LAMBDA": args.GAE_LAMBDA,
        "CLIP_EPS": args.CLIP_EPS,
        "ENT_COEF": args.ENT_COEF,
        "VF_COEF": args.VF_COEF,
        "MAX_GRAD_NORM": args.MAX_GRAD_NORM,
        "ACTIVATION": args.ACTIVATION,
        "ENV_NAME": args.ENV_NAME,
        "ANNEAL_LR": True,
    }
    print("training of", config["ENV_NAME"])
    rng = jax.random.PRNGKey(0)
    sta = time.time()
    out = train(config, rng)
    end = time.time()
    print("training: time", end - sta)
    train_state, _, _, key = out["runner_state"]
    ckpt_filename = f'params/{config["ENV_NAME"]}_epochs_{config["UPDATE_EPOCHS"]}_batch_size_{config["NUM_ENVS"]}.ckpt'
    with open(ckpt_filename, "wb") as writer:
            dic = {"params": train_state.params}
            pickle.dump(dic, writer)