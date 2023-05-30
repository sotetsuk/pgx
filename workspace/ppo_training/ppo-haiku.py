import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Literal
from flax.training.train_state import TrainState
import distrax
import pgx
from utils import auto_reset, single_play_step_vs_policy_in_backgammon, single_play_step_vs_policy_in_two, normal_step
import time
import os

import pickle
import argparse
from omegaconf import OmegaConf
from pydantic import BaseModel
import wandb


class PPOConfig(BaseModel):
    ENV_NAME: Literal[ 
        "leduc_holdem", 
        "kuhn_poker", 
        "minatar-breakout", 
        "minatar-freeway", 
        "minatar-space_invaders", 
        "minatar-asterix", 
        "minatar-seaquest", 
        "play2048",
        "backgammon"
        ] = "backgammon"
    LR: float = 2.5e-4
    NUM_ENVS: int = 64
    NUM_STEPS: int = 256
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
    VS_RANDOM: bool = False
    UPDATE_INTERVAL:int = 5
    MAKE_ANCHOR: bool = True


env = pgx.make(config.env_name)


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"
    env_name: str = "backgammon"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        if self.env_name not in ["backgammon", "2048", "kuhn_poker", "leduc_holdem"]:
            x = hk.Conv2d(features=32, kernel_shap=2)(x)
            x = nn.relu(x)
            x = hk.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
            x = x.reshape((x.shape[0], -1))  # flatten
        x = hk.Linear(64)(x)
        x = nn.relu(x)
        actor_mean = hk.Linear(
            64
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = hk.Linear(
            64
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = hk.Linear(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        critic = hk.Linear(
            64
        )(x)
        critic = activation(critic)
        critic = hk.Linear(
            64
        )(critic)
        critic = activation(critic)
        critic = hk.Linear(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return actor_mean, jnp.squeeze(critic, axis=-1)

def forward_pass(x, is_eval=False):
    net = ActorCritic(env.num_actions, activation=config.activation, env_name=config.env_name)
    logits, value = net(x, is_training=not is_eval, test_local_stats=False)
    return logits, value
forward_pass = hk.without_apply_rng(hk.transform_with_state(forward_pass))
optimizer = optax.adam(learning_rate=config.learning_rate)


def _make_step(env_name, network, params,):  # TODO
    env = pgx.make(env_name)
    auto_rese_step = auto_reset(env.step, env.init)
    if env_name == "backgammon":
        return single_play_step_vs_policy_in_backgammon(auto_rese_step, network, params)
    elif env_name in ["kuhn_poker", "leduc_holdem"]:
        return single_play_step_vs_policy_in_two(auto_rese_step, network, params)
    else:
        return normal_step(auto_rese_step)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    legal_action_mask: jnp.ndarray


def make_update_fn(config):
     # TRAIN LOOP
    def _update_step(runner_state):
        # COLLECT TRAJECTORIES
        step_fn = _make_step(config["ENV_NAME"], network, runner_state[0].params)  # TODO
        get_fn = _get if config["ENV_NAME"] in ["backgammon", "leduc_holdem", "kuhn_poker"] else _get_zero
        def _env_step(runner_state, unused):
            model, opt_state, env_state, last_obs, rng = runner_state  # DONE
            model_params, model_state = model
            actor = env_state.current_player
            # SELECT ACTION
            rng, _rng = jax.random.split(rng)
            logits, value = forward_pass.apply(model_params, model_state, last_obs, is_eval=True)  # DONE
            mask = env_state.legal_action_mask
            logits = logits + jnp.finfo(np.float64).min * (~mask)
            pi = distrax.Categorical(logits=logits)
            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            env_state = step_fn(
                env_state, action, _rng
            )
            transition = Transition(
                env_state.terminated, action, value, jax.vmap(get_fn)(env_state.rewards, actor), log_prob, last_obs, mask
            )
            runner_state = (model, opt_state, env_state, env_state.observation, rng)  # DONE
            return runner_state, transition

        runner_state, traj_batch = jax.lax.scan(
            _env_step, runner_state, None, config["NUM_STEPS"]
        )

        # CALCULATE ADVANTAGE
        model, opt_state, env_state, last_obs, rng = runner_state  # DONE
        model_params, model_state = model
        _, last_val = forward_pass.apply(model_params, model_state, last_obs, is_eval=False)  # DONE

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
            def _update_minbatch(model, opt_state, batch_info):
                traj_batch, advantages, targets = batch_info
                model_params, model_state = model

                def _loss_fn(model_params, model_state,  traj_batch, gae, targets):
                    # RERUN NETWORK
                    logits, value = ne.apply(model_params, model_state, traj_batch.obs, eval=False)  # DONE
                    mask = traj_batch.legal_action_mask
                    logits = logits + jnp.finfo(np.float64).min * (~mask)
                    pi = distrax.Categorical(logits=logits)
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
                    model_params, model_state, traj_batch, advantages, targets
                )  # DONE
                updates, opt_state = optimizer.update(grads, opt_state)
                model_params = optax.apply_updates(model_params, updates)  # DONE
                return (model_params, model_state), opt_state, total_loss  # DONE

            model, opt_state, traj_batch, advantages, targets, rng = update_state  # DONE
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
            model, opt_state, total_loss = jax.lax.scan(
                _update_minbatch, model, opt_state, minibatches
            )  # DONE
            update_state = (model, opt_state, traj_batch, advantages, targets, rng)  # DONE
            return update_state, total_loss

        update_state = (model, opt_state, traj_batch, advantages, targets, rng)  # DONE
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
        )
        model, opt_state , _, _, _, rng = update_state  # DONE

        runner_state = (model, opt_state, env_state, last_obs, rng)  # DONE
        return runner_state, loss_info
    return _update_step


def _get(x, i):
    return x[i]


def _get_zero(x, i):
    return x[0]


def evaluate(params, network, step_fn,  env, rng_key, config):
    rng_key, sub_key = jax.random.split(rng_key)
    subkeys = jax.random.split(sub_key, config["NUM_ENVS"])
    state = jax.vmap(env.init)(subkeys)
    cum_return = jnp.zeros(config["NUM_ENVS"])
    get_fn = _get if config["ENV_NAME"] in ["backgammon", "leduc_holdem", "kuhn_poker"] else _get_zero
    i = 0
    states = []
    def cond_fn(tup):
        state, _, _ = tup
        return ~state.terminated.all()
    def loop_fn(tup):
        state, cum_return, rng_key = tup
        actor = state.current_player
        logits, value = network.apply(params, state.observation)  # TODO
        logits = logits + jnp.finfo(np.float64).min * (~state.legal_action_mask)
        pi = distrax.Categorical(logits=logits)
        rng_key, _rng = jax.random.split(rng_key)
        action = pi.sample(seed=_rng)
        rng_key, _rng = jax.random.split(rng_key)
        state = step_fn(state, action, _rng)
        cum_return = cum_return + jax.vmap(get_fn)(state.rewards, actor)
        return state, cum_return ,rng_key
    state, cum_return, _ = jax.lax.while_loop(cond_fn, loop_fn, (state, cum_return, rng_key))
    return cum_return.mean()


def train(config, rng):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    # INIT NETWORK
    rng, _rng = jax.random.split(rng)
    init_x = jnp.zeros((1, ) + env.observation_shape)
    model = forward_pass.init(_rng, init_x)  # (params, state)  # DONE
    opt_state = optimizer.init(params=model[0])  # DONE
    

    # INIT UPDATE FUNCTION
    _update_step = make_update_fn(config)  # DONE
    jitted_update_step = jax.jit(_update_step)

    # INIT ENV
    rng, _rng = jax.random.split(rng)
    reset_rng= jax.random.split(_rng, config["NUM_ENVS"])
    env_state = jax.vmap(env.init)(reset_rng)

    rng, _rng = jax.random.split(rng)
    runner_state = (train_state, env_state, env_state.observation, _rng)  # TODO

    ckpt_params = None
    ckpt_filename = f'params/{config["ENV_NAME"]}/anchor.ckpt'
    if ckpt_filename != "" and os.path.isfile(ckpt_filename):
        with open(ckpt_filename, "rb") as reader:
            dic = pickle.load(reader)
        ckpt_params = dic["params"]
    
    steps = 0
    for i in range(config["NUM_UPDATES"]):
        if ckpt_params is not None:
            step_fn = _make_step(config["ENV_NAME"], network, ckpt_params, eval=True)  # TODO
            eval_R = evaluate(runner_state[0].params, network, step_fn, env, rng, config)   # TODO
        else:
            step_fn = _make_step(config["ENV_NAME"], network, runner_state[0].params)  # TODO
            eval_R = evaluate(runner_state[0].params, network, step_fn, env, rng, config, eval=True) # TODO
        log = {
            f"eval_R{config['ENV_NAME']}": float(eval_R),
            "steps": steps,
        }
        print(log)
        wandb.log(log)
        runner_state, loss_info = jitted_update_step(runner_state)  # TODO
        steps += config["NUM_ENVS"] * config["NUM_STEPS"]
        _, (value_loss, loss_actor, entropy) = loss_info
    return runner_state


if __name__ == "__main__":
    args = PPOConfig(**OmegaConf.to_object(OmegaConf.from_cli()))
    key = "483ca3866ab4eaa8f523bacae3cb603d27d69c3d" # please specify your wandb key
    wandb.login(key=key)
    mode = "make-anchor" if args.MAKE_ANCHOR else "train"
    wandb.init(project=f"ppo-{mode}", config=args.dict())
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
        "VS_RANDOM": args.VS_RANDOM,
        "UPDATE_INTERVAL": args.UPDATE_INTERVAL,
    }
    if config["ENV_NAME"] == "play2048":
        config["ENV_NAME"] = "2048"
    print("training of", config["ENV_NAME"])
    rng = jax.random.PRNGKey(0)
    sta = time.time()
    out = train(config, rng)
    end = time.time()
    print("training: time", end - sta)
    train_state, _, _, key = out["runner_state"]