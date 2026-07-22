""" 
Based on PureJaxRL & jaxmarl Implementation of PPO
"""
import jax
import jax.numpy as jnp
import distrax
import optax
from flax.training.train_state import TrainState
# from flax.training import checkpoints
from gymnax.wrappers.purerl import LogWrapper
import socialjax
from socialjax.wrappers.baselines import LogWrapper
import hydra
from omegaconf import OmegaConf
import wandb
import copy

# Import shared network architectures
from algorithms.utils import (
    ActorCritic,
    batchify,
    batchify_dict,
    unbatchify,
    save_params,
    load_params,
    evaluate_ippo as evaluate,
    Transition,
)

def make_train(config):
    env = socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    # No auxiliary "model of other agents" network for the influence reward, in either
    # PARAMETER_SHARING mode (see ippo_cnn_coins.py for the original version of this):
    #  - PARAMETER_SHARING=True (influence/enabled_shared.yaml): one policy shared
    #    across agents already IS the exact model of every other agent.
    #  - PARAMETER_SHARING=False (influence/enabled_independent.yaml): agents keep
    #    separate policies, but each agent's reward computation is given direct read
    #    access to every other agent's params during centralized training (never at
    #    execution time) -- decentralized execution isn't broken since the reward is
    #    only ever used for the training update, not for acting.
    if config["PARAMETER_SHARING"]:
        config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    else:
        config["NUM_ACTORS"] = config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    env = LogWrapper(env, replace_info=False)

    rew_shaping_anneal = optax.linear_schedule(
        init_value=0.,
        end_value=1.,
        transition_steps=config["REW_SHAPING_HORIZON"],
        transition_begin=config["SHAPING_BEGIN"]
    )

    rew_shaping_anneal_org = optax.linear_schedule(
        init_value=1.,
        end_value=0.,
        transition_steps=config["REW_SHAPING_HORIZON"],
        transition_begin=config["SHAPING_BEGIN"]
    )
    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):

        # INIT NETWORK
        influence_reward = config.get("INFLUENCE_REWARD", False)
        action_dim = env.action_space().n
        if config["PARAMETER_SHARING"]:
            network = ActorCritic(env.action_space().n, activation=config["ACTIVATION"])
        else:
            network = [ActorCritic(env.action_space().n, activation=config["ACTIVATION"]) for _ in range(env.num_agents)]
        
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros((1, *(env.observation_space()[0]).shape))

        if config["PARAMETER_SHARING"]:
            network_params = network.init(_rng, init_x)
        else:
            network_params = [network[i].init(_rng, init_x) for i in range(env.num_agents)]
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        if config["PARAMETER_SHARING"]:
            train_state = TrainState.create(
                apply_fn=network.apply,
                params=network_params,
                tx=tx,
            )
        else:
            train_state = [TrainState.create(
                apply_fn=network[i].apply,
                params=network_params[i],
                tx=tx,
            ) for i in range(env.num_agents)]

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, update_step, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)

                
                # obs_batch = jnp.stack([last_obs[a] for a in env.agents]).reshape(-1, *env.observation_space().shape)
                
                if config["PARAMETER_SHARING"]:
                    obs_batch = jnp.transpose(last_obs,(1,0,2,3,4)).reshape(-1, *(env.observation_space()[0]).shape)
                    pi, value = network.apply(train_state.params, obs_batch)
                    action = pi.sample(seed=_rng)
                    log_prob = pi.log_prob(action)
                    env_act = unbatchify(
                        action, env.agents, config["NUM_ENVS"], env.num_agents
                    )
                else:
                    obs_batch = jnp.transpose(last_obs,(1,0,2,3,4))
                    env_act = {}
                    log_prob = []
                    value = []
                    pi_list = []
                    for i in range(env.num_agents):
                        pi, value_i = network[i].apply(train_state[i].params, obs_batch[i])
                        action = pi.sample(seed=_rng)
                        log_prob.append(pi.log_prob(action))
                        env_act[env.agents[i]] = action
                        value.append(value_i)
                        pi_list.append(pi)

                # env_act = {k: v.flatten() for k, v in env_act.items()}
                env_act = [v for v in env_act.values()]
                
                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                env_state_t = env_state  # pre-step state, needed below for counterfactuals

                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state_t, env_act)

                # SOCIAL INFLUENCE REWARD, MOA-free. No auxiliary network: "what would
                # agent j do differently" is answered by literally re-running env.step
                # with the SAME rng_step and SAME pre-step env_state_t, swapping only
                # agent k's action, and asking a real policy about the resulting real
                # (not approximated) observation. Costs num_agents x action_dim extra
                # env.step calls per real step -- with 7 agents x 8 actions that's 56,
                # noticeably heavier than coins' 14. Two variants depending on
                # architecture:
                #  - PARAMETER_SHARING=True: one network answers for every agent.
                #  - PARAMETER_SHARING=False: each agent keeps its own network, but the
                #    reward computation is given direct read access to every other
                #    agent's params (fine for a training-time-only quantity -- it never
                #    touches how actions get chosen, so decentralized execution still
                #    holds).
                if influence_reward:
                    obs_shape = (env.observation_space()[0]).shape
                    not_self = jnp.array(
                        [[j != k for j in range(env.num_agents)] for k in range(env.num_agents)]
                    )  # (k, j)

                    # Only count influence over agents k can actually SEE this step --
                    # with up to 7 agents on an open grid, most pairs are nowhere near
                    # each other at any given moment, and there's no causal story for
                    # "I influenced an agent whose observation couldn't possibly have
                    # been affected by anything I did." Visibility is derived from the
                    # env's own observation-window geometry (env.get_obs_point), not
                    # re-implemented: agent j is visible to k iff j's (padded) position
                    # falls inside the OBS_SIZE x OBS_SIZE crop centered (direction-
                    # adjusted) on k, exactly the same window _get_obs uses to build k's
                    # actual observation.
                    def _visibility(agent_locs):  # agent_locs: (num_agents, 3) for one env
                        start_x, start_y = jax.vmap(env.get_obs_point)(agent_locs)  # each (num_agents,)
                        px = agent_locs[:, 0] + env.PADDING
                        py = agent_locs[:, 1] + env.PADDING
                        in_x = (px[None, :] >= start_x[:, None]) & (px[None, :] < start_x[:, None] + env.OBS_SIZE)
                        in_y = (py[None, :] >= start_y[:, None]) & (py[None, :] < start_y[:, None] + env.OBS_SIZE)
                        return in_x & in_y  # (k, j)

                    visible = jax.vmap(_visibility)(env_state_t.env_state.agent_locs)  # (NUM_ENVS, k, j)

                    def _safe_kl(cond_probs, marginal_probs):
                        # As training progresses the policy gets more peaked (low
                        # ENT_COEF), and any category can underflow to exact float32
                        # 0.0 after enough softmax/einsum compounding. Categorical KL
                        # is log(p/q) under the hood -- an exact-zero q with nonzero p
                        # there blows up to +inf, poisoning the whole update (Adam's
                        # moments stay NaN forever once hit). Clip-and-renormalize
                        # keeps every category strictly positive so KL stays finite.
                        eps = 1e-6
                        cond_safe = cond_probs + eps
                        cond_safe = cond_safe / cond_safe.sum(-1, keepdims=True)
                        marginal_safe = marginal_probs + eps
                        marginal_safe = marginal_safe / marginal_safe.sum(-1, keepdims=True)
                        return distrax.Categorical(probs=cond_safe).kl_divergence(
                            distrax.Categorical(probs=marginal_safe)
                        )

                    if config["PARAMETER_SHARING"]:
                        act_probs = pi.probs.reshape(env.num_agents, config["NUM_ENVS"], action_dim)

                        # Real conditional: what agents actually do next, given the real
                        # joint action that was actually taken -- already have obsv for free.
                        obsv_flat = jnp.transpose(obsv, (1, 0, 2, 3, 4)).reshape(-1, *obs_shape)
                        cond_pi, _ = network.apply(train_state.params, obsv_flat)
                        cond_probs = cond_pi.probs.reshape(env.num_agents, config["NUM_ENVS"], action_dim)

                        influence = []
                        for k in range(env.num_agents):
                            def _cf_obs(a_idx, k=k):
                                cf_act = [
                                    jnp.where(i == k, jnp.full_like(env_act[i], a_idx), env_act[i])
                                    for i in range(env.num_agents)
                                ]
                                cf_obsv, _, _, _, _ = jax.vmap(
                                    env.step, in_axes=(0, 0, 0)
                                )(rng_step, env_state_t, cf_act)
                                return cf_obsv  # (NUM_ENVS, num_agents, *obs_shape)

                            cf_obsv_all = jax.vmap(_cf_obs)(jnp.arange(action_dim))
                            cf_obsv_flat = jnp.transpose(
                                cf_obsv_all, (0, 2, 1, 3, 4, 5)
                            ).reshape(-1, *obs_shape)
                            cf_pi, _ = network.apply(train_state.params, cf_obsv_flat)
                            cf_probs = cf_pi.probs.reshape(
                                action_dim, env.num_agents, config["NUM_ENVS"], action_dim
                            )

                            # Marginalize the counterfactuals over agent k's own real policy.
                            marginal_probs = jnp.einsum("ea,ajet->jet", act_probs[k], cf_probs)
                            kl_per_j = _safe_kl(cond_probs, marginal_probs)  # (num_agents, NUM_ENVS)

                            # visible[:, k, :] is (NUM_ENVS, j); transpose to (j, NUM_ENVS)
                            # to line up with kl_per_j, and AND in not_self so k is never
                            # counted as influencing itself even though k trivially sees k.
                            mask_k = visible[:, k, :].T & not_self[k][:, None]
                            influence.append(jnp.sum(jnp.where(mask_k, kl_per_j, 0.0), axis=0))
                    else:
                        # Same idea, but each agent j's conditional/counterfactual query
                        # goes through agent j's OWN network/params rather than one
                        # shared call -- that's literally agent j's real policy, so this
                        # is the exact (not approximated) model of every other agent.
                        cond_probs = jnp.stack(
                            [
                                network[j].apply(train_state[j].params, obsv[:, j])[0].probs
                                for j in range(env.num_agents)
                            ],
                            axis=0,
                        )  # (num_agents, NUM_ENVS, action_dim)

                        influence = []
                        for k in range(env.num_agents):
                            def _cf_obs(a_idx, k=k):
                                cf_act = [
                                    jnp.where(i == k, jnp.full_like(env_act[i], a_idx), env_act[i])
                                    for i in range(env.num_agents)
                                ]
                                cf_obsv, _, _, _, _ = jax.vmap(
                                    env.step, in_axes=(0, 0, 0)
                                )(rng_step, env_state_t, cf_act)
                                return cf_obsv  # (NUM_ENVS, num_agents, *obs_shape)

                            cf_obsv_all = jax.vmap(_cf_obs)(jnp.arange(action_dim))  # (a, NUM_ENVS, num_agents, *obs_shape)
                            cf_probs = jnp.stack(
                                [
                                    network[j].apply(
                                        train_state[j].params,
                                        cf_obsv_all[:, :, j].reshape(-1, *obs_shape),
                                    )[0].probs.reshape(action_dim, config["NUM_ENVS"], action_dim)
                                    for j in range(env.num_agents)
                                ],
                                axis=1,
                            )  # (a, j, e, t)

                            # Marginalize the counterfactuals over agent k's own real policy.
                            marginal_probs = jnp.einsum("ea,ajet->jet", pi_list[k].probs, cf_probs)
                            kl_per_j = _safe_kl(cond_probs, marginal_probs)  # (num_agents, NUM_ENVS)

                            mask_k = visible[:, k, :].T & not_self[k][:, None]
                            influence.append(jnp.sum(jnp.where(mask_k, kl_per_j, 0.0), axis=0))

                    influence = jnp.stack(influence, axis=0)  # (num_agents, NUM_ENVS)

                    # Constant weight, no curriculum ramp -- full strength from step 0.
                    beta = config["INFLUENCE_WEIGHT"]
                    done_env = done["__all__"]
                    influence = influence * (1.0 - done_env.astype(jnp.float32))[None, :]
                    reward = reward + beta * influence.T

                # current_timestep = update_step*config["NUM_STEPS"]*config["NUM_ENVS"]
                # shaped_reward = compute_grouped_rewards(reward)
                # reward = jax.tree.map(lambda x,y: x*rew_shaping_anneal_org(current_timestep)+y*rew_shaping_anneal(current_timestep), reward, shaped_reward)


                if config["PARAMETER_SHARING"]:
                    info = jax.tree.map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                    if influence_reward:
                        info["influence_reward"] = influence.T.reshape(-1)
                    transition = Transition(
                        batchify_dict(done, env.agents, config["NUM_ACTORS"]).squeeze(),
                        action,
                        value,
                        batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                        log_prob,
                        obs_batch,
                        info,
                        )
                else:
                    transition = []
                    done = [v for v in done.values()]
                    for i in range(env.num_agents):
                        info_i = {key: jax.tree.map(lambda x: x.reshape((config["NUM_ACTORS"]),1), value[:,i]) for key, value in info.items()}
                        if influence_reward:
                            info_i["influence_reward"] = influence[i].reshape((config["NUM_ACTORS"], 1))
                        transition.append(Transition(
                            done[i],
                            env_act[i],
                            value[i],
                            reward[:,i],
                            log_prob[i],
                            obs_batch[i],
                            info_i,
                        ))
                runner_state = (train_state, env_state, obsv, update_step, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, update_step, rng = runner_state
            if config["PARAMETER_SHARING"]:
                last_obs_batch = jnp.transpose(last_obs,(1,0,2,3,4)).reshape(-1, *(env.observation_space()[0]).shape)
                _, last_val = network.apply(train_state.params, last_obs_batch)
            else:
                last_obs_batch = jnp.transpose(last_obs,(1,0,2,3,4))
                last_val = []
                for i in range(env.num_agents):
                    _, last_val_i = network[i].apply(train_state[i].params, last_obs_batch[i])
                    last_val.append(last_val_i)
                last_val = jnp.stack(last_val, axis=0)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    # reward_mean = jnp.mean(reward, axis=0)
                    # # reward_std = jnp.std(reward, axis=0) + 1e-8
                    # reward = (reward - reward_mean)# / reward_std
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
            if config["PARAMETER_SHARING"]:
                advantages, targets = _calculate_gae(traj_batch, last_val)
            else:
                advantages = []
                targets = []
                for i in range(env.num_agents):
                    advantages_i, targets_i = _calculate_gae(traj_batch[i], last_val[i])
                    advantages.append(advantages_i)
                    targets.append(targets_i)
                advantages = jnp.stack(advantages, axis=0)
                targets = jnp.stack(targets, axis=0)
            # UPDATE NETWORK
            def _update_epoch(update_state, unused, i):
                def _update_minbatch(train_state, batch_info, network_used):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets, network_used):
                        # RERUN NETWORK
                        pi, value = network_used.apply(params, traj_batch.obs)
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
                            train_state.params, traj_batch, advantages, targets, network_used
                        )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ACTORS"]
                ), "batch size must be equal to number of steps * number of actors"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                        lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                    )
                # if config["PARAMETER_SHARING"]:
                    
                # else:
                #     batch = jax.tree_util.tree_map(
                #         lambda x: x.reshape((batch_size,) + x.shape[2:]),  # 保持第一个维度为batch_size，自动计算第二个维度
                #         batch
                #     )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                if config["PARAMETER_SHARING"]:
                    train_state, total_loss = jax.lax.scan(
                        lambda state, batch_info: _update_minbatch(state, batch_info, network), train_state, minibatches
                    )
                else:
                    train_state, total_loss = jax.lax.scan(
                        lambda state, batch_info: _update_minbatch(state, batch_info, network[i]), train_state, minibatches
                    )

                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss
            
            if config["PARAMETER_SHARING"]:
                update_state = (train_state, traj_batch, advantages, targets, rng)
                update_state, loss_info = jax.lax.scan(
                    lambda state, unused: _update_epoch(state, unused, 0), update_state, None, config["UPDATE_EPOCHS"]
                )
                train_state = update_state[0]
                metric = traj_batch.info
                rng = update_state[-1]
            else:
                update_state_dict = []
                metric = []
                for i in range(env.num_agents):
                    update_state = (train_state[i], traj_batch[i], advantages[i], targets[i], rng)
                    update_state, loss_info = jax.lax.scan(
                        lambda state, unused: _update_epoch(state, unused, i), update_state, None, config["UPDATE_EPOCHS"]
                    )
                    update_state_dict.append(update_state)
                    train_state[i] = update_state[0]
                    metric_i = traj_batch[i].info
                    metric_i['loss'] = loss_info[0]
                    metric.append(metric_i)
                    rng = update_state[-1]
                
            def callback(metric):
                wandb.log(metric)

            update_step = update_step + 1
            metric = jax.tree.map(lambda x: x.mean(), metric)
            if config["PARAMETER_SHARING"]:
                metric["update_step"] = update_step
                metric["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
                jax.debug.callback(callback, metric)
            else:
                for i in range(env.num_agents):
                    metric[i]["update_step"] = update_step
                    metric[i]["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
                metric = metric[0]
                jax.debug.callback(callback, metric)

            runner_state = (train_state, env_state, last_obs, update_step, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, 0, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train

# Used by algorithms/train.py to dispatch through algorithms.IPPO._runner.
SINGLE_RUN_KWARGS = {"wandb_name": "ippo_cnn_harvest_open"}
TUNE_KWARGS       = {"sweep_name": "harvest_open"}
