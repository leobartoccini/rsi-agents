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
    ActorCriticMOARNN,
    ActorCriticLSTM,
    batchify,
    batchify_dict,
    unbatchify,
    save_params,
    load_params,
    evaluate_ippo as evaluate,
    Transition,
    MOATransition,
)


def reset_hstate(carry, done_env):
    """Zero an LSTM carry wherever done_env is True. done_env: (B,), broadcasts against
    carry=(c,h) each (B, hidden_dim). Episodes in this env reset synchronously for every
    agent, so a single per-environment boolean is the correct reset signal (see
    LogWrapper's docstring)."""
    c, h = carry
    mask = done_env[:, None]
    return (jnp.where(mask, 0.0, c), jnp.where(mask, 0.0, h))


def make_train(config):
    env = socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    # Four ways to get a social-influence reward here, picked by config flags:
    #  - PARAMETER_SHARING=True + INFLUENCE_REWARD=True (influence/enabled_shared.yaml):
    #    no MOA network -- the one shared policy already IS the exact model of every
    #    other agent.
    #  - PARAMETER_SHARING=False + INFLUENCE_REWARD=True (influence/enabled_independent.yaml):
    #    no MOA network either -- each agent's reward computation is given direct read
    #    access to every other agent's own params during centralized training.
    #  - PARAMETER_SHARING=False + INFLUENCE_REWARD=True + RECURRENT_MOA=True
    #    (influence/enabled_recurrent.yaml): the actual Jaques et al. MOA -- a recurrent
    #    auxiliary head trained to predict other agents' next actions from the ego
    #    agent's own observation, exactly like ippo_cnn_cleanup.py.
    #  - PARAMETER_SHARING=False + INFLUENCE_REWARD=True + LSTM_INFLUENCE=True
    #    (influence/enabled_lstm.yaml): ActorCriticLSTM -- the MAIN policy/value
    #    pathway itself (not a separate auxiliary head) conditions on the other
    #    agent's previous action, so the "real" conditional is just the ordinary
    #    policy output and the counterfactual is the same network re-called with a
    #    hypothetical previous action -- no MOA loss, no env.step resimulation.
    if config.get("RECURRENT_MOA", False):
        assert config.get("INFLUENCE_REWARD", False) and not config["PARAMETER_SHARING"], (
            "RECURRENT_MOA requires INFLUENCE_REWARD=True and PARAMETER_SHARING=False "
            "(see influence/enabled_recurrent.yaml)."
        )
    if config.get("LSTM_INFLUENCE", False):
        assert config.get("INFLUENCE_REWARD", False) and not config["PARAMETER_SHARING"], (
            "LSTM_INFLUENCE requires INFLUENCE_REWARD=True and PARAMETER_SHARING=False "
            "(see influence/enabled_lstm.yaml)."
        )
        assert not config.get("RECURRENT_MOA", False), (
            "LSTM_INFLUENCE and RECURRENT_MOA select different network architectures "
            "for the same INFLUENCE_REWARD flag -- pick one, not both."
        )
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
        recurrent_moa = config.get("RECURRENT_MOA", False)
        lstm_influence = config.get("LSTM_INFLUENCE", False)
        hidden_dim = config.get("LSTM_HIDDEN_DIM", 128)
        action_dim = env.action_space().n
        if config["PARAMETER_SHARING"]:
            network = ActorCritic(action_dim, activation=config["ACTIVATION"])
        elif recurrent_moa:
            network = [ActorCriticMOARNN(
                action_dim, num_agents=env.num_agents, hidden_dim=hidden_dim, activation=config["ACTIVATION"]
            ) for _ in range(env.num_agents)]
        elif lstm_influence:
            # NORMAL vs LSTM: ActorCritic is stateless -- output depends only on the
            # current observation. ActorCriticLSTM additionally carries a recurrent
            # hidden state and accepts the *other agents' previous joint action* as an
            # extra input, so the main policy itself (not an auxiliary head) already
            # conditions on what everyone did last step. This is what lets the
            # influence-reward computation below skip resimulating env.step for its
            # counterfactuals -- see the `elif lstm_influence:` block in _env_step.
            network = [ActorCriticLSTM(
                action_dim, num_agents=env.num_agents, hidden_dim=hidden_dim, activation=config["ACTIVATION"]
            ) for _ in range(env.num_agents)]
        else:
            network = [ActorCritic(action_dim, activation=config["ACTIVATION"]) for _ in range(env.num_agents)]

        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros((1, *(env.observation_space()[0]).shape))
        init_joint_action = jnp.zeros((1, env.num_agents, action_dim))

        if config["PARAMETER_SHARING"]:
            network_params = network.init(_rng, init_x)
        elif recurrent_moa:
            init_ph_1 = ActorCriticMOARNN.initialize_carry(1, hidden_dim)
            init_mh_1 = ActorCriticMOARNN.initialize_carry(1, hidden_dim)
            network_params = [
                network[i].init(_rng, init_ph_1, init_mh_1, init_x, init_joint_action, method=network[i].init_all)
                for i in range(env.num_agents)
            ]
        elif lstm_influence:
            # NORMAL vs LSTM: network[i].init() needs an example carry (init_ph_1) and
            # an example joint-action input (init_joint_action) to trace the shapes of
            # the recurrent weights, in addition to the example observation (init_x).
            # Batch size 1 here is only for shape-tracing -- it does not need to match
            # NUM_ENVS. Normal's init only ever needed init_x, since ActorCritic has no
            # recurrent state or extra input.
            init_ph_1 = ActorCriticLSTM.initialize_carry(1, hidden_dim)
            network_params = [
                network[i].init(_rng, init_ph_1, init_x, init_joint_action)
                for i in range(env.num_agents)
            ]
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

        if recurrent_moa:
            policy_hstate = [ActorCriticMOARNN.initialize_carry(config["NUM_ENVS"], hidden_dim) for _ in range(env.num_agents)]
            moa_hstate = [ActorCriticMOARNN.initialize_carry(config["NUM_ENVS"], hidden_dim) for _ in range(env.num_agents)]
        elif lstm_influence:
            # NORMAL vs LSTM: this whole block has no counterpart in Normal, because a
            # stateless policy has nothing to carry between timesteps. LSTM needs three
            # pieces of memory:
            #   - policy_hstate: the recurrent carry (c, h) per agent, batch = NUM_ENVS
            #     (distinct from init_ph_1 above, which was batch=1 and only used for
            #     shape-tracing).
            #   - prev_joint_action: the discrete action indices fed into the policy as
            #     "what did everyone do last step" -- sentinel -1 (not 0) means "no
            #     valid previous action."
            #   - prev_action_probs: the distribution the marginalization in the
            #     `elif lstm_influence:` reward block needs to weight counterfactuals by
            #     agent k's REAL policy from the step that produced prev_joint_action.
            #     Uniform (1/action_dim) is the neutral prior for "no real policy exists
            #     yet to weight by."
            policy_hstate = [ActorCriticLSTM.initialize_carry(config["NUM_ENVS"], hidden_dim) for _ in range(env.num_agents)]
            # No real action exists before the first step. -1 is an out-of-range
            # sentinel: jax.nn.one_hot(-1, action_dim) is a true all-zero vector,
            # unlike one_hot(0, ...) which is [1,0,...,0] -- a real (fake) claim that
            # every agent chose action 0. Using 0 here would make "episode just
            # started" and "everyone specifically picked action 0" indistinguishable
            # to the network, a systematic false signal repeated at every episode
            # boundary throughout training. Also use a uniform prior over what agent
            # k's policy would have said, so the very first step's reward doesn't
            # spuriously credit/blame a fictitious action.
            prev_joint_action = jnp.full((config["NUM_ENVS"], env.num_agents), -1, dtype=jnp.int32)
            prev_action_probs = jnp.full((env.num_agents, config["NUM_ENVS"], action_dim), 1.0 / action_dim)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            if recurrent_moa:
                # Snapshot the carries entering this rollout window -- they were already
                # correctly reset in real time during the previous window's collection, so
                # the loss's time-scan can use them as-is with no reset check at step 0.
                _, _, _, init_policy_hstate, init_moa_hstate, _, _ = runner_state
            elif lstm_influence:
                _, _, _, init_policy_hstate, _, _, _, _ = runner_state

            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                if recurrent_moa:
                    train_state, env_state, last_obs, policy_hstate, moa_hstate, update_step, rng = runner_state
                elif lstm_influence:
                    train_state, env_state, last_obs, policy_hstate, prev_joint_action, prev_action_probs, update_step, rng = runner_state
                else:
                    train_state, env_state, last_obs, update_step, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)

                if config["PARAMETER_SHARING"]:
                    obs_batch = jnp.transpose(last_obs,(1,0,2,3,4)).reshape(-1, *(env.observation_space()[0]).shape)
                    pi, value = network.apply(train_state.params, obs_batch)
                    action = pi.sample(seed=_rng)
                    log_prob = pi.log_prob(action)
                    env_act = unbatchify(
                        action, env.agents, config["NUM_ENVS"], env.num_agents
                    )
                elif recurrent_moa:
                    obs_batch = jnp.transpose(last_obs,(1,0,2,3,4))
                    env_act = {}
                    log_prob = []
                    value = []
                    pi_list = []
                    new_policy_hstate = []
                    for i in range(env.num_agents):
                        rng, agent_rng = jax.random.split(rng)
                        new_ph_i, (pi, value_i) = network[i].apply(train_state[i].params, policy_hstate[i], obs_batch[i])
                        action = pi.sample(seed=agent_rng)
                        log_prob.append(pi.log_prob(action))
                        env_act[env.agents[i]] = action
                        value.append(value_i)
                        pi_list.append(pi)
                        new_policy_hstate.append(new_ph_i)
                elif lstm_influence:
                    # NORMAL vs LSTM, per line:
                    #  - prev_joint_action_onehot: computed once, outside the per-agent
                    #    loop, since every agent's forward pass needs to see it
                    #    (including its own previous action). one_hot(-1, action_dim) is
                    #    all-zero at t=0/post-reset -- see the runtime-state-init block
                    #    above.
                    #  - rng, agent_rng = jax.random.split(rng)  [per agent, inside the
                    #    loop]: Normal reuses the SAME _rng for every agent's
                    #    pi.sample() call, correlating their samples. LSTM splits a
                    #    fresh key per agent -- the more correct JAX pattern, avoiding
                    #    that correlation.
                    #  - network[i].apply(params, policy_hstate[i], obs_batch[i],
                    #    prev_joint_action_onehot): two extra args vs Normal's
                    #    apply(params, obs_batch[i]) -- the incoming recurrent carry and
                    #    the previous joint action. Return shape also changes: Normal
                    #    returns (pi, value_i) flat; LSTM returns
                    #    (new_carry, (pi, value_i)) nested, because any recurrent cell
                    #    must hand back its updated carry alongside the normal output.
                    #  - new_policy_hstate.append(new_ph_i): Normal has nothing to
                    #    accumulate here: no state to propagate to the next timestep.
                    #    LSTM collects each agent's updated carry so it can be threaded
                    #    into the next _env_step call (see the runner_state repack at
                    #    the end of _env_step).
                    obs_batch = jnp.transpose(last_obs,(1,0,2,3,4))
                    prev_joint_action_onehot = jax.nn.one_hot(prev_joint_action, action_dim)
                    env_act = {}
                    log_prob = []
                    value = []
                    pi_list = []
                    new_policy_hstate = []
                    for i in range(env.num_agents):
                        rng, agent_rng = jax.random.split(rng)
                        new_ph_i, (pi, value_i) = network[i].apply(
                            train_state[i].params, policy_hstate[i], obs_batch[i], prev_joint_action_onehot
                        )
                        action = pi.sample(seed=agent_rng)
                        log_prob.append(pi.log_prob(action))
                        env_act[env.agents[i]] = action
                        value.append(value_i)
                        pi_list.append(pi)
                        new_policy_hstate.append(new_ph_i)
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

                # SOCIAL INFLUENCE REWARD -- recurrent MOA variant. Fully computable from
                # info already available at time t (own obs, real joint action, own
                # policy) -- doesn't need to wait for env.step's output, unlike the
                # MOA-free variants below which need the real/counterfactual next obs.
                if recurrent_moa:
                    joint_action = jnp.stack(env_act, axis=-1)  # (NUM_ENVS, num_agents)
                    joint_action_onehot = jax.nn.one_hot(joint_action, action_dim)

                    current_timestep = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
                    beta = config["INFLUENCE_WEIGHT"] * rew_shaping_anneal(current_timestep)

                    influence = []
                    new_moa_hstate = []
                    for k in range(env.num_agents):
                        params_k = train_state[k].params

                        # The action feeds into the MOA LSTM itself (paper Fig. 6), so a
                        # counterfactual action changes the LSTM's output, not just a
                        # downstream head -- every candidate action needs its own
                        # moa_step call. moa_features (CNN+FC+FC) is action-independent
                        # though, so it's computed once and reused for the real query and
                        # all counterfactuals below, rather than rerunning the CNN.
                        moa_feats_k = network[k].apply(
                            params_k, obs_batch[k], method=network[k].moa_features
                        )
                        new_mh_k, cond_logits = network[k].apply(
                            params_k, moa_hstate[k], moa_feats_k, joint_action_onehot,
                            method=network[k].moa_step,
                        )
                        new_moa_hstate.append(new_mh_k)

                        def _counterfactual_logits(a_idx, k=k, params_k=params_k, moa_feats_k=moa_feats_k):
                            cf_onehot = jax.nn.one_hot(a_idx, action_dim)
                            cf_joint = joint_action_onehot.at[:, k, :].set(cf_onehot)
                            # Uses the incoming (pre-this-step) moa_hstate[k], same as the
                            # real query -- "what if k had acted differently right now,
                            # holding everything remembered before now fixed." The
                            # counterfactual carry is discarded, never stored.
                            _, cf_logits = network[k].apply(
                                params_k, moa_hstate[k], moa_feats_k, cf_joint,
                                method=network[k].moa_step,
                            )
                            return cf_logits

                        cond_probs = jax.nn.softmax(cond_logits, axis=-1)
                        cf_logits = jax.vmap(_counterfactual_logits)(jnp.arange(action_dim))
                        cf_probs = jax.nn.softmax(cf_logits, axis=-1)

                        # Marginalize the counterfactuals over agent k's own real policy.
                        marginal_probs = jnp.einsum(
                            "ea,aejt->ejt", pi_list[k].probs, cf_probs
                        )

                        kl_per_j = distrax.Categorical(probs=cond_probs).kl_divergence(
                            distrax.Categorical(probs=marginal_probs)
                        )  # (NUM_ENVS, num_agents)

                        not_self = jnp.array([j != k for j in range(env.num_agents)])
                        influence.append(jnp.sum(jnp.where(not_self, kl_per_j, 0.0), axis=-1))

                    influence = jnp.stack(influence, axis=-1)  # (NUM_ENVS, num_agents)
                elif lstm_influence:
                    # SOCIAL INFLUENCE REWARD -- LSTM variant. No auxiliary network at
                    # all: the MAIN policy already conditions on the previous timestep's
                    # real joint action (fed in above for action selection), so agent j's
                    # REAL conditional is just pi_list[j] -- already computed for free.
                    # The counterfactual re-calls each agent j's SAME network with the
                    # SAME incoming carry/obs but agent k's slot in that previous-action
                    # input swapped to a hypothetical action -- no env.step needed at all.
                    cond_probs = jnp.stack(
                        [pi_list[j].probs for j in range(env.num_agents)], axis=0
                    )  # (j, e, t) -- REAL

                    current_timestep = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
                    beta = config["INFLUENCE_WEIGHT"] * rew_shaping_anneal(current_timestep)

                    def _safe_kl_lstm(cond_probs, marginal_probs):
                        eps = 1e-6
                        cond_safe = cond_probs + eps
                        cond_safe = cond_safe / cond_safe.sum(-1, keepdims=True)
                        marginal_safe = marginal_probs + eps
                        marginal_safe = marginal_safe / marginal_safe.sum(-1, keepdims=True)
                        return distrax.Categorical(probs=cond_safe).kl_divergence(
                            distrax.Categorical(probs=marginal_safe)
                        )

                    def _apply_pi_probs(j, other_action_onehot):
                        _, (pi_j, _) = network[j].apply(
                            train_state[j].params, policy_hstate[j], obs_batch[j], other_action_onehot
                        )
                        return pi_j.probs

                    influence = []
                    for k in range(env.num_agents):
                        def _cf_probs(a_idx, k=k):
                            cf_onehot = jax.nn.one_hot(a_idx, action_dim)
                            cf_joint = prev_joint_action_onehot.at[:, k, :].set(cf_onehot)
                            return jnp.stack(
                                [_apply_pi_probs(j, cf_joint) for j in range(env.num_agents)], axis=0
                            )  # (j, e, t)

                        cf_probs = jax.vmap(_cf_probs)(jnp.arange(action_dim))  # (a, j, e, t)
                        # Marginalize the counterfactuals over agent k's REAL policy from
                        # the step that actually produced prev_joint_action.
                        marginal_probs = jnp.einsum("ea,ajet->jet", prev_action_probs[k], cf_probs)
                        kl_per_j = _safe_kl_lstm(cond_probs, marginal_probs)  # (num_agents, NUM_ENVS)

                        not_self = jnp.array([j != k for j in range(env.num_agents)])
                        influence.append(jnp.sum(jnp.where(not_self[:, None], kl_per_j, 0.0), axis=0))

                    influence = jnp.stack(influence, axis=0)  # (num_agents=k, NUM_ENVS)

                    # NORMAL vs LSTM (Normal's MOA-free counterpart lives in the
                    # `elif influence_reward:` branch further down, after env.step):
                    # both compute the same Jaques et al. causal-influence KL, but they
                    # get the real/counterfactual quantities completely differently.
                    #
                    #  - cond_probs: Normal recomputes this by calling the network on
                    #    the REAL obs that resulted from env.step -- i.e. it waits for
                    #    the environment to tell it what happened. LSTM just reuses
                    #    pi_list[j], already computed above during action selection,
                    #    because the policy already conditioned on the real previous
                    #    joint action as an input -- nothing new needs to be asked of
                    #    the environment.
                    #
                    #  - counterfactual generation (_cf_probs vs Normal's _cf_obs):
                    #    Normal's _cf_obs calls jax.vmap(env.step) with the SAME
                    #    rng_step/env_state_t but agent k's action swapped -- a real
                    #    resimulation of physics. LSTM's _cf_probs never touches the
                    #    environment: it swaps agent k's slot in
                    #    prev_joint_action_onehot and re-calls the SAME network with the
                    #    SAME obs/carry. The "counterfactual" here is a hypothetical
                    #    belief fed to the policy, not a hypothetical physical outcome.
                    #
                    #  - marginal_probs einsum weight: Normal weights by
                    #    pi_list[k].probs (k's policy at THIS timestep, matching
                    #    cond_probs which is anchored to this timestep's real obs).
                    #    LSTM weights by prev_action_probs[k] (k's policy from the
                    #    PREVIOUS timestep, matching cond_probs which is anchored to the
                    #    previous action). This keeps both sides of the KL referring to
                    #    the same moment in time.
                    #
                    #  - beta uses rew_shaping_anneal(current_timestep) here
                    #    (curriculum-gated) -- Normal's `elif influence_reward:` branch
                    #    further down computes current_timestep the same way but never
                    #    multiplies beta by the anneal schedule. That asymmetry is a
                    #    pre-existing inconsistency in the file, not an intentional
                    #    design choice; if fixing it, apply the same
                    #    `* rew_shaping_anneal(current_timestep)` there too.
                    #
                    #  - cost: Normal pays action_dim extra env.step calls PER agent k,
                    #    per timestep -- expensive but exact. LSTM pays action_dim extra
                    #    network forward passes per agent k -- much cheaper, but the
                    #    resulting "influence" is always one step stale relative to the
                    #    action that produced it (see cond_probs note above).

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                env_state_t = env_state  # pre-step state, needed below for the MOA-free counterfactuals

                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state_t, env_act)

                if recurrent_moa:
                    done_env = done["__all__"]
                    # Don't reward "influence" on the step the episode ends -- the
                    # relationship/context between agents is about to reset, so a KL
                    # divergence computed right at that boundary isn't meaningful. Rare
                    # here (NUM_STEPS << num_inner_steps, so most windows never see a
                    # done step at all) but cheap and more correct.
                    reward = reward + beta * influence * (1.0 - done_env.astype(jnp.float32))[:, None]
                    # Auto-reset semantics: the obs env.step returns when done is True is
                    # already the fresh post-reset observation, so the carry entering the
                    # *next* step must be zeroed now, using *this* step's done.
                    policy_hstate_next = [reset_hstate(h, done_env) for h in new_policy_hstate]
                    moa_hstate_next = [reset_hstate(h, done_env) for h in new_moa_hstate]
                elif lstm_influence:
                    done_env = done["__all__"]
                    influence = influence * (1.0 - done_env.astype(jnp.float32))[None, :]
                    reward = reward + beta * influence.T
                    policy_hstate_next = [reset_hstate(h, done_env) for h in new_policy_hstate]
                    joint_action = jnp.stack(env_act, axis=-1)  # (NUM_ENVS, num_agents) -- THIS step's real action
                    # -1 (not 0) on reset -- see the sentinel comment at initialization.
                    prev_joint_action_next = jnp.where(done_env[:, None], -1, joint_action)
                    prev_action_probs_next = jnp.where(
                        done_env[None, :, None], jnp.full_like(cond_probs, 1.0 / action_dim), cond_probs
                    )
                    # NORMAL vs LSTM: the first two lines above (masking influence by
                    # done_env, adding it into reward) are IDENTICAL in spirit to
                    # Normal's `elif influence_reward:` branch further down -- don't
                    # reward influence on the step an episode ends, since the KL at
                    # that boundary isn't meaningful (the agent relationship is about
                    # to reset).
                    #
                    # Everything below that is LSTM-only, because Normal carries
                    # nothing between steps and so has nothing to reset when an episode
                    # boundary hits mid-rollout:
                    #  - policy_hstate_next: zero the recurrent carry wherever
                    #    done_env is True. Auto-reset means the obs env.step returns on
                    #    a done step is ALREADY the fresh post-reset observation, so the
                    #    carry entering the *next* step must be zeroed using THIS step's
                    #    done, not the next one.
                    #  - prev_joint_action_next: normally this would just be
                    #    joint_action (this step's real action). But if the episode
                    #    just ended, that action belongs to the OLD episode -- feeding
                    #    it forward as "previous action" into a brand-new episode would
                    #    leak stale context across an episode boundary. Reset to the
                    #    same -1 sentinel used at training-start init, since a fresh
                    #    episode has no more valid "previous action" than the very first
                    #    step of training did.
                    #  - prev_action_probs_next: same idea for the marginalization
                    #    weight -- cond_probs (this step's real policy) is what would
                    #    normally be carried forward, but on a done step it's replaced
                    #    with the same uniform prior used at init, for the same reason:
                    #    no real policy exists yet within the new episode to weight by.
                elif influence_reward:
                    # SOCIAL INFLUENCE REWARD, MOA-free. No auxiliary network: "what
                    # would agent j do differently" is answered by literally re-running
                    # env.step with the SAME rng_step and SAME pre-step env_state_t,
                    # swapping only agent k's action, and asking a real policy about the
                    # resulting real (not approximated) observation. Two variants
                    # depending on architecture:
                    #  - PARAMETER_SHARING=True: one network answers for every agent.
                    #  - PARAMETER_SHARING=False: each agent keeps its own network, but
                    #    the reward computation is given direct read access to every
                    #    other agent's params (fine for a training-time-only quantity --
                    #    it never touches how actions get chosen, so decentralized
                    #    execution still holds).
                    obs_shape = (env.observation_space()[0]).shape
                    not_self_mat = jnp.array(
                        [[j != k for j in range(env.num_agents)] for k in range(env.num_agents)]
                    )  # (k, j)

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

                            influence.append(
                                jnp.sum(jnp.where(not_self_mat[k][:, None], kl_per_j, 0.0), axis=0)
                            )
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

                            influence.append(
                                jnp.sum(jnp.where(not_self_mat[k][:, None], kl_per_j, 0.0), axis=0)
                            )

                    influence = jnp.stack(influence, axis=0)  # (num_agents, NUM_ENVS)

                    current_timestep = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
                    beta = config["INFLUENCE_WEIGHT"]
                    done_env = done["__all__"]
                    influence = influence * (1.0 - done_env.astype(jnp.float32))[None, :]
                    reward = reward + beta * influence.T

                # current_timestep = update_step*config["NUM_STEPS"]*config["NUM_ENVS"]
                # shaped_reward = compute_grouped_rewards(reward)
                # reward = jax.tree.map(lambda x,y: x*rew_shaping_anneal_org(current_timestep)+y*rew_shaping_anneal(current_timestep), reward, shaped_reward)


                if config["PARAMETER_SHARING"]:
                    info = jax.tree.map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                    if influence_reward and not recurrent_moa:
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
                elif recurrent_moa:
                    transition = []
                    done = [v for v in done.values()]
                    for i in range(env.num_agents):
                        info_i = {key: jax.tree.map(lambda x: x.reshape((config["NUM_ACTORS"]),1), value[:,i]) for key, value in info.items()}
                        info_i["influence_reward"] = influence[:, i].reshape((config["NUM_ACTORS"], 1))
                        transition.append(MOATransition(
                            done[i],
                            env_act[i],
                            value[i],
                            reward[:,i],
                            log_prob[i],
                            obs_batch[i],
                            info_i,
                            joint_action,
                            joint_action,  # placeholder, replaced with the real next-step
                                           # joint action once the scan below completes
                        ))
                elif lstm_influence:
                    transition = []
                    done = [v for v in done.values()]
                    for i in range(env.num_agents):
                        info_i = {key: jax.tree.map(lambda x: x.reshape((config["NUM_ACTORS"]),1), value[:,i]) for key, value in info.items()}
                        info_i["influence_reward"] = influence[i].reshape((config["NUM_ACTORS"], 1))
                        transition.append(MOATransition(
                            done[i],
                            env_act[i],
                            value[i],
                            reward[:,i],
                            log_prob[i],
                            obs_batch[i],
                            info_i,
                            prev_joint_action,  # the INPUT actually used to produce this
                                                # transition's action -- no shift needed,
                                                # unlike recurrent_moa's next_joint_action.
                            prev_joint_action,  # placeholder -- unused, no auxiliary loss here
                        ))
                    # NORMAL vs LSTM: Normal appends to a plain Transition namedtuple (7
                    # fields: done, action, value, reward, log_prob, obs, info). LSTM
                    # must use the wider MOATransition (9 fields) to additionally carry
                    # prev_joint_action, because the loss's replay scan (in
                    # `_loss_fn`'s RERUN NETWORK section) needs to know, for every
                    # stored timestep, exactly which "previous joint action" input the
                    # policy was actually conditioned on when it produced that
                    # timestep's action/log_prob -- without storing it, there'd be no
                    # way to reconstruct an identical forward pass later with updated
                    # params.
                    #
                    # The second prev_joint_action slot is a genuine placeholder:
                    # MOATransition has a 9th field designed for recurrent_moa's
                    # next_joint_action (the auxiliary MOA loss's training target).
                    # lstm_influence has no auxiliary loss, so that field is unused
                    # here -- it's just reusing the same namedtuple shape rather than
                    # defining a third transition type.
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
                if recurrent_moa:
                    runner_state = (train_state, env_state, obsv, policy_hstate_next, moa_hstate_next, update_step, rng)
                elif lstm_influence:
                    # NORMAL vs LSTM: Normal's runner_state carries 5 elements; LSTM's
                    # carries 8 -- the three extra being exactly the state introduced at
                    # runtime-state-init and updated just above (policy_hstate_next,
                    # prev_joint_action_next, prev_action_probs_next). This is what
                    # makes the memory persist correctly across the jax.lax.scan that
                    # drives _env_step across NUM_STEPS timesteps.
                    runner_state = (train_state, env_state, obsv, policy_hstate_next, prev_joint_action_next, prev_action_probs_next, update_step, rng)
                else:
                    runner_state = (train_state, env_state, obsv, update_step, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            if recurrent_moa:
                # traj_batch[i].joint_action has shape (NUM_STEPS, NUM_ENVS, num_agents).
                # Build the MOA's next-action cross-entropy target here, while the time
                # axis is still intact -- PPO minibatching below shuffles it away. Pad
                # with the last step's action rather than jnp.roll, to avoid wrapping the
                # first timestep's action into the last slot.
                joint_action_t = traj_batch[0].joint_action
                next_joint_action = jnp.concatenate(
                    [joint_action_t[1:], joint_action_t[-1:]], axis=0
                )
                traj_batch = [t._replace(next_joint_action=next_joint_action) for t in traj_batch]

            # CALCULATE ADVANTAGE
            if recurrent_moa:
                train_state, env_state, last_obs, policy_hstate, moa_hstate, update_step, rng = runner_state
            elif lstm_influence:
                train_state, env_state, last_obs, policy_hstate, prev_joint_action, prev_action_probs, update_step, rng = runner_state
            else:
                train_state, env_state, last_obs, update_step, rng = runner_state
            if config["PARAMETER_SHARING"]:
                last_obs_batch = jnp.transpose(last_obs,(1,0,2,3,4)).reshape(-1, *(env.observation_space()[0]).shape)
                _, last_val = network.apply(train_state.params, last_obs_batch)
            elif recurrent_moa:
                last_obs_batch = jnp.transpose(last_obs,(1,0,2,3,4))
                last_val = []
                for i in range(env.num_agents):
                    _, (_, last_val_i) = network[i].apply(train_state[i].params, policy_hstate[i], last_obs_batch[i])
                    last_val.append(last_val_i)
                last_val = jnp.stack(last_val, axis=0)
            elif lstm_influence:
                # NORMAL vs LSTM: exact same asymmetry as the action-selection call in
                # _env_step. Normal's apply(params, last_obs_batch[i]) returns
                # (pi, value) flat and needs only the obs. LSTM's
                # apply(params, policy_hstate[i], last_obs_batch[i],
                # prev_joint_action_onehot) needs the current carry and previous joint
                # action too, and returns the nested (new_carry, (pi, value)) shape --
                # the new_carry is discarded here (`_`) since this is only a one-off
                # bootstrap value for GAE, not a step that gets stored or replayed.
                last_obs_batch = jnp.transpose(last_obs,(1,0,2,3,4))
                prev_joint_action_onehot = jax.nn.one_hot(prev_joint_action, action_dim)
                last_val = []
                for i in range(env.num_agents):
                    _, (_, last_val_i) = network[i].apply(
                        train_state[i].params, policy_hstate[i], last_obs_batch[i], prev_joint_action_onehot
                    )
                    last_val.append(last_val_i)
                last_val = jnp.stack(last_val, axis=0)
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
                    if recurrent_moa:
                        traj_batch, advantages, targets, ph, mh = batch_info
                    elif lstm_influence:
                        traj_batch, advantages, targets, ph = batch_info
                    else:
                        traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets, network_used, ph=None, mh=None):
                        # RERUN NETWORK
                        if recurrent_moa:
                            dones = traj_batch.done
                            # reset entering step t = done[t-1]; step 0 uses ph/mh as-is
                            # (already correctly live-reset when captured before the rollout).
                            reset_in = jnp.concatenate(
                                [jnp.zeros_like(dones[:1]), dones[:-1]], axis=0
                            ).astype(bool)

                            def _policy_body(carry, xs):
                                obs_t, reset_t = xs
                                carry = reset_hstate(carry, reset_t)
                                new_carry, (pi_t, value_t) = network_used.apply(params, carry, obs_t)
                                return new_carry, (pi_t.logits, value_t)

                            _, (logits, value) = jax.lax.scan(
                                _policy_body, ph, (traj_batch.obs, reset_in)
                            )
                            pi = distrax.Categorical(logits=logits)
                            log_prob = pi.log_prob(traj_batch.action)
                        elif lstm_influence:
                            dones = traj_batch.done
                            reset_in = jnp.concatenate(
                                [jnp.zeros_like(dones[:1]), dones[:-1]], axis=0
                            ).astype(bool)

                            def _policy_body(carry, xs):
                                obs_t, other_action_t, reset_t = xs
                                carry = reset_hstate(carry, reset_t)
                                new_carry, (pi_t, value_t) = network_used.apply(
                                    params, carry, obs_t, jax.nn.one_hot(other_action_t, action_dim)
                                )
                                return new_carry, (pi_t.logits, value_t)

                            _, (logits, value) = jax.lax.scan(
                                _policy_body, ph, (traj_batch.obs, traj_batch.joint_action, reset_in)
                            )
                            pi = distrax.Categorical(logits=logits)
                            log_prob = pi.log_prob(traj_batch.action)
                            # NORMAL vs LSTM: PPO needs to recompute log_prob/value
                            # under the CURRENT (already partially updated) params, to
                            # build the importance-sampling ratio for clipping. Normal
                            # can do this with a single vectorized call over every
                            # stored observation at once, because ActorCritic is
                            # stateless -- no ordering dependency between samples.
                            #
                            # LSTM cannot: the recurrent carry at timestep t depends on
                            # the carry from t-1 under the CURRENT params, so timesteps
                            # must be replayed IN ORDER via jax.lax.scan, starting from
                            # ph (the real carry saved before this rollout window).
                            # Skipping straight to a middle timestep would use a carry
                            # from stale params, giving a wrong log_prob and poisoning
                            # the PPO ratio.
                            #  - reset_in: reconstructs where hidden-state resets belong
                            #    during this replay. reset_in[t] = dones[t-1] because a
                            #    done at t-1 means step t starts a fresh episode. Step 0
                            #    is left False because ph, the carry entering the scan,
                            #    was already correctly live-reset during collection --
                            #    no need to reset it again here.
                            #  - traj_batch.joint_action fed into _policy_body: this is
                            #    exactly the prev_joint_action stored per-transition in
                            #    the Transition-construction block -- without it, this
                            #    replay couldn't reconstruct the same input the live
                            #    policy actually saw when it produced the stored
                            #    action/log_prob.
                            #  - pi_t.logits (not .probs) is threaded through the scan
                            #    and turned into a distrax.Categorical only once, after
                            #    the full scan completes -- logits are more numerically
                            #    stable to accumulate across many scan steps than
                            #    probabilities.
                        else:
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

                        if recurrent_moa:
                            # MOA auxiliary loss: rerun the MOA head with fresh params
                            # (same rerun-from-stored-obs pattern as the actor/critic
                            # above) and predict the real next-step joint action. The
                            # action feeds into the MOA LSTM itself (paper Fig. 6), so
                            # the scan body needs it at every step -- can't defer to a
                            # single vectorized head call after the scan. moa_features
                            # (CNN+FC+FC) is still action-independent though: compute it
                            # for every timestep in one big vectorized call before the
                            # scan, so only the truly-sequential LSTM step runs inside it.
                            T, Am = traj_batch.obs.shape[0], traj_batch.obs.shape[1]
                            flat_obs = traj_batch.obs.reshape(T * Am, *traj_batch.obs.shape[2:])
                            flat_feats = network_used.apply(params, flat_obs, method=network_used.moa_features)
                            moa_feats = flat_feats.reshape(T, Am, *flat_feats.shape[1:])

                            def _moa_body(carry, xs):
                                feats_t, action_t, reset_t = xs
                                carry = reset_hstate(carry, reset_t)
                                new_carry, logits_t = network_used.apply(
                                    params, carry, feats_t, jax.nn.one_hot(action_t, action_dim),
                                    method=network_used.moa_step,
                                )
                                return new_carry, logits_t

                            _, moa_logits = jax.lax.scan(
                                _moa_body, mh, (moa_feats, traj_batch.joint_action, reset_in)
                            )
                            moa_ce = optax.softmax_cross_entropy_with_integer_labels(
                                moa_logits, traj_batch.next_joint_action
                            )  # (T, Am, num_agents)
                            not_self = jnp.array([j != i for j in range(env.num_agents)])
                            per_sample_moa_loss = jnp.sum(jnp.where(not_self, moa_ce, 0.0), axis=-1)
                            moa_correct = (
                                jnp.argmax(moa_logits, axis=-1) == traj_batch.next_joint_action
                            ).astype(jnp.float32)
                            per_sample_moa_acc = jnp.sum(
                                jnp.where(not_self, moa_correct, 0.0), axis=-1
                            ) / jnp.sum(not_self)

                            # The last timestep of each collected window has no real
                            # "next action" -- next_joint_action repeats the last real
                            # action as a filler there, which would otherwise be a
                            # spurious training target. Exclude it explicitly with a
                            # positional mask rather than `done`: NUM_STEPS is far
                            # shorter than num_inner_steps here, so episodes almost
                            # never actually end inside a window and a done-based mask
                            # would essentially never fire. Minibatching for this path
                            # keeps the time axis intact and unshuffled (see the
                            # actor-axis-only permutation above), so index -1 reliably
                            # is that one artificial sample for every actor.
                            valid_target = jnp.ones((per_sample_moa_loss.shape[0],)).at[-1].set(0.0)
                            denom = valid_target.sum() * per_sample_moa_loss.shape[1]
                            moa_loss = (per_sample_moa_loss * valid_target[:, None]).sum() / denom
                            moa_accuracy = (per_sample_moa_acc * valid_target[:, None]).sum() / denom

                            # Curriculum-gate the auxiliary loss the same way the reward
                            # is gated. Without this, the MOA loss reshapes the shared
                            # CNN trunk toward "predict teammates' actions" from update 0,
                            # well before the influence reward itself has ramped in,
                            # distorting early policy learning before it gets off the ground.
                            current_timestep = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
                            moa_weight = config["MOA_LOSS_WEIGHT"]
                            total_loss = total_loss + moa_weight * moa_loss
                            return total_loss, (value_loss, loss_actor, entropy, moa_loss, moa_accuracy)

                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    if recurrent_moa:
                        total_loss, grads = grad_fn(
                                train_state.params, traj_batch, advantages, targets, network_used, ph, mh
                            )
                    elif lstm_influence:
                        total_loss, grads = grad_fn(
                                train_state.params, traj_batch, advantages, targets, network_used, ph
                            )
                    else:
                        total_loss, grads = grad_fn(
                                train_state.params, traj_batch, advantages, targets, network_used
                            )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                if recurrent_moa:
                    train_state, traj_batch, advantages, targets, init_ph, init_mh, rng = update_state
                elif lstm_influence:
                    train_state, traj_batch, advantages, targets, init_ph, rng = update_state
                else:
                    train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)

                if recurrent_moa:
                    # Minibatch by permuting the ACTOR axis only, keeping the full time
                    # axis intact per minibatch -- required once temporal order matters.
                    # NUM_MINIBATCHES must divide NUM_ACTORS here (not NUM_STEPS*NUM_ACTORS).
                    A = config["NUM_ACTORS"]
                    M = config["NUM_MINIBATCHES"]
                    Am = A // M
                    perm = jax.random.permutation(_rng, A)
                    sh_traj = jax.tree_util.tree_map(lambda x: jnp.take(x, perm, axis=1), traj_batch)
                    sh_adv = jnp.take(advantages, perm, axis=1)
                    sh_tgt = jnp.take(targets, perm, axis=1)
                    sh_ph = jax.tree_util.tree_map(lambda x: jnp.take(x, perm, axis=0), init_ph)
                    sh_mh = jax.tree_util.tree_map(lambda x: jnp.take(x, perm, axis=0), init_mh)

                    def split_traj(x):  # (T,A,...) -> (M,T,Am,...)
                        x = x.reshape(x.shape[0], M, Am, *x.shape[2:])
                        return jnp.swapaxes(x, 0, 1)

                    def split_carry(x):  # (A,hidden) -> (M,Am,hidden)
                        return x.reshape(M, Am, *x.shape[1:])

                    mb_traj = jax.tree_util.tree_map(split_traj, sh_traj)
                    mb_adv = split_traj(sh_adv)
                    mb_tgt = split_traj(sh_tgt)
                    mb_ph = jax.tree_util.tree_map(split_carry, sh_ph)
                    mb_mh = jax.tree_util.tree_map(split_carry, sh_mh)

                    minibatches = (mb_traj, mb_adv, mb_tgt, mb_ph, mb_mh)
                    train_state, total_loss = jax.lax.scan(
                        lambda state, batch_info: _update_minbatch(state, batch_info, network[i]), train_state, minibatches
                    )
                elif lstm_influence:
                    # Same actor-axis-only minibatching as recurrent_moa, just without a
                    # second (moa) hidden state to shuffle along with it.
                    #
                    # NORMAL vs LSTM: Normal treats every (timestep, actor) pair as an
                    # independent training sample, so it flattens T*A into one axis and
                    # permutes freely across that whole flattened index -- order between
                    # samples is irrelevant to a stateless network.
                    #
                    # LSTM cannot do this: the replay scan in _loss_fn's RERUN NETWORK
                    # section needs each actor's FULL, IN-ORDER time sequence intact to
                    # reconstruct the correct hidden state. So here the permutation
                    # (`perm`) shuffles ONLY the actor axis (axis=1 for
                    # traj_batch/advantages/targets, axis=0 for the carry init_ph, since
                    # carries have no time axis) -- the time axis (axis 0 of traj_batch
                    # etc.) is never touched by `perm`. split_traj/split_carry then just
                    # partition that already-actor-shuffled data into M groups of Am
                    # actors each, still with T intact per group, which is exactly what
                    # that replay scan expects: full per-actor sequences, just fewer
                    # actors per minibatch.
                    A = config["NUM_ACTORS"]
                    M = config["NUM_MINIBATCHES"]
                    Am = A // M
                    perm = jax.random.permutation(_rng, A)
                    sh_traj = jax.tree_util.tree_map(lambda x: jnp.take(x, perm, axis=1), traj_batch)
                    sh_adv = jnp.take(advantages, perm, axis=1)
                    sh_tgt = jnp.take(targets, perm, axis=1)
                    sh_ph = jax.tree_util.tree_map(lambda x: jnp.take(x, perm, axis=0), init_ph)

                    def split_traj(x):  # (T,A,...) -> (M,T,Am,...)
                        x = x.reshape(x.shape[0], M, Am, *x.shape[2:])
                        return jnp.swapaxes(x, 0, 1)

                    def split_carry(x):  # (A,hidden) -> (M,Am,hidden)
                        return x.reshape(M, Am, *x.shape[1:])

                    mb_traj = jax.tree_util.tree_map(split_traj, sh_traj)
                    mb_adv = split_traj(sh_adv)
                    mb_tgt = split_traj(sh_tgt)
                    mb_ph = jax.tree_util.tree_map(split_carry, sh_ph)

                    minibatches = (mb_traj, mb_adv, mb_tgt, mb_ph)
                    train_state, total_loss = jax.lax.scan(
                        lambda state, batch_info: _update_minbatch(state, batch_info, network[i]), train_state, minibatches
                    )
                else:
                    batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                    assert (
                        batch_size == config["NUM_STEPS"] * config["NUM_ACTORS"]
                    ), "batch size must be equal to number of steps * number of actors"
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
                    if config["PARAMETER_SHARING"]:
                        train_state, total_loss = jax.lax.scan(
                            lambda state, batch_info: _update_minbatch(state, batch_info, network), train_state, minibatches
                        )
                    else:
                        train_state, total_loss = jax.lax.scan(
                            lambda state, batch_info: _update_minbatch(state, batch_info, network[i]), train_state, minibatches
                        )

                if recurrent_moa:
                    update_state = (train_state, traj_batch, advantages, targets, init_ph, init_mh, rng)
                elif lstm_influence:
                    update_state = (train_state, traj_batch, advantages, targets, init_ph, rng)
                else:
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
                    if recurrent_moa:
                        update_state = (train_state[i], traj_batch[i], advantages[i], targets[i], init_policy_hstate[i], init_moa_hstate[i], rng)
                    elif lstm_influence:
                        update_state = (train_state[i], traj_batch[i], advantages[i], targets[i], init_policy_hstate[i], rng)
                    else:
                        update_state = (train_state[i], traj_batch[i], advantages[i], targets[i], rng)
                    update_state, loss_info = jax.lax.scan(
                        lambda state, unused: _update_epoch(state, unused, i), update_state, None, config["UPDATE_EPOCHS"]
                    )
                    update_state_dict.append(update_state)
                    train_state[i] = update_state[0]
                    metric_i = traj_batch[i].info
                    metric_i['loss'] = loss_info[0]
                    if recurrent_moa:
                        # Surface the loss components that were being computed and
                        # immediately discarded -- needed to tell "policy instability"
                        # apart from "MOA-specific bug" instead of inferring blindly
                        # from environment-level behavior.
                        metric_i['value_loss'] = loss_info[1][0]
                        metric_i['loss_actor'] = loss_info[1][1]
                        metric_i['entropy'] = loss_info[1][2]
                        metric_i['moa_loss_raw'] = loss_info[1][3]
                        metric_i['moa_accuracy'] = loss_info[1][4]
                    metric.append(metric_i)
                    rng = update_state[-1]

            def callback(metric):
                wandb.log(metric)

            update_step = update_step + 1
            metric = jax.tree.map(lambda x: x.mean(), metric)
            if config["PARAMETER_SHARING"]:
                metric["update_step"] = update_step
                metric["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
                # jax.debug.callback(callback, metric)
            else:
                for i in range(env.num_agents):
                    metric[i]["update_step"] = update_step
                    metric[i]["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
                metric = metric[0]
                # jax.debug.callback(callback, metric)
            metric["update_step"] = update_step
            metric["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
            metric["eat_own_coins"] = metric["eat_own_coins"] * config["ENV_KWARGS"]["num_inner_steps"]
            jax.debug.callback(callback, metric)

            if recurrent_moa:
                runner_state = (train_state, env_state, last_obs, policy_hstate, moa_hstate, update_step, rng)
            elif lstm_influence:
                runner_state = (train_state, env_state, last_obs, policy_hstate, prev_joint_action, prev_action_probs, update_step, rng)
            else:
                runner_state = (train_state, env_state, last_obs, update_step, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        if recurrent_moa:
            runner_state = (train_state, env_state, obsv, policy_hstate, moa_hstate, 0, _rng)
        elif lstm_influence:
            runner_state = (train_state, env_state, obsv, policy_hstate, prev_joint_action, prev_action_probs, 0, _rng)
        else:
            runner_state = (train_state, env_state, obsv, 0, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train

# Used by algorithms/train.py to dispatch through algorithms.IPPO._runner.
SINGLE_RUN_KWARGS = {"wandb_name": "ippo_cnn_coins"}
TUNE_KWARGS       = {"sweep_name": "coins"}
