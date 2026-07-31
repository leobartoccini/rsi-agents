"""
Based on PureJaxRL & jaxmarl Implementation of PPO
"""
import itertools
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import spearmanr, pearsonr
import chex
import distrax
import optax
import flashbax as fbx
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
    ValueInfluenceEstimator,
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


# TrainState subclass for the separate Value-Influence Q-network used by the
# "_sep" reward modes. Holds two parameter sets:
#   - params: the ONLINE network, updated every learn step via gradient descent
#     on an extrinsic-only GAE target (see the qnet learn phase later in this
#     file).
#   - target_network_params: a FROZEN copy, synced from `params` only every
#     VALUE_INFLUENCE_TARGET_UPDATE_INTERVAL updates. The reward computation in
#     _env_step always reads target_network_params, never params directly --
#     this keeps the intrinsic reward signal from chasing a value function
#     that is simultaneously being trained on a reward that already includes
#     that signal (avoids circularity) and keeps the reward stable across
#     updates (target-network trick, same idea as in DQN).
class ValueInfluenceTrainState(TrainState):
    """TrainState for the Value-Influence Q-estimator (ValueInfluenceEstimator),
    carrying a frozen target_network_params alongside the trainable params -- mirrors
    CustomTrainState in algorithms/VDN/vdn_cnn_coins.py. `params` is the ONLINE network
    (trained periodically off the replay buffer below); `target_network_params` is what
    the reward computation in _env_step actually reads, synced from `params` only every
    VALUE_INFLUENCE_TARGET_UPDATE_INTERVAL updates -- this is what keeps the reward
    signal from chasing a network that's simultaneously chasing it.
    """
    target_network_params: Any


# One (obs, joint_action, target, done) sample stored in the qnet's replay
# buffer. `target` is the EXTRINSIC-only, self-bootstrapped GAE target computed
# for the qnet's own regression (see the second _calculate_gae call later in
# this file) -- never the policy's own (shaped-reward) GAE target.
@chex.dataclass(frozen=True)
class QTimestep:
    """One (obs, joint_action, target) sample for the Value-Influence Q-estimator's
    replay buffer -- mirrors the Timestep dataclass in algorithms/VDN/vdn_cnn_coins.py.
    `target` is the EXTRINSIC-only, self-bootstrapped GAE target (see the second
    _calculate_gae call in _update_step), never the policy's own (shaped-reward) GAE
    target."""
    obs: chex.Array          # (num_agents, *obs_shape)
    joint_action_onehot: chex.Array  # (num_agents, action_dim)
    target: chex.Array       # (num_agents,)
    done: chex.Array         # ()


def make_train(config):
    env = socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    # Five ways to get a social-influence reward here, picked by config flags:
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
    #  - PARAMETER_SHARING=False + INFLUENCE_REWARD=True + VALUE_INFLUENCE_MULT/SUM/DIFF=True
    #    (influence/value_influence_*.yaml): same MOA-free env.step-resimulation
    #    mechanism as enabled_independent.yaml (no LSTM, no auxiliary network), but
    #    the KL-based influence magnitude I is additionally combined with delta_v --
    #    reusing the same counterfactual observations/values already computed for the
    #    KL term. Same three combination modes also work with LSTM_INFLUENCE=True.
    #
    # LSTM_INFLUENCE can also be used with INFLUENCE_REWARD=False: that keeps the
    # recurrent, previous-action-conditioned ActorCriticLSTM architecture (still
    # useful on its own -- the policy gets to see what the other agent just did) but
    # skips the counterfactual/KL reward computation entirely rather than just
    # zeroing its contribution, since INFLUENCE_WEIGHT/beta only ever get read inside
    # `if influence_reward:` guards below.
    if config.get("RECURRENT_MOA", False):
        assert config.get("INFLUENCE_REWARD", False) and not config["PARAMETER_SHARING"], (
            "RECURRENT_MOA requires INFLUENCE_REWARD=True and PARAMETER_SHARING=False "
            "(see influence/enabled_recurrent.yaml)."
        )
    if config.get("LSTM_INFLUENCE", False):
        assert not config["PARAMETER_SHARING"], (
            "LSTM_INFLUENCE requires PARAMETER_SHARING=False -- each agent needs its "
            "own distinguishable network for 'the other agent's previous action' to "
            "mean anything (see influence/enabled_lstm.yaml)."
        )
        assert not config.get("RECURRENT_MOA", False), (
            "LSTM_INFLUENCE and RECURRENT_MOA select different network architectures "
            "for the same INFLUENCE_REWARD flag -- pick one, not both."
        )
    # VALUE_INFLUENCE_MULT/SUM/DIFF/DIFF2/VALUE/VALUE_DIFF are six mutually-exclusive
    # ways to turn delta_v (how much the acting agent's real action raised or lowered
    # another agent's value, relative to what its own real policy would have produced
    # on average) into an intrinsic reward, optionally combined with the KL-based
    # influence magnitude I. MULT/SUM/DIFF/DIFF2 replace the old
    # INFLUENCE_SIGNED/LSTM_INFLUENCE_SIGNED booleans (which only ever did "mult") and
    # work with either the feedforward MOA-free path (PARAMETER_SHARING True or
    # False) or LSTM_INFLUENCE. VALUE/VALUE_DIFF are LSTM_INFLUENCE-only (see the
    # assert below) -- see the value_influence_mode branches in _env_step:
    #  - mult:       r^WAI = I * delta_v (the original welfare-aware-influence formula)
    #  - sum:        r = I + lambda*delta_v
    #  - diff:       r = I + lambda*(delta_v_(k->j) - delta_v_(j->k)) -- how much k
    #    influenced j's value minus how much j influenced k's value back
    #  - diff2:      r = I * (delta_v_(k->j) - delta_v_(j->k)) -- same net value effect
    #    as diff, but multiplied by I instead of added to it
    #  - value:      r = delta_v alone -- no KL-influence term I at all
    #  - value_diff: r = delta_v_(k->j) - delta_v_(j->k) alone -- no I at all
    #
    # mult_sep/sum_sep/diff_sep/diff2_sep are FOUR SEPARATE, self-contained
    # implementations of the mult/sum/diff/diff2 formulas above -- same formulas, but
    # delta_v comes from a fully independent ValueInfluenceEstimator Q-network (own
    # CNN/FC/LSTM, own optimizer, periodically-synced frozen target copy) instead of
    # ActorCriticLSTM's own value head, so the intrinsic reward is never derived from
    # the exact critic PPO is simultaneously updating with that same reward (gradient
    # isolation + target stability). Deliberately NOT sharing code with
    # mult/sum/diff/diff2 above -- each _sep mode has its own complete KL +
    # qnet-value + combine block in _env_step, so nothing about the six modes above
    # is touched by this. LSTM_INFLUENCE-only, like VALUE/VALUE_DIFF -- the qnet
    # needs the same recurrent carry (qnet_hstate) threading LSTM_INFLUENCE already
    # sets up for the policy.
    # ---------------------------------------------------------------------------
    # VALUE-INFLUENCE REWARD MODES
    # ---------------------------------------------------------------------------
    # Six ways to turn delta_v (how much an agent's real action changed another
    # agent's estimated value, relative to what its own real policy would have
    # produced on average) into an intrinsic reward, optionally combined with the
    # KL-based causal-influence magnitude I (Jaques et al.):
    #   mult / sum / diff / diff2 / value / value_diff
    # Each also has a "_sep" counterpart (mult_sep, sum_sep, diff_sep, diff2_sep)
    # that computes delta_v from a fully separate ValueInfluenceEstimator Q-network
    # instead of the policy's own value head -- see the qnet setup and _env_step
    # branches below. At most one of these ten flags may be active at a time.
    value_influence_flags = {
        "mult": config.get("VALUE_INFLUENCE_MULT", False),
        "sum": config.get("VALUE_INFLUENCE_SUM", False),
        "diff": config.get("VALUE_INFLUENCE_DIFF", False),
        "diff2": config.get("VALUE_INFLUENCE_DIFF2", False),
        "value": config.get("VALUE_INFLUENCE_VALUE", False),
        "value_diff": config.get("VALUE_INFLUENCE_VALUE_DIFF", False),
        "mult_sep": config.get("VALUE_INFLUENCE_MULT_SEP", False),
        "sum_sep": config.get("VALUE_INFLUENCE_SUM_SEP", False),
        "diff_sep": config.get("VALUE_INFLUENCE_DIFF_SEP", False),
        "diff2_sep": config.get("VALUE_INFLUENCE_DIFF2_SEP", False),
    }
    assert sum(bool(v) for v in value_influence_flags.values()) <= 1, (
        "VALUE_INFLUENCE_MULT/SUM/DIFF/DIFF2/VALUE/VALUE_DIFF/*_SEP are mutually "
        "exclusive -- pick at most one combination mode."
    )
    if any(value_influence_flags.values()):
        assert config.get("INFLUENCE_REWARD", False) and not config.get("RECURRENT_MOA", False), (
            "VALUE_INFLUENCE_MULT/SUM/DIFF/DIFF2/VALUE/VALUE_DIFF/*_SEP requires "
            "INFLUENCE_REWARD=True and is incompatible with RECURRENT_MOA "
            "(see config/influence/value_influence_*.yaml, value_lstm.yaml, "
            "value_diff_lstm.yaml)."
        )
    # VALUE / VALUE_DIFF / all four *_SEP modes require LSTM_INFLUENCE=True --
    # they are not implemented for the feedforward (env.step-resimulation) path.
    _sep_modes = ("mult_sep", "sum_sep", "diff_sep", "diff2_sep")
    if value_influence_flags["value"] or value_influence_flags["value_diff"] or any(
        value_influence_flags[m] for m in _sep_modes
    ):
        assert config.get("LSTM_INFLUENCE", False), (
            "VALUE_INFLUENCE_VALUE/VALUE_DIFF/*_SEP are only implemented in the "
            "LSTM_INFLUENCE path -- the feedforward branches don't handle these modes "
            "(see value_lstm.yaml / value_diff_lstm.yaml / "
            "value_influence_*_lstm_sep.yaml). Use mult/sum/diff/diff2 outside "
            "LSTM_INFLUENCE."
        )
    # VALUE_INEQUITY_AVERSION: same u_k = r_k - penalty formula as
    # get_inequity_aversion_rewards_immediate() in coin_game.py, but built from the
    # critic's value estimates V_k instead of the immediate reward r_k -- V is a
    # smoother, longer-horizon proxy for "how well agent k is doing" than a single
    # step's reward, playing the same role the env's own `smooth_rewards`
    # exponential-average option does for the immediate-reward version. Computed here
    # (not in the env) because the env has no access to the critic. Works with every
    # branch below (PARAMETER_SHARING, recurrent_moa, lstm_influence, independent) --
    # all of them already compute a pre-step value estimate per agent during action
    # selection, which is what this reuses.
    value_inequity_aversion = config.get("VALUE_INEQUITY_AVERSION", False)
    if value_inequity_aversion:
        assert not config["ENV_KWARGS"].get("inequity_aversion", False), (
            "VALUE_INEQUITY_AVERSION and ENV_KWARGS.inequity_aversion both shape the "
            "reward for inequity -- one from the value function, one from the immediate "
            "reward -- so stacking them double-penalizes the same inequity. Pick one."
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
        # Resolve which single value-influence combination mode (if any) is active
        # this run, from the ten mutually-exclusive config flags checked above.
        # None means no value-influence term at all (plain KL-only influence reward,
        # if INFLUENCE_REWARD is on).
        # Which welfare-aware combination (if any) to apply to the KL-based influence
        # magnitude I -- see the value_influence_flags assertion above for what each
        # mode means. None means plain (unsigned) influence reward, same as before
        # INFLUENCE_SIGNED/LSTM_INFLUENCE_SIGNED existed. Works identically whether
        # the active path ends up being the LSTM-conditioned one (lstm_influence
        # below) or the feedforward MOA-free one.
        value_influence_mode = next(
            (mode for mode, flag in (
                ("mult", config.get("VALUE_INFLUENCE_MULT", False)),
                ("sum", config.get("VALUE_INFLUENCE_SUM", False)),
                ("diff", config.get("VALUE_INFLUENCE_DIFF", False)),
                ("diff2", config.get("VALUE_INFLUENCE_DIFF2", False)),
                ("value", config.get("VALUE_INFLUENCE_VALUE", False)),
                ("value_diff", config.get("VALUE_INFLUENCE_VALUE_DIFF", False)),
                ("mult_sep", config.get("VALUE_INFLUENCE_MULT_SEP", False)),
                ("sum_sep", config.get("VALUE_INFLUENCE_SUM_SEP", False)),
                ("diff_sep", config.get("VALUE_INFLUENCE_DIFF_SEP", False)),
                ("diff2_sep", config.get("VALUE_INFLUENCE_DIFF2_SEP", False)),
            ) if flag),
            None,
        )
        # True only for the four "_sep" modes -- everything below gated on this flag
        # (qnet instantiation, qnet_hstate, the replay buffer, the periodic learn
        # phase) exists only when one of these four modes is active. In every other
        # mode, delta_v comes from the policy's own value head instead.
        # mult_sep/sum_sep/diff_sep/diff2_sep use a fully separate Q-network
        # (ValueInfluenceEstimator) for delta_v instead of ActorCriticLSTM's value
        # head -- see the value_influence_flags comment above for why. Everything
        # below gated on this bool (qnet instantiation, qnet_hstate, the buffer, the
        # learn phase) only exists when one of these four modes is active.
        _sep_modes = ("mult_sep", "sum_sep", "diff_sep", "diff2_sep")
        value_influence_use_qnet = value_influence_mode in _sep_modes
        # Weight on the delta_v term in the LSTM path's "sum" mode (I + lambda*delta_v)
        # and "diff" mode (I + lambda*(delta_v_(k->j) - delta_v_(j->k))) -- a second,
        # independent knob alongside INFLUENCE_WEIGHT/beta (which scales the WHOLE
        # combined term when added to reward), letting delta_v's contribution be tuned
        # relative to the KL-influence term I itself. Only read/applied in the
        # lstm_influence branch's "sum"/"diff" cases below -- "mult" and "diff2" scale
        # delta_v multiplicatively through I already, and the non-LSTM feedforward path
        # doesn't use it.
        value_influence_lambda = config.get("VALUE_INFLUENCE_LAMBDA", 1.0)
        value_inequity_aversion = config.get("VALUE_INEQUITY_AVERSION", False)
        value_inequity_alpha = config.get("VALUE_INEQUITY_AVERSION_ALPHA", 5)
        value_inequity_beta = config.get("VALUE_INEQUITY_AVERSION_BETA", 0.05)
        # Number of update-windows' worth of raw per-timestep data to accumulate before
        # computing the live Group B behavioral metrics (conditional cooperation,
        # retaliation lag, forgiveness rate, influence/delta_v-vs-extrinsic-reward
        # correlation) -- see _flush_group_b_metrics below. A single NUM_STEPS window is
        # usually shorter than a real episode (num_inner_steps), so these are computed
        # over a K-update rolling buffer instead of every update.
        metrics_window_updates = config.get("METRICS_WINDOW_UPDATES", 50)
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

        # ---------------------------------------------------------------------------
        # SEPARATE Q-NETWORK SETUP (only for mult_sep/sum_sep/diff_sep/diff2_sep)
        # ---------------------------------------------------------------------------
        # Builds one ValueInfluenceEstimator (a recurrent Q-network estimating
        # Q_i(obs_i, joint_action) for agent i) per agent, with its own optimizer and
        # its own ValueInfluenceTrainState (online + frozen target params). Also
        # builds a flashbax trajectory replay buffer that stores contiguous sequences
        # (not i.i.d. samples), since the qnet's LSTM carry must be genuinely unrolled
        # across real consecutive timesteps. None of this shares any state, gradient,
        # or optimizer with the main policy's train_state.
        if value_influence_use_qnet:
            # Fully separate Q-network for delta_v in the mult_sep/sum_sep/diff_sep/
            # diff2_sep modes -- own architecture, own params, own optimizer, never
            # shares tx/optimizer state with the policy's train_state above. Only
            # `qnet_train_state[i].target_network_params` (synced from `.params` every
            # VALUE_INFLUENCE_TARGET_UPDATE_INTERVAL updates, see the learn phase in
            # _update_step) is ever read when computing the reward -- `.params` itself
            # is only touched by the periodic learn-phase gradient step.
            # Own learning rate (defaults to the policy's LR) and own Adam optimizer chain
            # -- gradients computed for the qnet never touch the policy's train_state.
            qnet_lr = config.get("VALUE_INFLUENCE_Q_LR", config["LR"])
            qnet_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(qnet_lr, eps=1e-5),
            )
            # One independent ValueInfluenceEstimator per agent -- estimates agent i's
            # expected return conditioned on the FULL joint action of all agents, not
            # just its own action.
            qnet = [
                ValueInfluenceEstimator(
                    num_agents=env.num_agents, action_dim=action_dim,
                    hidden_dim=hidden_dim, activation=config["ACTIVATION"],
                ) for _ in range(env.num_agents)
            ]
            rng, _rng = jax.random.split(rng)
            init_qh_1 = ValueInfluenceEstimator.initialize_carry(1, hidden_dim)
            qnet_params = [
                qnet[i].init(_rng, init_qh_1, init_x, init_joint_action)
                for i in range(env.num_agents)
            ]
            # target_network_params initialized equal to params; will only diverge once
            # the periodic sync (later in this file) starts copying updated params into it.
            qnet_train_state = [
                ValueInfluenceTrainState.create(
                    apply_fn=qnet[i].apply, params=p, target_network_params=p, tx=qnet_tx,
                ) for i, p in enumerate(qnet_params)
            ]

            # Replay buffer of short (obs, joint_action_onehot, extrinsic-only GAE
            # target) SEQUENCES -- a flashbax trajectory buffer, not a flat buffer,
            # specifically so the qnet's LSTM carry can be genuinely unrolled across a
            # sampled chunk (with burn-in for chunks starting mid-episode, see the
            # learn phase in _update_step) instead of fed a cold zero-carry per
            # independent sample.
            # Trajectory (not flat) replay buffer: stores overlapping sequences of length
            # qnet_buffer_seq_len so the qnet's recurrent carry can be unrolled across a
            # real chunk of consecutive timesteps during the learn phase, with burn-in for
            # chunks that start mid-episode (see _qnet_learn later in this file).
            qnet_buffer_seq_len = config.get("VALUE_INFLUENCE_BUFFER_SEQ_LEN", 12)
            qnet_buffer = fbx.make_trajectory_buffer(
                add_batch_size=config["NUM_ENVS"],
                sample_batch_size=config.get("VALUE_INFLUENCE_BUFFER_BATCH_SIZE", 32),
                sample_sequence_length=qnet_buffer_seq_len,
                period=config.get("VALUE_INFLUENCE_BUFFER_PERIOD", max(qnet_buffer_seq_len // 2, 1)),
                min_length_time_axis=config.get("VALUE_INFLUENCE_BUFFER_MIN_LENGTH", qnet_buffer_seq_len * 4),
                max_size=config.get("VALUE_INFLUENCE_BUFFER_MAX_SIZE", 200_000),
            )
            qnet_buffer = qnet_buffer.replace(
                init=jax.jit(qnet_buffer.init),
                add=jax.jit(qnet_buffer.add, donate_argnums=0),
                sample=jax.jit(qnet_buffer.sample),
                can_sample=jax.jit(qnet_buffer.can_sample),
            )
            # Shape-only placeholder used to pre-allocate the buffer's storage arrays --
            # these values are never actually used.
            _dummy_qtimestep = QTimestep(
                obs=jnp.zeros((env.num_agents, *(env.observation_space()[0]).shape)),
                joint_action_onehot=jnp.zeros((env.num_agents, action_dim)),
                target=jnp.zeros((env.num_agents,)),
                done=jnp.array(False),
            )
            qnet_buffer_state = qnet_buffer.init(_dummy_qtimestep)

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
            if value_influence_use_qnet:
                # qnet's own recurrent carry, threaded through runner_state exactly
                # like policy_hstate (updated from the REAL, non-counterfactual
                # forward call each _env_step, reset on done) -- separate from
                # policy_hstate since it belongs to a completely different network.
                qnet_hstate = [ValueInfluenceEstimator.initialize_carry(config["NUM_ENVS"], hidden_dim) for _ in range(env.num_agents)]

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # Outer carry is (env_runner_state, metrics_buffer) -- or, when a
            # mult_sep/sum_sep/diff_sep/diff2_sep mode is active,
            # (env_runner_state, metrics_buffer, qnet_train_state, qnet_buffer_state).
            # metrics_buffer/qnet_train_state/qnet_buffer_state are all untouched by
            # _env_step (only read/written here, once per update), so unpacking them
            # here and re-nesting at the bottom keeps every existing
            # env_runner_state-shaped branch below (recurrent_moa/lstm_influence/else)
            # completely unchanged.
            if value_influence_use_qnet:
                runner_state, metrics_buffer, qnet_train_state, qnet_buffer_state = runner_state
            else:
                runner_state, metrics_buffer = runner_state
            if recurrent_moa:
                # Snapshot the carries entering this rollout window -- they were already
                # correctly reset in real time during the previous window's collection, so
                # the loss's time-scan can use them as-is with no reset check at step 0.
                _, _, _, init_policy_hstate, init_moa_hstate, _, _ = runner_state
            elif lstm_influence:
                if value_influence_use_qnet:
                    _, _, _, init_policy_hstate, init_qnet_hstate, _, _, _, _ = runner_state
                else:
                    _, _, _, init_policy_hstate, _, _, _, _ = runner_state

            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                if recurrent_moa:
                    train_state, env_state, last_obs, policy_hstate, moa_hstate, update_step, rng = runner_state
                elif lstm_influence:
                    if value_influence_use_qnet:
                        train_state, env_state, last_obs, policy_hstate, qnet_hstate, prev_joint_action, prev_action_probs, update_step, rng = runner_state
                    else:
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
                    beta = config["INFLUENCE_WEIGHT"]

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
                    if influence_reward and not value_influence_use_qnet:
                        # ---------------------------------------------------------------------------
                        # NON-SEP VALUE-INFLUENCE PATH (mult/sum/diff/diff2/value/value_diff)
                        # ---------------------------------------------------------------------------
                        # delta_v here reuses ActorCriticLSTM's OWN value head -- no separate Q-network
                        # involved. cond_probs is the real conditional policy (already computed above
                        # during action selection); the counterfactual re-calls each agent's network
                        # with the SAME incoming carry/obs but agent k's slot in the previous joint
                        # action swapped to a hypothetical action, retrieving both the counterfactual
                        # policy AND the counterfactual value from the same forward pass "for free".
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

                        def _apply_pi_value(j, other_action_onehot):
                            # Grabbing value_j here is free -- the network already
                            # computes it alongside pi_j in the same forward pass;
                            # LSTM_INFLUENCE just never used to look at it.
                            _, (pi_j, value_j) = network[j].apply(
                                train_state[j].params, policy_hstate[j], obs_batch[j], other_action_onehot
                            )
                            return pi_j.probs, value_j

                        if value_influence_mode is not None:
                            # real_value[j] is agent j's estimated value under the REAL previous
                            # joint action -- reused from the `value` list already computed during
                            # action selection above, no extra forward pass needed.
                            # REAL value estimate for every agent, under the REAL
                            # prev_joint_action -- already computed for free during
                            # action selection (the `value` list), same free-lunch as
                            # cond_probs reusing pi_list above.
                            real_value = jnp.stack([value[j] for j in range(env.num_agents)], axis=0)  # (j, e)

                        kl_list = []
                        delta_v_list = []  # only populated when value_influence_mode is not None
                        for k in range(env.num_agents):
                            def _cf_probs_value(a_idx, k=k):
                                cf_onehot = jax.nn.one_hot(a_idx, action_dim)
                                cf_joint = prev_joint_action_onehot.at[:, k, :].set(cf_onehot)
                                probs_list, value_list = zip(
                                    *[_apply_pi_value(j, cf_joint) for j in range(env.num_agents)]
                                )
                                return jnp.stack(probs_list, axis=0), jnp.stack(value_list, axis=0)  # (j,e,t), (j,e)

                            cf_probs, cf_values = jax.vmap(_cf_probs_value)(jnp.arange(action_dim))  # (a,j,e,t), (a,j,e)
                            # Marginalize the counterfactuals over agent k's REAL policy from
                            # the step that actually produced prev_joint_action.
                            marginal_probs = jnp.einsum("ea,ajet->jet", prev_action_probs[k], cf_probs)
                            kl_per_j = _safe_kl_lstm(cond_probs, marginal_probs)  # (num_agents, NUM_ENVS)
                            kl_list.append(kl_per_j)

                            if value_influence_mode is not None:
                                # delta_v_(k->j) = real_value[j] (under k's REAL action) minus
                                # marginal_value[j] (the value j would have had on average, had k
                                # acted according to k's own real policy instead of the specific
                                # action it took). Positive means k's specific action helped j
                                # relative to k's typical behavior; negative means it hurt, scaled
                                # by how much.
                                # Same marginalization, applied to the value instead of
                                # the action distribution: "what would agent j's value
                                # have been on average, had agent k's action been drawn
                                # from k's own real policy instead of the specific
                                # action it actually took." Comparing that to the REAL
                                # value (which used k's actual action) gives delta_v,
                                # the welfare-aware-influence paper's full ΔV term --
                                # positive when this specific action helped j relative
                                # to k's own typical behavior, negative when it hurt,
                                # scaled by how MUCH it helped/hurt (not just the sign).
                                # delta_v_list[k][j] = delta_v_(k->j): how much k's real
                                # action changed j's value relative to k's own average
                                # behavior.
                                marginal_value = jnp.einsum("ea,aje->je", prev_action_probs[k], cf_values)  # (j, e)
                                delta_v_list.append(real_value - marginal_value)  # (j, e)

                        # Combine the KL-based influence magnitude I (kl_stack) with delta_v according
                        # to the active mode:
                        #   mult:       I * delta_v          (original welfare-aware-influence formula)
                        #   sum:        I + lambda*delta_v
                        #   value:      delta_v alone, no I
                        #   diff/diff2/value_diff: use the NET (reciprocal) delta_v -- see net_delta_v
                        #   below -- delta_v_(k->j) minus delta_v_(j->k), i.e. k's effect on j net of
                        #   j's effect back on k.
                        kl_stack = jnp.stack(kl_list, axis=0)  # (k, j, NUM_ENVS)
                        if value_influence_mode is None:
                            combined = kl_stack
                        else:
                            delta_v_stack = jnp.stack(delta_v_list, axis=0)  # (k, j, NUM_ENVS)
                            if value_influence_mode == "mult":
                                # r^WAI = I * ΔV: multiplying kl_per_j (the KL-based
                                # influence magnitude I) by delta_v directly reproduces
                                # that product -- using jnp.sign(delta_v) here instead
                                # would collapse ΔV's magnitude to +-1, discarding
                                # exactly the "how much" information the product is
                                # meant to carry.
                                combined = kl_stack * delta_v_stack
                                value_term = delta_v_stack
                            elif value_influence_mode == "sum":
                                combined = kl_stack + value_influence_lambda * delta_v_stack
                                value_term = delta_v_stack
                            elif value_influence_mode == "value":
                                # r = ΔV alone -- no KL-influence term I at all, unlike
                                # "sum"/"mult" which both fold I in.
                                combined = delta_v_stack
                                value_term = delta_v_stack
                            else:
                                # diff / diff2 / value_diff all need the net (reciprocal)
                                # delta_v: delta_v_(k->j) minus delta_v_(j->k) -- how much
                                # k influenced j's value, net of how much j influenced k's
                                # value back. delta_v_stack[j, k] (transpose's [k, j]
                                # entry) is exactly delta_v_(j->k), computed during
                                # iteration j above.
                                reverse = jnp.transpose(delta_v_stack, (1, 0, 2))
                                net_delta_v = delta_v_stack - reverse
                                if value_influence_mode == "diff":
                                    combined = kl_stack + value_influence_lambda * net_delta_v
                                elif value_influence_mode == "diff2":
                                    combined = kl_stack * net_delta_v
                                else:  # "value_diff": net delta_v alone, no I
                                    combined = net_delta_v
                                value_term = net_delta_v

                            # Raw, un-scaled per-agent breakdown of the two signals that
                            # go into `combined` above -- "just I", "just the value term"
                            # (delta_v for mult/sum, net delta_v for diff/diff2), and
                            # their raw product (only actually equal to `combined` for
                            # "mult", but logged for every mode as a diagnostic) -- summed
                            # over j != k the same way `influence` is below, but BEFORE
                            # beta/lambda/done-masking, so these read the underlying
                            # signal magnitudes directly instead of the applied reward.
                            not_self_mat_kv = jnp.array(
                                [[j != k for j in range(env.num_agents)] for k in range(env.num_agents)]
                            )  # (k, j)
                            influence_kl_component = jnp.sum(
                                jnp.where(not_self_mat_kv[:, :, None], kl_stack, 0.0), axis=1
                            )  # (k, NUM_ENVS)
                            influence_value_component = jnp.sum(
                                jnp.where(not_self_mat_kv[:, :, None], value_term, 0.0), axis=1
                            )
                            influence_kl_times_value_component = jnp.sum(
                                jnp.where(not_self_mat_kv[:, :, None], kl_stack * value_term, 0.0), axis=1
                            )

                        influence = []
                        for k in range(env.num_agents):
                            not_self = jnp.array([j != k for j in range(env.num_agents)])
                            influence.append(jnp.sum(jnp.where(not_self[:, None], combined[k], 0.0), axis=0))

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
                    elif influence_reward and value_influence_use_qnet:
                        # ---------------------------------------------------------------------------
                        # SEP VALUE-INFLUENCE PATH (mult_sep/sum_sep/diff_sep/diff2_sep)
                        # ---------------------------------------------------------------------------
                        # Same overall shape as the non-sep path above, but delta_v is computed from
                        # the separate qnet's FROZEN target_network_params instead of the policy's own
                        # value head -- deliberately re-derived from scratch here rather than sharing
                        # code with the non-sep block, so nothing about those modes is touched by this
                        # branch. The KL term I still comes from the policy (cond_probs/_cf_pi_probs_sep),
                        # same as before; only the value term switches to the qnet.
                        # mult_sep/sum_sep/diff_sep/diff2_sep -- FOUR SEPARATE, self-
                        # contained implementations of the mult/sum/diff/diff2 formulas
                        # above. Deliberately not sharing any code with that block: the
                        # KL-divergence/counterfactual-policy computation below is
                        # rederived from scratch here rather than calling into
                        # _safe_kl_lstm/_apply_pi_value above, so nothing about the
                        # non-sep modes is touched by this branch. The four modes DO
                        # share this setup with EACH OTHER (only the final combine
                        # formula differs between them, same as the non-sep block does
                        # for its six modes) -- that's new code shared among new code,
                        # not reuse of the existing implementation.
                        cond_probs = jnp.stack(
                            [pi_list[j].probs for j in range(env.num_agents)], axis=0
                        )  # (j, e, t) -- REAL

                        current_timestep = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
                        beta = config["INFLUENCE_WEIGHT"] * rew_shaping_anneal(current_timestep)

                        def _safe_kl_lstm_sep(cond_probs, marginal_probs):
                            eps = 1e-6
                            cond_safe = cond_probs + eps
                            cond_safe = cond_safe / cond_safe.sum(-1, keepdims=True)
                            marginal_safe = marginal_probs + eps
                            marginal_safe = marginal_safe / marginal_safe.sum(-1, keepdims=True)
                            return distrax.Categorical(probs=cond_safe).kl_divergence(
                                distrax.Categorical(probs=marginal_safe)
                            )

                        def _cf_pi_probs_sep(j, other_action_onehot):
                            # Policy-only counterfactual (no value head reused here --
                            # qnet supplies delta_v separately below).
                            _, (pi_j, _) = network[j].apply(
                                train_state[j].params, policy_hstate[j], obs_batch[j], other_action_onehot
                            )
                            return pi_j.probs

                        # THIS step's real joint action (not prev_joint_action_onehot, which only
                        # conditions the POLICY) -- the qnet estimates Q(s,a) for the CURRENT decision,
                        # so it needs the action actually taken now, not the previous one.
                        # THIS step's real joint action (not prev_joint_action_onehot,
                        # which conditions the POLICY) -- qnet estimates Q(s,a) for the
                        # CURRENT decision, so it needs the action actually taken now.
                        joint_action_onehot_current = jax.nn.one_hot(
                            jnp.stack(env_act, axis=-1), action_dim
                        )  # (NUM_ENVS, num_agents, action_dim)

                        # Real Q-value for every agent under the REAL current joint action, evaluated
                        # through the FROZEN target network. This same call also produces the qnet's
                        # updated recurrent carry (new_qnet_hstate), threaded forward exactly like
                        # new_policy_hstate.
                        # REAL Q-value for every agent under the REAL current joint
                        # action, via the FROZEN target network -- this is also the
                        # carry-updating call, so its returned carries become
                        # new_qnet_hstate (threaded forward below, after env.step,
                        # exactly like new_policy_hstate).
                        real_qvalue = []
                        new_qnet_hstate = []
                        for j in range(env.num_agents):
                            new_qh_j, real_qv_j = qnet[j].apply(
                                qnet_train_state[j].target_network_params, qnet_hstate[j],
                                obs_batch[j], joint_action_onehot_current,
                            )
                            real_qvalue.append(real_qv_j)
                            new_qnet_hstate.append(new_qh_j)
                        real_qvalue = jnp.stack(real_qvalue, axis=0)  # (j, e)

                        kl_list = []
                        delta_v_list = []
                        for k in range(env.num_agents):
                            # KL term I: counterfactual over agent k's PREVIOUS action
                            # (matches the policy's own conditioning), via
                            # ActorCriticLSTM -- counterfactual carry discarded, only
                            # the value (probs) is used.
                            def _cf_probs_sep(a_idx, k=k):
                                cf_onehot = jax.nn.one_hot(a_idx, action_dim)
                                cf_joint = prev_joint_action_onehot.at[:, k, :].set(cf_onehot)
                                return jnp.stack(
                                    [_cf_pi_probs_sep(j, cf_joint) for j in range(env.num_agents)],
                                    axis=0,
                                )  # (j, e, t)

                            cf_probs = jax.vmap(_cf_probs_sep)(jnp.arange(action_dim))  # (a, j, e, t)
                            marginal_probs = jnp.einsum("ea,ajet->jet", prev_action_probs[k], cf_probs)
                            kl_per_j = _safe_kl_lstm_sep(cond_probs, marginal_probs)  # (num_agents, NUM_ENVS)
                            kl_list.append(kl_per_j)

                            # Counterfactual over agent k's CURRENT action (unlike the KL term, which
                            # counterfactualizes k's PREVIOUS action), evaluated via the qnet's frozen
                            # target network, then marginalized by k's CURRENT real policy -- Q(s,a) is
                            # about the current decision, so both the counterfactual and the marginalizing
                            # weight must be anchored to "now", not to the previous timestep.
                            # delta_v term: counterfactual over agent k's CURRENT
                            # action, via qnet's FROZEN target network, marginalized by
                            # k's CURRENT real policy (Q(s,a) is about the current
                            # decision, unlike the KL term above).
                            def _cf_qvalue_sep(a_idx, k=k):
                                cf_onehot = jax.nn.one_hot(a_idx, action_dim)
                                cf_joint = joint_action_onehot_current.at[:, k, :].set(cf_onehot)
                                return jnp.stack(
                                    [
                                        qnet[j].apply(
                                            qnet_train_state[j].target_network_params, qnet_hstate[j],
                                            obs_batch[j], cf_joint,
                                        )[1]
                                        for j in range(env.num_agents)
                                    ],
                                    axis=0,
                                )  # (j, e)

                            cf_qvalue = jax.vmap(_cf_qvalue_sep)(jnp.arange(action_dim))  # (a, j, e)
                            marginal_value = jnp.einsum("ea,aje->je", pi_list[k].probs, cf_qvalue)  # (j, e)
                            # delta_v_(k->j): how much k's real action changed j's qnet
                            # value relative to k's own average behavior.
                            delta_v_list.append(real_qvalue - marginal_value)  # (j, e)

                        kl_stack = jnp.stack(kl_list, axis=0)  # (k, j, NUM_ENVS)
                        delta_v_stack = jnp.stack(delta_v_list, axis=0)  # (k, j, NUM_ENVS)

                        # Same four combination formulas as the non-sep path (mult/sum/diff/diff2),
                        # just applied to the qnet-derived delta_v_stack instead of the policy's own
                        # value head. "value"/"value_diff" have no "_sep" counterpart.
                        if value_influence_mode == "mult_sep":
                            # r = I * delta_v
                            combined = kl_stack * delta_v_stack
                            value_term = delta_v_stack
                        elif value_influence_mode == "sum_sep":
                            # r = I + lambda*delta_v
                            combined = kl_stack + value_influence_lambda * delta_v_stack
                            value_term = delta_v_stack
                        else:
                            # diff_sep/diff2_sep need the net (reciprocal) delta_v:
                            # delta_v_(k->j) minus delta_v_(j->k).
                            reverse = jnp.transpose(delta_v_stack, (1, 0, 2))
                            net_delta_v = delta_v_stack - reverse
                            if value_influence_mode == "diff_sep":
                                combined = kl_stack + value_influence_lambda * net_delta_v
                            else:  # "diff2_sep"
                                combined = kl_stack * net_delta_v
                            value_term = net_delta_v

                        # Raw per-agent breakdown -- same convention/naming as the
                        # non-sep block, so agent{i}/influence_kl etc. graphs work
                        # identically for _sep runs.
                        not_self_mat_sep = jnp.array(
                            [[j != k for j in range(env.num_agents)] for k in range(env.num_agents)]
                        )  # (k, j)
                        influence_kl_component = jnp.sum(
                            jnp.where(not_self_mat_sep[:, :, None], kl_stack, 0.0), axis=1
                        )
                        influence_value_component = jnp.sum(
                            jnp.where(not_self_mat_sep[:, :, None], value_term, 0.0), axis=1
                        )
                        influence_kl_times_value_component = jnp.sum(
                            jnp.where(not_self_mat_sep[:, :, None], kl_stack * value_term, 0.0), axis=1
                        )

                        influence = []
                        for k in range(env.num_agents):
                            not_self = jnp.array([j != k for j in range(env.num_agents)])
                            influence.append(jnp.sum(jnp.where(not_self[:, None], combined[k], 0.0), axis=0))
                        influence = jnp.stack(influence, axis=0)  # (num_agents=k, NUM_ENVS)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                env_state_t = env_state  # pre-step state, needed below for the MOA-free counterfactuals

                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state_t, env_act)

                # The one genuinely pre-shaping snapshot in this function -- every
                # branch below (VALUE_INEQUITY_AVERSION's subtraction, then whichever
                # influence-reward mechanism runs) mutates `reward` in place from here
                # on. Persisted into every branch's info/info_i below as
                # "extrinsic_reward" for live behavioral-metrics logging (P ratio uses
                # eat_own/eat_other directly and doesn't need this, but the
                # influence/delta_v-vs-extrinsic-reward correlation metric does, and
                # must never be computed against the shaped reward -- that would
                # trivially inflate the correlation since the shaped reward already
                # contains a scaled copy of the influence/delta_v signal).
                raw_env_reward = reward  # (NUM_ENVS, num_agents)

                if value_inequity_aversion:
                    # u_k = r_k - [alpha * sum_{j!=k} max(V_j - V_k, 0)
                    #            + beta  * sum_{j!=k} max(V_k - V_j, 0)] / (N - 1)
                    # Same shape as get_inequity_aversion_rewards_immediate() in
                    # coin_game.py, but built from the PRE-STEP value estimates
                    # (`value`, already computed above during action selection) rather
                    # than the immediate reward -- V_k is a standing critic estimate of
                    # agent k's expected return, so this penalizes an agent for being
                    # ahead/behind in expected long-run outcome, not just this step's
                    # payoff.
                    if config["PARAMETER_SHARING"]:
                        value_stack = value.reshape(env.num_agents, config["NUM_ENVS"])
                    else:
                        value_stack = jnp.stack(value, axis=0)  # (num_agents, NUM_ENVS)
                    v_k = value_stack[:, None, :]  # (k, 1, NUM_ENVS)
                    v_j = value_stack[None, :, :]  # (1, j, NUM_ENVS)
                    disadvantageous = jnp.maximum(v_j - v_k, 0.0)  # (k, j, NUM_ENVS)
                    advantageous = jnp.maximum(v_k - v_j, 0.0)     # (k, j, NUM_ENVS)
                    not_self = (1.0 - jnp.eye(env.num_agents))[:, :, None]
                    n_others = env.num_agents - 1
                    value_inequity_penalty = (
                        value_inequity_alpha * jnp.sum(disadvantageous * not_self, axis=1)
                        + value_inequity_beta * jnp.sum(advantageous * not_self, axis=1)
                    ) / n_others  # (k=num_agents, NUM_ENVS)
                    reward = reward - value_inequity_penalty.T  # (NUM_ENVS, num_agents)

                # Snapshot reward right before any influence-reward shaping gets added
                # below, so the actual per-agent bonus (whatever it turns out to be --
                # constant beta, ANNEALED beta in the lstm_influence branch below, with
                # or without a done-step mask) can be recovered afterwards as a measured
                # `reward - reward_before_influence` delta instead of re-deriving each
                # branch's exact formula/schedule here.
                reward_before_influence = reward

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
                    # Policy-state resets always happen -- the architecture (carry,
                    # previous-action input) exists regardless of whether an intrinsic
                    # reward is being computed from it.
                    policy_hstate_next = [reset_hstate(h, done_env) for h in new_policy_hstate]
                    if value_influence_use_qnet:
                        # new_qnet_hstate is produced by whichever mult_sep/sum_sep/
                        # diff_sep/diff2_sep block ran above, from its REAL (not
                        # counterfactual) qnet forward call -- same reset convention
                        # as policy_hstate_next.
                        qnet_hstate_next = [reset_hstate(h, done_env) for h in new_qnet_hstate]
                    joint_action = jnp.stack(env_act, axis=-1)  # (NUM_ENVS, num_agents) -- THIS step's real action
                    # -1 (not 0) on reset -- see the sentinel comment at initialization.
                    prev_joint_action_next = jnp.where(done_env[:, None], -1, joint_action)
                    if influence_reward:
                        influence = influence * (1.0 - done_env.astype(jnp.float32))[None, :]
                        reward = reward + beta * influence.T
                        prev_action_probs_next = jnp.where(
                            done_env[None, :, None], jnp.full_like(cond_probs, 1.0 / action_dim), cond_probs
                        )
                    else:
                        # Never read when influence_reward is off (it only feeds the
                        # marginalization above) -- carry it through unchanged rather
                        # than spend a jnp.where recomputing something unused.
                        prev_action_probs_next = prev_action_probs
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
                        # ---------------------------------------------------------------------------
                        # FEEDFORWARD (NO LSTM, NO QNET) VALUE-INFLUENCE PATH
                        # ---------------------------------------------------------------------------
                        # Used when the policy has no recurrent state at all (plain ActorCritic). No
                        # network shortcut is available for "what would agent j do/have differently"
                        # -- the counterfactual is obtained by literally re-running env.step with the
                        # SAME rng_step and pre-step env_state_t but agent k's action swapped, then
                        # feeding the resulting REAL counterfactual observation through the network.
                        # More expensive (action_dim extra env.step calls per agent k per timestep)
                        # but exact, unlike the LSTM paths' network-level approximation.
                        act_probs = pi.probs.reshape(env.num_agents, config["NUM_ENVS"], action_dim)

                        # Real conditional value: cond_value is agent j's value estimate under the REAL
                        # post-step observation (i.e. under the joint action that actually happened).
                        # Only used when value_influence_mode is set.
                        # Real conditional: what agents actually do next, given the real
                        # joint action that was actually taken -- already have obsv for free.
                        # cond_value (the REAL next-obs value estimate per agent) is only
                        # used when value_influence_mode is set -- it's free from the same forward
                        # pass that produces cond_probs.
                        obsv_flat = jnp.transpose(obsv, (1, 0, 2, 3, 4)).reshape(-1, *obs_shape)
                        cond_pi, cond_value = network.apply(train_state.params, obsv_flat)
                        cond_probs = cond_pi.probs.reshape(env.num_agents, config["NUM_ENVS"], action_dim)
                        cond_value = cond_value.reshape(env.num_agents, config["NUM_ENVS"])

                        kl_list = []
                        delta_v_list = []  # only populated when value_influence_mode is not None
                        for k in range(env.num_agents):
                            # Real resimulation of the environment: swap only agent k's action for each
                            # hypothetical a_idx, keep every other agent's real action, and re-run
                            # env.step with the SAME rng_step/env_state_t as the real step -- this
                            # produces the actual observation that would have resulted, not an
                            # approximation.
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
                            cf_pi, cf_value = network.apply(train_state.params, cf_obsv_flat)
                            cf_probs = cf_pi.probs.reshape(
                                action_dim, env.num_agents, config["NUM_ENVS"], action_dim
                            )
                            cf_value = cf_value.reshape(action_dim, env.num_agents, config["NUM_ENVS"])

                            # Marginalize the counterfactuals over agent k's own real policy.
                            marginal_probs = jnp.einsum("ea,ajet->jet", act_probs[k], cf_probs)
                            kl_per_j = _safe_kl(cond_probs, marginal_probs)  # (num_agents, NUM_ENVS)
                            kl_list.append(kl_per_j)

                            if value_influence_mode is not None:
                                # delta_v_(k->j): cond_value (real value under k's real action) minus
                                # marginal_value (expected value had k's action been drawn from k's own real
                                # policy instead) -- reuses the same counterfactual observations/weights
                                # already computed above for marginal_probs, just applied to the value head.
                                # ΔV_(k->j) is how much agent j's value under k's REAL
                                # action (cond_value, from the real resimulated next obs)
                                # differs from what j's value would have been on average
                                # had k's action been drawn from k's own real policy
                                # instead (marginal_value) -- reusing the exact same
                                # counterfactual observations/weights already computed
                                # for marginal_probs above, just applied to the value
                                # head instead of the policy head.
                                marginal_value = jnp.einsum("ea,aje->je", act_probs[k], cf_value)  # (j, e)
                                delta_v_list.append(cond_value - marginal_value)  # (j, e)

                        kl_stack = jnp.stack(kl_list, axis=0)  # (k, j, NUM_ENVS)
                        if value_influence_mode is None:
                            combined = kl_stack
                        else:
                            delta_v_stack = jnp.stack(delta_v_list, axis=0)  # (k, j, NUM_ENVS)
                            if value_influence_mode == "mult":
                                # r^WAI = I * ΔV
                                combined = kl_stack * delta_v_stack
                                value_term = delta_v_stack
                            elif value_influence_mode == "sum":
                                combined = kl_stack + delta_v_stack
                                value_term = delta_v_stack
                            else:
                                # delta_v_(k->j) minus delta_v_(j->k): how much k
                                # influenced j's value, net of how much j influenced k's
                                # value back.
                                reverse = jnp.transpose(delta_v_stack, (1, 0, 2))
                                net_delta_v = delta_v_stack - reverse
                                combined = (
                                    kl_stack + net_delta_v if value_influence_mode == "diff"
                                    else kl_stack * net_delta_v  # "diff2"
                                )
                                value_term = net_delta_v

                            # Raw per-agent breakdown -- see the matching comment in the
                            # lstm_influence branch above.
                            influence_kl_component = jnp.sum(
                                jnp.where(not_self_mat[:, :, None], kl_stack, 0.0), axis=1
                            )  # (k, NUM_ENVS)
                            influence_value_component = jnp.sum(
                                jnp.where(not_self_mat[:, :, None], value_term, 0.0), axis=1
                            )
                            influence_kl_times_value_component = jnp.sum(
                                jnp.where(not_self_mat[:, :, None], kl_stack * value_term, 0.0), axis=1
                            )

                        influence = [
                            jnp.sum(jnp.where(not_self_mat[k][:, None], combined[k], 0.0), axis=0)
                            for k in range(env.num_agents)
                        ]
                    else:
                        # Same idea, but each agent j's conditional/counterfactual query
                        # goes through agent j's OWN network/params rather than one
                        # shared call -- that's literally agent j's real policy, so this
                        # is the exact (not approximated) model of every other agent.
                        cond_out = [
                            network[j].apply(train_state[j].params, obsv[:, j])
                            for j in range(env.num_agents)
                        ]
                        cond_probs = jnp.stack(
                            [out[0].probs for out in cond_out], axis=0
                        )  # (num_agents, NUM_ENVS, action_dim)
                        # cond_value (REAL next-obs value per agent) is only used when
                        # value_influence_mode is set -- free from the same forward pass as cond_probs.
                        cond_value = jnp.stack(
                            [out[1] for out in cond_out], axis=0
                        )  # (num_agents, NUM_ENVS)

                        kl_list = []
                        delta_v_list = []  # only populated when value_influence_mode is not None
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
                            cf_out = [
                                network[j].apply(
                                    train_state[j].params,
                                    cf_obsv_all[:, :, j].reshape(-1, *obs_shape),
                                )
                                for j in range(env.num_agents)
                            ]
                            cf_probs = jnp.stack(
                                [
                                    out[0].probs.reshape(action_dim, config["NUM_ENVS"], action_dim)
                                    for out in cf_out
                                ],
                                axis=1,
                            )  # (a, j, e, t)
                            cf_value = jnp.stack(
                                [out[1].reshape(action_dim, config["NUM_ENVS"]) for out in cf_out],
                                axis=1,
                            )  # (a, j, e)

                            # Marginalize the counterfactuals over agent k's own real policy.
                            marginal_probs = jnp.einsum("ea,ajet->jet", pi_list[k].probs, cf_probs)
                            kl_per_j = _safe_kl(cond_probs, marginal_probs)  # (num_agents, NUM_ENVS)
                            kl_list.append(kl_per_j)

                            if value_influence_mode is not None:
                                # Same ΔV_(k->j) as the PARAMETER_SHARING branch above,
                                # just using each agent's OWN network for both the real
                                # and counterfactual value estimates.
                                marginal_value = jnp.einsum("ea,aje->je", pi_list[k].probs, cf_value)  # (j, e)
                                delta_v_list.append(cond_value - marginal_value)  # (j, e)

                        kl_stack = jnp.stack(kl_list, axis=0)  # (k, j, NUM_ENVS)
                        if value_influence_mode is None:
                            combined = kl_stack
                        else:
                            delta_v_stack = jnp.stack(delta_v_list, axis=0)  # (k, j, NUM_ENVS)
                            if value_influence_mode == "mult":
                                # r^WAI = I * ΔV
                                combined = kl_stack * delta_v_stack
                                value_term = delta_v_stack
                            elif value_influence_mode == "sum":
                                combined = kl_stack + delta_v_stack
                                value_term = delta_v_stack
                            else:
                                # delta_v_(k->j) minus delta_v_(j->k): how much k
                                # influenced j's value, net of how much j influenced k's
                                # value back.
                                reverse = jnp.transpose(delta_v_stack, (1, 0, 2))
                                net_delta_v = delta_v_stack - reverse
                                combined = (
                                    kl_stack + net_delta_v if value_influence_mode == "diff"
                                    else kl_stack * net_delta_v  # "diff2"
                                )
                                value_term = net_delta_v

                            # Raw per-agent breakdown -- see the matching comment in the
                            # lstm_influence branch above.
                            influence_kl_component = jnp.sum(
                                jnp.where(not_self_mat[:, :, None], kl_stack, 0.0), axis=1
                            )  # (k, NUM_ENVS)
                            influence_value_component = jnp.sum(
                                jnp.where(not_self_mat[:, :, None], value_term, 0.0), axis=1
                            )
                            influence_kl_times_value_component = jnp.sum(
                                jnp.where(not_self_mat[:, :, None], kl_stack * value_term, 0.0), axis=1
                            )

                        influence = [
                            jnp.sum(jnp.where(not_self_mat[k][:, None], combined[k], 0.0), axis=0)
                            for k in range(env.num_agents)
                        ]

                    influence = jnp.stack(influence, axis=0)  # (num_agents, NUM_ENVS)

                    current_timestep = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
                    beta = config["INFLUENCE_WEIGHT"]
                    done_env = done["__all__"]
                    influence = influence * (1.0 - done_env.astype(jnp.float32))[None, :]
                    reward = reward + beta * influence.T

                # current_timestep = update_step*config["NUM_STEPS"]*config["NUM_ENVS"]
                # shaped_reward = compute_grouped_rewards(reward)
                # reward = jax.tree.map(lambda x,y: x*rew_shaping_anneal_org(current_timestep)+y*rew_shaping_anneal(current_timestep), reward, shaped_reward)

                # The actual per-agent influence-reward bonus that ended up in `reward`
                # this step, whatever branch/schedule/mask produced it above -- see the
                # reward_before_influence comment. Always (NUM_ENVS, num_agents), unlike
                # `influence` itself whose shape/orientation differs per branch.
                influence_contribution = reward - reward_before_influence

                if config["PARAMETER_SHARING"]:
                    info = jax.tree.map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                    if influence_reward and not recurrent_moa:
                        info["influence_reward"] = influence_contribution.reshape(-1)
                    if value_influence_mode is not None:
                        # Raw (un-scaled, pre-beta/lambda) breakdown of the two signals
                        # combined into influence_reward above -- lets you see "just I",
                        # "just the value term" (delta_v for mult/sum, net delta_v for
                        # diff/diff2), and their raw product separately, rather than only
                        # the final applied reward.
                        info["influence_kl"] = influence_kl_component.T.reshape(-1)
                        info["influence_value"] = influence_value_component.T.reshape(-1)
                        info["influence_kl_times_value"] = influence_kl_times_value_component.T.reshape(-1)
                    if value_inequity_aversion:
                        info["value_inequity_penalty"] = value_inequity_penalty.T.reshape(-1)
                    info["extrinsic_reward"] = raw_env_reward.reshape(-1)
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
                        info_i["influence_reward"] = influence_contribution[:, i].reshape((config["NUM_ACTORS"], 1))
                        if value_inequity_aversion:
                            info_i["value_inequity_penalty"] = value_inequity_penalty[i].reshape((config["NUM_ACTORS"], 1))
                        info_i["extrinsic_reward"] = raw_env_reward[:, i].reshape((config["NUM_ACTORS"], 1))
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
                        if influence_reward:
                            info_i["influence_reward"] = influence_contribution[:, i].reshape((config["NUM_ACTORS"], 1))
                        if value_influence_mode is not None:
                            # Raw per-agent breakdown -- see the matching comment in the
                            # PARAMETER_SHARING branch above.
                            info_i["influence_kl"] = influence_kl_component[i].reshape((config["NUM_ACTORS"], 1))
                            info_i["influence_value"] = influence_value_component[i].reshape((config["NUM_ACTORS"], 1))
                            info_i["influence_kl_times_value"] = influence_kl_times_value_component[i].reshape((config["NUM_ACTORS"], 1))
                        if value_inequity_aversion:
                            info_i["value_inequity_penalty"] = value_inequity_penalty[i].reshape((config["NUM_ACTORS"], 1))
                        info_i["extrinsic_reward"] = raw_env_reward[:, i].reshape((config["NUM_ACTORS"], 1))
                        if value_influence_use_qnet:
                            # The REAL (non-counterfactual) qnet forward value on
                            # joint_action_onehot_current -- reused (not recomputed) as
                            # the "value" side of the extrinsic-only GAE target computed
                            # for qnet's own regression target in _update_step.
                            info_i["qnet_value"] = real_qvalue[i].reshape((config["NUM_ACTORS"], 1))
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
                            info_i["influence_reward"] = influence_contribution[:, i].reshape((config["NUM_ACTORS"], 1))
                        if value_influence_mode is not None:
                            # Raw per-agent breakdown -- see the matching comment in the
                            # PARAMETER_SHARING branch above.
                            info_i["influence_kl"] = influence_kl_component[i].reshape((config["NUM_ACTORS"], 1))
                            info_i["influence_value"] = influence_value_component[i].reshape((config["NUM_ACTORS"], 1))
                            info_i["influence_kl_times_value"] = influence_kl_times_value_component[i].reshape((config["NUM_ACTORS"], 1))
                        if value_inequity_aversion:
                            info_i["value_inequity_penalty"] = value_inequity_penalty[i].reshape((config["NUM_ACTORS"], 1))
                        info_i["extrinsic_reward"] = raw_env_reward[:, i].reshape((config["NUM_ACTORS"], 1))
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
                    if value_influence_use_qnet:
                        runner_state = (train_state, env_state, obsv, policy_hstate_next, qnet_hstate_next, prev_joint_action_next, prev_action_probs_next, update_step, rng)
                    else:
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
                if value_influence_use_qnet:
                    train_state, env_state, last_obs, policy_hstate, qnet_hstate, prev_joint_action, prev_action_probs, update_step, rng = runner_state
                else:
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

                if value_influence_use_qnet:
                    # ---------------------------------------------------------------------------
                    # TERMINAL BOOTSTRAP VALUE FOR THE QNET (last_qnet_val)
                    # ---------------------------------------------------------------------------
                    # At the end of the rollout there is no real "next action" to hold the other
                    # agents fixed at (unlike the mid-rollout delta_v counterfactual, which holds
                    # every other agent at their real taken action). Instead, take the full
                    # expectation over the joint policy: enumerate every possible joint action
                    # (the static action_dim**num_agents grid, cheap since both are fixed Python
                    # ints), weight each combination by the product of each agent's real policy
                    # probability, and evaluate the qnet (via its frozen target params) at every
                    # combination.
                    # Terminal GAE bootstrap for qnet's OWN regression target (see the
                    # extrinsic-only _calculate_gae calls below). There's no real
                    # action at this boundary (the rollout just ended), so unlike the
                    # mid-rollout delta_v counterfactual -- which holds every other
                    # agent at their REAL taken action -- there's nothing to hold "the
                    # others" fixed at here. Full expectation over the joint policy
                    # instead: enumerate the static action_dim**num_agents joint-action
                    # grid once (both fixed Python ints -- cheap, done once per update,
                    # not per step), weight each combo by the product of each agent's
                    # real policy probability, evaluate qnet at every combo. A separate
                    # (not reused) forward pass through the policy here, so nothing
                    # about the last_val computation above is touched.
                    last_pi_probs = []
                    for i in range(env.num_agents):
                        _, (last_pi_i, _) = network[i].apply(
                            train_state[i].params, policy_hstate[i], last_obs_batch[i], prev_joint_action_onehot
                        )
                        last_pi_probs.append(last_pi_i.probs)  # (NUM_ENVS, action_dim)

                    # All possible joint actions, enumerated once per update (not per step).
                    joint_action_grid = jnp.array(
                        list(itertools.product(range(action_dim), repeat=env.num_agents))
                    )  # (action_dim**num_agents, num_agents)
                    joint_action_grid_onehot = jax.nn.one_hot(joint_action_grid, action_dim)  # (A, num_agents, action_dim)

                    # Probability of each joint-action combination under the current real joint
                    # policy: product of each agent's probability of its assigned action in that
                    # combination.
                    # Weight of each joint-action combo under the current REAL joint
                    # policy: product of each agent's probability of its assigned
                    # action in that combo.
                    combo_weights = jnp.ones((joint_action_grid.shape[0], config["NUM_ENVS"]))
                    for a in range(env.num_agents):
                        combo_weights = combo_weights * last_pi_probs[a][:, joint_action_grid[:, a]].T

                    last_qnet_val = []
                    for i in range(env.num_agents):
                        def _q_at_combo(combo_onehot, i=i):
                            combo_onehot_batched = jnp.broadcast_to(
                                combo_onehot[None, :, :], (config["NUM_ENVS"], env.num_agents, action_dim)
                            )
                            _, qv = qnet[i].apply(
                                qnet_train_state[i].target_network_params, qnet_hstate[i],
                                last_obs_batch[i], combo_onehot_batched,
                            )
                            return qv  # (NUM_ENVS,)

                        # Expected Q-value under the full joint-action distribution -- this is the
                        # qnet's terminal bootstrap value, used by qnet_targets' GAE call above.
                        q_grid = jax.vmap(_q_at_combo)(joint_action_grid_onehot)  # (A, NUM_ENVS)
                        last_qnet_val.append(jnp.sum(combo_weights * q_grid, axis=0))  # (NUM_ENVS,)
                    last_qnet_val = jnp.stack(last_qnet_val, axis=0)  # (num_agents, NUM_ENVS)
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

            if value_influence_use_qnet:
                # ---------------------------------------------------------------------------
                # QNET'S OWN GAE TARGET (extrinsic reward only)
                # ---------------------------------------------------------------------------
                # The qnet must never regress toward a return that already has delta_v folded
                # into it (that would be circular -- the reward-generating critic training on
                # a target that already contains its own signal). So this reuses the same
                # generic _calculate_gae closure but substitutes traj_batch[i].info["qnet_value"]
                # for `value` and traj_batch[i].info["extrinsic_reward"] (raw, unshaped env
                # reward) for `reward`, bootstrapped from last_qnet_val (computed below). This
                # never touches the policy's own `targets` or ActorCriticLSTM's value head.
                # Independent, extrinsic-only GAE target for qnet's own regression --
                # reuses the same _calculate_gae closure (it's generic over
                # .done/.value/.reward) but with EXTRINSIC reward and qnet's OWN value
                # estimates substituted in via ._replace(), and last_qnet_val (computed
                # above, self-contained) as the bootstrap. Never touches the policy's
                # own `targets` above or ActorCriticLSTM's value head -- this is the
                # fix for the circularity the mentor's guidance was about: Q_j must
                # never regress toward a return that already has delta_v folded in.
                qnet_targets = []
                for i in range(env.num_agents):
                    _, qnet_targets_i = _calculate_gae(
                        traj_batch[i]._replace(
                            value=traj_batch[i].info["qnet_value"].squeeze(-1),
                            reward=traj_batch[i].info["extrinsic_reward"].squeeze(-1),
                        ),
                        last_qnet_val[i],
                    )
                    qnet_targets.append(qnet_targets_i)
                qnet_targets = jnp.stack(qnet_targets, axis=0)  # (num_agents, NUM_STEPS, NUM_ENVS)

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
                    # The actual (post-shaping) reward this agent received isn't part
                    # of `info` -- it's traj_batch[i].reward, a separate Transition
                    # field -- so it never got logged to WandB on its own before.
                    metric_i['reward'] = traj_batch[i].reward
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

            def _flush_group_b_metrics(buffer, flush_update_step):
                # Host-side numpy port of the Group B behavioral metrics (conditional
                # cooperation, retaliation lag, forgiveness rate, influence/delta_v vs
                # extrinsic-reward correlation) -- only ever called every
                # metrics_window_updates updates (see the jax.lax.cond gate below), so
                # the Python-level loops here (retaliation lag's forward search in
                # particular isn't easily vectorized -- it needs early termination on
                # the first retaliation or episode boundary) are an acceptable
                # trade-off, not something that runs every step. Assumes 2 agents
                # (Coin Game) -- "the partner" of agent i is agent 1-i; extending to
                # more agents would need a real per-pair-of-agents redesign, since
                # eat_other_coins doesn't distinguish WHICH other agent was harmed.
                num_a = env.num_agents
                num_envs = config["NUM_ENVS"]
                T = metrics_window_updates * num_steps
                own = np.asarray(buffer["own"]).reshape(T, num_a, num_envs)
                other = np.asarray(buffer["other"]).reshape(T, num_a, num_envs)
                extrinsic = np.asarray(buffer["extrinsic_reward"]).reshape(T, num_a, num_envs)
                done_arr = np.asarray(buffer["done"]).reshape(T, num_envs)
                has_kl = "influence_kl" in buffer
                has_value = "influence_value" in buffer
                # Present whenever ANY influence-reward mechanism is active, including
                # "plain Jaques" (unsigned KL-only influence, no delta_v at all --
                # value_influence_mode is None but influence_reward is still True) --
                # the guide explicitly wants metric 6 to cover that case too, not just
                # the value-based variants that also populate influence_kl/_value.
                has_reward = "influence_reward" in buffer
                infl_kl = np.asarray(buffer["influence_kl"]).reshape(T, num_a, num_envs) if has_kl else None
                infl_value = np.asarray(buffer["influence_value"]).reshape(T, num_a, num_envs) if has_value else None
                infl_reward = np.asarray(buffer["influence_reward"]).reshape(T, num_a, num_envs) if has_reward else None

                window, n_bins, max_lag = 20, 5, 20
                log_payload = {"update_step": int(flush_update_step)}

                def episode_sums(values):
                    sums = []
                    for e in range(num_envs):
                        running = 0.0
                        for t in range(T):
                            running += values[t, e]
                            if done_arr[t, e]:
                                sums.append(running)
                                running = 0.0
                    return np.array(sums)

                for i in range(num_a):
                    j = 1 - i
                    own_i, other_i = own[:, i, :], other[:, i, :]
                    own_j, other_j = own[:, j, :], other[:, j, :]

                    # Metric 3: conditional cooperation -- partner's trailing
                    # coop-rate (rolling window, via cumsum so this stays vectorized)
                    # bucketed, then focal agent's coop rate within each bucket.
                    bucket_sums = np.zeros(n_bins)
                    bucket_counts = np.zeros(n_bins)
                    bins = np.linspace(0, 1, n_bins + 1)
                    for e in range(num_envs):
                        own_cum = np.concatenate([[0.0], np.cumsum(own_j[:, e])])  # length T+1
                        oth_cum = np.concatenate([[0.0], np.cumsum(other_j[:, e])])
                        # own_cum[t] - own_cum[t-window] = trailing sum over the window
                        # ending at t, for t in [window, T-1] -- matches
                        # partner_rate[window:]'s (T-window)-length range exactly.
                        o = own_cum[window:T] - own_cum[0:T - window]
                        d = oth_cum[window:T] - oth_cum[0:T - window]
                        denom = o + d
                        partner_rate = np.full(T, np.nan)
                        partner_rate[window:] = np.where(denom > 0, o / np.maximum(denom, 1e-8), np.nan)
                        eating_events = (own_i[:, e] + other_i[:, e]) > 0
                        valid = eating_events & ~np.isnan(partner_rate)
                        if not valid.any():
                            continue
                        bucket_idx = np.clip(np.digitize(partner_rate[valid], bins) - 1, 0, n_bins - 1)
                        focal_coop = (own_i[valid, e] > 0).astype(float)
                        for b in range(n_bins):
                            mask = bucket_idx == b
                            if mask.any():
                                bucket_sums[b] += focal_coop[mask].sum()
                                bucket_counts[b] += mask.sum()
                    for b in range(n_bins):
                        if bucket_counts[b] > 0:
                            log_payload[f"agent{i}/conditional_coop_bin{b}"] = bucket_sums[b] / bucket_counts[b]

                    # Metrics 4-5: retaliation lag + forgiveness rate. other_i[t] = i
                    # harmed the partner (retaliation); other_j[t] = i WAS harmed at
                    # t. Censored (never-retaliated-within-max_lag) events are counted
                    # separately, never dropped -- right-censoring, not zero.
                    lags = []
                    censored = 0
                    for e in range(num_envs):
                        oth_i_e, oth_j_e, d_e = other_i[:, e], other_j[:, e], done_arr[:, e]
                        for t in range(T):
                            if oth_j_e[t] > 0:
                                retaliated = False
                                end = min(T, t + 1 + max_lag)
                                for t_prime in range(t + 1, end):
                                    if d_e[t_prime - 1]:
                                        break
                                    if oth_i_e[t_prime] > 0:
                                        lags.append(t_prime - t)
                                        retaliated = True
                                        break
                                if not retaliated:
                                    censored += 1
                    total_harm_events = len(lags) + censored
                    if total_harm_events > 0:
                        log_payload[f"agent{i}/forgiveness_rate"] = censored / total_harm_events
                    if lags:
                        log_payload[f"agent{i}/retaliation_lag_mean"] = float(np.mean(lags))
                        log_payload[f"agent{i}/retaliation_lag_median"] = float(np.median(lags))

                    # Metric 6: influence/delta_v vs EXTRINSIC (never shaped) reward,
                    # segmented into done-delimited episodes -- correlating against the
                    # shaped reward would trivially inflate this, since it already
                    # contains a scaled copy of the influence/delta_v signal.
                    if has_kl or has_value or has_reward:
                        reward_per_ep = episode_sums(extrinsic[:, i, :])
                        for name, arr in (("kl", infl_kl), ("value", infl_value), ("reward", infl_reward)):
                            if arr is None:
                                continue
                            infl_per_ep = episode_sums(arr[:, i, :])
                            if (
                                len(infl_per_ep) > 1
                                and len(infl_per_ep) == len(reward_per_ep)
                                and np.std(infl_per_ep) > 0
                                and np.std(reward_per_ep) > 0
                            ):
                                rho, _ = spearmanr(infl_per_ep, reward_per_ep)
                                r, _ = pearsonr(infl_per_ep, reward_per_ep)
                                log_payload[f"agent{i}/influence_{name}_reward_corr_spearman"] = rho
                                log_payload[f"agent{i}/influence_{name}_reward_corr_pearson"] = r

                wandb.log(log_payload, step=int(flush_update_step))

            update_step = update_step + 1
            num_agents = env.num_agents
            num_steps = config["NUM_STEPS"]
            num_inner_steps = config["ENV_KWARGS"]["num_inner_steps"]

            def _scale_eat_keys(d):
                # eat_own_coins/eat_other_coins average near-zero per step (coin
                # pickup is a rare event) -- scaling by num_inner_steps turns that
                # tiny per-step expectation into an expected count over an inner
                # episode, which is what's actually being reported here.
                for key in ("eat_own_coins", "eat_other_coins"):
                    if key in d:
                        d[key] = d[key] * num_inner_steps
                return d

            def _welfare_metrics(reward_means):
                # reward_means: (num_agents,) mean reward this window, one entry per
                # agent -- the same "reward by agent" values being logged below, so
                # U/E are trivially consistent with them.
                total = jnp.sum(reward_means)
                diffs = jnp.abs(reward_means[:, None] - reward_means[None, :])
                # Standard Gini coefficient (mean-absolute-difference form); abs()
                # + eps on the denominator guards against a near-zero or negative
                # total (early training / heavily-penalized windows) blowing this up
                # to +-inf instead of just reporting a degenerate value.
                gini = jnp.sum(diffs) / (2.0 * num_agents * jnp.abs(total) + 1e-8)
                return {
                    "Overall/U_utilitarian_reward": total,
                    "Overall/E_equality": 1.0 - gini,
                }

            if config["PARAMETER_SHARING"]:
                # traj_batch.info's fields were flattened ENV-major a few hundred
                # lines up in _env_step (plain `.reshape((NUM_ACTORS,))` on a
                # (NUM_ENVS, num_agents) array: index = env_idx*num_agents +
                # agent_idx). traj_batch.reward was flattened AGENT-major instead
                # (via batchify(), which stacks per-agent slices before reshaping:
                # index = agent_idx*NUM_ENVS + env_idx). Both conventions collapse
                # to the same scalar under a full mean(), which is why this
                # discrepancy never mattered before -- but recovering a per-agent
                # breakdown means respecting whichever convention actually produced
                # each field, not applying one blindly to both.
                num_envs = config["NUM_ENVS"]
                per_agent_info = {
                    key: jnp.mean(
                        value.reshape(num_steps, num_envs, num_agents), axis=(0, 1)
                    )
                    for key, value in traj_batch.info.items()
                    if value.shape == (num_steps, config["NUM_ACTORS"])
                }
                reward_time = traj_batch.reward.reshape(num_steps, num_agents, num_envs)
                reward_means = jnp.mean(reward_time, axis=(0, 2))  # (num_agents,)

                metric = jax.tree.map(lambda x: x.mean(), traj_batch.info)
                metric = _scale_eat_keys(metric)
                metric["update_step"] = update_step
                metric["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]

                for i in range(num_agents):
                    metric[f"agent{i}/reward"] = reward_means[i]
                    for key, value in _scale_eat_keys(
                        {k: v[i] for k, v in per_agent_info.items()}
                    ).items():
                        metric[f"agent{i}/{key}"] = value
                    # P (own-coin ratio): fraction of this agent's coin pickups that
                    # were its own, vs the other agent's -- the direct behavioral
                    # signature of cooperation vs defection. _scale_eat_keys' constant
                    # multiplier cancels out in a ratio, so raw per_agent_info is fine
                    # here without going through it. Logged every update, so this same
                    # scalar viewed as a wandb line chart already IS the "P(Own Coin)
                    # trajectory over training" plot -- no separate computation needed.
                    own_i = per_agent_info["eat_own_coins"][i]
                    other_i = per_agent_info["eat_other_coins"][i]
                    metric[f"agent{i}/P_own_coin_ratio"] = own_i / (own_i + other_i + 1e-8)

                metric.update(_welfare_metrics(reward_means))
                # original_rewards is coin_game.py's own pre-ANY-shaping snapshot --
                # set in every branch (plain/shared/svo/interest/inequity_aversion), so
                # reading it directly is simpler AND more robust than reconstructing it
                # by adding back individual training-loop-level shaping components
                # (influence_reward/value_inequity_penalty): it also correctly excludes
                # ENV_KWARGS.inequity_aversion's penalty, which reward_means alone
                # cannot recover (that shaping happens inside the env, before `reward`
                # is ever returned -- there's no separate delta to add back here).
                value_inequity_penalty_means = per_agent_info.get(
                    "value_inequity_penalty", jnp.zeros(num_agents)
                )
                original_reward_means = per_agent_info["original_rewards"]
                metric["Overall/env_reward"] = jnp.sum(original_reward_means)
                # Just the intrinsic/inequity-aversion component on its own (not netted
                # against env reward like Overall/env_reward above) -- zero whenever
                # VALUE_INEQUITY_AVERSION is disabled.
                metric["Overall/value_inequity_penalty"] = jnp.sum(value_inequity_penalty_means)
                # Sustainability: average within-window timestep at which each
                # agent's reward was positive, averaged across agents -- higher
                # means reward collection is more spread out/deferred rather than
                # front-loaded. reward_time is agent-major, see the comment above.
                time_idx = jnp.arange(num_steps, dtype=jnp.float32)[:, None, None]
                positive = (reward_time > 0).astype(jnp.float32)
                per_agent_s = jnp.sum(time_idx * positive, axis=(0, 2)) / jnp.maximum(
                    jnp.sum(positive, axis=(0, 2)), 1.0
                )
                metric["Overall/S_sustainability"] = jnp.mean(per_agent_s)
            else:
                # Sustainability needs the raw per-timestep reward (before the
                # per-agent mean below collapses the time axis away).
                time_idx = jnp.arange(num_steps, dtype=jnp.float32)[:, None]
                per_agent_s = []
                for i in range(num_agents):
                    positive = (traj_batch[i].reward > 0).astype(jnp.float32)
                    per_agent_s.append(
                        jnp.sum(time_idx * positive) / jnp.maximum(jnp.sum(positive), 1.0)
                    )
                sustainability = jnp.mean(jnp.stack(per_agent_s))

                metric = jax.tree.map(lambda x: x.mean(), metric)
                for i in range(num_agents):
                    _scale_eat_keys(metric[i])
                    # P (own-coin ratio) -- see the matching comment in the
                    # PARAMETER_SHARING branch above. _scale_eat_keys' constant
                    # multiplier already ran above but cancels out in a ratio regardless.
                    own_i = metric[i]["eat_own_coins"]
                    other_i = metric[i]["eat_other_coins"]
                    metric[i]["P_own_coin_ratio"] = own_i / (own_i + other_i + 1e-8)

                reward_means = jnp.stack([metric[i]["reward"] for i in range(num_agents)])
                value_inequity_penalty_means = jnp.stack([
                    metric[i].get("value_inequity_penalty", jnp.array(0.0))
                    for i in range(num_agents)
                ])
                # original_rewards is coin_game.py's own pre-ANY-shaping snapshot --
                # see the matching comment in the PARAMETER_SHARING branch above for why
                # reading it directly (rather than reconstructing from reward_means +
                # influence/value_inequity components) is the robust choice, including
                # for ENV_KWARGS.inequity_aversion runs.
                original_reward_means = jnp.stack([
                    metric[i]["original_rewards"] for i in range(num_agents)
                ])

                merged = {}
                for i in range(num_agents):
                    for key, value in metric[i].items():
                        merged[f"agent{i}/{key}"] = value
                metric = merged
                metric["update_step"] = update_step
                metric["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
                metric.update(_welfare_metrics(reward_means))
                metric["Overall/env_reward"] = jnp.sum(original_reward_means)
                metric["Overall/value_inequity_penalty"] = jnp.sum(value_inequity_penalty_means)
                metric["Overall/S_sustainability"] = sustainability

            if value_influence_use_qnet:
                # ---------------------------------------------------------------------------
                # QNET REPLAY BUFFER WRITE + PERIODIC LEARN PHASE + TARGET-NETWORK SYNC
                # ---------------------------------------------------------------------------
                # Runs once per update (not once per env step). Three things happen here:
                #   1. Package this update's (obs, joint_action, qnet_targets, done) into a
                #      QTimestep and add it to the trajectory buffer.
                #   2. If the buffer has enough data (can_sample), sample a batch of sequences
                #      and run one gradient step on the qnet's ONLINE params (_qnet_learn);
                #      otherwise skip (_qnet_skip).
                #   3. Every VALUE_INFLUENCE_TARGET_UPDATE_INTERVAL updates, sync
                #      target_network_params from params (the only way target_network_params
                #      ever changes -- it never receives a gradient directly).
                # Fill the qnet replay buffer with this update's (obs, joint_action,
                # extrinsic-only target) sequences, then -- gated by can_sample, same
                # "cheap write every step, expensive work only when ready" pattern as
                # the Group B metrics flush below -- run a periodic learn phase and
                # target-network sync. See QTimestep's docstring for why `target` here
                # is qnet_targets (extrinsic-only, self-bootstrapped), never the
                # policy's own `targets`.
                qnet_obs = jnp.stack([traj_batch[i].obs for i in range(env.num_agents)], axis=2)
                qnet_joint_action = jnp.stack([traj_batch[i].action for i in range(env.num_agents)], axis=-1)
                qnet_joint_action_onehot = jax.nn.one_hot(qnet_joint_action, action_dim)
                qnet_target_batch = jnp.transpose(qnet_targets, (1, 2, 0))  # (NUM_STEPS, NUM_ENVS, num_agents)
                qnet_done_batch = traj_batch[0].done  # (NUM_STEPS, NUM_ENVS), synchronized across agents

                # swapaxes(0, 1) moves NUM_ENVS to the first axis, matching the buffer's
                # add_batch_size convention (set to NUM_ENVS at buffer construction).
                q_timestep = QTimestep(
                    obs=jnp.swapaxes(qnet_obs, 0, 1),                      # (NUM_ENVS, NUM_STEPS, num_agents, *obs_shape)
                    joint_action_onehot=jnp.swapaxes(qnet_joint_action_onehot, 0, 1),
                    target=jnp.swapaxes(qnet_target_batch, 0, 1),
                    done=jnp.swapaxes(qnet_done_batch, 0, 1),
                )
                qnet_buffer_state = qnet_buffer.add(qnet_buffer_state, q_timestep)

                # Sample a batch of contiguous sequences from the buffer and run one gradient
                # step per agent's qnet on its ONLINE params. The recurrent carry starts at
                # zero for each sampled sequence (since a sequence may start mid-episode) and
                # is unrolled in order via jax.lax.scan, resetting at episode boundaries
                # (reset_in). The first qnet_burn_in steps of each sequence are excluded from
                # the loss (R2D2-style burn-in): the carry hasn't had time to reach a
                # plausible mid-sequence state yet, so predictions there are unreliable and
                # would poison the gradient if included.
                qnet_burn_in = config.get("VALUE_INFLUENCE_BURN_IN", 4)

                def _qnet_learn(carry):
                    qnet_train_state, rng = carry
                    rng, sample_rng = jax.random.split(rng)
                    experience = qnet_buffer.sample(qnet_buffer_state, sample_rng).experience
                    done_seq = experience.done  # (B, T)
                    # reset_in[t] = done at t-1 (episode ended just before t) -- same
                    # convention as the PPO replay scan's reset_in in _loss_fn above.
                    reset_in = jnp.concatenate(
                        [jnp.zeros_like(done_seq[:, :1]), done_seq[:, :-1]], axis=1
                    ).astype(bool)  # (B, T)

                    new_qnet_train_state = []
                    losses = []
                    qpreds = []
                    for i in range(env.num_agents):
                        obs_i = experience.obs[:, :, i]              # (B, T, *obs_shape)
                        joint_action_onehot_seq = experience.joint_action_onehot  # (B, T, num_agents, action_dim) -- SAME joint action for every agent's Q
                        target_i = experience.target[:, :, i]        # (B, T)

                        def _loss_fn(params, i=i, obs_i=obs_i, joint_action_onehot_seq=joint_action_onehot_seq, target_i=target_i):
                            batch_size = obs_i.shape[0]
                            init_carry = ValueInfluenceEstimator.initialize_carry(batch_size, hidden_dim)

                            def _step(carry, xs):
                                obs_t, ja_t, reset_t = xs
                                carry = reset_hstate(carry, reset_t)
                                new_carry, q_t = qnet[i].apply(params, carry, obs_t, ja_t)
                                return new_carry, q_t

                            _, q_seq = jax.lax.scan(
                                _step, init_carry,
                                (
                                    jnp.swapaxes(obs_i, 0, 1),
                                    jnp.swapaxes(joint_action_onehot_seq, 0, 1),
                                    jnp.swapaxes(reset_in, 0, 1),
                                ),
                            )
                            q_seq = jnp.swapaxes(q_seq, 0, 1)  # (B, T)
                            # Burn-in: the first qnet_burn_in steps of each sampled
                            # chunk run forward (warming the carry from zero toward a
                            # plausible mid-sequence state, R2D2-style) but don't
                            # contribute to the loss -- most sampled chunks start
                            # mid-episode, so a bare zero carry misrepresents the real
                            # accumulated history at that point.
                            sq_err = (q_seq - target_i) ** 2
                            loss = jnp.mean(sq_err[:, qnet_burn_in:])
                            return loss, q_seq[:, qnet_burn_in:].mean()

                        (loss_i, qpred_i), grads_i = jax.value_and_grad(_loss_fn, has_aux=True)(
                            qnet_train_state[i].params
                        )
                        new_qnet_train_state.append(qnet_train_state[i].apply_gradients(grads=grads_i))
                        losses.append(loss_i)
                        qpreds.append(qpred_i)

                    return (new_qnet_train_state, rng), (jnp.stack(losses), jnp.stack(qpreds))

                # No-op used when the buffer doesn't yet have enough data to sample from --
                # returns the qnet_train_state unchanged and zero-valued diagnostic metrics.
                def _qnet_skip(carry):
                    qnet_train_state, rng = carry
                    return (qnet_train_state, rng), (jnp.zeros(env.num_agents), jnp.zeros(env.num_agents))

                # jax.lax.cond is required (not a Python if) since is_learn_time is a
                # dynamically-traced value under jit/scan.
                is_learn_time = qnet_buffer.can_sample(qnet_buffer_state)
                rng, _qnet_rng = jax.random.split(rng)
                (qnet_train_state, _qnet_rng), (qnet_losses, qnet_qpreds) = jax.lax.cond(
                    is_learn_time, _qnet_learn, _qnet_skip, (qnet_train_state, _qnet_rng)
                )

                # Periodic (every target_update_interval updates) sync of target_network_params
                # from params via optax.incremental_update -- tau=1.0 (default) is a full
                # hard copy; tau<1.0 would be a soft/interpolated (Polyak) update. This is the
                # ONLY way target_network_params ever changes; it never receives a direct
                # gradient. Keeping it frozen between syncs is what makes the reward signal
                # read by _env_step (Sections 5-6 above) stable across many updates, instead
                # of tracking a target that moves every gradient step.
                # Periodic hard/soft target sync -- the reward-facing copy
                # (target_network_params, read in _env_step) only changes every
                # VALUE_INFLUENCE_TARGET_UPDATE_INTERVAL updates, regardless of how
                # often the learn phase above actually fires.
                target_update_interval = config.get("VALUE_INFLUENCE_TARGET_UPDATE_INTERVAL", 10)
                tau = config.get("VALUE_INFLUENCE_TAU", 1.0)
                is_sync_time = (update_step % target_update_interval) == 0
                qnet_train_state = [
                    ts.replace(
                        target_network_params=jax.lax.cond(
                            is_sync_time,
                            lambda ts=ts: optax.incremental_update(ts.params, ts.target_network_params, tau),
                            lambda ts=ts: ts.target_network_params,
                        )
                    )
                    for ts in qnet_train_state
                ]

                # Diagnostic logging: qnet's own regression loss and mean predicted value per
                # agent, to check the qnet is actually learning to predict extrinsic return
                # correctly (independent from whether the resulting delta_v reward improves
                # policy behavior).
                for i in range(num_agents):
                    metric[f"agent{i}/qnet_loss"] = qnet_losses[i]
                    metric[f"agent{i}/qnet_pred_mean"] = qnet_qpreds[i]

            jax.debug.callback(callback, metric)

            # Write this update's raw per-timestep data into the Group B metrics
            # buffer, then flush (host-side numpy) every metrics_window_updates
            # updates. See _flush_group_b_metrics below for what actually gets
            # computed -- this just maintains the rolling (K, NUM_STEPS, ...) buffer
            # and gates the (comparatively expensive) flush to a periodic cadence,
            # same "cheap write every step, expensive work only periodically" pattern
            # already used for the Value-Influence target-network sync (Part 1 plan).
            def _stack_field(key):
                if config["PARAMETER_SHARING"]:
                    return traj_batch.info[key].reshape(
                        num_steps, config["NUM_ENVS"], num_agents
                    ).transpose(0, 2, 1)
                return jnp.stack(
                    [traj_batch[i].info[key].squeeze(-1) for i in range(num_agents)],
                    axis=1,
                )  # (NUM_STEPS, num_agents, NUM_ENVS)

            done_this = (
                traj_batch.done.reshape(num_steps, config["NUM_ENVS"], num_agents)[:, :, 0]
                if config["PARAMETER_SHARING"] else traj_batch[0].done
            )  # episodes reset synchronously, see reset_hstate's docstring
            write_idx = update_step % metrics_window_updates
            metrics_buffer = dict(metrics_buffer)
            metrics_buffer["own"] = metrics_buffer["own"].at[write_idx].set(_stack_field("eat_own_coins"))
            metrics_buffer["other"] = metrics_buffer["other"].at[write_idx].set(_stack_field("eat_other_coins"))
            metrics_buffer["extrinsic_reward"] = metrics_buffer["extrinsic_reward"].at[write_idx].set(_stack_field("extrinsic_reward"))
            metrics_buffer["done"] = metrics_buffer["done"].at[write_idx].set(done_this)
            if value_influence_mode is not None:
                metrics_buffer["influence_kl"] = metrics_buffer["influence_kl"].at[write_idx].set(_stack_field("influence_kl"))
                metrics_buffer["influence_value"] = metrics_buffer["influence_value"].at[write_idx].set(_stack_field("influence_value"))
            if influence_reward:
                metrics_buffer["influence_reward"] = metrics_buffer["influence_reward"].at[write_idx].set(_stack_field("influence_reward"))

            is_flush_time = (update_step % metrics_window_updates) == (metrics_window_updates - 1)
            jax.lax.cond(
                is_flush_time,
                lambda: jax.debug.callback(_flush_group_b_metrics, metrics_buffer, update_step),
                lambda: None,
            )

            if recurrent_moa:
                env_runner_state = (train_state, env_state, last_obs, policy_hstate, moa_hstate, update_step, rng)
            elif lstm_influence:
                if value_influence_use_qnet:
                    env_runner_state = (train_state, env_state, last_obs, policy_hstate, qnet_hstate, prev_joint_action, prev_action_probs, update_step, rng)
                else:
                    env_runner_state = (train_state, env_state, last_obs, policy_hstate, prev_joint_action, prev_action_probs, update_step, rng)
            else:
                env_runner_state = (train_state, env_state, last_obs, update_step, rng)
            if value_influence_use_qnet:
                runner_state = (env_runner_state, metrics_buffer, qnet_train_state, qnet_buffer_state)
            else:
                runner_state = (env_runner_state, metrics_buffer)
            return runner_state, metric

        # Rolling raw-per-timestep buffer feeding the live Group B behavioral metrics
        # (_flush_group_b_metrics below) -- separate from, and unrelated to, any
        # training data structure. Always (num_agents, NUM_ENVS)-shaped per field
        # regardless of PARAMETER_SHARING/independent/recurrent_moa/lstm_influence,
        # since eat_own_coins/eat_other_coins/extrinsic_reward exist for every branch.
        metrics_buffer = {
            "own": jnp.zeros((metrics_window_updates, config["NUM_STEPS"], env.num_agents, config["NUM_ENVS"])),
            "other": jnp.zeros((metrics_window_updates, config["NUM_STEPS"], env.num_agents, config["NUM_ENVS"])),
            "extrinsic_reward": jnp.zeros((metrics_window_updates, config["NUM_STEPS"], env.num_agents, config["NUM_ENVS"])),
            "done": jnp.zeros((metrics_window_updates, config["NUM_STEPS"], config["NUM_ENVS"])),
        }
        if value_influence_mode is not None:
            metrics_buffer["influence_kl"] = jnp.zeros((metrics_window_updates, config["NUM_STEPS"], env.num_agents, config["NUM_ENVS"]))
            metrics_buffer["influence_value"] = jnp.zeros((metrics_window_updates, config["NUM_STEPS"], env.num_agents, config["NUM_ENVS"]))
        if influence_reward:
            metrics_buffer["influence_reward"] = jnp.zeros((metrics_window_updates, config["NUM_STEPS"], env.num_agents, config["NUM_ENVS"]))

        rng, _rng = jax.random.split(rng)
        if recurrent_moa:
            env_runner_state = (train_state, env_state, obsv, policy_hstate, moa_hstate, 0, _rng)
        elif lstm_influence:
            if value_influence_use_qnet:
                env_runner_state = (train_state, env_state, obsv, policy_hstate, qnet_hstate, prev_joint_action, prev_action_probs, 0, _rng)
            else:
                env_runner_state = (train_state, env_state, obsv, policy_hstate, prev_joint_action, prev_action_probs, 0, _rng)
        else:
            env_runner_state = (train_state, env_state, obsv, 0, _rng)
        if value_influence_use_qnet:
            # qnet_train_state/qnet_buffer_state change once per _update_step (not
            # once per _env_step, unlike everything in env_runner_state) -- kept as
            # separate sibling elements of the outer scan carry rather than threaded
            # through _env_step's inner scan, exactly like metrics_buffer above.
            runner_state = (env_runner_state, metrics_buffer, qnet_train_state, qnet_buffer_state)
        else:
            runner_state = (env_runner_state, metrics_buffer)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        # Unwrap back to the original flat env_runner_state shape
        # (train_state, env_state, ..., update_step, rng) before returning --
        # metrics_buffer/qnet_train_state/qnet_buffer_state only needed to exist for
        # the jax.lax.scan carry above, not downstream. algorithms/IPPO/_runner.py is
        # shared across every env file and indexes out["runner_state"][0] expecting
        # train_state directly; keeping the nesting internal-only here means
        # single_run()'s checkpoint-saving code needs no changes and other env files
        # (which never gained this nesting) stay unaffected.
        env_runner_state = runner_state[0]
        return {"runner_state": env_runner_state, "metrics": metric}

    return train

# Used by algorithms/train.py to dispatch through algorithms.IPPO._runner.
SINGLE_RUN_KWARGS = {"wandb_name": "ippo_cnn_coins"}
TUNE_KWARGS       = {"sweep_name": "coins"}
