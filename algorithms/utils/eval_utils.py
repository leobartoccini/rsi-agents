"""
Shared evaluation utilities for all algorithms.

This module provides evaluation functions for trained MARL policies.
Different algorithms have slightly different evaluation patterns:
- IPPO: Uses ActorCritic with PARAMETER_SHARING logic + WandB GIF logging
- MAPPO/IRAT: Use Actor network (no critic in eval) without WandB logging
- SVO/TRANSFER: Use ActorCritic without PARAMETER_SHARING logic, no WandB logging
"""

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from pathlib import Path
import wandb

from algorithms.utils.networks import ActorCritic, ActorCriticMOA, ActorCriticMOARNN, ActorCriticLSTM, SmallActor
from algorithms.utils.data_utils import unbatchify


def evaluate_ippo(params, env, save_path, config, eval_seed=0):
    """
    Evaluation function for IPPO algorithm.

    Supports both parameter sharing and individual networks.
    Logs evaluation GIF to WandB.

    Args:
        params: Model parameters (single or list for multi-agent)
        env: Environment instance
        save_path: Path where params were saved (unused but kept for API compatibility)
        config: Configuration dictionary
        eval_seed: RNG seed for this evaluation episode. Defaults to 0 (the original,
            single-gif-per-run behavior); pass a different value per call to get a
            different rollout/gif from the same checkpoint instead of always
            replaying the identical episode.
    """
    rng = jax.random.PRNGKey(eval_seed)

    rng, _rng = jax.random.split(rng)
    obs, state = env.reset(_rng)
    done = False

    pics = []
    img = env.render(state)
    pics.append(img)

    # Extract environment name for root_dir
    env_name = config["ENV_NAME"]
    # Map environment names to evaluation directories
    env_dir_mapping = {
        "clean_up": "cleanup",
        "coin_game": "coins",
        "coop_mining": "coop_mining",
        "gift": "gift",
        "harvest_common_open": "harvest_common",
        "harvest_common_closed": "harvest_closed",
        "harvest_common_partnership": "harvest_partnership",
        "mushrooms": "mushrooms",
        "pd_arena": "pd_arena",
        "territory_open": "territory_open",
    }
    env_dir = env_dir_mapping.get(env_name, env_name)
    root_dir = f"evaluation/{env_dir}"
    path = Path(root_dir + "/state_pics")
    path.mkdir(parents=True, exist_ok=True)
    
    # --- Setup: build one recurrent MOA network + zeroed LSTM carry per agent,
    # only when RECURRENT_MOA is enabled in the config ---
    recurrent_moa = config.get("RECURRENT_MOA", False)
    if recurrent_moa:
        hidden_dim = config.get("LSTM_HIDDEN_DIM", 128)
        network = [ActorCriticMOARNN(
            action_dim=env.action_space().n,
            num_agents=env.num_agents,
            hidden_dim=hidden_dim,
            activation=config.get("ACTIVATION", "relu"),
        ) for _ in range(env.num_agents)]
        policy_hstate = [ActorCriticMOARNN.initialize_carry(1, hidden_dim) for _ in range(env.num_agents)]

    # --- Setup: build one ActorCriticLSTM network + zeroed LSTM carry + zeroed
    # previous joint action per agent, only when LSTM_INFLUENCE is enabled. Mirrors
    # ippo_cnn_coins.py's action-selection input exactly (including the observation-
    # window visibility mask), since the trained policy expects that input
    # distribution -- feeding it something different here would look degenerate in
    # the GIF even though the checkpoint itself is fine.
    lstm_influence = config.get("LSTM_INFLUENCE", False)
    if lstm_influence:
        hidden_dim = config.get("LSTM_HIDDEN_DIM", 128)
        action_dim = env.action_space().n
        network = [ActorCriticLSTM(
            action_dim=action_dim,
            num_agents=env.num_agents,
            hidden_dim=hidden_dim,
            activation=config.get("ACTIVATION", "relu"),
        ) for _ in range(env.num_agents)]
        policy_hstate = [ActorCriticLSTM.initialize_carry(1, hidden_dim) for _ in range(env.num_agents)]
        # -1, not 0: jax.nn.one_hot(-1, action_dim) is a true all-zero vector, so this
        # reliably means "no previous action" -- one_hot(0, ...) would instead claim
        # every agent specifically chose action 0, indistinguishable to the network
        # from a real action (see ippo_cnn_coins.py's matching sentinel).
        prev_joint_action = jnp.full((1, env.num_agents), -1, dtype=jnp.int32)

        def _eval_visibility(agent_locs):  # agent_locs: (num_agents, 3), single env
            start_x, start_y = jax.vmap(env.get_obs_point)(agent_locs)
            px = agent_locs[:, 0] + env.PADDING
            py = agent_locs[:, 1] + env.PADDING
            in_x = (px[None, :] >= start_x[:, None]) & (px[None, :] < start_x[:, None] + env.OBS_SIZE)
            in_y = (py[None, :] >= start_y[:, None]) & (py[None, :] < start_y[:, None] + env.OBS_SIZE)
            return in_x & in_y  # (viewer_i, other_j)

    for o_t in range(config["GIF_NUM_FRAMES"]):
        # Use model to select actions
        if config.get("PARAMETER_SHARING", True):
            obs_batch = jnp.stack([obs[a] for a in env.agents]).reshape(-1, *env.observation_space()[0].shape)
            network = ActorCritic(action_dim=env.action_space().n, activation=config.get("ACTIVATION", "relu"))
            pi, _ = network.apply(params, obs_batch)
            rng, _rng = jax.random.split(rng)
            actions = pi.sample(seed=_rng)
            # Convert action format
            env_act = {k: v.squeeze() for k, v in unbatchify(
                actions, env.agents, 1, env.num_agents
            ).items()}
        
        # --- Recurrent MOA rollout step: for each agent, run its RNN policy
        # forward with its own hidden state + current obs, update the hidden
        # state, and sample an action from the resulting policy ---
        elif recurrent_moa:
            obs_batch = jnp.stack([obs[a] for a in env.agents])
            env_act = {}
            for i in range(env.num_agents):
                obs_i = jnp.expand_dims(obs_batch[i], axis=0)
                policy_hstate[i], (pi, _) = network[i].apply(params[i], policy_hstate[i], obs_i)
                rng, _rng = jax.random.split(rng)
                single_action = pi.sample(seed=_rng)
                env_act[env.agents[i]] = single_action
        
        # --- LSTM-influence rollout step: same as recurrent MOA, but each
        # agent's policy also conditions on the (visibility-masked) previous
        # joint action -- mirrors ippo_cnn_coins.py's action-selection exactly ---
        elif lstm_influence:
            obs_batch = jnp.stack([obs[a] for a in env.agents])
            prev_joint_action_onehot = jax.nn.one_hot(prev_joint_action, action_dim)
            visible = _eval_visibility(state.agent_locs)  # (viewer_i, other_j)
            env_act = {}
            for i in range(env.num_agents):
                obs_i = jnp.expand_dims(obs_batch[i], axis=0)
                masked_i = jnp.where(visible[i][None, :, None], prev_joint_action_onehot, 0.0)
                policy_hstate[i], (pi, _) = network[i].apply(params[i], policy_hstate[i], obs_i, masked_i)
                rng, _rng = jax.random.split(rng)
                single_action = pi.sample(seed=_rng)
                env_act[env.agents[i]] = single_action

        # --- Non-recurrent rollout step: (re)build a feedforward network per
        # agent -- ActorCriticMOA only for the true cleanup-style MOA variant
        # (identifiable by MOA_LOSS_WEIGHT, which only that variant's config
        # sets), plain ActorCritic for both the no-influence baseline and the
        # MOA-free influence mechanisms (shared/independent), which trained
        # plain ActorCritic params and would fail to load into ActorCriticMOA ---
        else:
            obs_batch = jnp.stack([obs[a] for a in env.agents])
            env_act = {}
            if config.get("INFLUENCE_REWARD", False) and "MOA_LOSS_WEIGHT" in config:
                network = [ActorCriticMOA(
                    action_dim=env.action_space().n,
                    num_agents=env.num_agents,
                    activation=config.get("ACTIVATION", "relu"),
                ) for _ in range(env.num_agents)]
            else:
                network = [ActorCritic(action_dim=env.action_space().n, activation=config.get("ACTIVATION", "relu")) for _ in range(env.num_agents)]
            for i in range(env.num_agents):
                obs = jnp.expand_dims(obs_batch[i], axis=0)
                pi, _ = network[i].apply(params[i], obs)
                rng, _rng = jax.random.split(rng)
                single_action = pi.sample(seed=_rng)
                env_act[env.agents[i]] = single_action

        # Execute actions
        real_actions = [v.item() for v in env_act.values()]
        rng, _rng = jax.random.split(rng)
        obs, state, reward, done, info = env.step(_rng, state, real_actions)
        done = done["__all__"]

        # --- Reset LSTM carry to zero after an episode ends: the auto-reset
        # env already returned a fresh post-reset observation, so the hidden
        # state must not carry over stale info from the finished episode ---
        if recurrent_moa and done:
            policy_hstate = [jax.tree.map(jnp.zeros_like, h) for h in policy_hstate]
        if lstm_influence:
            real_action_arr = jnp.array([real_actions], dtype=jnp.int32)  # (1, num_agents)
            prev_joint_action = jnp.full_like(real_action_arr, -1) if done else real_action_arr
            if done:
                policy_hstate = [jax.tree.map(jnp.zeros_like, h) for h in policy_hstate]

        # Render
        img = env.render(state)
        pics.append(img)

    # Save GIF
    print(f"Saving Episode GIF")
    pics = [Image.fromarray(np.array(img)) for img in pics]
    n_agents = len(env.agents)
    gif_path = f"{root_dir}/{n_agents}-agents_seed-{config['SEED']}_eval-{eval_seed}_frames-{o_t + 1}.gif"
    pics[0].save(
        gif_path,
        format="GIF",
        save_all=True,
        optimize=False,
        append_images=pics[1:],
        duration=200,
        loop=0,
    )

    # Log the GIF to WandB
    print("Logging GIF to WandB")
    wandb.log({"Episode GIF": wandb.Video(gif_path, caption="Evaluation Episode", format="gif")})


def evaluate_mappo_style(params, env, save_path, config, use_actor_only=True):
    """
    Evaluation function for MAPPO/IRAT/SVO/TRANSFER algorithms.

    These algorithms use simpler evaluation without PARAMETER_SHARING logic
    and without WandB GIF logging.

    Args:
        params: Model parameters
        env: Environment instance
        save_path: Path where params were saved (unused but kept for API compatibility)
        config: Configuration dictionary
        use_actor_only: If True, use Actor network; if False, use ActorCritic (for SVO/TRANSFER)
    """
    rng = jax.random.PRNGKey(0)

    rng, _rng = jax.random.split(rng)
    obs, state = env.reset(_rng)
    done = False

    pics = []
    img = env.render(state)
    pics.append(img)

    # Extract environment name for root_dir
    env_name = config["ENV_NAME"]
    # Map environment names to evaluation directories
    env_dir_mapping = {
        "clean_up": "cleanup",
        "coin_game": "coins",
        "coop_mining": "coop_mining",
        "gift": "gift",
        "harvest_common_open": "harvest_common",
        "harvest_common_closed": "harvest_closed",
        "harvest_common_partnership": "harvest_partnership",
        "mushrooms": "mushrooms",
        "pd_arena": "pd_arena",
        "territory_open": "territory_open",
        "harvest_open": "harvest_open",
        "harvest_closed": "harvest_closed",
        "harvest_partnership": "harvest_partnership",
    }
    env_dir = env_dir_mapping.get(env_name, env_name)
    root_dir = f"evaluation/{env_dir}"
    path = Path(root_dir + "/state_pics")
    path.mkdir(parents=True, exist_ok=True)

    for o_t in range(config["GIF_NUM_FRAMES"]):
        obs_batch = jnp.stack([obs[a] for a in env.agents]).reshape(-1, *env.observation_space()[0].shape)

        # Use model to select actions
        if use_actor_only:
            # MAPPO uses SmallActor (features=16), not Actor (features=64)
            network = SmallActor(action_dim=env.action_space().n, activation=config.get("ACTIVATION", "relu"))
            pi = network.apply(params, obs_batch)
        else:
            network = ActorCritic(action_dim=env.action_space().n, activation=config.get("ACTIVATION", "relu"))
            pi, _ = network.apply(params, obs_batch)

        rng, _rng = jax.random.split(rng)
        actions = pi.sample(seed=_rng)

        # Convert action format
        env_act = {k: v.squeeze() for k, v in unbatchify(
            actions, env.agents, 1, env.num_agents
        ).items()}

        # Execute actions
        rng, _rng = jax.random.split(rng)
        obs, state, reward, done, info = env.step(_rng, state, [v.item() for v in env_act.values()])
        done = done["__all__"]

        # Render
        img = env.render(state)
        pics.append(img)

    # Save GIF
    print(f"Saving Episode GIF")
    pics = [Image.fromarray(np.array(img)) for img in pics]
    pics[0].save(
        f"{root_dir}/state_outer_step_{o_t+1}.gif",
        format="GIF",
        save_all=True,
        optimize=False,
        append_images=pics[1:],
        duration=200,
        loop=0,
    )
