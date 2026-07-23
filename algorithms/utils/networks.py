"""
Shared neural network architectures for multi-agent reinforcement learning algorithms.

This module contains CNN-based network architectures used across different MARL algorithms
including IPPO, MAPPO, SVO, Inequity Aversion, and AAA.
"""

import flax.linen as nn
import numpy as np
from flax.linen.initializers import constant, orthogonal
from typing import Sequence
import distrax
import jax.numpy as jnp


class CNN(nn.Module):
    """
    Convolutional Neural Network for visual feature extraction.

    Architecture:
        - Conv2D (32 filters, 5x5 kernel)
        - Conv2D (32 filters, 3x3 kernel)
        - Conv2D (32 filters, 3x3 kernel)
        - Dense (64 units)

    All layers use orthogonal initialization with sqrt(2) scaling.

    Attributes:
        activation: Activation function name ("relu" or "tanh")
    """
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        x = nn.Conv(
            features=32,
            kernel_size=(5, 5),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)

        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)

        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)

        x = x.reshape((x.shape[0], -1))  # Flatten

        x = nn.Dense(
            features=64,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(x)
        x = activation(x)

        return x


class ActorCritic(nn.Module):
    """
    Combined Actor-Critic network for IPPO, SVO, and Inequity Aversion algorithms.

    This network uses a shared CNN backbone with separate actor and critic heads.
    Used in algorithms that don't require separate actor/critic networks.

    Architecture:
        - Shared CNN feature extractor
        - Actor head: Dense(64) -> Dense(action_dim) -> Categorical distribution
        - Critic head: Dense(64) -> Dense(1) -> Value estimate

    Attributes:
        action_dim: Number of discrete actions
        activation: Activation function name ("relu" or "tanh")

    Returns:
        Tuple of (policy distribution, value estimate)
    """
    action_dim: Sequence[int]
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        embedding = CNN(self.activation)(x)

        # Actor head
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        # Critic head
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class ActorCriticMOA(nn.Module):
    action_dim: int
    num_agents: int
    activation: str = "relu"

    def setup(self):
        self.cnn = CNN(self.activation)

        self.policy_fc1 = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.policy_fc2 = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.actor_out = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))
        self.critic_out = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))

        self.moa_fc1 = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.moa_fc2 = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.moa_out = nn.Dense(
            self.num_agents * self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )

    def _activation_fn(self, x):
        return nn.relu(x) if self.activation == "relu" else nn.tanh(x)

    def __call__(self, obs):
        embedding = self.cnn(obs)
        h = self._activation_fn(self.policy_fc1(embedding))
        h = self._activation_fn(self.policy_fc2(h))

        pi = distrax.Categorical(logits=self.actor_out(h))
        value = jnp.squeeze(self.critic_out(h), axis=-1)

        return pi, value

    def moa(self, obs, joint_action_onehot):
        embedding = self.cnn(obs)
        flat_actions = joint_action_onehot.reshape(*joint_action_onehot.shape[:-2], -1)
        x = jnp.concatenate([embedding, flat_actions], axis=-1)
        h = self._activation_fn(self.moa_fc1(x))
        h = self._activation_fn(self.moa_fc2(h))
        logits = self.moa_out(h)
        return logits.reshape(*joint_action_onehot.shape[:-1], self.action_dim)

    def init_all(self, obs, joint_action_onehot):
        pi, value = self.__call__(obs)
        moa_logits = self.moa(obs, joint_action_onehot)
        return pi, value, moa_logits


class ActorCriticMOARNN(nn.Module):
    action_dim: int
    num_agents: int
    hidden_dim: int = 128
    activation: str = "relu"

    def setup(self):
        self.cnn = CNN(self.activation)

        self.policy_fc1 = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.policy_fc2 = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.policy_lstm = nn.LSTMCell(features=self.hidden_dim)
        self.actor_out = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))
        self.critic_out = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))

        self.moa_fc1 = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.moa_fc2 = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.moa_lstm = nn.LSTMCell(features=self.hidden_dim)
        self.moa_out = nn.Dense(
            self.num_agents * self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )

    def _activation_fn(self, x):
        return nn.relu(x) if self.activation == "relu" else nn.tanh(x)

    def __call__(self, policy_carry, obs):
        embedding = self.cnn(obs)
        h = self._activation_fn(self.policy_fc1(embedding))
        h = self._activation_fn(self.policy_fc2(h))
        new_policy_carry, ph = self.policy_lstm(policy_carry, h)

        pi = distrax.Categorical(logits=self.actor_out(ph))
        value = jnp.squeeze(self.critic_out(ph), axis=-1)

        return new_policy_carry, (pi, value)

    def moa_features(self, obs):
        embedding = self.cnn(obs)
        h = self._activation_fn(self.moa_fc1(embedding))
        h = self._activation_fn(self.moa_fc2(h))
        return h

    def moa_step(self, moa_carry, moa_features, joint_action_onehot):
        flat_actions = joint_action_onehot.reshape(*joint_action_onehot.shape[:-2], -1)
        x = jnp.concatenate([moa_features, flat_actions], axis=-1)
        new_moa_carry, mh = self.moa_lstm(moa_carry, x)
        logits = self.moa_out(mh)
        return new_moa_carry, logits.reshape(*joint_action_onehot.shape[:-1], self.action_dim)

    def moa(self, moa_carry, obs, joint_action_onehot):
        feats = self.moa_features(obs)
        return self.moa_step(moa_carry, feats, joint_action_onehot)

    def init_all(self, policy_carry, moa_carry, obs, joint_action_onehot):
        pc, (pi, value) = self.__call__(policy_carry, obs)
        mc, logits = self.moa(moa_carry, obs, joint_action_onehot)
        return (pi, value, logits)

    @staticmethod
    def initialize_carry(batch_size, hidden_dim):
        z = jnp.zeros((batch_size, hidden_dim))
        return (z, z)


class ActorCriticLSTM(nn.Module):
    """
    Recurrent Actor-Critic whose main policy/value pathway is conditioned on the
    other agent(s)' action, not just the ego agent's own observation.

    Unlike ActorCriticMOARNN (which keeps the other-agent-conditioned head separate,
    as an auxiliary MOA prediction bolted onto an otherwise ordinary policy), this
    network feeds the other agent's action one-hot directly into the main pathway:
    CNN -> concat with other agent's action -> two FC layers -> LSTM -> actor/critic
    heads. So the acting policy itself, not just an auxiliary head, sees what the
    other agent did.

    `other_action_onehot` follows the same joint-action convention used elsewhere in
    this file (shape (..., num_agents, action_dim)) so counterfactual callers can
    swap in a hypothetical action the same way they do for ActorCriticMOA/RNN's
    `joint_action_onehot` -- it's on the caller to zero out the ego agent's own slot
    if "other" should exclude self.

    Attributes:
        action_dim: Number of discrete actions
        num_agents: Total number of agents (sizes the flattened action input)
        hidden_dim: LSTM hidden size
        activation: Activation function name ("relu" or "tanh")
    """
    action_dim: int
    num_agents: int
    hidden_dim: int = 128
    activation: str = "relu"

    def setup(self):
        self.cnn = CNN(self.activation)

        self.fc1 = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.fc2 = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.lstm = nn.LSTMCell(features=self.hidden_dim)
        self.actor_out = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))
        self.critic_out = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))

    def _activation_fn(self, x):
        return nn.relu(x) if self.activation == "relu" else nn.tanh(x)

    def __call__(self, carry, obs, other_action_onehot):
        embedding = self.cnn(obs)
        flat_other_action = other_action_onehot.reshape(*other_action_onehot.shape[:-2], -1)
        x = jnp.concatenate([embedding, flat_other_action], axis=-1)
        h = self._activation_fn(self.fc1(x))
        h = self._activation_fn(self.fc2(h))
        new_carry, h = self.lstm(carry, h)

        pi = distrax.Categorical(logits=self.actor_out(h))
        value = jnp.squeeze(self.critic_out(h), axis=-1)

        return new_carry, (pi, value)

    @staticmethod
    def initialize_carry(batch_size, hidden_dim):
        z = jnp.zeros((batch_size, hidden_dim))
        return (z, z)


class Actor(nn.Module):
    """
    Standalone Actor network for MAPPO algorithm.

    MAPPO uses separate actor and critic networks to allow independent parameter updates.
    The actor takes per-agent observations and outputs action distributions.

    Architecture:
        - CNN feature extractor
        - Dense(64) -> Dense(action_dim) -> Categorical distribution

    Attributes:
        action_dim: Number of discrete actions
        activation: Activation function name ("relu" or "tanh")

    Returns:
        Categorical policy distribution over actions
    """
    action_dim: int
    activation: str = "relu"

    @nn.compact
    def __call__(self, obs):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        embedding = CNN(self.activation)(obs)

        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=actor_mean)

        return pi


class Critic(nn.Module):
    """
    Standalone Critic network for MAPPO algorithm.

    MAPPO critic takes world state (concatenated observations from all agents)
    as input to estimate centralized value function.

    Architecture:
        - CNN feature extractor (processes world state)
        - Dense(64) -> Dense(1) -> Value estimate

    Attributes:
        activation: Activation function name ("relu" or "tanh")

    Returns:
        Scalar value estimate (squeezed to remove last dimension)
    """
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        world_state = x

        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        embedding = CNN(self.activation)(world_state)

        hidden = nn.Dense(
            features=64,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(embedding)
        hidden = activation(hidden)

        value = nn.Dense(
            features=1,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(hidden)

        # Squeeze to remove last dimension
        return jnp.squeeze(value, axis=-1)


# ============================================================================
# MAPPO Small Network Architectures (features=16)
# ============================================================================
# MAPPO uses smaller networks compared to IPPO/SVO for efficiency


class SmallCNN(nn.Module):
    """
    Small Convolutional Neural Network for MAPPO algorithm.

    This is a lighter version of CNN used specifically by MAPPO for faster training
    with reduced model capacity.

    Architecture:
        - Conv2D (16 filters, 3x3 kernel) - single conv layer
        - Dense (16 units)

    All layers use orthogonal initialization with sqrt(2) scaling.

    Attributes:
        activation: Activation function name ("relu" or "tanh")
    """
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        x = nn.Conv(
            features=16,
            kernel_size=(3, 3),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)

        x = x.reshape((x.shape[0], -1))  # Flatten

        x = nn.Dense(
            features=16,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )(x)
        x = activation(x)

        return x


class SmallActor(nn.Module):
    """
    Small Actor network for MAPPO algorithm.

    Uses SmallCNN backbone with reduced hidden layer size (16 instead of 64).
    Designed for faster training in multi-agent scenarios.

    Architecture:
        - SmallCNN feature extractor
        - Dense(16) -> Dense(action_dim) -> Categorical distribution

    Attributes:
        action_dim: Number of discrete actions
        activation: Activation function name ("relu" or "tanh")

    Returns:
        Categorical policy distribution over actions
    """
    action_dim: int
    activation: str = "relu"

    @nn.compact
    def __call__(self, obs):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        embedding = SmallCNN(self.activation)(obs)

        actor_mean = nn.Dense(
            16, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=actor_mean)

        return pi


class SmallCritic(nn.Module):
    """
    Small Critic network for MAPPO algorithm.

    Uses SmallCNN backbone with reduced hidden layer size (16 instead of 64).
    Processes world state (concatenated agent observations) for centralized value estimation.

    Architecture:
        - SmallCNN feature extractor (processes world state)
        - Dense(16) -> Dense(1) -> Value estimate

    Attributes:
        activation: Activation function name ("relu" or "tanh")

    Returns:
        Scalar value estimate (squeezed to remove last dimension)
    """
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        world_state = x

        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        embedding = SmallCNN(self.activation)(world_state)

        hidden = nn.Dense(
            features=16,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(embedding)
        hidden = activation(hidden)

        value = nn.Dense(
            features=1,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(hidden)

        # Squeeze to remove last dimension
        return jnp.squeeze(value, axis=-1)


