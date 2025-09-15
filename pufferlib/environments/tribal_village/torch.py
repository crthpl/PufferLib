"""
PyTorch policy architectures for Tribal Village environment.

Ultra-simplified policy for high performance, following metta pattern.
"""

import torch
import torch.nn as nn
import pufferlib.pytorch
import pufferlib.models


class Policy(nn.Module):
    """
    Ultra-simplified policy for Tribal Village environment.

    Optimized for speed over sophistication - simple embeddings and small hidden sizes.
    """

    def __init__(self, env, hidden_size=128, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.is_continuous = False

        # Token-based observation processing
        obs_space = env.single_observation_space
        self.max_tokens = obs_space.shape[0]

        # Token-native processing: embed tokens then pool to hidden size
        self.token_embed = pufferlib.pytorch.layer_init(nn.Linear(3, 32))
        self.pooling = pufferlib.pytorch.layer_init(nn.Linear(32, hidden_size))

        # Action heads for multi-discrete actions
        action_space = env.single_action_space
        self.actor = nn.ModuleList([
            pufferlib.pytorch.layer_init(nn.Linear(hidden_size, n), std=0.01)
            for n in action_space.nvec
        ])

        # Value head
        self.value = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, 1), std=1.0)

    def forward(self, observations: torch.Tensor, state=None):
        """Forward pass returning tuple format like metta."""
        hidden = self.encode_observations(observations, state)
        actions, value = self.decode_actions(hidden)
        return (actions, value), hidden

    def forward_eval(self, observations: torch.Tensor, state=None):
        """Forward pass for inference."""
        hidden = self.encode_observations(observations, state)
        return self.decode_actions(hidden)

    def encode_observations(self, observations: torch.Tensor, state=None) -> torch.Tensor:
        """Token-native processing - no grid rebuilding, direct token embedding."""
        # observations shape: (batch, max_tokens, 3)

        # Simple token embedding - process tokens directly without grid conversion
        token_features = torch.relu(self.token_embed(observations.float()))  # (batch, max_tokens, 32)

        # Simple mean pooling over valid tokens (ignore padding where coord_byte == 255)
        coord_bytes = observations[:, :, 0]  # (batch, max_tokens)
        valid_mask = (coord_bytes != 255).float().unsqueeze(-1)  # (batch, max_tokens, 1)

        # Masked mean pooling
        masked_features = token_features * valid_mask
        valid_counts = valid_mask.sum(dim=1).clamp(min=1)  # (batch, 1)
        pooled_features = masked_features.sum(dim=1) / valid_counts  # (batch, 32)

        # Final processing
        hidden = torch.relu(self.pooling(pooled_features))  # (batch, hidden_size)
        return hidden

    def decode_actions(self, hidden: torch.Tensor):
        """Decode actions from hidden features."""
        # Generate logits for each action dimension (return as list like metta)
        logits = [head(hidden) for head in self.actor]

        # Value prediction
        values = self.value(hidden)

        return logits, values


class Recurrent(pufferlib.models.LSTMWrapper):
    """Recurrent policy using PufferLib's LSTM wrapper."""

    def __init__(self, env, policy, input_size=128, hidden_size=128):
        super().__init__(env, policy, input_size, hidden_size)