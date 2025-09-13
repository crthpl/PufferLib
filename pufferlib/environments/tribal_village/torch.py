"""
PyTorch policy architectures for Tribal Village environment.

Simplified policy following the metta pattern for token-based observations.
"""

import torch
import torch.nn as nn
import pufferlib.pytorch
import pufferlib.models


class Policy(nn.Module):
    """
    Simplified policy for Tribal Village environment.

    Follows the metta pattern with token-based observations and multi-discrete actions.
    """

    def __init__(self, env, hidden_size=512, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.is_continuous = False

        # Token-based observation processing
        obs_space = env.single_observation_space
        self.max_tokens = obs_space.shape[0]  # Should be 2541

        # Simple token embedding - map each token [coord_byte, layer, value] to features
        self.token_embed = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(3, 128)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(128, 256)),
            nn.ReLU(),
        )

        # Global pooling to fixed size
        self.pooling = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(256, hidden_size)),
            nn.ReLU(),
        )

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
        """Encode token observations to features."""
        batch_size = observations.shape[0]

        # observations shape: (batch, max_tokens, 3)
        # Embed each token
        token_features = self.token_embed(observations.float())  # (batch, max_tokens, 256)

        # Simple mean pooling over tokens (ignoring padding tokens where coord_byte == 255)
        coord_bytes = observations[:, :, 0]  # (batch, max_tokens)
        valid_mask = (coord_bytes != 255).float().unsqueeze(-1)  # (batch, max_tokens, 1)

        # Masked mean pooling
        masked_features = token_features * valid_mask
        valid_counts = valid_mask.sum(dim=1).clamp(min=1)  # (batch, 1)
        pooled_features = masked_features.sum(dim=1) / valid_counts  # (batch, 256)

        # Final processing
        hidden = self.pooling(pooled_features)  # (batch, hidden_size)
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

    def __init__(self, env, policy, input_size=512, hidden_size=512):
        super().__init__(env, policy, input_size, hidden_size)