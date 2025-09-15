"""
Ultra-minimal PyTorch policy for Tribal Village - optimized for maximum SPS.
"""

import torch
import torch.nn as nn
import pufferlib.pytorch
import pufferlib.models


class Policy(nn.Module):
    """Ultra-minimal policy optimized for speed over sophistication."""

    def __init__(self, env, hidden_size=64, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.is_continuous = False

        # Minimal token processing - just sum non-padding tokens
        self.token_proj = pufferlib.pytorch.layer_init(nn.Linear(3, 1))  # 3->1 projection
        self.hidden_proj = pufferlib.pytorch.layer_init(nn.Linear(256, hidden_size))  # Fixed input size

        # Action heads
        action_space = env.single_action_space
        self.actor = nn.ModuleList([
            pufferlib.pytorch.layer_init(nn.Linear(hidden_size, n), std=0.01)
            for n in action_space.nvec
        ])

        # Value head
        self.value = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, 1), std=1.0)

    def forward(self, observations: torch.Tensor, state=None):
        hidden = self.encode_observations(observations, state)
        actions, value = self.decode_actions(hidden)
        return (actions, value), hidden

    def forward_eval(self, observations: torch.Tensor, state=None):
        hidden = self.encode_observations(observations, state)
        return self.decode_actions(hidden)

    def encode_observations(self, observations: torch.Tensor, state=None) -> torch.Tensor:
        """Ultra-minimal token processing - just count and sum."""
        # observations shape: (batch, 256, 3)

        # Simple projection of each token to scalar
        token_values = self.token_proj(observations.float()).squeeze(-1)  # (batch, 256)

        # Mask out padding tokens (coord_byte == 255)
        coord_bytes = observations[:, :, 0]
        valid_mask = (coord_bytes != 255).float()

        # Simple masked sum - very fast
        masked_values = token_values * valid_mask
        features = masked_values.sum(dim=1, keepdim=True)  # (batch, 1)

        # Pad to fixed size for hidden projection
        padded_features = torch.zeros(observations.shape[0], 256, device=observations.device)
        padded_features[:, :1] = features

        # Project to hidden size
        hidden = torch.relu(self.hidden_proj(padded_features))
        return hidden

    def decode_actions(self, hidden: torch.Tensor):
        """Simple action decoding."""
        logits = [head(hidden) for head in self.actor]
        values = self.value(hidden)
        return logits, values


class Recurrent(pufferlib.models.LSTMWrapper):
    """Minimal LSTM wrapper."""

    def __init__(self, env, policy, input_size=64, hidden_size=64):
        super().__init__(env, policy, input_size, hidden_size)