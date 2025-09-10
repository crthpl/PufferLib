"""
PyTorch policy architectures for Tribal environment.

This module provides neural network policies optimized for the tribal environment's
token-based observation format and multi-discrete action space.
"""

import torch
import torch.nn as nn
import pufferlib.pytorch
from typing import Optional, Tuple


class Policy(nn.Module):
    """
    Default policy for Tribal environment.
    
    Optimized for token-based observations with coordinate-byte encoding.
    Uses attention mechanisms to process variable-length token sequences.
    """
    
    def __init__(self, env):
        super().__init__()
        
        # Environment specifications
        obs_space = env.single_observation_space
        action_space = env.single_action_space
        
        # Token-based observation processing
        # obs_space.shape should be (MAX_TOKENS_PER_AGENT, 3) where 3 = [coord_byte, layer, value]
        self.max_tokens = obs_space.shape[0]
        self.token_dim = obs_space.shape[1]  # Should be 3
        
        # Token embedding layers
        self.coord_embed = nn.Embedding(256, 64)  # coord_byte values 0-255
        self.layer_embed = nn.Embedding(16, 32)   # layer values
        self.value_embed = nn.Embedding(256, 32)  # value values 0-255
        
        # Token fusion
        embed_dim = 64 + 32 + 32  # 128 total
        self.token_fusion = pufferlib.pytorch.layer_init(nn.Linear(embed_dim, 128))
        
        # Self-attention for token sequence processing
        self.attention = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=8,
            batch_first=True,
            dropout=0.1
        )
        
        # Policy head for multi-discrete actions
        self.action_head = nn.ModuleList([
            pufferlib.pytorch.layer_init(nn.Linear(128, n), std=0.01)
            for n in action_space.nvec
        ])
        
        # Value head
        self.value_head = pufferlib.pytorch.layer_init(nn.Linear(128, 1), std=1.0)
        
    def forward(self, observations: torch.Tensor, state=None) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward_eval(observations, state)
    
    def forward_eval(self, observations: torch.Tensor, state=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for tribal environment.
        
        Args:
            observations: Token observations of shape (batch_size, max_tokens, 3)
        
        Returns:
            logits: Action logits for multi-discrete action space
            values: State values
        """
        batch_size = observations.shape[0]
        
        # Extract token components
        coord_bytes = observations[:, :, 0].long()  # (batch, max_tokens)
        layers = observations[:, :, 1].long()       # (batch, max_tokens)
        values = observations[:, :, 2].long()       # (batch, max_tokens)
        
        # Embed each component
        coord_embeds = self.coord_embed(coord_bytes)  # (batch, max_tokens, 64)
        layer_embeds = self.layer_embed(layers)       # (batch, max_tokens, 32)
        value_embeds = self.value_embed(values)       # (batch, max_tokens, 32)
        
        # Concatenate embeddings
        token_embeds = torch.cat([coord_embeds, layer_embeds, value_embeds], dim=-1)  # (batch, max_tokens, 128)
        
        # Apply token fusion
        tokens = torch.relu(self.token_fusion(token_embeds))  # (batch, max_tokens, 128)
        
        # Create attention mask (tokens with coord_byte == 255 are padding)
        padding_mask = (coord_bytes == 255)  # (batch, max_tokens)
        
        # Self-attention over tokens
        if padding_mask.all():
            # All tokens are padding, use zero features
            attended_tokens = torch.zeros_like(tokens)
        else:
            attended_tokens, _ = self.attention(
                tokens, tokens, tokens,
                key_padding_mask=padding_mask
            )  # (batch, max_tokens, 128)
        
        # Global pooling over valid tokens
        if padding_mask.all():
            pooled_features = torch.zeros(batch_size, 128, device=observations.device)
        else:
            # Use mean pooling over non-padding tokens
            valid_mask = ~padding_mask.unsqueeze(-1)  # (batch, max_tokens, 1)
            masked_tokens = attended_tokens * valid_mask.float()
            valid_counts = valid_mask.sum(dim=1, keepdim=True).float().clamp(min=1)
            pooled_features = masked_tokens.sum(dim=1) / valid_counts.squeeze(-1)  # (batch, 128)
        
        # Generate action logits for each action dimension
        action_logits = []
        for head in self.action_head:
            logits = head(pooled_features)
            action_logits.append(logits)
        
        # Concatenate all action logits
        all_logits = torch.cat(action_logits, dim=-1)
        
        # Value prediction
        values = self.value_head(pooled_features)
        
        return all_logits, values


class Recurrent(Policy):
    """
    Recurrent policy for Tribal environment with LSTM memory.
    
    Extends the base Policy with LSTM layers for temporal modeling.
    """
    
    def __init__(self, env):
        super().__init__(env)
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Update heads to use LSTM output size
        action_space = env.single_action_space
        self.action_head = nn.ModuleList([
            pufferlib.pytorch.layer_init(nn.Linear(256, n), std=0.01)
            for n in action_space.nvec
        ])
        self.value_head = pufferlib.pytorch.layer_init(nn.Linear(256, 1), std=1.0)
        
    def forward_eval(self, observations: torch.Tensor, state=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with LSTM temporal modeling."""
        batch_size = observations.shape[0]
        
        # Get base features from parent class
        # First get token embeddings and attention (without final pooling)
        coord_bytes = observations[:, :, 0].long()
        layers = observations[:, :, 1].long()
        values = observations[:, :, 2].long()
        
        coord_embeds = self.coord_embed(coord_bytes)
        layer_embeds = self.layer_embed(layers)
        value_embeds = self.value_embed(values)
        
        token_embeds = torch.cat([coord_embeds, layer_embeds, value_embeds], dim=-1)
        tokens = torch.relu(self.token_fusion(token_embeds))
        
        padding_mask = (coord_bytes == 255)
        
        if padding_mask.all():
            attended_tokens = torch.zeros_like(tokens)
        else:
            attended_tokens, _ = self.attention(tokens, tokens, tokens, key_padding_mask=padding_mask)
        
        if padding_mask.all():
            pooled_features = torch.zeros(batch_size, 128, device=observations.device)
        else:
            valid_mask = ~padding_mask.unsqueeze(-1)
            masked_tokens = attended_tokens * valid_mask.float()
            valid_counts = valid_mask.sum(dim=1, keepdim=True).float().clamp(min=1)
            pooled_features = masked_tokens.sum(dim=1) / valid_counts.squeeze(-1)
        
        # Add sequence dimension for LSTM
        lstm_input = pooled_features.unsqueeze(1)  # (batch, 1, 128)
        
        # LSTM forward pass
        if state is not None:
            hidden, cell = state
            lstm_output, (new_hidden, new_cell) = self.lstm(lstm_input, (hidden, cell))
        else:
            lstm_output, (new_hidden, new_cell) = self.lstm(lstm_input)
        
        # Remove sequence dimension
        lstm_features = lstm_output.squeeze(1)  # (batch, 256)
        
        # Generate action logits
        action_logits = []
        for head in self.action_head:
            logits = head(lstm_features)
            action_logits.append(logits)
        
        all_logits = torch.cat(action_logits, dim=-1)
        values = self.value_head(lstm_features)
        
        return all_logits, values, (new_hidden, new_cell)