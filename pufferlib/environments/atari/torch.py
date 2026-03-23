import torch.nn as nn
import pufferlib.models
import pufferlib.pytorch

class Encoder(nn.Module):
    def __init__(self, env, hidden_size=128, framestack=1, flat_size=64*6*9):
        super().__init__()
        self.network = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Conv2d(framestack, 32, 8, stride=4)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            pufferlib.pytorch.layer_init(nn.Linear(flat_size, hidden_size)),
            nn.ReLU(),
        )

    def forward(self, observations):
        return self.network(observations.float() / 255.0)
