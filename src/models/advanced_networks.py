import torch
import torch.nn as nn


class DQN(nn.Module):
    """Deep Q-Network supporting both vector (MLP) and grid (CNN) inputs."""

    def __init__(self, input_shape, num_actions):
        super().__init__()

        if len(input_shape) == 1:
            # --- MLP branch for 1D state inputs ---
            self.net = nn.Sequential(
                nn.Linear(input_shape[0], 32),
                nn.ReLU(),
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, num_actions)
            )

        else:
            # --- CNN branch for 3D state inputs ---
            c, h, w = input_shape[2], input_shape[0], input_shape[1]
            self.conv = nn.Sequential(
                nn.Conv2d(c, 32, kernel_size=3, stride=1), nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=1), nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=1), nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3, stride=1), nn.ReLU(),
                nn.Flatten()
            )

            # Compute flattened output size dynamically
            with torch.no_grad():
                dummy = torch.zeros(1, c, h, w)
                conv_out_size = self.conv(dummy).shape[1]

            self.fc = nn.Sequential(
                nn.Linear(conv_out_size, 256),
                nn.ReLU(),
                nn.Linear(256, num_actions)
            )

    def forward(self, x):
        if hasattr(self, "net"):
            # Vector (MLP) mode
            return self.net(x)
        else:
            # Grid (CNN) mode
            x = self.conv(x)
            return self.fc(x)
