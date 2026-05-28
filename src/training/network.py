"""DQN convolutional network parameterised by board size."""

from __future__ import annotations
import torch
import torch.nn as nn


class DQNNetwork(nn.Module):
    """Q-value network for C_lines.

    Input:  (B, 6, N, N)
    Output: (B, N*N)  — one Q-value per board cell
    """

    def __init__(self, board_size: int):
        super().__init__()
        n = board_size
        self.board_size = n
        self.conv = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
        )
        flat = 64 * n * n
        self.fc = nn.Sequential(
            nn.Linear(flat, 256), nn.ReLU(),
            nn.Linear(256, n * n),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.flatten(start_dim=1)
        return self.fc(x)
