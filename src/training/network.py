"""DQN network architectures: plain CNN (v1) and residual-block (v1 resnet)."""

from __future__ import annotations
import torch
import torch.nn as nn

from src.config import NUM_RES_BLOCKS, RES_CHANNELS, STATE_CHANNELS_V1, STATE_CHANNELS_V2


class DQNNetworkPlain(nn.Module):
    """Original 3-conv plain CNN. Kept for back-compat with v1 snapshots.

    Input:  (B, in_channels, N, N)
    Output: (B, N*N)
    """

    def __init__(self, board_size: int, in_channels: int = STATE_CHANNELS_V1):
        super().__init__()
        n = board_size
        self.board_size = n
        self.in_channels = in_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1), nn.ReLU(),
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


class _ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.block(x) + x)


class DQNNetwork(nn.Module):
    """Residual-block Q-value network.

    Input:  (B, in_channels, N, N)
    Output: (B, N*N)
    """

    def __init__(
        self,
        board_size: int,
        in_channels: int = STATE_CHANNELS_V2,
        num_res_blocks: int = NUM_RES_BLOCKS,
        res_channels: int = RES_CHANNELS,
    ):
        super().__init__()
        n = board_size
        self.board_size = n
        self.in_channels = in_channels

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, res_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(res_channels),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.Sequential(
            *[_ResBlock(res_channels) for _ in range(num_res_blocks)]
        )
        # Q-head
        self.head = nn.Sequential(
            nn.Conv2d(res_channels, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * n * n, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, n * n),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.res_blocks(x)
        return self.head(x)


def q_to_policy_prior(
    q_values: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Convert masked Q-values to a policy prior via softmax.

    Used by MCTSAgent to initialise prior probabilities P(s,a).
    """
    masked_q = q_values.clone()
    masked_q[~mask] = float("-inf")
    return torch.softmax(masked_q / max(temperature, 1e-8), dim=-1)


def build_network(
    board_size: int,
    in_channels: int,
    network_arch: str = "resnet_v1",
) -> nn.Module:
    """Factory: return the right network class for the given arch tag."""
    if network_arch == "plain_v1":
        return DQNNetworkPlain(board_size, in_channels=in_channels)
    return DQNNetwork(board_size, in_channels=in_channels)
