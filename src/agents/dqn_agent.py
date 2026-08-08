"""DQN agent with epsilon-greedy, target network, and legal-move masking."""

from __future__ import annotations
import copy
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.agents.base_agent import BaseAgent
from src.training.network import build_network
from src.config import (
    LR, GAMMA, GRADIENT_CLIP, EPS_START,
    STATE_CHANNELS_V2, NETWORK_ARCH,
)

_NEG_INF = torch.finfo(torch.float32).min / 2   # float32-safe large negative


def _best_device() -> torch.device:
    """Return the fastest available device.

    DirectML (Intel UHD integrated) is excluded: Adam's lerp op falls back to
    CPU on the DML backend, causing 100× overhead from repeated shared-memory
    transfers.  Only use DirectML if a discrete NVIDIA/AMD GPU is present.
    CUDA (discrete NVIDIA) is supported and fast.

    The FLAT4_DEVICE environment variable overrides the automatic choice
    ("cpu" / "cuda" / "auto").  Two uses:
      * Episode-collection workers force "cpu" — they only ever run single-state
        inference, so a per-worker CUDA context is pure overhead and VRAM waste.
      * On a host where the tiny 8x8 ResNet is faster on CPU than on GPU (kernel
        launch latency dominates single-state forwards), the whole run can be
        pinned to CPU without a code change.
    """
    requested = os.environ.get("FLAT4_DEVICE", "auto").strip().lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("  [device] FLAT4_DEVICE=cuda but CUDA is unavailable — using CPU.")
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class DQNAgent(BaseAgent):
    """Online + target DQN. Call update() each gradient step."""

    def __init__(
        self,
        board_size: int,
        eval_only: bool = False,
        in_channels: int = STATE_CHANNELS_V2,
        network_arch: str = NETWORK_ARCH,
    ):
        self.board_size = board_size
        self.in_channels = in_channels
        self.network_arch = network_arch
        self.epsilon = EPS_START
        self._device = _best_device()

        self._online = build_network(board_size, in_channels, network_arch).to(self._device)
        if not eval_only:
            self._target = copy.deepcopy(self._online)
            self._target.eval()
            self._optimizer = optim.Adam(self._online.parameters(), lr=LR)
        else:
            self._target = None
            self._optimizer = None

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray, legal_mask: np.ndarray) -> int:
        if np.random.random() < self.epsilon:
            legal_indices = np.where(legal_mask)[0]
            return int(np.random.choice(legal_indices))
        return self._greedy_action(obs, legal_mask)

    def q_values(self, obs: np.ndarray) -> np.ndarray:
        """Return raw Q-values for all cells (for use as MCTS priors)."""
        self._online.eval()
        with torch.no_grad():
            t = torch.from_numpy(obs).unsqueeze(0).to(self._device)
            q = self._online(t).squeeze(0).cpu().numpy()
        self._online.train()
        return q

    # ------------------------------------------------------------------
    # Training utilities
    # ------------------------------------------------------------------

    def update(self, batch: dict[str, np.ndarray]) -> float:
        assert self._target is not None, "Cannot update an eval_only agent"
        # Cast all tensors to float32 explicitly — DirectML does not support float64
        states      = torch.from_numpy(batch["states"]).float().to(self._device)
        actions     = torch.from_numpy(batch["actions"]).long().to(self._device)
        rewards     = torch.from_numpy(batch["rewards"]).float().to(self._device)
        next_states = torch.from_numpy(batch["next_states"]).float().to(self._device)
        dones       = torch.from_numpy(batch["dones"]).float().to(self._device)
        legal_next  = torch.from_numpy(batch["legal_masks_next"]).to(self._device)
        # Per-transition bootstrap discount: GAMMA^n for n-step returns, GAMMA for 1-step.
        if "gammas" in batch:
            gammas = torch.from_numpy(batch["gammas"]).float().to(self._device)
        else:
            gammas = torch.full((rewards.shape[0],), GAMMA, dtype=torch.float32,
                                device=self._device)

        with torch.no_grad():
            # Double DQN: online net selects the action, target net evaluates it.
            # Negamax sign: next_state is encoded from the OPPONENT's perspective,
            # so max_a Q(s',a) is the opponent's value.  In a zero-sum game that
            # value must be SUBTRACTED, not added, to get the correct TD target.
            q_next_online = self._online(next_states)
            q_next_online[~legal_next] = _NEG_INF
            next_actions = q_next_online.argmax(dim=1, keepdim=True)

            q_next_target = self._target(next_states)
            next_q = q_next_target.gather(1, next_actions).squeeze(1)

            # Use integer 1 (not 1.0) — float literals are float64 on DirectML
            targets = rewards - gammas * next_q * (1 - dones)

        q_pred = self._online(states)
        q_pred_actions = q_pred.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Huber loss: less sensitive to large TD errors than plain MSE
        loss = nn.functional.smooth_l1_loss(q_pred_actions, targets)
        self._optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._online.parameters(), GRADIENT_CLIP)
        self._optimizer.step()
        return loss.item()

    def sync_target(self) -> None:
        assert self._target is not None
        self._target.load_state_dict(self._online.state_dict())

    def set_epsilon(self, epsilon: float) -> None:
        self.epsilon = float(epsilon)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return self._online.state_dict()

    def load_state_dict(self, sd: dict) -> None:
        # sd may come from torch.load(map_location="cpu") — move to our device
        sd = {k: v.to(self._device) if isinstance(v, torch.Tensor) else v
              for k, v in sd.items()}
        self._online.load_state_dict(sd)
        if self._target is not None:
            self._target.load_state_dict(sd)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _greedy_action(self, obs: np.ndarray, legal_mask: np.ndarray) -> int:
        self._online.eval()
        with torch.no_grad():
            t = torch.from_numpy(obs).unsqueeze(0).to(self._device)
            q = self._online(t).squeeze(0).cpu().numpy()
        self._online.train()
        q[~legal_mask] = _NEG_INF
        return int(np.argmax(q))
