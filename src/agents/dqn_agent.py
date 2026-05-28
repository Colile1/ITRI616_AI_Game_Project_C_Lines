"""DQN agent with epsilon-greedy, target network, and legal-move masking."""

from __future__ import annotations
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.agents.base_agent import BaseAgent
from src.training.network import DQNNetwork
from src.config import LR, GAMMA, GRADIENT_CLIP, EPS_START

_NEG_INF = -1e9


class DQNAgent(BaseAgent):
    """Online + target DQN. Call update() each gradient step."""

    def __init__(self, board_size: int, eval_only: bool = False):
        self.board_size = board_size
        self.epsilon = EPS_START
        self._device = torch.device("cpu")

        self._online = DQNNetwork(board_size).to(self._device)
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

    # ------------------------------------------------------------------
    # Training utilities
    # ------------------------------------------------------------------

    def update(self, batch: dict[str, np.ndarray]) -> float:
        """One gradient step. Returns scalar TD loss."""
        assert self._target is not None, "Cannot update an eval_only agent"
        states = torch.from_numpy(batch["states"]).to(self._device)
        actions = torch.from_numpy(batch["actions"]).long().to(self._device)
        rewards = torch.from_numpy(batch["rewards"]).to(self._device)
        next_states = torch.from_numpy(batch["next_states"]).to(self._device)
        dones = torch.from_numpy(batch["dones"]).to(self._device)
        legal_next = torch.from_numpy(batch["legal_masks_next"]).to(self._device)

        with torch.no_grad():
            q_next = self._target(next_states)
            q_next[~legal_next] = _NEG_INF
            max_q_next = q_next.max(dim=1).values
            targets = rewards + GAMMA * max_q_next * (1.0 - dones)

        q_pred = self._online(states)
        q_pred_actions = q_pred.gather(1, actions.unsqueeze(1)).squeeze(1)

        loss = nn.functional.mse_loss(q_pred_actions, targets)
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
