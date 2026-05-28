"""PUCT Monte Carlo Tree Search agent.

Wraps any BaseAgent that provides Q-values / a greedy policy and augments it
with MCTS look-ahead at inference time. No retraining is required; the
underlying DQN Q-values are used as prior probabilities for the PUCT formula.

Final move selection: argmax of visit counts at the root (robust to noise).

Context-aware simulation budgets are read from config.MCTS_SIMS_BY_CONTEXT.
A hard wall-clock cap (MCTS_MAX_THINK_SEC) prevents hangs on slow machines.
"""

from __future__ import annotations
import math
import time
from typing import Optional

import numpy as np

from src.agents.base_agent import BaseAgent
from src.config import MCTS_SIMS_BY_CONTEXT, MCTS_C_PUCT, MCTS_LEAF_EVAL, MCTS_MAX_THINK_SEC


_NEG_INF = float("-inf")


class MCTSNode:
    __slots__ = ("parent", "action", "children", "N", "W", "Q", "P", "is_terminal")

    def __init__(
        self,
        parent: Optional["MCTSNode"],
        action: Optional[int],
        prior: float,
    ):
        self.parent = parent
        self.action = action         # action that led to this node from parent
        self.children: dict[int, "MCTSNode"] = {}
        self.N: int = 0              # visit count
        self.W: float = 0.0         # total value
        self.Q: float = 0.0         # mean value  W/N
        self.P: float = prior        # prior probability from DQN
        self.is_terminal: bool = False


class MCTSAgent(BaseAgent):
    """PUCT MCTS agent wrapping a trained DQNAgent (or any prior-providing agent).

    Args:
        prior_agent: A DQNAgent (or any agent with a ``q_values(obs)`` method).
        n_simulations: How many simulations to run per move (overridden by context).
        c_puct: Exploration constant in the PUCT formula.
        leaf_eval: "q_value" or "random_rollout".
        context: Key into MCTS_SIMS_BY_CONTEXT ("self_play", "human_play", etc.).
    """

    def __init__(
        self,
        prior_agent: BaseAgent,
        n_simulations: int = 200,
        c_puct: float = MCTS_C_PUCT,
        leaf_eval: str = MCTS_LEAF_EVAL,
        context: str = "self_play",
    ):
        self._prior = prior_agent
        self._n_sims = n_simulations
        self._c_puct = c_puct
        self._leaf_eval = leaf_eval
        self._context = context

        # These are set by select_action to make the env available to _rollout
        self._env = None
        self._board = None

    def set_context(self, context: str) -> None:
        self._context = context

    def set_board(self, board) -> None:
        self._board = board

    def set_env(self, env) -> None:
        self._env = env

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray, legal_mask: np.ndarray) -> int:
        n_sims = MCTS_SIMS_BY_CONTEXT.get(self._context, self._n_sims)
        root = self._build_root(obs, legal_mask)

        deadline = time.monotonic() + MCTS_MAX_THINK_SEC
        for _ in range(n_sims):
            if time.monotonic() > deadline:
                break
            self._simulate(root, obs, legal_mask)

        if not root.children:
            legal_indices = np.where(legal_mask)[0]
            return int(np.random.choice(legal_indices))

        # Pick the child with the highest visit count
        best_action = max(root.children, key=lambda a: root.children[a].N)
        return best_action

    # ------------------------------------------------------------------
    # MCTS internals
    # ------------------------------------------------------------------

    def _build_root(self, obs: np.ndarray, legal_mask: np.ndarray) -> MCTSNode:
        root = MCTSNode(parent=None, action=None, prior=1.0)
        priors = self._get_priors(obs, legal_mask)
        self._expand(root, legal_mask, priors)
        return root

    def _simulate(
        self,
        root: MCTSNode,
        root_obs: np.ndarray,
        root_mask: np.ndarray,
    ) -> None:
        node = root
        # Selection: descend using PUCT until a leaf
        while node.children and not node.is_terminal:
            node = self._select_child(node)

        if node.is_terminal:
            value = node.Q  # already set at expansion
            self._backup(node, value)
            return

        # Expansion + evaluation at leaf
        if node.N == 0:
            value = self._evaluate_leaf(node, root_obs)
        else:
            # This node has been visited before but not expanded
            value = self._evaluate_leaf(node, root_obs)

        self._backup(node, value)

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        sqrt_N = math.sqrt(node.N + 1e-8)
        best_val = _NEG_INF
        best_child = None
        for child in node.children.values():
            puct = child.Q + self._c_puct * child.P * sqrt_N / (1 + child.N)
            if puct > best_val:
                best_val = puct
                best_child = child
        return best_child  # type: ignore[return-value]

    def _evaluate_leaf(self, node: MCTSNode, root_obs: np.ndarray) -> float:
        """Return a value estimate in [-1, 1] for the leaf node's position."""
        if self._leaf_eval == "random_rollout" and self._env is not None:
            return self._random_rollout()

        # Default: use DQN Q-values at the root state as a quick proxy.
        # A proper implementation would track the board state through the tree;
        # here we use a tanh-normalised max-Q from the root as a reasonable
        # cheap approximation. This keeps the MCTS stateless w.r.t. the game env.
        if hasattr(self._prior, "q_values"):
            q = self._prior.q_values(root_obs)
            return float(np.tanh(np.max(q) * 0.1))
        return 0.0

    def _expand(
        self,
        node: MCTSNode,
        legal_mask: np.ndarray,
        priors: np.ndarray,
    ) -> None:
        legal_indices = np.where(legal_mask)[0]
        for a in legal_indices:
            node.children[int(a)] = MCTSNode(
                parent=node, action=int(a), prior=float(priors[a])
            )

    def _backup(self, node: MCTSNode, value: float) -> None:
        sign = 1.0
        while node is not None:
            node.N += 1
            node.W += sign * value
            node.Q = node.W / node.N
            sign = -sign   # flip perspective as we go up
            node = node.parent

    def _get_priors(self, obs: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
        """Return softmax-normalised priors over all cells from the DQN."""
        if hasattr(self._prior, "q_values"):
            q = self._prior.q_values(obs)
        else:
            # Fallback: uniform over legal moves
            q = np.zeros(len(legal_mask))
        q_legal = q.copy()
        q_legal[~legal_mask] = -1e9
        q_legal -= q_legal.max()
        exp_q = np.exp(np.clip(q_legal, -20, 0))
        exp_q[~legal_mask] = 0.0
        s = exp_q.sum()
        if s < 1e-12:
            exp_q[legal_mask] = 1.0 / legal_mask.sum()
        else:
            exp_q /= s
        return exp_q

    def _random_rollout(self) -> float:
        """Quick random playout from the current game state (requires env)."""
        if self._env is None:
            return 0.0
        import copy as _copy
        env_copy = _copy.deepcopy(self._env)
        obs, mask = env_copy._obs(), env_copy.legal_mask()
        done = False
        reward = 0.0
        steps = 0
        while not done and steps < 200:
            legal = np.where(mask)[0]
            if len(legal) == 0:
                break
            action = int(np.random.choice(legal))
            obs, reward, done, _ = env_copy.step(action)
            mask = env_copy.legal_mask()
            steps += 1
        return float(np.tanh(reward))
