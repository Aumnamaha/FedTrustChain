"""Dynamic Node Trust Manager for Federated Learning.

Evaluates incoming client model updates against a server-side holdout validation
set, tracks historical node reputation scores (T_i in [0.0, 1.0]), generates
SHA-256 payload hashes for tamper-proof logging, and filters out low-reputation
or adversarial updates before tree union pooling.

This module implements Phase 2 Step 1 of the FedTrustChain architecture:
dynamic node trust scoring with pre-aggregation payload isolation.
"""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


@dataclass
class _NodeState:
    """Internal tracking state for a single client node."""

    client_id: int
    trust_score: float = 1.0
    total_rounds: int = 0
    accepted_updates: int = 0
    rejected_updates: int = 0


class DynamicTrustManager:
    """Evaluate and score client model updates for federated learning.

    Parameters
    ----------
    holdout_X : np.ndarray
        Small server-side validation feature set used to evaluate incoming
        client models before aggregation.
    holdout_y : array-like
        True labels corresponding to *holdout_X*.
    min_trust_threshold : float, default=0.50
        Minimum trust score T_i required for a client's model estimators to be
        included in the global aggregation. Nodes below this threshold are
        isolated and their updates discarded.
    reward_rate : float, default=0.05
        Increment applied to T_i when a client submits an acceptable-quality
        update (validation accuracy within delta of expected performance).
    decay_rate : float, default=0.25
        Penalty magnitude subtracted from T_i for severely degraded or
        adversarial updates.
    accuracy_delta : float, default=0.15
        Maximum allowed gap between a client's validation accuracy and the
        server-side baseline (best observed so far) to qualify as acceptable.

    Attributes
    ----------
    trust_scores : dict[int, float]
        Current per-node reputation scores (T_i in [0.0, 1.0]).
    node_registry : dict[int, _NodeState]
        Full tracking state for each registered client node.
    """

    def __init__(
        self,
        holdout_X: np.ndarray,
        holdout_y,
        min_trust_threshold: float = 0.50,
        reward_rate: float = 0.05,
        decay_rate: float = 0.25,
        accuracy_delta: float = 0.15,
    ):
        self.holdout_X = np.asarray(holdout_X)
        self.holdout_y = np.asarray(holdout_y)
        self.min_trust_threshold = min_trust_threshold
        self.reward_rate = reward_rate
        self.decay_rate = decay_rate
        self.accuracy_delta = accuracy_delta

        self._registry: Dict[int, _NodeState] = {}
        self._best_accuracy: float = 0.0

    # ------------------------------------------------------------------
    @property
    def trust_scores(self) -> Dict[int, float]:
        """Current per-node reputation scores (T_i in [0.0, 1.0])."""
        return {cid: st.trust_score for cid, st in self._registry.items()}

    @property
    def node_registry(self) -> Dict[int, _NodeState]:
        """Full tracking state for each registered client node."""
        return dict(self._registry)

    # ------------------------------------------------------------------
    def _ensure_node(self, client_id: int) -> _NodeState:
        """Register a new node if it hasn't been seen before."""
        if client_id not in self._registry:
            self._registry[client_id] = _NodeState(client_id=client_id)
        return self._registry[client_id]

    # ------------------------------------------------------------------
    @staticmethod
    def compute_payload_hash(payload_bytes: bytes) -> str:
        """Compute a deterministic SHA-256 hash of the serialized model payload.

        Parameters
        ----------
        payload_bytes : bytes
            The raw pickled model byte stream received over UDP.

        Returns
        -------
        str
            Hex-encoded SHA-256 digest (64 characters).
        """
        return hashlib.sha256(payload_bytes).hexdigest()

    # ------------------------------------------------------------------
    def evaluate_update(
        self, client_id: int, model: RandomForestClassifier
    ) -> Tuple[bool, float, str]:
        """Evaluate a single client model against the holdout validation set.

        Parameters
        ----------
        client_id : int
            Identifier of the submitting client node.
        model : RandomForestClassifier
            Fitted scikit-learn Random Forest model received from the client.

        Returns
        -------
        accepted : bool
            True if the model passes quality checks and should be aggregated.
        accuracy : float
            Validation accuracy achieved on the holdout set.
        payload_hash : str
            SHA-256 hex digest of the pickled model bytes (for audit logging).
        """
        # --- compute hash ---------------------------------------------------
        payload_bytes = pickle.dumps(model)
        payload_hash = self.compute_payload_hash(payload_bytes)

        # --- validate structural integrity ----------------------------------
        if not hasattr(model, "estimators_") or len(model.estimators_) == 0:
            _log_warning(
                "Client %d: rejected — empty estimators list", client_id
            )
            state = self._ensure_node(client_id)
            state.rejected_updates += 1
            state.trust_score = max(0.0, state.trust_score - self.decay_rate)
            return False, 0.0, payload_hash

        # --- evaluate on holdout set ----------------------------------------
        y_pred = model.predict(self.holdout_X)
        accuracy = float(accuracy_score(self.holdout_y, y_pred))

        # --- decision logic -------------------------------------------------
        state = self._ensure_node(client_id)
        state.total_rounds += 1

        if accuracy >= (self._best_accuracy - self.accuracy_delta):
            # Acceptable quality — reward
            state.trust_score = min(1.0, state.trust_score + self.reward_rate)
            state.accepted_updates += 1
            if accuracy > self._best_accuracy:
                self._best_accuracy = accuracy

            _log_info(
                "Client %d: accepted (accuracy=%.4f, T_i=%.3f)",
                client_id, accuracy, state.trust_score,
            )
            return True, accuracy, payload_hash

        else:
            # Poor quality — penalize
            state.trust_score = max(0.0, state.trust_score - self.decay_rate)
            state.rejected_updates += 1

            _log_warning(
                "Client %d: rejected (accuracy=%.4f, T_i=%.3f)",
                client_id, accuracy, state.trust_score,
            )
            return False, accuracy, payload_hash

    # ------------------------------------------------------------------
    def filter_trusted_clients(
        self,
        client_ids: List[int],
        models: List[RandomForestClassifier],
    ) -> Tuple[List[int], List[RandomForestClassifier]]:
        """Filter out model estimators from nodes below the trust threshold.

        Parameters
        ----------
        client_ids : list[int]
            Client IDs corresponding to each model in *models*.
        models : list[RandomForestClassifier]
            Models received from clients, ordered to match *client_ids*.

        Returns
        -------
        trusted_ids : list[int]
            Subset of client IDs with T_i >= min_trust_threshold.
        trusted_models : list[RandomForestClassifier]
            Corresponding models that passed the trust filter.
        """
        trusted_ids = []
        trusted_models = []

        for cid, model in zip(client_ids, models):
            state = self._registry.get(cid)
            if state is None:
                # Unknown node — register with default score and accept (first seen)
                state = self._ensure_node(cid)
            if state.trust_score >= self.min_trust_threshold:
                trusted_ids.append(cid)
                trusted_models.append(model)
            else:
                _log_warning(
                    "Client %d: isolated (T_i=%.3f < threshold %.2f)",
                    cid, state.trust_score, self.min_trust_threshold,
                )

        return trusted_ids, trusted_models


# ------------------------------------------------------------------
def aggregate_random_forests(
    client_models: List[RandomForestClassifier],
) -> RandomForestClassifier:
    """Merge a list of trained RandomForestClassifiers into one global model.

    Convenience wrapper around src.models.rf_federated.aggregate_rf_models for
    use after the trust filter has already pruned adversarial updates.

    Parameters
    ----------
    client_models : list[RandomForestClassifier]
        Filtered list of models from trusted clients only.

    Returns
    -------
    RandomForestClassifier
        Aggregated global forest containing all estimators from trusted nodes.
    """
    from src.models.rf_federated import aggregate_rf_models as _agg
    return _agg(client_models)


# ------------------------------------------------------------------
# Logging helpers (avoid circular imports at module level)

def _log_info(msg: str, *args):
    """Convenience logging wrapper for info messages."""
    from src.utils.logger import get_logger
    logger = get_logger("DynamicTrustManager")
    logger.info(msg, *args)


def _log_warning(msg: str, *args):
    """Convenience logging wrapper for warnings."""
    from src.utils.logger import get_logger
    logger = get_logger("DynamicTrustManager")
    logger.warning(msg, *args)
