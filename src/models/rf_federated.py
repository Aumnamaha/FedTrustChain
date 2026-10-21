"""Federated Random Forest ensemble aggregation engine.

Merges locally-fitted sklearn RandomForestClassifier instances from multiple
client nodes into a single global model by concatenating their individual
decision trees — the standard "bag-of-trees" federation strategy used in
FedAvg-style approaches for tree-based ensembles.
"""

from __future__ import annotations

import copy
from typing import List

from sklearn.ensemble import RandomForestClassifier


def aggregate_rf_models(client_models: list) -> RandomForestClassifier:
    """Merge a list of trained RandomForestClassifiers into one global model.

    Parameters
    ----------
    client_models : list[RandomForestClassifier]
        A non-empty list of fitted scikit-learn Random Forest models received
        from federated clients.  All must share the same ``n_features_in_`` and
        ``n_classes_``.

    Returns
    -------
    RandomForestClassifier
        A deep-copied global model whose ``estimators_`` attribute contains
        every tree from every client, concatenated in order.

    Raises
    ------
    ValueError
        If *client_models* is empty / ``None``, or if models disagree on
        feature count or class count.
    """
    # ---- validation --------------------------------------------------------
    if not client_models:
        raise ValueError("client_models must be a non-empty list of trained "
                         "RandomForestClassifier instances.")

    if len(client_models) == 1:
        return copy.deepcopy(client_models[0])

    ref = client_models[0]
    for idx, m in enumerate(client_models[1:], start=1):
        if m.n_features_in_ != ref.n_features_in_:
            raise ValueError(
                f"Feature dimension mismatch at index {idx}: "
                f"{m.n_features_in_} vs expected {ref.n_features_in_}"
            )
        if m.n_classes_ != ref.n_classes_:
            raise ValueError(
                f"Class count mismatch at index {idx}: "
                f"{m.n_classes_} vs expected {ref.n_classes_}"
            )

    # ---- aggregation -------------------------------------------------------
    global_model = copy.deepcopy(ref)

    combined_estimators: list = []
    for model in client_models:
        combined_estimators.extend(model.estimators_)

    global_model.estimators_ = combined_estimators
    global_model.n_estimators = len(combined_estimators)

    return global_model
