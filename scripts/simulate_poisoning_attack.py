"""Adversarial Poisoning Simulation for FedTrustChain.

Runs a controlled label-flipping attack benchmark across two modes:

  Mode A (Unprotected): min_trust_threshold=0.0 - all client models blindly
      aggregated into the global forest regardless of quality.

  Mode B (FedTrustChain Protected): min_trust_threshold=0.50 - incoming model
      payloads are evaluated by DynamicTrustManager; adversarial updates with
      degraded validation accuracy are flagged, penalised, and REJECTED in the
      blockchain audit log before tree-union pooling.

Usage
-----
    python scripts/simulate_poisoning_attack.py [--port PORT]

Outputs
-------
    results/poisoning_benchmark_results.json   - Mode A vs Mode B metrics
    results/blockchain_audit_log.json          - Per-client ACCEPTED/REJECTED ledger
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.client.client_node import FederatedClient
from src.server.server_node import FederatedServer
from src.server.dynamic_trust import DynamicTrustManager
from data.dataset_loader import IoTDatasetLoader

# constants
NUM_CLIENTS = 3
N_ESTIMATORS = 50
MAX_DEPTH = 15
RANDOM_STATE = 42
DEFAULT_PORT = 5080
RESULTS_DIR = "results"


def _load_wustl_data(test_size: float = 0.2, max_samples: int = None):
    """Load and preprocess the WUSTL-EHMS dataset."""
    # Use absolute path to avoid config resolution issues
    project_root = str(Path(__file__).resolve().parent.parent)
    dataset_path = os.path.join(project_root, "data", "Datasets", "wustl-ehms-2020_with_attacks_categories.csv")
    loader = IoTDatasetLoader()
    X_train, X_test, y_train, y_test, scaler = loader.load_and_preprocess(
        dataset_path_or_key=dataset_path,
        test_size=test_size,
        max_samples=max_samples,
    )
    return X_train, X_test, y_train, y_test


def _partition_data(X: np.ndarray, y, num_clients: int):
    """Split data into num_clients contiguous equal chunks."""
    n = len(X)
    chunk_size = n // num_clients
    remainder = n % num_clients
    partitions = []
    start = 0
    for i in range(num_clients):
        end = start + chunk_size + (1 if i < remainder else 0)
        partitions.append((X[start:end], y[start:end]))
        start = end
    return partitions


def _label_flip(X: np.ndarray, y):
    """Flip all binary labels (y_poisoned = 1 - y_clean)."""
    y_flipped = 1.0 - y
    return X.copy(), y_flipped


def _run_single_round(
    port: int,
    partitions: List[Tuple[np.ndarray, np.ndarray]],
    X_test: np.ndarray,
    y_test,
    trust_manager_kwargs: Dict[str, Any] | None = None,
    blockchain_enabled: bool = False,
    audit_log_path: str | None = None,
):
    """Run one FL round with clients sending models to server over UDP.

    Returns (global_model, metrics).
    """
    num_clients = len(partitions)
    results: Dict[str, Any] = {}
    server_ready = threading.Event()

    # Create holdout set for trust evaluation when needed
    val_split = max(50, int(len(X_test) * 0.3))
    X_val, y_val = X_test[:val_split], y_test[:val_split]

    def _server_worker():
        srv_kwargs = {"host": "127.0.0.1", "port": port, "min_clients": num_clients}
        if trust_manager_kwargs is not None:
            srv_obj = FederatedServer(**srv_kwargs)
            srv_obj._trust_threshold = trust_manager_kwargs.get("min_trust_threshold", 0.50)
            srv_obj._reward_rate = trust_manager_kwargs.get("reward_rate", 0.05)
            srv_obj._decay_rate = trust_manager_kwargs.get("decay_rate", 0.25)
            srv_obj._audit_log_path = audit_log_path or os.path.join(RESULTS_DIR, "blockchain_audit_log.json")
        else:
            srv_obj = FederatedServer(**srv_kwargs)

        server_ready.set()
        if trust_manager_kwargs is not None and blockchain_enabled:
            gmodel, metrics = srv_obj.start(
                X_test, y_test,
                X_val=X_val, y_val=y_val,
                blockchain_enabled=blockchain_enabled,
            )
        else:
            gmodel, metrics = srv_obj.start(X_test, y_test, blockchain_enabled=blockchain_enabled)
        results["global_model"] = gmodel
        results["metrics"] = metrics

    def _client_worker(cid: int, X_c: np.ndarray, y_c):
        client = FederatedClient(
            client_id=cid + 1,
            server_host="127.0.0.1",
            server_port=port,
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            random_state=RANDOM_STATE + cid,
        )
        server_ready.wait(timeout=15)
        client.train_and_send(X_c, y_c)

    # Launch server thread
    srv_thread = threading.Thread(target=_server_worker, daemon=True)
    srv_thread.start()

    # Send clients sequentially to avoid UDP interleaving
    for cid, (X_c, y_c) in enumerate(partitions):
        t = threading.Thread(
            target=_client_worker, args=(cid, X_c, y_c), daemon=True
        )
        t.start()
        t.join(timeout=30)
        time.sleep(2.0)  # gap for reassembly

    srv_thread.join(timeout=30)

    gmodel = results.get("global_model")
    metrics = results.get("metrics", {})
    if gmodel is None:
        raise RuntimeError("Server did not produce a global model within timeout.")
    return gmodel, metrics


def run_benchmark(port_base: int = DEFAULT_PORT) -> Dict[str, Any]:
    """Execute the full adversarial poisoning benchmark.

    1. Load WUSTL-EHMS data and partition into 3 clients (2 clean + 1 adv).
    2. Mode A: blind aggregation with trust threshold = 0.0.
    3. Mode B: FedTrustChain protected with trust threshold = 0.50.
    4. Export results to JSON artifacts.
    """
    print("=" * 72)
    print("  FedTrustChain - Adversarial Poisoning Benchmark")
    print("=" * 72)

    # Step 1: Load data and partition
    print("\n[1/4] Loading WUSTL-EHMS dataset ...")
    X_train, X_test, y_train, y_test = _load_wustl_data()
    partitions = _partition_data(X_train, y_train, NUM_CLIENTS)

    # Client 0 (id=1): clean partition
    X_clean_1, y_clean_1 = partitions[0]
    # Client 1 (id=2): clean partition
    X_clean_2, y_clean_2 = partitions[1]
    # Client 2 (id=3): adversarial - label flipped
    X_adv, y_adv = _label_flip(*partitions[2])

    print(f"      Clean partition 0: {X_clean_1.shape[0]} samples")
    print(f"      Clean partition 1: {X_clean_2.shape[0]} samples")
    print(f"      Adversarial (flipped): {X_adv.shape[0]} samples")

    # Step 2: Mode A - Unprotected (blind aggregation)
    port_a = port_base
    print("\n[2/4] Mode A - Unprotected (min_trust_threshold=0.0, blind agg)")
    print("-" * 60)

    mode_a_partitions = [
        (X_clean_1, y_clean_1),
        (X_clean_2, y_clean_2),
        (X_adv, y_adv),
    ]

    gmodel_a, metrics_a = _run_single_round(
        port=port_a,
        partitions=mode_a_partitions,
        X_test=X_test,
        y_test=y_test,
        trust_manager_kwargs={"min_trust_threshold": 0.0},
    )

    # Compute per-client local accuracy on test set for reference
    local_accs_a = []
    for cid in range(NUM_CLIENTS):
        X_c, y_c = mode_a_partitions[cid]
        tmp_model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=RANDOM_STATE + cid
        )
        tmp_model.fit(X_c, y_c)
        local_accs_a.append(float(accuracy_score(y_test, tmp_model.predict(X_test))))

    acc_key = "accuracy"
    print(f"      Global accuracy: {metrics_a[acc_key]:.4f}")
    print(f"      Local accs (C1..C3): {[round(a, 4) for a in local_accs_a]}")

    # Step 3: Mode B - FedTrustChain Protected
    port_b = port_base + 100
    print("\n[3/4] Mode B - FedTrustChain Protected (min_trust_threshold=0.50)")
    print("-" * 60)

    mode_b_partitions = [
        (X_clean_1, y_clean_1),
        (X_clean_2, y_clean_2),
        (X_adv, y_adv),
    ]

    gmodel_b, metrics_b = _run_single_round(
        port=port_b,
        partitions=mode_b_partitions,
        X_test=X_test,
        y_test=y_test,
        trust_manager_kwargs={
            "min_trust_threshold": 0.50,
            "reward_rate": 0.05,
            "decay_rate": 0.60,  # single rejection drops T below threshold
        },
        blockchain_enabled=True,
        audit_log_path=os.path.join(RESULTS_DIR, "blockchain_audit_log.json"),
    )

    local_accs_b = []
    for cid in range(NUM_CLIENTS):
        X_c, y_c = mode_b_partitions[cid]
        tmp_model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=RANDOM_STATE + cid
        )
        tmp_model.fit(X_c, y_c)
        local_accs_b.append(float(accuracy_score(y_test, tmp_model.predict(X_test))))

    print(f"      Global accuracy: {metrics_b[acc_key]:.4f}")
    print(f"      Local accs (C1..C3): {[round(a, 4) for a in local_accs_b]}")

    # Step 4: Export results
    os.makedirs(RESULTS_DIR, exist_ok=True)

    benchmark_results = {
        "benchmark": "adversarial_poisoning_simulation",
        "dataset": "WUSTL-EHMS-2020",
        "num_clients": NUM_CLIENTS,
        "n_estimators_per_client": N_ESTIMATORS,
        "max_depth": MAX_DEPTH,
        "random_state": RANDOM_STATE,
        "label_flip_strategy": "1 - y (binary inversion)",
        "modes": {
            "mode_a_unprotected": {
                "min_trust_threshold": 0.0,
                "blockchain_enabled": False,
                "global_accuracy": metrics_a[acc_key],
                "f1_score": metrics_a.get("f1_score", 0.0),
                "precision": metrics_a.get("precision", 0.0),
                "recall": metrics_a.get("recall", 0.0),
                "local_client_accuracies": {
                    f"client_{cid + 1}": round(acc, 4) for cid, acc in enumerate(local_accs_a)
                },
            },
            "mode_b_fedtrustchain_protected": {
                "min_trust_threshold": 0.50,
                "blockchain_enabled": True,
                "global_accuracy": metrics_b[acc_key],
                "f1_score": metrics_b.get("f1_score", 0.0),
                "precision": metrics_b.get("precision", 0.0),
                "recall": metrics_b.get("recall", 0.0),
                "local_client_accuracies": {
                    f"client_{cid + 1}": round(acc, 4) for cid, acc in enumerate(local_accs_b)
                },
            },
        },
    }

    output_path = os.path.join(RESULTS_DIR, "poisoning_benchmark_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)

    print(f"\n[4/4] Results exported to {output_path}")
    print("=" * 72)
    return benchmark_results


def main():
    parser = argparse.ArgumentParser(description="Adversarial Poisoning Simulation for FedTrustChain")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Base UDP port (default: 5080)")
    args = parser.parse_args()

    results = run_benchmark(port_base=args.port)

    # Print comparison matrix to stdout
    ma = results["modes"]["mode_a_unprotected"]
    mb = results["modes"]["mode_b_fedtrustchain_protected"]
    print("\n" + "=" * 72)
    print("  MODE A vs MODE B - COMPARISON MATRIX")
    print("=" * 72)
    header = f"  {'Metric':<35} {'Mode A (Unprotected)':>25} {'Mode B (Protected)':>25}"
    print(header)
    print("-" * 72)
    print(f"  {'Global Accuracy':<35} {ma['global_accuracy']:>24.4f} {mb['global_accuracy']:>24.4f}")
    print(f"  {'F1-Score (weighted)':<35} {ma['f1_score']:>24.4f} {mb['f1_score']:>24.4f}")
    print(f"  {'Precision (weighted)':<35} {ma['precision']:>24.4f} {mb['precision']:>24.4f}")
    print(f"  {'Recall (weighted)':<35} {ma['recall']:>24.4f} {mb['recall']:>24.4f}")
    acc_diff = mb["global_accuracy"] - ma["global_accuracy"]
    print("-" * 72)
    print(f"  {'Accuracy Delta (B - A)':<35} {'':>25} {acc_diff:>23.4f}")
    print("=" * 72)

    return results


if __name__ == "__main__":
    main()
