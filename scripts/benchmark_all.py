#!/usr/bin/env python3
"""End-to-end multi-dataset benchmarking suite for FedTrustChain.

Loads each configured dataset, runs a simulated federated learning round with
length-prefixed TCP socket communication between clients and server, aggregates
the global Random Forest model, evaluates it, and exports structured metrics to
``results/benchmark_summary.json``.

Usage:
    python scripts/benchmark_all.py
"""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import yaml
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

# Import project modules
from data.dataset_loader import IoTDatasetLoader
from src.models.rf_federated import aggregate_rf_models
from src.client.client_node import FederatedClient
from src.server.server_node import FederatedServer
from src.utils.logger import get_logger


# ------------------------------------------------------------------
def _find_free_port(host: str = "127.0.0.1", start_port: int = 5060) -> int:
    """Find a TCP port that is currently free to bind on."""
    for port in range(start_port, start_port + 100):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((host, port))
                return port
        except OSError:
            continue
    raise RuntimeError("No free ports found in range 5060-5159")


# ------------------------------------------------------------------
def _run_benchmark_round(
    dataset_name: str,
    data_path: str,
    num_clients: int = 2,
    n_estimators_per_client: int = 50,
    max_depth: int = 15,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Execute one full federated learning benchmark round on a single dataset.

    Parameters
    ----------
    dataset_name : str
        Human-readable dataset identifier (used in logging and output).
    data_path : str
        Path to the CSV or Parquet file.
    num_clients : int
        Number of simulated client nodes.
    n_estimators_per_client : int
        Trees per local forest.
    max_depth : int
        Maximum depth per tree.
    random_state : int
        Reproducibility seed.

    Returns
    -------
    metrics : dict
        Structured evaluation results for this dataset run.
    """
    logger = get_logger(f"Benchmark-{dataset_name}")
    round_start = time.time()

    # --- load & preprocess ---------------------------------------------
    loader = IoTDatasetLoader(config_path=os.path.join(PROJECT_ROOT, "configs/config.yaml"))
    # Apply stratified downsampling for ultra-large datasets
    is_ciciot = "ciciot" in data_path.lower()
    max_samples = 400_000 if is_ciciot else None

    X_train, X_test, y_train, y_test, scaler = loader.load_and_preprocess(
        dataset_path_or_key=data_path,
        test_size=0.2,
        max_samples=max_samples,
    )

    logger.info("Dataset '%s': %d train samples (%d features), %d test",
                dataset_name, X_train.shape[0], X_train.shape[1], len(X_test))

    # --- partition across clients --------------------------------------
    n = len(X_train)
    chunk_size = n // num_clients
    remainder = n % num_clients
    partitions: List[Tuple[np.ndarray, np.ndarray]] = []
    start_idx = 0
    for i in range(num_clients):
        end_idx = start_idx + chunk_size + (1 if i < remainder else 0)
        partitions.append((X_train[start_idx:end_idx], y_train[start_idx:end_idx]))
        start_idx = end_idx

    # --- allocate port -------------------------------------------------
    port = _find_free_port()
    logger.info("Using TCP port %d for benchmark round", port)

    # --- synchronization primitives ------------------------------------
    server_ready = threading.Event()
    results: Dict[str, Any] = {
        "global_model": None,
        "metrics": None,
    }

    def _server_worker():
        srv = FederatedServer(
            host="127.0.0.1", port=port, min_clients=num_clients
        )
        server_ready.set()
        gmodel, metrics = srv.start(X_test, y_test)
        results["global_model"] = gmodel
        results["metrics"] = metrics

    def _client_worker(client_id: int, X_local, y_local):
        client = FederatedClient(
            client_id=client_id,
            server_host="127.0.0.1",
            server_port=port,
            n_estimators=n_estimators_per_client,
            max_depth=max_depth,
            random_state=random_state + client_id,
        )
        server_ready.wait(timeout=15)
        client.train_and_send(X_local, y_local)

    # --- launch threads ------------------------------------------------
    srv_thread = threading.Thread(target=_server_worker, daemon=True)
    srv_thread.start()

    client_threads: List[threading.Thread] = []
    for cid, (X_c, y_c) in enumerate(partitions):
        t = threading.Thread(
            target=_client_worker, args=(cid, X_c, y_c), daemon=True
        )
        client_threads.append(t)

    for t in client_threads:
        t.start()
    for t in client_threads:
        t.join(timeout=600)  # extended timeout for large datasets (e.g. CICIOT23)

    srv_thread.join(timeout=120)

    elapsed = time.time() - round_start

    # --- assemble metrics dict -----------------------------------------
    gm = results.get("global_model")
    rm = results.get("metrics") or {}

    metrics = {
        "dataset": dataset_name,
        "samples_train": int(X_train.shape[0]),
        "samples_test": int(len(X_test)),
        "n_features": int(X_train.shape[1]),
        "num_clients": num_clients,
        "total_trees": int(gm.n_estimators) if gm else 0,
        "accuracy": round(float(rm.get("accuracy", 0.0)), 4),
        "f1_score": round(float(rm.get("f1_score", 0.0)), 4),
        "precision": round(float(rm.get("precision", 0.0)), 4),
        "recall": round(float(rm.get("recall", 0.0)), 4),
        "execution_time_sec": round(elapsed, 3),
    }

    logger.info(
        "Benchmark complete — Acc: %.2f%% | F1: %.2f%% | Time: %.3fs",
        metrics["accuracy"] * 100,
        metrics["f1_score"] * 100,
        elapsed,
    )

    return metrics


# ------------------------------------------------------------------
def _print_table(results: List[Dict[str, Any]]) -> None:
    """Print a formatted ASCII summary table of all benchmark results."""
    sep = "-" * 96
    header = (
        f"{'Dataset':<20} {'Samples':>8} {'Clients':>8}"
        f" {'Trees':>7} {'Accuracy':>10} {'F1-Score':>10} {'Time(s)':>8}"
    )

    print()
    print(sep)
    print(header)
    print(sep)

    for r in results:
        print(
            f"{r['dataset']:<20} {r['samples_train']:>8d} {r['num_clients']:>8d}"
            f" {r['total_trees']:>7d} {r['accuracy']:>10.4f} "
            f"{r['f1_score']:>10.4f} {r['execution_time_sec']:>8.3f}"
        )

    print(sep)
    total = sum(r["samples_train"] for r in results)
    avg_acc = np.mean([r["accuracy"] for r in results])
    avg_f1 = np.mean([r["f1_score"] for r in results])
    total_time = sum(r["execution_time_sec"] for r in results)
    print(
        f"{'TOTAL / AVG':<20} {total:>8d}"
        f" {'—':>7s} {'—':>6s} {avg_acc:>10.4f} {avg_f1:>10.4f} {total_time:>8.3f}"
    )
    print(sep)
    print()


# ------------------------------------------------------------------
def main():
    """Run the full multi-dataset benchmarking suite."""
    # --- load config ---------------------------------------------------
    config_path = os.path.join(PROJECT_ROOT, "configs/config.yaml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    server_cfg = cfg.get("server", {})
    model_cfg = cfg.get("model", {})
    paths_cfg = cfg.get("paths", {})

    num_clients = server_cfg.get("min_clients", 2)
    n_estimators = model_cfg.get("n_estimators_per_client", 50)
    max_depth = model_cfg.get("max_depth", 15)
    random_state = model_cfg.get("random_state", 42)

    dataset_configs = [
        {"name": "wustl_ehms", "path": paths_cfg.get("wustl_ehms")},
        {"name": "ciciot23",   "path": paths_cfg.get("ciciot23")},
        {"name": "nf_ton_iot", "path": paths_cfg.get("nf_ton_iot")},
    ]

    # --- run benchmarks ------------------------------------------------
    results: List[Dict[str, Any]] = []
    for ds in dataset_configs:
        path = ds["path"]
        if not path or not os.path.isfile(path):
            print(f"[WARN] Dataset '{ds['name']}' not found at {path} — skipping.")
            continue

        try:
            metrics = _run_benchmark_round(
                dataset_name=ds["name"],
                data_path=path,
                num_clients=num_clients,
                n_estimators_per_client=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
            )
            results.append(metrics)
        except Exception as exc:
            print(f"[ERROR] Benchmark failed for '{ds['name']}': {exc}")

    # --- export results ------------------------------------------------
    os.makedirs("results", exist_ok=True)
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "num_clients": num_clients,
            "n_estimators_per_client": n_estimators,
            "max_depth": max_depth,
            "random_state": random_state,
        },
        "results": results,
    }

    output_path = os.path.join("results", "benchmark_summary.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # --- display table -------------------------------------------------
    if results:
        _print_table(results)
    else:
        print("[WARN] No benchmarks completed successfully.")


if __name__ == "__main__":
    main()
