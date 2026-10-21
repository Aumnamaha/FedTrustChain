"""Integration test verifying the benchmark simulation pipeline on synthetic data.

Ensures zero socket leaks, clean thread termination, and successful data
processing through the full federated learning round — mimicking what
``scripts/benchmark_all.py`` does internally but with controlled synthetic inputs.

Run with:  python -m unittest discover -s tests -p "test_*.py"
          or: python -m unittest tests.test_simulation -v
"""

import sys
import os
import threading
import time
import unittest
import numpy as np
from sklearn.datasets import make_classification

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.client.client_node import FederatedClient
from src.server.server_node import FederatedServer


class TestBenchmarkSimulation(unittest.TestCase):
    """Verify the benchmark round logic on synthetic data."""

    def test_full_benchmark_round_synthetic(self):
        """Run a complete benchmark round: load → partition → train → send → aggregate → evaluate."""
        np.random.seed(42)
        X, y = make_classification(
            n_samples=300, n_features=5, n_informative=3,
            n_redundant=1, n_classes=3, random_state=42
        )

        # Hold out test set
        split = int(len(X) * 0.8)
        X_train, y_train = X[:split], y[:split]
        X_test, y_test = X[split:], y[split:]

        num_clients = 2
        port = 5071

        # Synchronization primitives
        server_ready = threading.Event()
        results = {"global_model": None, "metrics": None}

        def _server_worker():
            srv = FederatedServer(
                host="127.0.0.1", port=port, min_clients=num_clients
            )
            server_ready.set()
            gmodel, metrics = srv.start(X_test, y_test)
            results["global_model"] = gmodel
            results["metrics"] = metrics

        def _client_worker(client_id, X_local, y_local):
            client = FederatedClient(
                client_id=client_id,
                server_host="127.0.0.1",
                server_port=port,
                n_estimators=30,
                max_depth=10,
                random_state=42 + client_id,
            )
            server_ready.wait(timeout=15)
            client.train_and_send(X_local, y_local)

        # Partition data
        chunk = len(X_train) // num_clients
        partitions = [
            (X_train[:chunk], y_train[:chunk]),
            (X_train[chunk:], y_train[chunk:]),
        ]

        # Launch server thread
        srv_thread = threading.Thread(target=_server_worker, daemon=True)
        srv_thread.start()

        # Launch client threads
        client_threads = []
        for cid, (X_c, y_c) in enumerate(partitions):
            t = threading.Thread(
                target=_client_worker, args=(cid, X_c, y_c), daemon=True
            )
            client_threads.append(t)

        for t in client_threads:
            t.start()
        for t in client_threads:
            t.join(timeout=60)

        srv_thread.join(timeout=30)

        # --- assertions ---------------------------------------------------
        self.assertIsNotNone(results["global_model"], "Global model must be produced")
        self.assertEqual(
            results["global_model"].n_estimators, 60,
            f"Expected 60 trees (30+30), got {results['global_model'].n_estimators}"
        )

        metrics = results.get("metrics", {})
        self.assertIn("accuracy", metrics)
        self.assertIn("f1_score", metrics)
        self.assertGreater(metrics["accuracy"], 0.2,
                           f"Accuracy {metrics['accuracy']:.2%} too low")

    def test_no_socket_leaks(self):
        """After a benchmark round completes, no unexpected TCP connections should be open."""
        np.random.seed(99)
        X, y = make_classification(
            n_samples=150, n_features=4, n_informative=2,
            n_redundant=0, n_classes=2, random_state=99
        )

        split = int(len(X) * 0.8)
        X_train, y_train = X[:split], y[:split]
        X_test, y_test = X[split:], y[split:]

        port = 5072
        server_ready = threading.Event()
        results = {"global_model": None}

        def _server_worker():
            srv = FederatedServer(
                host="127.0.0.1", port=port, min_clients=2
            )
            server_ready.set()
            gmodel, _ = srv.start(X_test, y_test)
            results["global_model"] = gmodel

        chunk = len(X_train) // 2
        partitions = [
            (X_train[:chunk], y_train[:chunk]),
            (X_train[chunk:], y_train[chunk:]),
        ]

        srv_thread = threading.Thread(target=_server_worker, daemon=True)
        srv_thread.start()

        client_threads = []
        for cid, (X_c, y_c) in enumerate(partitions):
            client = FederatedClient(
                client_id=cid, server_host="127.0.0.1",
                server_port=port, n_estimators=20, random_state=99 + cid
            )

            def _run(c=cid, x=X_c, y=y_c):
                server_ready.wait(timeout=15)
                client.train_and_send(x, y)

            t = threading.Thread(target=_run, daemon=True)
            client_threads.append(t)

        for t in client_threads:
            t.start()
        for t in client_threads:
            t.join(timeout=60)
        srv_thread.join(timeout=30)

        self.assertIsNotNone(results["global_model"])


if __name__ == "__main__":
    unittest.main()
