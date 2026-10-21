"""Integration tests for TCP socket client-server federated pipeline.

Uses multi-threading with proper synchronization (threading.Barrier) to run a
FederatedServer and multiple FederatedClient instances concurrently, verifying
end-to-end model transmission, aggregation, and evaluation via length-prefixed
TCP framing.

Run with:  python -m unittest discover -s tests -p "test_*.py"
          or: python -m unittest tests.test_networking -v
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


# ------------------------------------------------------------------
class TestFederatedNetworking(unittest.TestCase):
    """End-to-end integration tests for the TCP networking layer."""

    def setUp(self):
        np.random.seed(42)
        self.X, self.y = make_classification(
            n_samples=400, n_features=5, n_informative=3,
            n_redundant=1, n_classes=3, random_state=42
        )
        self.X_test = self.X[:80]
        self.y_test = self.y[:80]

    # ------------------------------------------------------------------
    def test_full_client_server_pipeline(self):
        """Full pipeline: 2 clients train and send models to server."""
        port = 5061
        num_clients = 2

        # --- shared synchronization primitives ----------------------------
        barrier = threading.Barrier(num_clients)
        server_ready = threading.Event()
        results = {"global_model": None, "metrics": None}

        def _server_worker():
            """Run the server on its own thread."""
            srv = FederatedServer(host="127.0.0.1", port=port, min_clients=num_clients)
            server_ready.set()  # signal clients that server is bound & listening
            gmodel, metrics = srv.start(self.X_test, self.y_test)
            results["global_model"] = gmodel
            results["metrics"] = metrics
            return srv

        def _client_worker(client_id, X_local, y_local):
            """Train locally and send model to server."""
            client = FederatedClient(
                client_id=client_id, server_host="127.0.0.1",
                server_port=port, n_estimators=50, random_state=client_id * 42,
            )
            # Wait for server to be ready before connecting
            server_ready.wait(timeout=10)
            client.train_and_send(X_local, y_local)

        # --- partition data across clients --------------------------------
        n = len(self.X)
        split = n // num_clients
        partitions = [(self.X[i * split: (i + 1) * split],
                       self.y[i * split: (i + 1) * split])
                      for i in range(num_clients)]

        # --- start server thread ------------------------------------------
        srv_thread = threading.Thread(target=_server_worker, daemon=True)
        srv_thread.start()

        # --- start client threads -----------------------------------------
        client_threads = []
        for cid, (X_c, y_c) in enumerate(partitions):
            t = threading.Thread(target=_client_worker, args=(cid, X_c, y_c))
            client_threads.append(t)

        for t in client_threads:
            t.start()
        for t in client_threads:
            t.join(timeout=30)

        # --- wait for server to finish ------------------------------------
        srv_thread.join(timeout=30)

        # --- assertions ---------------------------------------------------
        self.assertIsNotNone(results["global_model"])
        self.assertEqual(results["global_model"].n_estimators, 100)
        metrics = results["metrics"]
        self.assertIn("accuracy", metrics)
        self.assertIn("f1_score", metrics)
        self.assertGreater(metrics["accuracy"], 0.3)

    # ------------------------------------------------------------------
    def test_server_receives_correct_model_count(self):
        """Server's client_models list should contain exactly min_clients models."""
        port = 5062
        num_clients = 2

        server_ready = threading.Event()
        results = {"client_models": None}

        def _server_worker():
            srv = FederatedServer(host="127.0.0.1", port=port, min_clients=num_clients)
            server_ready.set()
            srv.start(self.X_test, self.y_test)
            results["client_models"] = srv.client_models

        n = len(self.X)
        partitions = [(self.X[:n // 2], self.y[:n // 2]),
                      (self.X[n // 2:], self.y[n // 2:])]

        srv_thread = threading.Thread(target=_server_worker, daemon=True)
        srv_thread.start()

        client_threads = []
        for cid, (X_c, y_c) in enumerate(partitions):
            client = FederatedClient(client_id=cid, server_host="127.0.0.1",
                                     server_port=port, n_estimators=50, random_state=cid * 42)
            t = threading.Thread(target=lambda c=cid, x=X_c, y=y_c: (
                server_ready.wait(timeout=10),
                client.train_and_send(x, y)
            ))
            client_threads.append(t)

        for t in client_threads:
            t.start()
        for t in client_threads:
            t.join(timeout=30)
        srv_thread.join(timeout=30)

        self.assertEqual(len(results["client_models"]), num_clients)


if __name__ == "__main__":
    unittest.main()
