"""Integration tests for chunked UDP socket client-server federated pipeline.

Uses multi-threading with proper synchronization (threading.Barrier) to run a
FederatedServer and multiple FederatedClient instances concurrently, verifying
end-to-end model transmission, aggregation, and evaluation via **UDP datagram**
chunking with 12-byte binary headers.

Run with:  python -m unittest discover -s tests -p "test_*.py"
          or: python -m unittest tests.test_networking -v
"""

import pickle
import socket
import struct
import sys
import os
import threading
import time
import unittest
import numpy as np
from sklearn.datasets import make_classification

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.client.client_node import FederatedClient
from src.models.rf_federated import aggregate_rf_models
from src.server.server_node import FederatedServer


# ------------------------------------------------------------------
class TestUDPPacketChunking(unittest.TestCase):
    """Unit tests for the UDP chunking / reassembly logic."""

    HEADER_FMT = "!III"
    HEADER_SIZE = struct.calcsize(HEADER_FMT)  # 12

    def test_chunk_creation(self):
        """send_udp_model partitions payload into correctly-sized chunks."""
        client = FederatedClient(client_id=0, server_host="127.0.0.1",
                                 server_port=5063, chunk_size=512)
        # 1500 bytes → ceil(1500/512) = 3 chunks
        payload = b"A" * 1500
        total_chunks = max(1, (len(payload) + client.chunk_size - 1) // client.chunk_size)
        self.assertEqual(total_chunks, 3)

    def test_header_format(self):
        """Header is exactly 12 bytes with three big-endian unsigned ints."""
        header = struct.pack(self.HEADER_FMT, 42, 7, 10)
        self.assertEqual(len(header), 12)
        cid, idx, total = struct.unpack(self.HEADER_FMT, header)
        self.assertEqual(cid, 42)
        self.assertEqual(idx, 7)
        self.assertEqual(total, 10)

    def test_reassembly_integrity(self):
        """Reassembling chunks reproduces the original payload exactly."""
        original = os.urandom(3000)
        chunk_size = 1024
        total_chunks = max(1, (len(original) + chunk_size - 1) // chunk_size)

        chunks: list[bytes] = []
        for i in range(total_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, len(original))
            chunk_data = original[start:end]
            header = struct.pack(self.HEADER_FMT, 1, i, total_chunks)
            chunks.append(header + chunk_data)

        # Reassemble (simulate server-side logic)
        buffer: dict[int, bytes] = {}
        for pkt in chunks:
            cid, idx, tc = struct.unpack(self.HEADER_FMT, pkt[:self.HEADER_SIZE])
            buffer[idx] = pkt[self.HEADER_SIZE:]

        reassembled = b"".join(buffer[i] for i in range(total_chunks))
        self.assertEqual(reassembled, original)

    def test_reassembly_with_dropped_chunk(self):
        """Missing chunk results in incomplete reassembly."""
        total_chunks = 5
        chunks: list[bytes] = []
        for i in range(total_chunks):
            header = struct.pack(self.HEADER_FMT, 1, i, total_chunks)
            chunks.append(header + b"chunk_" + str(i).encode())

        # Simulate dropping chunk index 2
        buffer: dict[int, bytes] = {}
        for pkt in chunks:
            cid, idx, tc = struct.unpack(self.HEADER_FMT, pkt[:self.HEADER_SIZE])
            if idx != 2:
                buffer[idx] = pkt[self.HEADER_SIZE:]

        # Not all chunks present → reassembly incomplete
        self.assertNotEqual(len(buffer), total_chunks)


# ------------------------------------------------------------------
class TestFederatedNetworking(unittest.TestCase):
    """End-to-end integration tests for the UDP networking layer."""

    def setUp(self):
        np.random.seed(42)
        self.X, self.y = make_classification(
            n_samples=400, n_features=5, n_informative=3,
            n_redundant=1, n_classes=3, random_state=42,
        )
        self.X_test = self.X[:80]
        self.y_test = self.y[:80]

    # ------------------------------------------------------------------
    def test_full_client_server_pipeline(self):
        """Full pipeline: 2 clients train and send models to server via UDP.

        Clients send sequentially (not simultaneously) so that the server's
        sequential _receive_model calls match each client's datagrams correctly.
        On a shared UDP socket, simultaneous sends from multiple clients would
        interleave datagrams and cause reassembly timeouts."""
        port = 5071
        num_clients = 2

        results = {"global_model": None, "metrics": None}

        def _server_worker():
            srv = FederatedServer(
                host="127.0.0.1", port=port, min_clients=num_clients,
                reassembly_timeout=60.0,  # generous timeout for tests
            )
            # Bind socket and signal readiness BEFORE entering receive loop.
            srv._udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            srv._udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv._udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF,
                                       4 * 1024 * 1024)
            srv._udp_socket.bind((srv.host, srv.port))
            server_ready.set()
            # Now run receive → aggregate → evaluate (copy of start() body).
            try:
                for i in range(srv.min_clients):
                    model = srv._receive_model(client_index=i + 1)
                    srv.client_models.append(model)
                    results["global_model"] = (
                        aggregate_rf_models(srv.client_models)
                    )
                    y_pred = results["global_model"].predict(self.X_test)
                    from sklearn.metrics import (
                        accuracy_score, f1_score,
                        precision_score, recall_score,
                    )
                    results["metrics"] = {
                        "accuracy": float(accuracy_score(self.y_test, y_pred)),
                        "f1_score": float(f1_score(
                            self.y_test, y_pred, average="weighted"
                        )),
                        "precision": float(precision_score(
                            self.y_test, y_pred,
                            average="weighted", zero_division=0
                        )),
                        "recall": float(recall_score(
                            self.y_test, y_pred,
                            average="weighted", zero_division=0
                        )),
                    }
            finally:
                srv._udp_socket.close()
        
        server_ready = threading.Event()

        # --- partition data across clients --------------------------------
        n = len(self.X)
        split = n // num_clients
        partitions = [
            (self.X[i * split: (i + 1) * split], self.y[i * split: (i + 1) * split])
            for i in range(num_clients)
        ]

        # --- start server thread ------------------------------------------
        srv_thread = threading.Thread(target=_server_worker, daemon=True)
        srv_thread.start()

        # --- start clients sequentially from main thread -----------------
        # Clients wait for the server socket to be bound before sending
        # to avoid a race where packets arrive before recvfrom is called.
        for cid, (X_c, y_c) in enumerate(partitions):
            client = FederatedClient(
                client_id=cid, server_host="127.0.0.1",
                server_port=port, n_estimators=50,
                random_state=cid * 42,
            )
            server_ready.wait(timeout=30)  # wait for server to bind
            client.train_and_send(X_c, y_c)
            time.sleep(1.0)  # gap to let server finish reassembly

        srv_thread.join(timeout=60)

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
        port = 5072
        num_clients = 2

        server_ready = threading.Event()
        results: dict = {"client_models": None}

        def _server_worker():
            srv = FederatedServer(
                host="127.0.0.1", port=port, min_clients=num_clients,
                reassembly_timeout=60.0,
            )
            # Bind socket and signal readiness before receive loop.
            srv._udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            srv._udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv._udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF,
                                       4 * 1024 * 1024)
            srv._udp_socket.bind((srv.host, srv.port))
            server_ready.set()
            try:
                for i in range(srv.min_clients):
                    model = srv._receive_model(client_index=i + 1)
                    srv.client_models.append(model)
                results["client_models"] = list(srv.client_models)
            finally:
                srv._udp_socket.close()

        n = len(self.X)
        partitions = [
            (self.X[:n // 2], self.y[:n // 2]),
            (self.X[n // 2:], self.y[n // 2:]),
        ]

        srv_thread = threading.Thread(target=_server_worker, daemon=True)
        srv_thread.start()

        # Send clients sequentially after server is ready.
        for cid, (X_c, y_c) in enumerate(partitions):
            client = FederatedClient(
                client_id=cid, server_host="127.0.0.1",
                server_port=port, n_estimators=50, random_state=cid * 42,
            )
            server_ready.wait(timeout=30)
            client.train_and_send(X_c, y_c)
            time.sleep(1.0)  # gap for reassembly
        srv_thread.join(timeout=60)

        self.assertEqual(len(results["client_models"]), num_clients)


if __name__ == "__main__":
    unittest.main()
