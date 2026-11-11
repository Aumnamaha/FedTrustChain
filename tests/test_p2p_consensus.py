"""Integration tests for P2P Consensus Network (Phase 3 Step 2).

Tests:
- Asynchronous block broadcasting over UDP sockets.
- Simultaneous block mining race conditions with automatic fork resolution.
- Network disconnection and state recovery when a node re-joins and syncs chain.

Run with: python -m unittest discover -s tests -p "test_p2p_consensus.py" -v
"""

import json
import os
import pickle
import socket
import struct
import sys
import threading
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from src.blockchain.consensus import ProofOfWork, ValidatorRegistry, ChainValidator
from src.server.dynamic_trust import DynamicTrustManager


# ------------------------------------------------------------------
HEADER_FMT = "!III"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
BLOCK_HEADER_FMT = "!I"
SERVER_HOST = "127.0.0.1"

# ------------------------------------------------------------------
def _generate_data(n_samples: int = 50, n_features: int = 4, seed: int = 42):
    """Generate synthetic data for test clients."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


def _train_and_pickle(X, y, seed=42):
    """Train a local RF model and pickle it."""
    model = RandomForestClassifier(n_estimators=20, max_depth=5, random_state=seed)
    model.fit(X, y)
    return pickle.dumps(model), model


def _send_model_over_udp(sock: socket.socket, client_id: int, payload_bytes: bytes,
                         host: str, port: int, chunk_size: int = 2048):
    """Send a pickled model over UDP with the standard header."""
    total_chunks = max(1, (len(payload_bytes) + chunk_size - 1) // chunk_size)
    for i in range(total_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, len(payload_bytes))
        hdr = struct.pack(HEADER_FMT, client_id, i, total_chunks)
        sock.sendto(hdr + payload_bytes[start:end], (host, port))


def _send_block_broadcast(sock: socket.socket, block_data: Dict[str, Any],
                          host: str, port: int):
    """Send a consensus block broadcast over UDP."""
    body = json.dumps(block_data, default=str).encode("utf-8")
    hdr = struct.pack(BLOCK_HEADER_FMT, block_data["index"])
    sock.sendto(hdr + body, (host, port))


# ------------------------------------------------------------------
class TestUDPBlockBroadcasting(unittest.TestCase):
    """Test asynchronous block broadcasting over UDP sockets."""

    def setUp(self):
        self.server_port = 7050
        self.listen_port = 8050
        # Start a minimal server socket to receive models
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind((SERVER_HOST, self.server_port))

    def tearDown(self):
        if self.server_sock:
            try:
                self.server_sock.close()
            except OSError:
                pass

    def test_block_broadcast_reception(self):
        """A block broadcast sent via UDP must be received and parsed by a listener."""
        # Start listener on listen_port
        listener = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((SERVER_HOST, self.listen_port))
        listener.settimeout(3.0)

        try:
            # Create and send a test block broadcast
            test_block = {
                "index": 5,
                "timestamp": "2025-06-01T12:00:00+00:00",
                "transactions": [{"client_id": 1, "model_hash": "abc123"}],
                "previous_hash": "0" * 64,
                "nonce": 12345,
                "hash": "000valid_hash_xyz",
                "difficulty": 3,
            }
            _send_block_broadcast(self.server_sock, test_block, SERVER_HOST, self.listen_port)

            # Receive on listener
            data, addr = listener.recvfrom(65535)
            recv_index = struct.unpack(BLOCK_HEADER_FMT, data[:4])[0]
            recv_body = json.loads(data[4:].decode("utf-8"))

            self.assertEqual(recv_index, 5)
            self.assertEqual(recv_body["index"], 5)
            self.assertEqual(recv_body["hash"][:3], "000")
        finally:
            listener.close()

    def test_broadcast_latency_under_100ms(self):
        """Block broadcast latency over localhost must be < 100ms."""
        # Start a passive listener that just drops packets
        listener = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((SERVER_HOST, self.listen_port))
        listener.settimeout(3.0)

        try:
            test_block = {
                "index": 10, "timestamp": "2025-06-01T12:00:00+00:00",
                "transactions": [], "previous_hash": "0" * 64,
                "nonce": 999, "hash": "000latency_test_hash", "difficulty": 3,
            }

            start = time.time()
            _send_block_broadcast(self.server_sock, test_block, SERVER_HOST, self.listen_port)
            data, _addr = listener.recvfrom(65535)
            latency_ms = (time.time() - start) * 1000

            self.assertLess(latency_ms, 100.0, f"Broadcast latency {latency_ms:.1f}ms exceeds 100ms")
        finally:
            listener.close()


# ------------------------------------------------------------------
class TestForkResolutionRaceConditions(unittest.TestCase):
    """Test simultaneous block mining race conditions and fork resolution."""

    def test_simultaneous_mining_same_difficulty(self):
        """Two independent miners producing blocks at the same difficulty must both be valid."""
        pox_a = ProofOfWork(default_difficulty=3)
        pox_b = ProofOfWork(default_difficulty=3)

        txs_a = [{"client_id": 1, "model_hash": "hash_a", "trust_score": 0.9}]
        txs_b = [{"client_id": 2, "model_hash": "hash_b", "trust_score": 0.8}]

        # Mine in separate threads to simulate race condition
        block_a = None
        block_b = None

        def mine_a():
            nonlocal block_a
            block_a = pox_a.mine_block(1, "2025-06-01T00:00:00+00:00", txs_a, "0" * 64)

        def mine_b():
            nonlocal block_b
            # Different previous hash to simulate different chain state
            block_b = pox_b.mine_block(1, "2025-06-01T00:00:01+00:00", txs_b, "deadbeef" * 8)

        t_a = threading.Thread(target=mine_a)
        t_b = threading.Thread(target=mine_b)
        t_a.start()
        t_b.start()
        t_a.join(timeout=5.0)
        t_b.join(timeout=5.0)

        self.assertIsNotNone(block_a, "Block A mining timed out")
        self.assertIsNotNone(block_b, "Block B mining timed out")
        # Both must be independently valid
        self.assertTrue(ProofOfWork.verify_mined_block(block_a))
        self.assertTrue(ProofOfWork.verify_mined_block(block_b))

    def test_fork_resolution_longest_chain_wins(self):
        """ChainValidator.resolve_fork must select the longest valid chain."""
        cv = ChainValidator(difficulty=3)
        pox = ProofOfWork(default_difficulty=3)

        # Build a short valid chain (2 blocks)
        prev_hash = "0" * 64
        short_chain = []
        for i in range(1, 3):
            b = pox.mine_block(i, f"2025-01-{i:02d}T00:00:00+00:00", [], prev_hash)
            short_chain.append(b)
            prev_hash = b["hash"]

        # Build a longer valid chain (5 blocks) from the same genesis
        prev_hash = "0" * 64
        long_chain = []
        for i in range(1, 6):
            b = pox.mine_block(i, f"2025-01-{i:02d}T00:00:00+00:00", [], prev_hash)
            long_chain.append(b)
            prev_hash = b["hash"]

        best, forked = cv.resolve_fork(short_chain, long_chain)
        self.assertTrue(forked)
        self.assertEqual(len(best), 5)
        self.assertIs(best, long_chain)

    def test_invalid_fork_not_adopted(self):
        """A shorter invalid chain must never replace a longer valid one."""
        cv = ChainValidator(difficulty=3)
        pox = ProofOfWork(default_difficulty=3)

        # Valid 4-block chain
        prev_hash = "0" * 64
        valid_chain = []
        for i in range(1, 5):
            b = pox.mine_block(i, f"2025-01-{i:02d}T00:00:00+00:00", [], prev_hash)
            valid_chain.append(b)
            prev_hash = b["hash"]

        # Invalid 3-block chain (tampered block 1)
        bad_b = pox.mine_block(1, "2025-01-01T00:00:00+00:00", [], "0" * 64)
        bad_b["hash"] = "tampered_invalid_hash_xyz"
        invalid_chain = [bad_b]

        best, forked = cv.resolve_fork(valid_chain, invalid_chain)
        self.assertFalse(forked)
        self.assertEqual(len(best), 4)


# ------------------------------------------------------------------
class TestNetworkDisconnectionRecovery(unittest.TestCase):
    """Test network disconnection and state recovery when a node re-joins."""

    def test_node_rejoin_and_sync(self):
        """A disconnected node that re-joins must sync to the latest chain height."""
        cv = ChainValidator(difficulty=3)
        pox = ProofOfWork(default_difficulty=3)

        # Build a 5-block chain on the "server" side
        prev_hash = "0" * 64
        server_chain: List[Dict[str, Any]] = []
        for i in range(1, 6):
            b = pox.mine_block(i, f"2025-01-{i:02d}T00:00:00+00:00", [], prev_hash)
            server_chain.append(b)
            prev_hash = b["hash"]

        # Client starts with empty chain (disconnected)
        client_chain: List[Dict[str, Any]] = []

        # Simulate reconnection: client receives all blocks via broadcast
        for block in server_chain:
            valid, errors = cv.validate_chain([block])
            if not valid and len(client_chain) > 0:
                # If standalone invalid due to prev_hash mismatch, add with correct link
                continue
            client_chain.append(block)

        # Verify convergence
        self.assertEqual(len(client_chain), len(server_chain))
        self.assertTrue(cv.validate_chain(client_chain)[0])

    def test_node_disconnects_mid_chain(self):
        """A node that disconnects after receiving 3 blocks of a 5-block chain
        must have its partial chain still valid."""
        cv = ChainValidator(difficulty=3)
        pox = ProofOfWork(default_difficulty=3)

        # Build full chain
        prev_hash = "0" * 64
        full_chain: List[Dict[str, Any]] = []
        for i in range(1, 6):
            b = pox.mine_block(i, f"2025-01-{i:02d}T00:00:00+00:00", [], prev_hash)
            full_chain.append(b)
            prev_hash = b["hash"]

        # Node disconnects after block 3
        partial_chain = full_chain[:3]

        valid, errors = cv.validate_chain(partial_chain)
        self.assertTrue(valid)
        self.assertEqual(len(errors), 0)
        # Partial chain height is 3
        max_height = max(b["index"] for b in partial_chain)
        self.assertEqual(max_height, 3)


# ------------------------------------------------------------------
class TestEndToEndConsensusNetwork(unittest.TestCase):
    """End-to-end integration test: server + clients over UDP with consensus."""

    @classmethod
    def setUpClass(cls):
        cls.server_port = 7150
        cls.listen_ports = [8150, 8160, 8170]

    def test_full_consensus_round(self):
        """Full round: server receives models, mines blocks, broadcasts to clients."""
        # Start server socket
        srv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        srv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv_sock.bind((SERVER_HOST, self.server_port))

        try:
            # Start client listener sockets
            listeners = []
            for lp in self.listen_ports:
                ls = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                ls.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                ls.bind((SERVER_HOST, lp))
                ls.settimeout(3.0)
                listeners.append(ls)

            try:
                # Generate and send models from 2 honest clients
                X_held, y_held = _generate_data(30, 4, seed=100)
                payload_a, model_a = _train_and_pickle(X_held, y_held, seed=1)
                acc_a = accuracy_score(y_held, model_a.predict(X_held))

                payload_b, model_b = _train_and_pickle(X_held, y_held, seed=2)
                acc_b = accuracy_score(y_held, model_b.predict(X_held))

                print(f"  [Test] Client A accuracy: {acc_a:.4f}, Client B accuracy: {acc_b:.4f}")

                # Send models to server
                _send_model_over_udp(srv_sock, 1, payload_a, SERVER_HOST, self.server_port)
                time.sleep(0.3)
                _send_model_over_udp(srv_sock, 2, payload_b, SERVER_HOST, self.server_port)

                # Simulate server mining a block after processing (consensus simulation)
                pox = ProofOfWork(default_difficulty=3)
                mined_block = pox.mine_block(
                    index=1, timestamp="2025-06-01T12:00:00+00:00",
                    transactions=[
                        {"client_id": 1, "model_hash": "hash_a", "trust_score": acc_a},
                        {"client_id": 2, "model_hash": "hash_b", "trust_score": acc_b},
                    ],
                    previous_hash="0" * 64, difficulty=3,
                )

                # Broadcast block to all client listeners
                for ls in listeners:
                    _send_block_broadcast(srv_sock, mined_block, SERVER_HOST, ls.getsockname()[1])

                time.sleep(0.5)

                # Verify each listener received the broadcast
                received_blocks = []
                for i, ls in enumerate(listeners):
                    try:
                        data, addr = ls.recvfrom(65535)
                        recv_index = struct.unpack(BLOCK_HEADER_FMT, data[:4])[0]
                        recv_body = json.loads(data[4:].decode("utf-8"))
                        received_blocks.append(recv_body)

                        # Verify block integrity using ChainValidator
                        valid, errors = cv.validate_chain([recv_body]) if (cv := ChainValidator(3)) else (False, [])
                        self.assertTrue(valid, f"Client {i+1} received invalid block: {errors}")
                    except socket.timeout:
                        self.fail(f"Client listener {i+1} did not receive broadcast within timeout")

                # Verify all honest clients converged on same block height
                heights = [b["index"] for b in received_blocks]
                self.assertEqual(len(set(heights)), 1, "Clients did not converge on the same block height")
                self.assertEqual(heights[0], 1)

            finally:
                for ls in listeners:
                    try:
                        ls.close()
                    except OSError:
                        pass

        finally:
            srv_sock.close()


if __name__ == "__main__":
    unittest.main()
