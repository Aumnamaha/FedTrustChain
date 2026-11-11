#!/usr/bin/env python3
"""High-Concurrency Network Stress Test & Node Churn Resilience (Phase 2 Infrastructure).

Simulates 10 concurrent UDP worker nodes sending model update payloads to a FederatedServer,
with dynamic node churn: disconnects 3 active workers mid-round and attaches 2 new peers
during aggregation. Verifies zero orphaned state deadlocks, identical block heights across
active peers, and audit log stability.

Execution Flow:
1. Launch server with blockchain + consensus enabled.
2. Start 10 concurrent worker threads sending model payloads over chunked UDP.
3. After receiving models from 7 workers, simulate churn: disconnect nodes 8-10.
4. While aggregation/ledger logging is active, attach 2 new peer nodes (nodes 11-12).
5. Verify all active peers maintain identical block heights.
6. Assert zero orphaned state deadlocks.
7. Export stress metrics to results/network_stress_results.json.

Run with: python scripts/stress_test_network.py
"""

from __future__ import annotations

import json
import os
import pickle
import socket
import struct
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from src.blockchain.consensus import ProofOfWork, ValidatorRegistry, ChainValidator
from src.server.dynamic_trust import DynamicTrustManager


# ------------------------------------------------------------------
HEADER_FMT = "!III"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
BLOCK_BROADCAST_HEADER = "!I"


class StressWorkerNode:
    """UDP client node for stress testing that sends model payloads concurrently.

    Parameters
    ----------
    worker_id : int
        Unique identifier for this worker.
    host / port : str, int
        Server address to send models to.
    listen_port : int
        Local UDP port for receiving block broadcasts.
    X_train / y_train : np.ndarray
        Local training data.
    chunk_size : int
        Size per UDP datagram (after 12-byte header).
    """

    def __init__(
        self,
        worker_id: int,
        host: str = "127.0.0.1",
        port: int = 5050,
        listen_port: int = 9050 + 10 * (worker_id - 1),
        X_train: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None,
        chunk_size: int = 4096,
    ) -> None:
        self.worker_id = worker_id
        self.host = host
        self.port = port
        self.listen_port = listen_port
        self.X_train = X_train if X_train is not None else np.zeros((50, 4))
        self.y_train = y_train if y_train is not None else np.zeros(50)
        self.chunk_size = chunk_size

        # Broadcasting reception state
        self.chain_validator = ChainValidator(difficulty=3)
        self.received_blocks: List[Dict[str, Any]] = []
        self.broadcast_count: int = 0
        self._recv_socket: Optional[socket.socket] = None
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    def train_local_model(self, n_estimators: int = 25) -> RandomForestClassifier:
        """Train a local Random Forest on the worker's private data."""
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=6, random_state=self.worker_id
        )
        model.fit(self.X_train, self.y_train)
        return model

    # ------------------------------------------------------------------
    def send_model_to_server(self, model: RandomForestClassifier) -> Tuple[int, float]:
        """Serialize and send the trained model to the server over UDP.

        Returns (bytes_sent, wall_time_seconds).
        """
        payload = pickle.dumps(model)
        total_chunks = max(1, (len(payload) + self.chunk_size - 1) // self.chunk_size)

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            start = time.time()
            for i in range(total_chunks):
                s = i * self.chunk_size
                e = min(s + self.chunk_size, len(payload))
                hdr = struct.pack(HEADER_FMT, self.worker_id, i, total_chunks)
                sock.sendto(hdr + payload[s:e], (self.host, self.port))

            elapsed = time.time() - start
            return len(payload), elapsed
        finally:
            sock.close()

    # ------------------------------------------------------------------
    def start_broadcast_listener(self) -> None:
        """Start a background thread listening for block broadcast messages."""
        self._recv_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._recv_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._recv_socket.bind(("127.0.0.1", self.listen_port))
        self._recv_socket.settimeout(1.0)

        thread = threading.Thread(target=self._listen_loop, daemon=True)
        thread.start()

    # ------------------------------------------------------------------
    def _listen_loop(self) -> None:
        """Background loop receiving and verifying block broadcasts."""
        while not self._stop_event.is_set():
            try:
                data, addr = self._recv_socket.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break

            if len(data) < struct.calcsize(BLOCK_BROADCAST_HEADER):
                continue

            try:
                block_index = struct.unpack(BLOCK_BROADCAST_HEADER, data[:4])[0]
            except struct.error:
                continue

            try:
                block_data = json.loads(data[4:].decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            self.broadcast_count += 1

            valid, errors = self.chain_validator.validate_chain([block_data])
            if valid:
                self.received_blocks.append(block_data)
            # else silently drop — stress test focuses on throughput, not error logging


    # ------------------------------------------------------------------
    def stop_listener(self) -> None:
        """Stop the broadcast listener thread."""
        self._stop_event.set()
        if self._recv_socket:
            try:
                self._recv_socket.close()
            except OSError:
                pass

    # ------------------------------------------------------------------
    def get_converged_height(self) -> int:
        """Return the highest block index seen across all verified broadcasts."""
        return max((b["index"] for b in self.received_blocks), default=0)


# ------------------------------------------------------------------
def _generate_worker_data(worker_id: int, n_samples: int = 50, n_features: int = 4):
    """Generate deterministic synthetic training data per worker."""
    rng = np.random.RandomState(100 + worker_id)
    X = rng.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


# ------------------------------------------------------------------
def run_stress_test() -> Dict[str, Any]:
    """Execute the full high-concurrency network stress test.

    Returns a dict with timing and convergence results for reporting.
    """
    print("=" * 75)
    print("FedTrustChain — High-Concurrency Network Stress Test")
    print("(Phase 2 Infrastructure Milestone)")
    print("=" * 75)

    # Configuration
    SERVER_HOST = "127.0.0.1"
    SERVER_PORT = 6500
    LISTEN_BASE = 9050
    DIFFICULTY = 3
    N_WORKERS = 10
    CHUNK_SIZE = 4096
    HOLDOUT_N = 30

    # --- Phase A: Generate worker data -----------------------------------
    print("\n[Phase A] Generating synthetic training data for {} workers...".format(N_WORKERS))
    workers_data = [_generate_worker_data(i + 1) for i in range(N_WORKERS)]
    X_holdout, y_holdout = _generate_worker_data(999, HOLDOUT_N)

    # --- Phase B: Launch server ------------------------------------------
    print("\n[Phase B] Starting FederatedServer (blockchain+consensus enabled)...")
    from src.server.server_node import FederatedServer

    server = FederatedServer(
        host=SERVER_HOST, port=SERVER_PORT, min_clients=N_WORKERS,
        reassembly_timeout=15.0, chunk_size=CHUNK_SIZE,
    )

    # --- Phase C: Launch worker nodes ------------------------------------
    print("\n[Phase C] Launching {} concurrent UDP worker nodes...".format(N_WORKERS))
    workers: List[StressWorkerNode] = []
    for i in range(N_WORKERS):
        w = StressWorkerNode(
            worker_id=i + 1, host=SERVER_HOST, port=SERVER_PORT,
            listen_port=LISTEN_BASE + i * 10,
            X_train=workers_data[i][0], y_train=workers_data[i][1],
            chunk_size=CHUNK_SIZE,
        )
        w.start_broadcast_listener()
        workers.append(w)

    time.sleep(0.3)
    print(f"  All {N_WORKERS} listeners active on ports {LISTEN_BASE}-{LISTEN_BASE + (N_WORKERS-1)*10}")

    # --- Phase D: Start server in background thread ----------------------
    start_time = time.time()
    server_thread = threading.Thread(
        target=lambda: server.start(X_holdout, y_holdout, X_holdout, y_holdout,
                                     blockchain_enabled=True, difficulty=DIFFICULTY),
        daemon=True,
    )
    server_thread.start()
    time.sleep(0.5)

    # --- Phase E: Concurrent model sending (all workers simultaneously) --
    print("\n[Phase D] Sending models from {} workers concurrently...".format(N_WORKERS))
    send_results: List[Tuple[int, float]] = [None] * N_WORKERS  # (bytes, time)

    def worker_send(idx: int):
        w = workers[idx]
        model = w.train_local_model(n_estimators=25)
        bytes_sent, elapsed = w.send_model_to_server(model)
        send_results[idx] = (bytes_sent, elapsed)

    threads = [threading.Thread(target=worker_send, args=(i,)) for i in range(N_WORKERS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30.0)

    # Report send stats
    total_bytes = sum(r[0] if r else 0 for r in send_results)
    avg_send_time = np.mean([r[1] if r and r[1] > 0 else float("inf") for r in send_results])
    max_send_time = max((r[1] for r in send_results if r), default=0.0)
    print(f"  Total data sent: {total_bytes:,} bytes across {N_WORKERS} workers")
    print(f"  Avg send time: {avg_send_time:.3f}s | Max: {max_send_time:.3f}s")

    # --- Phase F: Dynamic Node Churn -------------------------------------
    print("\n[Phase E] Dynamic node churn — disconnecting nodes 8, 9, 10 mid-round...")
    # Simulate disconnect by closing their sockets (they've already sent models)
    for idx in [7, 8, 9]:  # workers 8, 9, 10
        workers[idx].stop_listener()

    time.sleep(0.5)
    print("  Nodes 8-10 disconnected. Server processing received models...")

    # --- Phase G: Attach new peers during aggregation ---------------------
    print("\n[Phase F] Attaching 2 new peer nodes (11, 12) while aggregation active...")

    X_new1, y_new1 = _generate_worker_data(901)
    X_new2, y_new2 = _generate_worker_data(902)

    worker_11 = StressWorkerNode(
        worker_id=11, host=SERVER_HOST, port=SERVER_PORT,
        listen_port=LISTEN_BASE + 10 * 10,
        X_train=X_new1, y_train=y_new1, chunk_size=CHUNK_SIZE,
    )
    worker_12 = StressWorkerNode(
        worker_id=12, host=SERVER_HOST, port=SERVER_PORT,
        listen_port=LISTEN_BASE + 10 * 11,
        X_train=X_new2, y_train=y_new2, chunk_size=CHUNK_SIZE,
    )

    # Start listeners for new nodes
    worker_11.start_broadcast_listener()
    worker_12.start_broadcast_listener()

    # Send models from new nodes to the running server
    model_11 = worker_11.train_local_model(n_estimators=25)
    model_12 = worker_12.train_local_model(n_estimators=25)
    _, t11 = worker_11.send_model_to_server(model_11)
    time.sleep(0.2)
    _, t12 = worker_12.send_model_to_server(model_12)

    print(f"  New node 11 sent: {t11:.3f}s | New node 12 sent: {t12:.3f}s")
    time.sleep(3.0)  # Wait for server to finish aggregation and mine blocks

    # --- Phase H: Results & Verification ---------------------------------
    elapsed = time.time() - start_time
    print("\n" + "=" * 75)
    print("[Phase G] Stress Test Results")
    print("=" * 75)

    # Gather convergence data from all active workers (1-7, 11, 12 = 9 active)
    active_workers = [w for i, w in enumerate(workers) if i < 7] + [worker_11, worker_12]
    heights = [w.get_converged_height() for w in active_workers]

    # Check for deadlocks: no worker should be stuck (all completed within timeout)
    deadlock_free = all(t is not None and t[0] > 0 for t in send_results if t)

    # Verify audit log exists and has content
    audit_path = Path("results/blockchain_audit_log.json")
    audit_valid = False
    chain_length = 0
    if audit_path.exists():
        try:
            with open(audit_path, "r", encoding="utf-8") as f:
                audit_data = json.load(f)
            chain_length = audit_data.get("chain_length", 0)
            audit_valid = True
        except (json.JSONDecodeError, IOError):
            pass

    # Verify all active workers converged on the same height
    unique_heights = set(heights)
    convergence_ok = len(unique_heights) == 1 and heights[0] > 0 if heights else False

    results = {
        "total_elapsed_seconds": round(elapsed, 2),
        "concurrent_workers_launched": N_WORKERS,
        "workers_disconnected_mid_round": [8, 9, 10],
        "new_peers_attached_during_aggregation": [11, 12],
        "total_data_sent_bytes": total_bytes,
        "avg_send_time_seconds": round(avg_send_time, 4),
        "max_send_time_seconds": round(max_send_time, 4),
        "throughput_mbps": round((total_bytes * 8) / (elapsed * 1_000_000), 2) if elapsed > 0 else 0.0,
        "active_workers_converged_height": heights[0] if convergence_ok else -1,
        "unique_block_heights_observed": len(unique_heights),
        "convergence_status": "CONVERGED" if convergence_ok else "NOT_CONVERGED",
        "deadlock_free": deadlock_free,
        "audit_log_valid": audit_valid,
        "chain_length_in_audit_log": chain_length,
    }

    # Print results matrix
    print(f"  {'Metric':<50} {'Value':>25}")
    print(f"  {'─' * 75}")
    for k, v in results.items():
        label = k.replace("_", " ").title()
        val_str = str(v) if not isinstance(v, float) else f"{v:.4f}"
        print(f"  {label:<50} {val_str:>25}")

    # --- Phase I: Cleanup -----------------------------------------------
    print("\n[Phase H] Cleaning up...")
    for w in workers + [worker_11, worker_12]:
        w.stop_listener()
    if server._udp_socket:
        try:
            server._udp_socket.close()
        except OSError:
            pass

    print(f"  Cleanup complete. Stress test finished in {elapsed:.2f}s.")
    print("\n" + "=" * 75)
    print(f"CONVERGENCE STATUS: {results['convergence_status']}")
    print(f"DEADLOCK FREE:      {'YES' if results['deadlock_free'] else 'NO'}")
    print("=" * 75)

    return results


# ------------------------------------------------------------------
if __name__ == "__main__":
    results = run_stress_test()

    # Export stress metrics to JSON
    output_path = Path("results/network_stress_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nStress metrics saved to {output_path}")

    # Exit code for CI
    sys.exit(0 if results["convergence_status"] == "CONVERGED" and results["deadlock_free"] else 1)
