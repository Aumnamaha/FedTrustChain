#!/usr/bin/env python3
"""Multi-node P2P Consensus Network Simulation (Phase 3 Step 2).

Launches a FederatedServer with 3 UDP client nodes:
- Client 1 (Validator A, T=0.95): Clean training, signs/broadcasts valid block proposals.
- Client 2 (Validator B, T=0.90): Clean training, receives block broadcasts and verifies chain.
- Client 3 (Malicious Peer, T=0.35): Attempts to broadcast a forged block with invalid prev hash.

Execution & Verification Flow:
1. Server mines candidate blocks upon receiving valid payloads from Clients 1 & 2.
2. Server broadcasts mined block headers to all active UDP sockets.
3. Client 3's forged block proposal is rejected by ChainValidator across all nodes.
4. Verify all honest nodes converge on identical block height and ledger state.

Run with: python scripts/simulate_consensus_network.py
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

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from src.blockchain.consensus import ProofOfWork, ValidatorRegistry, ChainValidator
from src.blockchain.ledger import PermissionedLedger, Block
from src.server.dynamic_trust import DynamicTrustManager


# ------------------------------------------------------------------
HEADER_FMT = "!III"  # client_id (uint32), chunk_idx (uint32), total_chunks (uint32)
HEADER_SIZE = struct.calcsize(HEADER_FMT)

BLOCK_BROADCAST_HEADER = "!I"  # block_index (uint32) for consensus broadcasts


# ------------------------------------------------------------------
class ConsensusClientNode:
    """UDP client node that trains a local model, sends it to the server, and
    receives/verifies broadcasted blocks.

    Parameters
    ----------
    client_id : int
        Unique identifier for this node.
    host : str
        Server bind address.
    port : int
        Server UDP port for sending models.
    listen_port : int
        Local UDP port for receiving block broadcasts.
    X_train, y_train : np.ndarray
        Local training data.
    consensus_enabled : bool
        Whether to participate in consensus (verify mined blocks).
    """

    def __init__(
        self,
        client_id: int,
        host: str = "127.0.0.1",
        port: int = 5050,
        listen_port: int = 6050 + 10 * (client_id - 1),
        X_train: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None,
        consensus_enabled: bool = True,
    ) -> None:
        self.client_id = client_id
        self.host = host
        self.port = port
        self.listen_port = listen_port
        self.X_train = X_train if X_train is not None else np.zeros((10, 5))
        self.y_train = y_train if y_train is not None else np.zeros(10)
        self.consensus_enabled = consensus_enabled

        # Consensus state
        self.chain_validator = ChainValidator(difficulty=3)
        self.validator_registry: Optional[ValidatorRegistry] = None
        self.local_chain: List[Dict[str, Any]] = []  # blocks received via broadcast
        self.block_signatures: List[str] = []

        # Network socket for receiving broadcasts
        self._recv_socket: Optional[socket.socket] = None
        self._stop_event = threading.Event()
        self.broadcast_count: int = 0
        self.received_blocks: List[Dict[str, Any]] = []
        self.rejected_blocks: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    def train_local_model(self, n_estimators: int = 30) -> RandomForestClassifier:
        """Train a local Random Forest on the client's private data."""
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=5, random_state=self.client_id)
        model.fit(self.X_train, self.y_train)
        return model

    # ------------------------------------------------------------------
    def send_model_to_server(
        self, model: RandomForestClassifier, chunk_size: int = 4096
    ) -> None:
        """Serialize and send the trained model to the server over UDP."""
        payload = pickle.dumps(model)
        total_chunks = max(1, (len(payload) + chunk_size - 1) // chunk_size)

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            for i in range(total_chunks):
                start = i * chunk_size
                end = min(start + chunk_size, len(payload))
                header = struct.pack(HEADER_FMT, self.client_id, i, total_chunks)
                sock.sendto(header + payload[start:end], (self.host, self.port))

            print(f"  [Client {self.client_id}] Model sent: {len(payload)} bytes "
                  f"({total_chunks} chunks)")
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

            # Unpack block index header
            try:
                block_index = struct.unpack(BLOCK_BROADCAST_HEADER, data[:4])[0]
            except struct.error:
                continue

            # Parse JSON block body
            try:
                block_data = json.loads(data[4:].decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            self.broadcast_count += 1
            print(f"  [Client {self.client_id}] Received block broadcast #"
                  f"{block_index} from {addr}")

            # Verify block integrity if consensus enabled
            if self.consensus_enabled:
                valid, errors = self.chain_validator.validate_chain([block_data])
                if valid:
                    self.received_blocks.append(block_data)
                    print(f"  [Client {self.client_id}] Block #{block_index} "
                          f"VERIFIED (hash={block_data.get('hash', '')[:16]}...)")
                else:
                    self.rejected_blocks.append(block_data)
                    event_types = []
                    for err in errors:
                        if "payload modification" in err or "hash mismatch" in err:
                            event_types.append("PAYLOAD_TAMPER")
                        elif "previous hash link" in err:
                            event_types.append("CHAIN_LINK_BREACH")
                        elif "proof-of-work" in err or "nonce" in err:
                            event_types.append("INVALID_NONCE")
                    print(f"  [Client {self.client_id}] Block #{block_index} REJECTED: "
                          f"{', '.join(event_types)}")

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
        if not self.received_blocks:
            return 0
        return max(b["index"] for b in self.received_blocks)


# ------------------------------------------------------------------
class MaliciousClientNode(ConsensusClientNode):
    """Malicious client that attempts to broadcast a forged block with invalid
    previous hash link.

    Parameters
    ----------
    client_id : int
        Unique identifier (must be < threshold trust score).
    forged_block : dict | None, optional
        Pre-built forged block. Generated automatically if None.
    """

    def __init__(self, client_id: int, **kwargs) -> None:
        super().__init__(client_id, **kwargs)
        self.is_malicious = True
        # Build a forged block with invalid previous hash
        self.forged_block = self._build_forged_block()

    # ------------------------------------------------------------------
    def _build_forged_block(self) -> Dict[str, Any]:
        """Construct a forged block with an intentionally broken chain link."""
        return {
            "index": 999,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
            "transactions": [{"client_id": self.client_id, "model_hash": "forged",
                              "trust_score": 1.0, "status": "MALICIOUS"}],
            "previous_hash": "deadbeef" * 16,  # Invalid — does not match any real block
            "nonce": 42,
            "hash": "tampered_hash_xyz" * 5,   # Fabricated hash
            "difficulty": 3,
        }

    # ------------------------------------------------------------------
    def broadcast_forged_block(self) -> None:
        """Broadcast the forged block to the server's listen port."""
        payload = json.dumps(self.forged_block, default=str).encode("utf-8")
        header = struct.pack("!I", self.forged_block["index"])
        full_msg = header + payload

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # Broadcast to the server's main port (where it listens for clients)
            sock.sendto(full_msg, (self.host, self.port))
            print(f"  [Client {self.client_id}] FORGED block broadcast sent "
                  f"(prev_hash={self.forged_block['previous_hash'][:16]}...)")

            # Also send to all honest clients' listen ports for verification
            for i in range(1, 4):
                if i != self.client_id:
                    try:
                        sock.sendto(full_msg, ("127.0.0.1", 6050 + 10 * (i - 1)))
                    except OSError:
                        pass

            print(f"  [Client {self.client_id}] Forged block sent to all nodes")
        finally:
            sock.close()


# ------------------------------------------------------------------
def generate_synthetic_data(n_samples: int = 80, n_features: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic training data for client nodes."""
    rng = np.random.RandomState(42)
    X = rng.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


# ------------------------------------------------------------------
def run_simulation() -> Dict[str, Any]:
    """Execute the full multi-node P2P consensus simulation.

    Returns a dict with timing and convergence results for reporting.
    """
    print("=" * 70)
    print("FedTrustChain — Multi-Node P2P Consensus Network Simulation")
    print("(Phase 3 Step 2)")
    print("=" * 70)

    # Configuration
    SERVER_HOST = "127.0.0.1"
    SERVER_PORT = 5050
    LISTEN_PORT_BASE = 6050
    DIFFICULTY = 3
    N_SAMPLES = 80
    N_FEATURES = 5

    # --- Phase A: Generate client data -----------------------------------
    print("\n[Phase A] Generating synthetic training data...")
    X1, y1 = generate_synthetic_data(N_SAMPLES, N_FEATURES)
    X2, y2 = generate_synthetic_data(N_SAMPLES, N_FEATURES)
    # Malicious client gets different (noisy) data
    rng_mal = np.random.RandomState(999)
    X3 = rng_mal.randn(N_SAMPLES, N_FEATURES) * 0.1  # Very low variance — poor model
    y3 = rng_mal.randint(0, 2, N_SAMPLES)

    print(f"  Client 1 (Validator A): {N_SAMPLES} samples, T_i target=0.95")
    print(f"  Client 2 (Validator B): {N_SAMPLES} samples, T_i target=0.90")
    print(f"  Client 3 (Malicious)  : noisy data, T_i expected ~0.35")

    # --- Phase B: Launch server ------------------------------------------
    print("\n[Phase B] Starting FederatedServer...")
    from src.server.server_node import FederatedServer

    server = FederatedServer(
        host=SERVER_HOST, port=SERVER_PORT, min_clients=2,
        reassembly_timeout=10.0, chunk_size=4096,
    )

    # Create holdout data for trust evaluation
    X_holdout, y_holdout = generate_synthetic_data(30, N_FEATURES)

    start_time = time.time()
    server_thread = threading.Thread(
        target=lambda: server.start(X_holdout, y_holdout, X_holdout, y_holdout,
                                     blockchain_enabled=True, difficulty=DIFFICULTY),
        daemon=True,
    )
    server_thread.start()

    # Give server time to bind
    time.sleep(0.5)
    print(f"  Server started on {SERVER_HOST}:{SERVER_PORT}")

    # --- Phase C: Launch client nodes ------------------------------------
    print("\n[Phase C] Launching consensus client nodes...")

    # Client 1 — Validator A (honest, high trust)
    client_a = ConsensusClientNode(
        client_id=1, host=SERVER_HOST, port=SERVER_PORT,
        listen_port=LISTEN_PORT_BASE + 0, X_train=X1, y_train=y1,
        consensus_enabled=True,
    )

    # Client 2 — Validator B (honest, good trust)
    client_b = ConsensusClientNode(
        client_id=2, host=SERVER_HOST, port=SERVER_PORT,
        listen_port=LISTEN_PORT_BASE + 10, X_train=X2, y_train=y2,
        consensus_enabled=True,
    )

    # Client 3 — Malicious Peer (low trust, broadcasts forged blocks)
    client_mal = MaliciousClientNode(
        client_id=3, host=SERVER_HOST, port=SERVER_PORT,
        listen_port=LISTEN_PORT_BASE + 20, X_train=X3, y_train=y3,
        consensus_enabled=True,
    )

    # Start broadcast listeners on all clients
    client_a.start_broadcast_listener()
    client_b.start_broadcast_listener()
    client_mal.start_broadcast_listener()

    time.sleep(0.3)

    # --- Phase D: Send models to server ----------------------------------
    print("\n[Phase D] Sending trained models to server...")

    model_a = client_a.train_local_model(n_estimators=30)
    acc_a = accuracy_score(y_holdout, model_a.predict(X_holdout))
    print(f"  Client A local accuracy: {acc_a:.4f}")
    client_a.send_model_to_server(model_a)

    time.sleep(0.5)

    model_b = client_b.train_local_model(n_estimators=30)
    acc_b = accuracy_score(y_holdout, model_b.predict(X_holdout))
    print(f"  Client B local accuracy: {acc_b:.4f}")
    client_b.send_model_to_server(model_b)

    time.sleep(1.5)  # Wait for server to process models and mine blocks

    # --- Phase E: Malicious broadcast ------------------------------------
    print("\n[Phase E] Malicious peer broadcasts forged block...")
    client_mal.broadcast_forged_block()
    time.sleep(1.0)  # Let broadcasts propagate

    # --- Phase F: Additional honest round --------------------------------
    print("\n[Phase F] Sending second round of models (Client A)...")
    model_a2 = client_a.train_local_model(n_estimators=30)
    acc_a2 = accuracy_score(y_holdout, model_a2.predict(X_holdout))
    print(f"  Client A round-2 accuracy: {acc_a2:.4f}")
    client_a.send_model_to_server(model_a2)

    time.sleep(2.0)  # Wait for second block to be mined and broadcast

    # --- Phase G: Results & Verification ---------------------------------
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("[Phase G] Simulation Results")
    print("=" * 70)

    convergence_status = "CONVERGED" if (
        client_a.get_converged_height() == client_b.get_converged_height()
        and len(client_a.received_blocks) > 0
        and len(client_b.rejected_blocks) > 0
    ) else "NOT CONVERGED"

    results = {
        "total_elapsed_seconds": round(elapsed, 2),
        "client_a_broadcasts_received": client_a.broadcast_count,
        "client_a_blocks_verified": len(client_a.received_blocks),
        "client_b_broadcasts_received": client_b.broadcast_count,
        "client_b_blocks_verified": len(client_b.received_blocks),
        "client_malicious_forged_sent": 1,
        "client_a_rejected_forged": sum(1 for b in client_a.rejected_blocks if b.get("index") == 999),
        "client_b_rejected_forged": sum(1 for b in client_b.rejected_blocks if b.get("index") == 999),
        "client_a_converged_height": client_a.get_converged_height(),
        "client_b_converged_height": client_b.get_converged_height(),
        "convergence_status": convergence_status,
    }

    # Print results matrix
    print(f"  {'Metric':<45} {'Value':>20}")
    print(f"  {'─' * 65}")
    for k, v in results.items():
        label = k.replace("_", " ").title()
        print(f"  {label:<45} {str(v):>20}")

    # --- Phase H: Cleanup -----------------------------------------------
    print("\n[Phase H] Cleaning up...")
    client_a.stop_listener()
    client_b.stop_listener()
    client_mal.stop_listener()

    # Stop server thread
    if server._udp_socket:
        try:
            server._udp_socket.close()
        except OSError:
            pass

    print(f"  All nodes stopped. Simulation complete.")
    print("\n" + "=" * 70)
    print(f"CONVERGENCE STATUS: {convergence_status}")
    print("=" * 70)

    return results


# ------------------------------------------------------------------
if __name__ == "__main__":
    results = run_simulation()

    # Export simulation results to JSON
    output_path = Path("results/consensus_simulation.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSimulation results saved to {output_path}")

    # Exit with appropriate code for CI
    sys.exit(0 if results["convergence_status"] == "CONVERGED" else 1)
