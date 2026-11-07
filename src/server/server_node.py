"""Federated learning server that receives, aggregates, and evaluates models."""

from __future__ import annotations

import pickle
import socket
import struct
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.blockchain.consensus import ProofOfWork, ValidatorRegistry
from src.blockchain.ledger import PermissionedLedger
from src.models.rf_federated import aggregate_rf_models
from src.server.dynamic_trust import DynamicTrustManager, aggregate_random_forests
from src.utils.logger import get_logger


class FederatedServer:
    """Listen for client models over UDP, aggregate them into a global model, and evaluate.

    Uses **chunked UDP datagram** transport with custom application-layer
    framing (12-byte binary header).  Incomplete streams time out after a
    configurable interval; dropped packets reduce the sender's trust score.

    Parameters
    ----------
    host : str
        Bind address (e.g. ``"127.0.0.1"``, ``"0.0.0.0"``).
    port : int
        Port to listen on for incoming UDP datagrams.
    min_clients : int
        Minimum number of client models required before aggregation triggers.
    reassembly_timeout : float, default=5.0
        Seconds to wait for missing chunks from a single client stream.
    chunk_size : int, default=8192
        Expected payload bytes per datagram (after 12-byte header).
        Set to 8192 to stay under typical MTU and avoid OS-level IP fragmentation.
    trust_manager : DynamicTrustManager or None, optional
        Pre-configured trust manager for evaluating client updates before
        aggregation. If provided, models are scored, hashed, and filtered by
        reputation before tree union pooling.
    blockchain_enabled : bool, default=False
        When True, creates a PermissionedLedger to immutably log every client
        evaluation event (ACCEPTED / REJECTED) with SHA-256 payload hashes,
        trust scores, and aggregation decisions.
    audit_log_path : str | Path, optional
        File path for JSON audit log export. Only used when *blockchain_enabled*
        is True. Defaults to ``results/blockchain_audit_log.json``.
    """

    # ------------------------------------------------------------------
    HEADER_FMT = "!III"          # big-endian unsigned int × 3 → 12 bytes
    HEADER_SIZE = struct.calcsize(HEADER_FMT)  # always 12

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5050,
        min_clients: int = 2,
        reassembly_timeout: float = 5.0,
        chunk_size: int = 8192,
    ):
        self.host = host
        self.port = port
        self.min_clients = min_clients
        self.reassembly_timeout = reassembly_timeout
        if chunk_size < 512:
            raise ValueError("chunk_size must be >= 512 bytes")
        self.chunk_size = chunk_size
        self.trust_manager: Optional[DynamicTrustManager] = None
        self.blockchain_enabled: bool = False
        self.audit_log_path: Optional[str | Path] = None
        self.ledger: Optional[PermissionedLedger] = None
        # --- Phase 3 Step 1: Consensus Engine integration -----------------
        self.consensus_enabled: bool = False
        self.proof_of_work: Optional[ProofOfWork] = None
        self.validator_registry: Optional[ValidatorRegistry] = None

        self.logger = get_logger("FederatedServer")
        self.client_models: List[RandomForestClassifier] = []
        self._trust_scores: Dict[int, float] = defaultdict(lambda: 1.0)
        self._udp_socket: Optional[socket.socket] = None

    # ------------------------------------------------------------------
    def start(
        self,
        X_test: np.ndarray,
        y_test,
        X_val: Optional[np.ndarray] = None,
        y_val=None,
        blockchain_enabled: bool = False,
        audit_log_path: Optional[str | Path] = None,
        difficulty: int = 2,
    ) -> Tuple[RandomForestClassifier, Dict[str, float]]:
        """Start the server, receive client models, aggregate, and evaluate.

        Parameters
        ----------
        X_test : np.ndarray
            Test features for global model evaluation.
        y_test : array-like
            True test labels.

        Returns
        -------
        global_model : RandomForestClassifier
            The aggregated global forest.
        metrics : dict[str, float]
            Computed evaluation metrics (accuracy, f1_score, precision, recall).
        """
        self.logger.info(
            "Starting UDP server on %s:%d (min_clients=%d, timeout=%.1fs)",
            self.host, self.port, self.min_clients, self.reassembly_timeout,
        )

        # --- bind UDP socket ----------------------------------------------
        self._udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Increase receive buffer to handle burst UDP datagrams from clients
        self._udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        self._udp_socket.bind((self.host, self.port))

        try:
            # --- initialize trust manager if holdout data provided --------
            if X_val is not None and y_val is not None:
                self.trust_manager = DynamicTrustManager(
                    holdout_X=X_val,
                    holdout_y=y_val,
                    min_trust_threshold=getattr(self, "_trust_threshold", 0.50),
                    reward_rate=getattr(self, "_reward_rate", 0.05),
                    decay_rate=getattr(self, "_decay_rate", 0.25),
                )

            # --- initialize blockchain ledger if enabled ------------------
            self.blockchain_enabled = bool(blockchain_enabled)
            self.audit_log_path = audit_log_path or getattr(
                self, "_audit_log_path", "results/blockchain_audit_log.json"
            )
            if self.blockchain_enabled:
                from pathlib import Path
                self.ledger = PermissionedLedger(
                    difficulty=difficulty,
                    audit_log_path=self.audit_log_path,
                )
                self.logger.info(
                    "Blockchain ledger initialized (path=%s, difficulty=%d)",
                    self.audit_log_path, difficulty,
                )

            # --- initialize consensus engine if enabled -------------------
            if blockchain_enabled:
                self.consensus_enabled = True
                self.proof_of_work = ProofOfWork(default_difficulty=difficulty)
                self.validator_registry = ValidatorRegistry(
                    min_trust_threshold=self.trust_manager.min_trust_threshold
                    if self.trust_manager else 0.50,
                )
                self.logger.info(
                    "Consensus engine initialized (PoW difficulty=%d)",
                    difficulty,
                )

            # --- receive models (one client at a time) ----------------------
            received_ids: List[int] = []
            for i in range(self.min_clients):
                model = self._receive_model(client_index=i + 1)
                cid = i + 1  # client_id maps to loop index
                received_ids.append(cid)

                # --- trust evaluation (if manager configured) ---------------
                if self.trust_manager is not None:
                    accepted, accuracy, payload_hash = (
                        self.trust_manager.evaluate_update(cid, model)
                    )
                    self.logger.info(
                        "Client %d: hash=%s accuracy=%.4f accepted=%s",
                        cid, payload_hash[:16], accuracy, accepted,
                    )
                    # --- record to blockchain ledger ------------------------
                    if self.ledger is not None:
                        status = "ACCEPTED" if accepted else "REJECTED"
                        current_trust = (
                            self.trust_manager.node_registry[cid].trust_score
                            if self.trust_manager is not None
                            else 1.0
                        )
                        self.ledger.record_transaction(
                            client_id=cid,
                            model_hash=payload_hash,
                            trust_score=current_trust,
                            status=status,
                        )
                        # --- Phase 3 Step 1: Consensus block mining ---------------
                        if (self.consensus_enabled and accepted
                                and self.proof_of_work is not None
                                and self.validator_registry is not None):
                            # Verify proposer authorization via ValidatorRegistry
                            if self.validator_registry.verify_proposer(cid):
                                self.validator_registry.sign_block(
                                    cid, payload_hash[:16] + "_block"
                                )
                                # Append transaction to consensus mining pool
                                consensus_tx = {
                                    "client_id": cid,
                                    "model_hash": payload_hash,
                                    "trust_score": round(current_trust, 6),
                                    "status": status,
                                    "timestamp": self.ledger._now_iso(),
                                }
                                # Mine a consensus block (single-transaction blocks per round)
                                prev_block = self.ledger.chain[-1]
                                mined = self.proof_of_work.mine_block(
                                    index=prev_block.index + 1,
                                    timestamp=self.ledger._now_iso(),
                                    transactions=[consensus_tx],
                                    previous_hash=prev_block._hash or prev_block.calculate_hash(),
                                    difficulty=difficulty,
                                )
                                # Broadcast mined block header (logged)
                                self.logger.info(
                                    "Consensus block #%d mined — hash=%s nonce=%d "
                                    "(diff=%d, proposer=node%d)",
                                    mined["index"],
                                    mined["hash"][:16],
                                    mined["nonce"],
                                    mined["difficulty"],
                                    cid,
                                )
                            else:
                                self.logger.warning(
                                    "Consensus: node %d not authorized to propose block",
                                    cid,
                                )
                    if not accepted:
                        # Still store but mark as rejected — filter later
                        pass
                self.client_models.append(model)

            # --- trust-filtered aggregation ---------------------------------
            if self.trust_manager is not None:
                trusted_ids, trusted_models = (
                    self.trust_manager.filter_trusted_clients(
                        received_ids, self.client_models
                    )
                )
                rejected_count = len(received_ids) - len(trusted_ids)
                if rejected_count > 0:
                    self.logger.warning(
                        "Trust filter: %d/%d clients isolated",
                        rejected_count, len(received_ids),
                    )
                    # Use filtered models for aggregation
                    aggregate_models = trusted_models
                    final_ids = trusted_ids
                else:
                    aggregate_models = self.client_models
                    final_ids = received_ids
            else:
                aggregate_models = self.client_models
                final_ids = received_ids

            self.logger.info("Aggregating %d client models ...",
                             len(aggregate_models))
            global_model = aggregate_random_forests(aggregate_models)
            self.logger.info("Global model: %d total trees",
                             global_model.n_estimators)

            # --- mine blockchain block and export audit log ---------------
            if self.ledger is not None:
                mined_block = self.ledger.mine_pending_block()
                audit_path = self.ledger.export_audit_log_json()
                summary = self.ledger.get_chain_summary()
                self.logger.info(
                    "Blockchain: block #%d mined (nonce=%d, txns=%d) → %s",
                    mined_block.index, mined_block.nonce,
                    len(mined_block.transactions), audit_path,
                )
                self.logger.info("Chain validation: %s", summary["is_valid"])

            # --- evaluate ---------------------------------------------------
            y_pred = global_model.predict(X_test)
            metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "f1_score": float(f1_score(y_test, y_pred, average="weighted")),
                "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
            }

            self.logger.info("Global model evaluation: %s", metrics)
            return global_model, metrics

        finally:
            if self._udp_socket is not None:
                self._udp_socket.close()
            self.logger.info("Server UDP socket closed")

    # ------------------------------------------------------------------
    def _receive_model(
        self, client_index: int = 0
    ) -> RandomForestClassifier:
        """Receive a pickled model from a single client over chunked UDP.

        Maintains an in-memory reassembly buffer keyed by ``(client_id,
        chunk_index)``.  Once ``total_chunks`` chunks arrive for the same
        client_id the payload is reassembled and deserialized.

        Parameters
        ----------
        client_index : int
            Index of the client for logging purposes.

        Returns
        -------
        RandomForestClassifier
            The deserialized model.
        """
        self.logger.info("Waiting for UDP datagrams from client %d ...",
                         client_index)

        # Fragment buffer: {client_id: {chunk_idx: bytearray}}
        frag_buf: Dict[int, Dict[int, bytearray]] = defaultdict(dict)
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time
            if self.reassembly_timeout > 0 and elapsed >= self.reassembly_timeout:
                raise TimeoutError(
                    f"Client {client_index} reassembly timed out after "
                    f"{elapsed:.1f}s — buffer incomplete."
                )

            # Block with timeout so we can check the deadline.
            self._udp_socket.settimeout(max(0.5, self.reassembly_timeout - elapsed))
            try:
                data, _addr = self._udp_socket.recvfrom(65535)
            except socket.timeout:
                continue

            if len(data) < self.HEADER_SIZE:
                # Discard undersized datagrams (malformed or truncated).
                continue

            # --- unpack header --------------------------------------------
            client_id, chunk_idx, total_chunks = struct.unpack(
                self.HEADER_FMT, data[:self.HEADER_SIZE]
            )

            payload_frag = data[self.HEADER_SIZE:]

            if client_id not in frag_buf:
                frag_buf[client_id] = {}

            # Append payload data for this chunk.
            buf_slot = frag_buf[client_id].setdefault(chunk_idx, bytearray())
            buf_slot.extend(payload_frag)

            self.logger.debug(
                "Client %d [%s]: chunk %d/%d — got %d bytes total",
                client_index, client_id, chunk_idx + 1, total_chunks,
                len(buf_slot),
            )

            # --- check reassembly complete --------------------------------
            # Verify that we have received ALL chunk indices.
            known_indices = set(frag_buf[client_id].keys())
            if len(known_indices) < total_chunks:
                continue

            # Verify every chunk has the expected size before reassembling.
            # Non-final chunks must be exactly self.chunk_size bytes;
            # the last chunk just needs to be non-empty (≥ 1 byte).
            all_done = True
            for ci in range(total_chunks):
                got = len(frag_buf[client_id].get(ci, b""))
                if ci < total_chunks - 1:
                    exp = self.chunk_size
                else:
                    exp = 1  # last chunk: at least one byte
                if got < exp:
                    all_done = False
                    break

            if not all_done:
                continue

            # --- reassemble -----------------------------------------------
            chunks = [bytes(frag_buf[client_id][i]) for i in range(total_chunks)]
            payload = b"".join(chunks)

            model = pickle.loads(payload)

            self.logger.info(
                "Client %d [%s]: reassembled %d bytes → %s (%d chunks)",
                client_index, client_id, len(payload),
                type(model).__name__, total_chunks,
            )

            return model

    @property
    def trust_scores(self) -> Dict[int, float]:
        """Current per-client trust scores (1.0 = fully trusted)."""
        return dict(self._trust_scores)
