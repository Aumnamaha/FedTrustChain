"""Federated learning client that trains locally and sends models over UDP."""

from __future__ import annotations

import pickle
import socket
import struct
import time
from typing import Any, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.utils.logger import get_logger


class FederatedClient:
    """Train a local Random Forest and transmit it to the federated server.

    Uses **chunked UDP datagram** transport with custom application-layer
    framing (12-byte binary header) for low-latency wireless edge networking.

    Parameters
    ----------
    client_id : int
        Unique identifier for this client node.
    server_host : str
        Hostname or IP of the federated server.
    server_port : int
        Port number the server is listening on.
    n_estimators : int, default=50
        Number of decision trees per local forest.
    max_depth : int, default=15
        Maximum depth of each decision tree.
    random_state : int, default=42
        Random seed for reproducibility.
    chunk_size : int, default=8192
        Size of each UDP payload chunk (excluding 12-byte header).
        Larger chunks reduce per-packet overhead for wireless edge networks.
    """

    # ------------------------------------------------------------------
    # Header format: !III = big-endian unsigned int × 3 → 12 bytes
    #   client_id | chunk_index | total_chunks
    HEADER_FMT = "!III"          # struct packing format string
    HEADER_SIZE = struct.calcsize(HEADER_FMT)  # always 12

    def __init__(
        self,
        client_id: int,
        server_host: str = "127.0.0.1",
        server_port: int = 5050,
        n_estimators: int = 50,
        max_depth: int = 15,
        random_state: int = 42,
        chunk_size: int = 8192,
    ):
        self.client_id = client_id
        self.server_host = server_host
        self.server_port = server_port
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        if chunk_size < 512:
            raise ValueError("chunk_size must be >= 512 bytes")
        self.chunk_size = chunk_size

        self.logger = get_logger(f"FederatedClient-{client_id}")
        self.local_model: Optional[RandomForestClassifier] = None

    # ------------------------------------------------------------------
    def train_and_send(self, X_local: np.ndarray, y_local) -> RandomForestClassifier:
        """Train a local Random Forest and send it to the server.

        Parameters
        ----------
        X_local : np.ndarray
            Local training features (n_samples, n_features).
        y_local : array-like
            Local training labels.

        Returns
        -------
        RandomForestClassifier
            The fitted local model.
        """
        self.logger.info("Starting local training (%d samples, %d features)",
                         X_local.shape[0], X_local.shape[1])

        # --- train --------------------------------------------------------
        start = time.time()
        model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        model.fit(X_local, y_local)
        elapsed = time.time() - start

        self.logger.info("Local training complete (%.3fs, %d trees)",
                         elapsed, self.n_estimators)

        # --- serialize ----------------------------------------------------
        payload = pickle.dumps(model)
        payload_len = len(payload)
        self.logger.info("Serialized model: %d bytes", payload_len)

        # --- send over UDP (chunked) --------------------------------------
        total_packets = self.send_udp_model(
            payload,
            chunk_size=self.chunk_size,
        )

        self.logger.info("Model sent successfully (%d/%d bytes, %d UDP packets)",
                         payload_len, payload_len, total_packets)

        self.local_model = model
        return model

    # ------------------------------------------------------------------
    def send_udp_model(self, payload_bytes: bytes, chunk_size: int = 8192) -> int:
        """Partition *payload_bytes* into UDP datagrams with sequencing headers.

        Each datagram carries a 12-byte header::

            [client_id (4B)][chunk_index (4B)][total_chunks (4B)]

        followed by up to ``chunk_size`` bytes of the serialized model payload.

        Parameters
        ----------
        payload_bytes : bytes
            The full pickled model payload.
        chunk_size : int, default=8192
            Maximum number of payload bytes per datagram (after header).
            Set to 8192 to stay under typical MTU and avoid IP fragmentation.

        Returns
        -------
        int
            Number of UDP packets transmitted.
        """
        total_chunks = max(1, (len(payload_bytes) + chunk_size - 1) // chunk_size)
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Increase send buffer so rapid sends don't drop on localhost
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
        try:
            for idx in range(total_chunks):
                start_byte = idx * chunk_size
                end_byte = min(start_byte + chunk_size, len(payload_bytes))
                chunk_data = payload_bytes[start_byte:end_byte]

                header = struct.pack(
                    self.HEADER_FMT,
                    self.client_id,
                    idx,
                    total_chunks,
                )
                packet = header + chunk_data
                n_sent = sock.sendto(packet, (self.server_host, self.server_port))
                if n_sent != len(packet):
                    self.logger.warning(
                        "Chunk %d/%d: sendto returned %d < expected %d",
                        idx + 1, total_chunks, n_sent, len(packet),
                    )

            return total_chunks
        finally:
            sock.close()
