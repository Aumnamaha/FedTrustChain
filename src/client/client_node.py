"""Federated learning client that trains locally and sends models over TCP."""

from __future__ import annotations

import pickle
import socket
import time
from typing import Any, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.utils.logger import get_logger


class FederatedClient:
    """Train a local Random Forest and transmit it to the federated server.

    Uses length-prefixed TCP framing (4-byte big-endian unsigned int prefix)
    to avoid pickle/JSON payload fragmentation issues on stream sockets.

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
    """

    def __init__(
        self,
        client_id: int,
        server_host: str = "127.0.0.1",
        server_port: int = 5050,
        n_estimators: int = 50,
        max_depth: int = 15,
        random_state: int = 42,
    ):
        self.client_id = client_id
        self.server_host = server_host
        self.server_port = server_port
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

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

        # --- send over TCP with length prefix -----------------------------
        self.logger.info("Connecting to server at %s:%d ...",
                         self.server_host, self.server_port)

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect((self.server_host, self.server_port))
            # 4-byte big-endian length prefix
            sock.sendall(payload_len.to_bytes(4, byteorder="big"))
            # Full payload
            sent = 0
            while sent < len(payload):
                chunk = min(4096, len(payload) - sent)
                sock.sendall(payload[sent: sent + chunk])
                sent += chunk

            self.logger.info("Model sent successfully (%d/%d bytes)",
                             sent, payload_len)
        finally:
            sock.close()

        self.local_model = model
        return model
