"""Federated learning server that receives, aggregates, and evaluates models."""

from __future__ import annotations

import pickle
import socket
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.models.rf_federated import aggregate_rf_models
from src.utils.logger import get_logger


class FederatedServer:
    """Listen for client models, aggregate them into a global model, and evaluate.

    Parameters
    ----------
    host : str
        Bind address (e.g. ``"127.0.0.1"``, ``"0.0.0.0"``).
    port : int
        Port to listen on for incoming client connections.
    min_clients : int
        Minimum number of client models required before aggregation triggers.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 5050,
                 min_clients: int = 2):
        self.host = host
        self.port = port
        self.min_clients = min_clients

        self.logger = get_logger("FederatedServer")
        self.client_models: List[RandomForestClassifier] = []
        self._server_socket: Optional[socket.socket] = None

    # ------------------------------------------------------------------
    def start(self, X_test: np.ndarray, y_test) -> Tuple[RandomForestClassifier, Dict[str, float]]:
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
        self.logger.info("Starting server on %s:%d (min_clients=%d)",
                         self.host, self.port, self.min_clients)

        # --- bind and listen ----------------------------------------------
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind((self.host, self.port))
        self._server_socket.listen(5)

        try:
            # --- receive models ---------------------------------------------
            for i in range(self.min_clients):
                model = self._receive_model(client_index=i + 1)
                self.client_models.append(model)
                self.logger.info("Received model from client %d (n_estimators=%d)",
                                 i + 1, len(model.estimators_))

            # --- aggregate --------------------------------------------------
            self.logger.info("Aggregating %d client models ...",
                             len(self.client_models))
            global_model = aggregate_rf_models(self.client_models)
            self.logger.info("Global model: %d total trees",
                             global_model.n_estimators)

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
            self._server_socket.close()
            self.logger.info("Server socket closed")

    # ------------------------------------------------------------------
    def _receive_model(self, client_index: int = 0) -> RandomForestClassifier:
        """Receive a pickled model from a single client using length-prefixed framing.

        Parameters
        ----------
        client_index : int
            Index of the client for logging purposes.

        Returns
        -------
        RandomForestClassifier
            The deserialized model.
        """
        self.logger.info("Waiting for connection from client %d ...",
                         client_index)

        conn, addr = self._server_socket.accept()
        self.logger.info("Connection accepted from %s:%d", *addr)

        try:
            # Read 4-byte length prefix
            header = b""
            while len(header) < 4:
                chunk = conn.recv(4 - len(header))
                if not chunk:
                    raise ConnectionError(f"Client {client_index} disconnected "
                                          f"before sending payload header")
                header += chunk

            payload_len = int.from_bytes(header, byteorder="big")
            self.logger.info("Client %d: expecting %d bytes",
                             client_index, payload_len)

            # Read exact payload
            payload = b""
            while len(payload) < payload_len:
                remaining = payload_len - len(payload)
                chunk = conn.recv(min(4096, remaining))
                if not chunk:
                    raise ConnectionError(f"Client {client_index} disconnected "
                                          f"during payload transfer "
                                          f"(got {len(payload)}/{payload_len})")
                payload += chunk

            model = pickle.loads(payload)
            self.logger.info("Client %d: deserialized model (%s)",
                             client_index, type(model).__name__)
            return model

        finally:
            conn.close()
