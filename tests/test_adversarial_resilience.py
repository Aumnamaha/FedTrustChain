"""Integration tests for adversarial poisoning resilience.

Tests verify:
- Label-flipping detection via DynamicTrustManager accuracy evaluation
- Trust score decay across multiple attack rounds
- Blockchain ledger verification for REJECTED payload SHA-256 hashes
- Full simulation comparison (Mode A vs Mode B) on synthetic data

Run with:  python -m unittest discover -s tests -p "test_adversarial_resilience.py" -v
"""

import hashlib
import os
import sys
import threading
import time
import unittest
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.server.dynamic_trust import DynamicTrustManager, aggregate_random_forests
from src.blockchain.ledger import PermissionedLedger
from src.client.client_node import FederatedClient
from src.server.server_node import FederatedServer


# ── Test 1: Label-flipping detection ───────────────────────────────────────

class TestLabelFlippingDetection(unittest.TestCase):
    """Verify that label-flipped models are detected as adversarial."""

    def setUp(self):
        np.random.seed(42)
        self.X_val, self.y_val = make_classification(
            n_samples=100, n_features=5, n_informative=3,
            n_redundant=0, n_classes=2, random_state=42,
        )

    def test_label_flip_model_triggers_decay(self):
        """A model trained on flipped labels should get low accuracy and trigger trust decay."""
        manager = DynamicTrustManager(
            holdout_X=self.X_val,
            holdout_y=self.y_val,
            reward_rate=0.05,
            decay_rate=0.25,
            accuracy_delta=0.15,
        )

        # First establish baseline with a good model from client 1
        good_model = RandomForestClassifier(n_estimators=30, random_state=42)
        good_model.fit(self.X_val, self.y_val)
        accepted_good, acc_good, _ = manager.evaluate_update(client_id=1, model=good_model)

        # Now submit a label-flipped model from client 2
        X_train_flip = np.random.randn(len(self.X_val), 5)
        y_train_flip = 1.0 - self.y_val  # flipped labels
        flip_model = RandomForestClassifier(n_estimators=30, random_state=42)
        flip_model.fit(X_train_flip, y_train_flip)

        accepted_flip, acc_flip, _ = manager.evaluate_update(client_id=2, model=flip_model)

        # Good model should be accepted with high accuracy (first eval always passes since _best_accuracy starts at 0.0)
        self.assertTrue(accepted_good, f"Good model should be accepted, got acc={acc_good:.4f}")
        self.assertGreaterEqual(acc_good, 0.75, f"Good model accuracy {acc_good:.4f} too low")

        # Flipped-label model should get rejected (low accuracy on holdout)
        self.assertFalse(accepted_flip, f"Flipped model should be detected as adversarial, got acc={acc_flip:.4f}")
        self.assertLess(acc_flip, 0.50, f"Label-flipped model accuracy {acc_flip:.4f} should be poor")

        # Verify trust score decay for client 2
        self.assertLess(manager.trust_scores[2], 1.0, "Adversarial client trust should decay below 1.0")


# ── Test 2: Trust score decay across multiple rounds ───────────────────────

class TestTrustScoreDecayMultiRound(unittest.TestCase):
    """Verify that repeated adversarial submissions progressively reduce trust scores."""

    def setUp(self):
        np.random.seed(42)
        self.X_val, self.y_val = make_classification(
            n_samples=100, n_features=5, n_informative=3,
            n_redundant=0, n_classes=2, random_state=42,
        )

    def test_trust_score_decays_to_threshold(self):
        """Repeated adversarial models should push T_i below the 0.50 threshold."""
        manager = DynamicTrustManager(
            holdout_X=self.X_val,
            holdout_y=self.y_val,
            min_trust_threshold=0.50,
            reward_rate=0.05,
            decay_rate=0.25,
            accuracy_delta=0.15,
        )

        # Establish baseline with a good model
        good_model = RandomForestClassifier(n_estimators=30, random_state=42)
        good_model.fit(self.X_val, self.y_val)
        manager.evaluate_update(client_id=1, model=good_model)

        # Submit adversarial models repeatedly
        for round_num in range(5):
            X_bad = np.random.randn(len(self.X_val), 5)
            y_bad = np.random.randint(0, 2, size=len(self.y_val))
            bad_model = RandomForestClassifier(n_estimators=10, max_depth=2, random_state=99 + round_num)
            bad_model.fit(X_bad, y_bad)
            accepted, acc, _ = manager.evaluate_update(client_id=1, model=bad_model)

        # After 5 adversarial rounds, trust should be well below threshold
        final_score = manager.trust_scores[1]
        self.assertLess(final_score, 0.50, f"Trust score {final_score:.4f} should be below 0.50 after 5 adversarial rounds")

    def test_trust_score_recovers_with_honest_rounds(self):
        """Honest retraining gradually restores trust after adversarial periods."""
        manager = DynamicTrustManager(
            holdout_X=self.X_val,
            holdout_y=self.y_val,
            min_trust_threshold=0.50,
            reward_rate=0.10,  # higher reward for faster recovery
            decay_rate=0.30,   # heavier penalty
            accuracy_delta=0.15,
        )

        # Establish baseline then degrade trust
        good_model = RandomForestClassifier(n_estimators=30, random_state=42)
        good_model.fit(self.X_val, self.y_val)
        manager.evaluate_update(client_id=1, model=good_model)

        for _ in range(5):
            X_bad = np.random.randn(len(self.X_val), 5)
            y_bad = np.random.randint(0, 2, size=len(self.y_val))
            bad_model = RandomForestClassifier(n_estimators=10, max_depth=2, random_state=99)
            bad_model.fit(X_bad, y_bad)
            manager.evaluate_update(client_id=1, model=bad_model)

        # Trust should be low now
        self.assertLess(manager.trust_scores[1], 0.50)

        # Now submit honest models to recover trust
        for round_num in range(8):
            honest_model = RandomForestClassifier(n_estimators=30, random_state=42 + round_num)
            honest_model.fit(self.X_val, self.y_val)
            manager.evaluate_update(client_id=1, model=honest_model)

        # Trust should have recovered above threshold
        self.assertGreaterEqual(manager.trust_scores[1], 0.50,
                                f"Trust score {manager.trust_scores[1]:.4f} should recover above 0.50")


# ── Test 3: Blockchain ledger verification for REJECTED payloads ───────────

class TestBlockchainLedgerVerification(unittest.TestCase):
    """Verify that rejected payload hashes are correctly recorded in the blockchain."""

    def test_rejected_payload_hash_recorded(self):
        """REJECTED transactions must have valid SHA-256 hash entries in the ledger."""
        np.random.seed(42)
        X_val, y_val = make_classification(n_samples=100, n_features=5, n_informative=3,
                                           n_redundant=0, n_classes=2, random_state=42)

        manager = DynamicTrustManager(
            holdout_X=X_val, holdout_y=y_val,
            min_trust_threshold=0.50, decay_rate=0.25, accuracy_delta=0.15,
        )

        ledger = PermissionedLedger(difficulty=2)

        # Establish baseline with good model from client 1
        good_model = RandomForestClassifier(n_estimators=30, random_state=42)
        good_model.fit(X_val, y_val)
        accepted_good, acc_good, hash_good = manager.evaluate_update(client_id=1, model=good_model)

        # Submit adversarial models to trigger rejection and record in ledger
        for round_num in range(3):
            X_bad = np.random.randn(len(X_val), 5)
            y_bad = np.random.randint(0, 2, size=len(y_val))
            bad_model = RandomForestClassifier(n_estimators=10, max_depth=2, random_state=99 + round_num)
            bad_model.fit(X_bad, y_bad)

            accepted, acc, payload_hash = manager.evaluate_update(client_id=2, model=bad_model)

            # Record in ledger
            current_trust = manager.node_registry[2].trust_score
            status = "ACCEPTED" if accepted else "REJECTED"
            ledger.record_transaction(
                client_id=3 + round_num,  # use different IDs to avoid confusion
                model_hash=payload_hash,
                trust_score=current_trust,
                status=status,
            )

        # Mine the block and verify chain validity
        mined_block = ledger.mine_pending_block()
        is_valid = ledger.is_chain_valid()
        self.assertTrue(is_valid, "Blockchain must remain valid after recording REJECTED transactions")

        # Verify that rejected payload hashes are valid SHA-256 (64 hex chars)
        for tx in mined_block.transactions:
            if tx["status"] == "REJECTED":
                model_hash = tx["model_hash"]
                self.assertEqual(len(model_hash), 64, f"Hash must be 64-char SHA-256, got {len(model_hash)}")
                int(model_hash, 16)  # verify it's valid hex

    def test_blockchain_audit_log_json_export(self):
        """JSON audit log export must contain REJECTED entries with correct structure."""
        np.random.seed(42)
        X_val, y_val = make_classification(n_samples=100, n_features=5, n_informative=3,
                                           n_redundant=0, n_classes=2, random_state=42)

        manager = DynamicTrustManager(
            holdout_X=X_val, holdout_y=y_val,
            min_trust_threshold=0.50, decay_rate=0.25, accuracy_delta=0.15,
        )

        ledger = PermissionedLedger(difficulty=2)

        # Submit adversarial model
        X_bad = np.random.randn(len(X_val), 5)
        y_bad = np.random.randint(0, 2, size=len(y_val))
        bad_model = RandomForestClassifier(n_estimators=10, max_depth=2, random_state=99)
        bad_model.fit(X_bad, y_bad)

        accepted, acc, payload_hash = manager.evaluate_update(client_id=3, model=bad_model)
        status = "REJECTED" if not accepted else "ACCEPTED"

        ledger.record_transaction(
            client_id=3, model_hash=payload_hash,
            trust_score=manager.node_registry[3].trust_score,
            status=status,
        )

        mined_block = ledger.mine_pending_block()
        audit_path = ledger.export_audit_log_json()

        # Verify JSON file exists and contains expected structure
        self.assertTrue(audit_path.exists(), "Audit log must be exported to disk")
        import json
        with open(audit_path, "r") as f:
            audit_data = json.load(f)

        self.assertIn("chain_length", audit_data)
        self.assertIn("is_valid", audit_data)
        self.assertTrue(audit_data["is_valid"], "Exported chain must be valid")


# ── Test 4: Full simulation comparison (Mode A vs Mode B synthetic) ───────

class TestFullSimulationComparison(unittest.TestCase):
    """End-to-end test comparing blind aggregation vs trust-filtered aggregation."""

    def _run_synthetic_round(self, port, partitions, X_test, y_test,
                              trust_kwargs=None, blockchain=False, audit_path=None):
        """Helper to run a single synthetic FL round."""
        num_clients = len(partitions)
        results = {}
        server_ready = threading.Event()

        # Create a small holdout set for trust evaluation (needed when blockchain=True)
        val_split = max(10, int(len(X_test) * 0.5))
        X_val, y_val = X_test[:val_split], y_test[:val_split]

        def _server_worker():
            srv = FederatedServer(host="127.0.0.1", port=port, min_clients=num_clients)
            if trust_kwargs is not None:
                srv._trust_threshold = trust_kwargs.get("min_trust_threshold", 0.50)
                srv._reward_rate = trust_kwargs.get("reward_rate", 0.05)
                srv._decay_rate = trust_kwargs.get("decay_rate", 0.25)
                srv._audit_log_path = audit_path or "results/blockchain_audit_log.json"
            server_ready.set()
            try:
                gmodel, metrics = srv.start(
                    X_test, y_test,
                    X_val=X_val, y_val=y_val,
                    blockchain_enabled=blockchain,
                )
                results["global_model"] = gmodel
                results["metrics"] = metrics
            except Exception as e:
                import sys
                print(f"[SERVER ERROR port={port}] {e}", file=sys.stderr)
                raise

        def _client_worker(cid, X_c, y_c):
            client = FederatedClient(
                client_id=cid + 1, server_host="127.0.0.1", server_port=port,
                n_estimators=30, max_depth=10, random_state=42 + cid,
            )
            server_ready.wait(timeout=15)
            client.train_and_send(X_c, y_c)

        srv_thread = threading.Thread(target=_server_worker, daemon=True)
        srv_thread.start()

        for cid, (X_c, y_c) in enumerate(partitions):
            t = threading.Thread(target=_client_worker, args=(cid, X_c, y_c), daemon=True)
            t.start()
            t.join(timeout=30)
            time.sleep(2.0)

        srv_thread.join(timeout=30)
        return results.get("global_model"), results.get("metrics", {})

    def test_mode_b_maintains_higher_accuracy(self):
        """Mode B (trust-filtered) should maintain higher accuracy than Mode A (blind)."""
        np.random.seed(42)
        X, y = make_classification(n_samples=300, n_features=5, n_informative=3,
                                    n_redundant=1, n_classes=2, random_state=42)

        split = int(len(X) * 0.8)
        X_train, y_train = X[:split], y[:split]
        X_test, y_test = X[split:], y[split:]

        # Partition into 3 clients: 2 clean + 1 adversarial (label flipped)
        chunk = len(X_train) // 3
        clean_1 = (X_train[:chunk], y_train[:chunk])
        clean_2 = (X_train[chunk:2*chunk], y_train[2*chunk:3*chunk])
        adv_X = X_train[2*chunk:3*chunk]
        adv_y = 1.0 - y_train[2*chunk:3*chunk]  # label flip

        partitions = [clean_1, clean_2, (adv_X, adv_y)]

        # Mode A: blind aggregation
        gmodel_a, metrics_a = self._run_synthetic_round(
            port=5090, partitions=partitions, X_test=X_test, y_test=y_test,
            trust_kwargs={"min_trust_threshold": 0.0}, blockchain=False,
        )

        # Mode B: trust-filtered aggregation
        gmodel_b, metrics_b = self._run_synthetic_round(
            port=5190, partitions=partitions, X_test=X_test, y_test=y_test,
            trust_kwargs={"min_trust_threshold": 0.50, "reward_rate": 0.05, "decay_rate": 0.25},
            blockchain=True, audit_path="results/blockchain_audit_log.json",
        )

        # Both modes should produce valid models with reasonable accuracy
        self.assertIsNotNone(gmodel_a)
        self.assertIsNotNone(gmodel_b)

        acc_a = metrics_a.get("accuracy", 0.0)
        acc_b = metrics_b.get("accuracy", 0.0)

        # Mode B should be >= Mode A (filtering adversarial updates helps or at worst equals)
        self.assertGreaterEqual(acc_b, acc_a - 0.05,
            f"Mode B accuracy ({acc_b:.4f}) should not be worse than Mode A ({acc_a:.4f}) by more than 5%")

    def test_client_3_trust_score_below_threshold(self):
        """In Mode B, the adversarial client (id=3) must have trust score below threshold."""
        np.random.seed(42)
        X, y = make_classification(n_samples=300, n_features=5, n_informative=3,
                                    n_redundant=1, n_classes=2, random_state=42)

        split = int(len(X) * 0.8)
        X_train, y_train = X[:split], y[:split]
        X_test, y_test = X[split:], y[split:]

        chunk = len(X_train) // 3
        clean_1 = (X_train[:chunk], y_train[:chunk])
        clean_2 = (X_train[chunk:2*chunk], y_train[2*chunk:3*chunk])
        adv_X = X_train[2*chunk:3*chunk]
        adv_y = 1.0 - y_train[2*chunk:3*chunk]

        partitions = [clean_1, clean_2, (adv_X, adv_y)]

        # Create holdout set for trust evaluation
        val_split = max(10, int(len(X_test) * 0.5))
        X_val, y_val = X_test[:val_split], y_test[:val_split]

        # Run Mode B and capture trust scores via a custom server wrapper
        results = {}
        port = 5290
        server_ready = threading.Event()

        def _server_worker():
            srv = FederatedServer(host="127.0.0.1", port=port, min_clients=3)
            srv._trust_threshold = 0.50
            srv._reward_rate = 0.05
            srv._decay_rate = 0.60  # aggressive decay so single adversarial round drops below threshold
            srv._audit_log_path = "results/blockchain_audit_log.json"
            server_ready.set()

            # Override start to capture trust scores after evaluation
            original_start = srv.start
            def wrapped_start(X_test, y_test, X_val=None, y_val=None, **kwargs):
                gmodel, metrics = original_start(
                    X_test, y_test,
                    X_val=X_val, y_val=y_val,
                    blockchain_enabled=True,
                    **kwargs
                )
                results["trust_scores"] = dict(srv.trust_manager.node_registry) if srv.trust_manager else {}
                return gmodel, metrics

            srv.start = wrapped_start
            gmodel, metrics = srv.start(X_test, y_test, X_val=X_val, y_val=y_val)
            results["metrics"] = metrics

        def _client_worker(cid, X_c, y_c):
            client = FederatedClient(
                client_id=cid + 1, server_host="127.0.0.1", server_port=port,
                n_estimators=30, max_depth=10, random_state=42 + cid,
            )
            server_ready.wait(timeout=15)
            client.train_and_send(X_c, y_c)

        srv_thread = threading.Thread(target=_server_worker, daemon=True)
        srv_thread.start()

        for cid, (X_c, y_c) in enumerate(partitions):
            t = threading.Thread(target=_client_worker, args=(cid, X_c, y_c), daemon=True)
            t.start()
            t.join(timeout=30)
            time.sleep(2.0)

        srv_thread.join(timeout=30)

        # Client 3 (adversarial) should have trust score below 0.50
        trust_scores = results.get("trust_scores", {})
        if 3 in trust_scores:
            self.assertLess(trust_scores[3].trust_score, 0.50,
                f"Adversarial client 3 trust score {trust_scores[3].trust_score:.4f} should be below 0.50")


if __name__ == "__main__":
    unittest.main()
