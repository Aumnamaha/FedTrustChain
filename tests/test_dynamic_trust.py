"""Unit tests for the Dynamic Node Trust Manager.

Verifies trust score initialization, decay behavior under adversarial updates,
SHA-256 hash consistency, and filtering of low-trust nodes before aggregation.

Run with:  python -m unittest discover -s tests -p "test_dynamic_trust.py" -v
"""

import pickle
import sys
import os
import unittest

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.server.dynamic_trust import (
    DynamicTrustManager,
    aggregate_random_forests,
)


# ------------------------------------------------------------------
class TestDynamicTrustInitialization(unittest.TestCase):
    """Verify default trust score initialization and registry setup."""

    def setUp(self):
        np.random.seed(42)
        self.X_val = np.random.randn(50, 5)
        self.y_val = (self.X_val[:, 0] + self.X_val[:, 1] > 0).astype(int)

    def test_default_trust_score_is_1_0(self):
        """New nodes must start with trust score T_i = 1.0."""
        manager = DynamicTrustManager(
            holdout_X=self.X_val,
            holdout_y=self.y_val,
            min_trust_threshold=0.50,
        )
        # No nodes registered yet
        self.assertEqual(len(manager.trust_scores), 0)

        # First evaluation registers the node with default score
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(self.X_val, self.y_val)
        accepted, accuracy, _ = manager.evaluate_update(client_id=1, model=model)
        self.assertTrue(accepted)
        scores = manager.trust_scores
        self.assertIn(1, scores)
        self.assertAlmostEqual(scores[1], 1.0, places=6)  # clamped at max 1.0

    def test_node_registry_tracks_rounds_and_updates(self):
        """Registry must track total_rounds, accepted_updates, rejected_updates."""
        manager = DynamicTrustManager(
            holdout_X=self.X_val,
            holdout_y=self.y_val,
        )
        good_model = RandomForestClassifier(n_estimators=10, random_state=42)
        good_model.fit(self.X_val, self.y_val)

        accepted, _, _ = manager.evaluate_update(client_id=42, model=good_model)
        self.assertTrue(accepted)

        registry = manager.node_registry[42]
        self.assertEqual(registry.total_rounds, 1)
        self.assertEqual(registry.accepted_updates, 1)
        self.assertEqual(registry.rejected_updates, 0)


# ------------------------------------------------------------------
class TestTrustScoreDecay(unittest.TestCase):
    """Verify score decay when clients submit low-accuracy or corrupted models."""

    def setUp(self):
        np.random.seed(42)
        # Create a clear signal so good models achieve high accuracy
        self.X_val, self.y_val = make_classification(
            n_samples=100, n_features=5, n_informative=3,
            n_redundant=0, n_classes=2, random_state=42,
        )

    def test_good_model_rewards_trust_score(self):
        """A well-performing model should increase T_i by reward_rate."""
        manager = DynamicTrustManager(
            holdout_X=self.X_val,
            holdout_y=self.y_val,
            reward_rate=0.05,
            accuracy_delta=0.15,
        )
        # Train a good model on the same distribution
        model = RandomForestClassifier(n_estimators=30, random_state=42)
        model.fit(self.X_val, self.y_val)

        accepted, accuracy, _ = manager.evaluate_update(client_id=1, model=model)
        self.assertTrue(accepted)
        self.assertGreaterEqual(accuracy, 0.85)
        # Score should be 1.0 + 0.05 (capped at 1.0)
        self.assertAlmostEqual(manager.trust_scores[1], 1.0, places=6)

    def test_poor_model_decays_trust_score(self):
        """A random/noisy model should trigger decay of T_i by decay_rate."""
        manager = DynamicTrustManager(
            holdout_X=self.X_val,
            holdout_y=self.y_val,
            reward_rate=0.05,
            decay_rate=0.25,
            accuracy_delta=0.15,
        )
        # Train a good baseline first to establish _best_accuracy
        good_model = RandomForestClassifier(n_estimators=30, random_state=42)
        good_model.fit(self.X_val, self.y_val)
        manager.evaluate_update(client_id=1, model=good_model)

        # Now submit a poor model (random predictions via shuffled labels during fit)
        X_bad = np.random.randn(len(self.X_val), 5)
        y_bad = np.random.randint(0, 2, size=len(self.y_val))
        bad_model = RandomForestClassifier(n_estimators=10, max_depth=2, random_state=99)
        bad_model.fit(X_bad, y_bad)

        accepted, accuracy, _ = manager.evaluate_update(client_id=1, model=bad_model)
        # Poor accuracy should trigger rejection and decay
        self.assertFalse(accepted)
        expected_score = max(0.0, 1.0 - 0.25)
        self.assertAlmostEqual(manager.trust_scores[1], expected_score, places=6)

    def test_trust_score_clamped_to_zero(self):
        """Repeated penalties must not push T_i below 0.0."""
        manager = DynamicTrustManager(
            holdout_X=self.X_val,
            holdout_y=self.y_val,
            decay_rate=0.25,
            accuracy_delta=0.15,
        )
        # Establish a high baseline first
        good_model = RandomForestClassifier(n_estimators=30, random_state=42)
        good_model.fit(self.X_val, self.y_val)
        manager.evaluate_update(client_id=1, model=good_model)

        # Submit many poor models to push score down
        for _ in range(10):
            X_bad = np.random.randn(len(self.X_val), 5)
            y_bad = np.random.randint(0, 2, size=len(self.y_val))
            bad_model = RandomForestClassifier(n_estimators=5, max_depth=2, random_state=99)
            bad_model.fit(X_bad, y_bad)
            manager.evaluate_update(client_id=1, model=bad_model)

        self.assertGreaterEqual(manager.trust_scores[1], 0.0)
        # Score should be clamped at 0.0 after enough penalties
        self.assertEqual(manager.trust_scores[1], 0.0)


# ------------------------------------------------------------------
class TestTrustFiltering(unittest.TestCase):
    """Verify that low-trust nodes are excluded from aggregation."""

    def setUp(self):
        np.random.seed(42)
        self.X_val = np.random.randn(50, 5)
        self.y_val = (self.X_val[:, 0] + self.X_val[:, 1] > 0).astype(int)

    def test_low_trust_node_excluded_from_aggregation(self):
        """Nodes with T_i < threshold must be filtered out by filter_trusted_clients."""
        manager = DynamicTrustManager(
            holdout_X=self.X_val,
            holdout_y=self.y_val,
            min_trust_threshold=0.50,
            decay_rate=0.25,
            accuracy_delta=0.15,
        )

        # Client 1: good model → accepted, T_i = 1.05 → clamped to 1.0
        good_model = RandomForestClassifier(n_estimators=20, random_state=42)
        good_model.fit(self.X_val, self.y_val)
        manager.evaluate_update(client_id=1, model=good_model)

        # Client 2: poor models to push trust below threshold
        for _ in range(3):
            X_bad = np.random.randn(len(self.X_val), 5)
            y_bad = np.random.randint(0, 2, size=len(self.y_val))
            bad_model = RandomForestClassifier(n_estimators=5, max_depth=2, random_state=99)
            bad_model.fit(X_bad, y_bad)
            manager.evaluate_update(client_id=2, model=bad_model)

        # Verify trust scores
        self.assertGreaterEqual(manager.trust_scores[1], 0.50)
        self.assertLess(manager.trust_scores[2], 0.50)

        # Filter trusted clients
        models = [good_model, bad_model]
        trusted_ids, trusted_models = manager.filter_trusted_clients(
            client_ids=[1, 2], models=models
        )
        self.assertEqual(trusted_ids, [1])
        self.assertEqual(len(trusted_models), 1)

    def test_aggregation_after_filtering(self):
        """aggregate_random_forests must work on filtered trusted model list."""
        manager = DynamicTrustManager(
            holdout_X=self.X_val,
            holdout_y=self.y_val,
            min_trust_threshold=0.50,
            decay_rate=0.25,
            accuracy_delta=0.15,
        )

        # Client 1: good → trusted
        model_1 = RandomForestClassifier(n_estimators=10, random_state=42)
        model_1.fit(self.X_val, self.y_val)
        manager.evaluate_update(client_id=1, model=model_1)

        # Client 2: bad → untrusted
        X_bad = np.random.randn(len(self.X_val), 5)
        y_bad = np.random.randint(0, 2, size=len(self.y_val))
        model_2 = RandomForestClassifier(n_estimators=5, max_depth=2, random_state=99)
        model_2.fit(X_bad, y_bad)
        for _ in range(4):
            manager.evaluate_update(client_id=2, model=model_2)

        # Filter
        trusted_ids, trusted_models = manager.filter_trusted_clients(
            [1, 2], [model_1, model_2]
        )
        self.assertEqual(len(trusted_ids), 1)
        self.assertEqual(trusted_ids[0], 1)

        # Aggregate only trusted models — must not raise
        global_model = aggregate_random_forests(trusted_models)
        self.assertEqual(global_model.n_estimators, 10)


# ------------------------------------------------------------------
class TestSHA256HashConsistency(unittest.TestCase):
    """Verify SHA-256 hash generation is deterministic and consistent."""

    def test_identical_payloads_produce_same_hash(self):
        """Same model serialized twice must yield identical hashes."""
        manager = DynamicTrustManager(
            holdout_X=np.random.randn(30, 3),
            holdout_y=(np.random.randn(30) > 0).astype(int),
        )
        X = np.random.randn(50, 3)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        hash1 = manager.compute_payload_hash(pickle.dumps(model))
        hash2 = manager.compute_payload_hash(pickle.dumps(model))
        self.assertEqual(hash1, hash2)
        self.assertEqual(len(hash1), 64)  # SHA-256 hex digest is 64 chars

    def test_different_models_produce_different_hashes(self):
        """Different models must yield different hashes."""
        manager = DynamicTrustManager(
            holdout_X=np.random.randn(30, 3),
            holdout_y=(np.random.randn(30) > 0).astype(int),
        )
        X = np.random.randn(50, 3)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        model_a = RandomForestClassifier(n_estimators=10, random_state=42)
        model_b = RandomForestClassifier(n_estimators=20, random_state=99)
        model_a.fit(X, y)
        model_b.fit(X, y)

        hash_a = manager.compute_payload_hash(pickle.dumps(model_a))
        hash_b = manager.compute_payload_hash(pickle.dumps(model_b))
        self.assertNotEqual(hash_a, hash_b)


# ------------------------------------------------------------------
class TestEmptyModelHandling(unittest.TestCase):
    """Verify that corrupted or empty models are rejected and penalized."""

    def test_empty_estimators_list_rejected(self):
        """A model with no estimators must be rejected and T_i decayed."""
        manager = DynamicTrustManager(
            holdout_X=np.random.randn(30, 3),
            holdout_y=(np.random.randn(30) > 0).astype(int),
            decay_rate=0.25,
        )
        # Create a model with empty estimators list (simulates corruption)
        bad_model = RandomForestClassifier(n_estimators=10, random_state=42)
        bad_model.fit(manager.holdout_X, manager.holdout_y)
        bad_model.estimators_ = []  # Corrupt: remove all trees

        accepted, accuracy, payload_hash = manager.evaluate_update(
            client_id=99, model=bad_model
        )
        self.assertFalse(accepted)
        self.assertEqual(accuracy, 0.0)
        self.assertEqual(len(payload_hash), 64)
        # Trust score should decay from 1.0 to 0.75
        self.assertAlmostEqual(manager.trust_scores[99], 0.75, places=6)


if __name__ == "__main__":
    unittest.main()
