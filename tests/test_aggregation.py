"""Unit tests for federated Random Forest aggregation engine.

Run with:  python -m unittest discover -s tests -p "test_*.py"
          or: python -m unittest tests.test_aggregation -v
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

from src.models.rf_federated import aggregate_rf_models


class TestAggregateRFModels(unittest.TestCase):
    """Tests for the federated ensemble aggregation function."""

    def setUp(self):
        """Create synthetic classification data used across tests."""
        np.random.seed(42)
        self.X_train, self.y_train = make_classification(
            n_samples=200, n_features=5, n_informative=3,
            n_redundant=1, n_classes=3, random_state=42
        )
        self.X_test, _ = make_classification(
            n_samples=50, n_features=5, n_informative=3,
            n_redundant=1, n_classes=3, random_state=99
        )

    # ------------------------------------------------------------------
    def test_aggregate_empty_list_raises(self):
        """aggregate_rf_models([]) should raise ValueError."""
        with self.assertRaises(ValueError):
            aggregate_rf_models([])

    def test_aggregate_none_raises(self):
        """aggregate_rf_models(None) should raise ValueError."""
        with self.assertRaises(ValueError):
            aggregate_rf_models(None)

    # ------------------------------------------------------------------
    def test_aggregate_single_model_returns_deep_copy(self):
        """Single model → deep copy; modifying global must not affect original."""
        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        rf.fit(self.X_train, self.y_train)

        global_model = aggregate_rf_models([rf])

        # Should be a deep copy (different object identity)
        self.assertIsNot(global_model, rf)
        self.assertEqual(global_model.n_estimators, rf.n_estimators)
        self.assertEqual(len(global_model.estimators_), len(rf.estimators_))

        # Modifying global model should not affect original
        global_model.estimators_.clear()
        self.assertEqual(len(rf.estimators_), 10)   # original untouched

    # ------------------------------------------------------------------
    def test_aggregate_multiple_models(self):
        """Two models with 20 trees each → global has 40 trees."""
        rf_1 = RandomForestClassifier(n_estimators=20, random_state=42)
        rf_2 = RandomForestClassifier(n_estimators=20, random_state=99)

        rf_1.fit(self.X_train, self.y_train)
        rf_2.fit(self.X_train, self.y_train)

        global_model = aggregate_rf_models([rf_1, rf_2])

        # Tree count
        self.assertEqual(global_model.n_estimators, 40)
        self.assertEqual(len(global_model.estimators_), 40)

        # Should still have correct feature & class metadata
        self.assertEqual(global_model.n_features_in_, 5)
        self.assertEqual(global_model.n_classes_, 3)

        # Predictions should run without error and produce valid output
        preds = global_model.predict(self.X_test)
        self.assertEqual(len(preds), len(self.X_test))
        self.assertTrue(all(p in [0, 1, 2] for p in preds))

    def test_aggregate_three_models(self):
        """Three models → sum of all trees."""
        n_trees = [15, 25, 30]
        models = []
        for nt in n_trees:
            m = RandomForestClassifier(n_estimators=nt, random_state=42)
            m.fit(self.X_train, self.y_train)
            models.append(m)

        global_model = aggregate_rf_models(models)

        expected_total = sum(n_trees)  # 70
        self.assertEqual(global_model.n_estimators, expected_total)
        self.assertEqual(len(global_model.estimators_), expected_total)

    # ------------------------------------------------------------------
    def test_mismatched_features_raises(self):
        """Models with different feature counts should raise ValueError."""
        rf_1 = RandomForestClassifier(n_estimators=10, random_state=42)
        rf_1.fit(self.X_train, self.y_train)  # 5 features

        X_other, y_other = make_classification(
            n_samples=200, n_features=8, n_informative=3,
            n_redundant=1, n_classes=3, random_state=42
        )
        rf_2 = RandomForestClassifier(n_estimators=10, random_state=99)
        rf_2.fit(X_other, y_other)  # 8 features

        with self.assertRaises(ValueError) as ctx:
            aggregate_rf_models([rf_1, rf_2])

        self.assertIn("Feature dimension mismatch", str(ctx.exception))

    def test_mismatched_classes_raises(self):
        """Models trained on different class sets should raise ValueError."""
        # Model 1: 3 classes (0, 1, 2)
        rf_1 = RandomForestClassifier(n_estimators=10, random_state=42)
        rf_1.fit(self.X_train, self.y_train)

        # Model 2: 2 classes only
        X_binary, y_binary = make_classification(
            n_samples=200, n_features=5, n_informative=3,
            n_redundant=1, n_classes=2, random_state=42
        )
        rf_2 = RandomForestClassifier(n_estimators=10, random_state=99)
        rf_2.fit(X_binary, y_binary)

        with self.assertRaises(ValueError) as ctx:
            aggregate_rf_models([rf_1, rf_2])

        self.assertIn("Class count mismatch", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
