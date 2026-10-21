"""Unit tests for IoTDatasetLoader (multi-format loading + client partitioning).

Uses synthetic numeric data so no real datasets or network access are needed.
Run with:  python -m pytest tests/test_loader.py -v
          or: python -m unittest tests.test_loader
"""

import os
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd

# Ensure the project root is on sys.path so we can import data.dataset_loader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset_loader import IoTDatasetLoader


class TestIoTDatasetLoader(unittest.TestCase):
    """Tests for dataset loading, preprocessing, and client partitioning."""

    def setUp(self):
        """Create a temporary directory with synthetic CSV and Parquet files."""
        self.tmpdir = tempfile.mkdtemp()

        # --- synthetic data: 100 rows x 5 features + 1 target (3 classes) ---
        np.random.seed(42)
        n_samples = 100
        X_syn = np.random.randn(n_samples, 5)
        y_syn = np.random.choice(["benign", "dos", "probe"], size=n_samples)

        self.csv_path = os.path.join(self.tmpdir, "synthetic.csv")
        pd.DataFrame(
            {**{f"feat_{i}": X_syn[:, i] for i in range(5)}, "label": y_syn}
        ).to_csv(self.csv_path, index=False)

        self.parquet_path = os.path.join(self.tmpdir, "synthetic.parquet")
        pd.DataFrame(
            {**{f"feat_{i}": X_syn[:, i] for i in range(5)}, "label": y_syn}
        ).to_parquet(self.parquet_path, engine="pyarrow", index=False)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # ------------------------------------------------------------------
    def test_load_csv_returns_correct_shapes(self):
        loader = IoTDatasetLoader()
        X_train, X_test, y_train, y_test, scaler = loader.load_and_preprocess(
            self.csv_path, target_col="label", test_size=0.2
        )

        # 80/20 split on 100 samples → 80 train, 20 test
        self.assertEqual(X_train.shape[0], 80)
        self.assertEqual(X_test.shape[0], 20)
        self.assertEqual(X_train.shape[1], 5)   # 5 features
        self.assertEqual(y_train.shape[0], 80)
        self.assertEqual(y_test.shape[0], 20)

    def test_load_parquet_returns_correct_shapes(self):
        loader = IoTDatasetLoader()
        X_train, X_test, y_train, y_test, scaler = loader.load_and_preprocess(
            self.parquet_path, target_col="label", test_size=0.2
        )

        self.assertEqual(X_train.shape[0], 80)
        self.assertEqual(X_test.shape[0], 20)
        self.assertEqual(y_train.shape[0], 80)
        self.assertEqual(y_test.shape[0], 20)

    def test_load_csv_auto_target_last_column(self):
        """When target_col is None, the last column should be used as label."""
        loader = IoTDatasetLoader()
        X_train, X_test, y_train, y_test, scaler = loader.load_and_preprocess(
            self.csv_path  # no target_col → 'label' (last col)
        )

        self.assertEqual(X_train.shape[0], 80)
        self.assertIn(y_train.iloc[0] if hasattr(y_train, "iloc") else y_train[0],
                      ["benign", "dos", "probe"])

    def test_no_nan_after_preprocessing(self):
        """After cleaning, X_train and X_test must contain no NaN or inf."""
        loader = IoTDatasetLoader()
        X_train, X_test, y_train, y_test, scaler = loader.load_and_preprocess(
            self.csv_path, target_col="label"
        )

        self.assertFalse(np.isnan(X_train).any(), "X_train contains NaN")
        self.assertFalse(np.isnan(X_test).any(), "X_test contains NaN")
        self.assertFalse(np.isinf(X_train).any(), "X_train contains inf")
        self.assertFalse(np.isinf(X_test).any(), "X_test contains inf")

    def test_scaler_is_fitted(self):
        """Scaler should have mean_ and scale_ attributes after fit."""
        loader = IoTDatasetLoader()
        X_train, X_test, y_train, y_test, scaler = loader.load_and_preprocess(
            self.csv_path, target_col="label"
        )

        self.assertTrue(hasattr(scaler, "mean_"))
        self.assertTrue(hasattr(scaler, "scale_"))
        self.assertEqual(scaler.mean_.shape[0], 5)

    def test_partition_equal_chunks(self):
        """Partition should produce num_clients chunks with correct total."""
        loader = IoTDatasetLoader()
        X_train, X_test, y_train, y_test, scaler = loader.load_and_preprocess(
            self.csv_path, target_col="label"
        )

        num_clients = 4
        partitions = loader.partition_for_clients(X_train, y_train, num_clients)

        self.assertEqual(len(partitions), num_clients)

        # Total samples across all clients should equal original train size
        total_samples = sum(p[0].shape[0] for p in partitions)
        self.assertEqual(total_samples, X_train.shape[0])  # 80

    def test_partition_no_nan(self):
        """Partitioned data must not contain NaN."""
        loader = IoTDatasetLoader()
        X_train, X_test, y_train, y_test, scaler = loader.load_and_preprocess(
            self.csv_path, target_col="label"
        )

        partitions = loader.partition_for_clients(X_train, y_train, num_clients=4)
        for i, (X_c, y_c) in enumerate(partitions):
            self.assertFalse(np.isnan(X_c).any(), f"Client {i} X has NaN")
            self.assertEqual(len(y_c), len(X_c))

    def test_unsupported_format_raises(self):
        loader = IoTDatasetLoader()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = f.name
        try:
            with self.assertRaises(ValueError):
                loader.load_and_preprocess(tmp_path, target_col="label")
        finally:
            os.unlink(tmp_path)

    def test_file_not_found_raises(self):
        loader = IoTDatasetLoader()
        with self.assertRaises(FileNotFoundError):
            loader.load_and_preprocess("/nonexistent/path/to/data.csv", target_col="label")


if __name__ == "__main__":
    unittest.main()
