"""Multi-format dataset loader with client partitioning for federated learning."""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class IoTDatasetLoader:
    """Load, preprocess, and partition IoT cybersecurity datasets.

    Supports CSV and Parquet formats, with optional config.yaml path resolution,
    missing-value imputation, scaling, and equal-sized client chunking for
    simulated federated learning nodes.
    """

    def __init__(self, dataset_dir: str = None, config_path: str = "configs/config.yaml"):
        self.dataset_dir = dataset_dir
        self._config_paths = {}
        if os.path.isfile(config_path):
            try:
                import yaml
                with open(config_path, "r") as f:
                    cfg = yaml.safe_load(f)
                if cfg and "paths" in cfg:
                    for key, val in cfg["paths"].items():
                        if isinstance(val, str) and os.path.isfile(val):
                            self._config_paths[key] = val
            except ImportError:
                pass  # pyyaml not available; rely on explicit paths

    # ------------------------------------------------------------------
    def load_and_preprocess(
        self,
        dataset_path_or_key: str,
        target_col: str | None = None,
        test_size: float = 0.2,
    ):
        """Load a CSV or Parquet file and return train/test splits + scaler.

        Parameters
        ----------
        dataset_path_or_key : str
            Absolute path to the data file, *or* a key present in config.yaml
            (e.g. ``'wustl_ehms'``, ``'ciciot23'``, ``'nf_ton_iot'``).
        target_col : str or None
            Name of the label column.  If ``None`` the last column is used.
        test_size : float
            Fraction reserved for testing (default 0.2 → 80/20 split).

        Returns
        -------
        X_train, X_test, y_train, y_test, scaler
            All feature arrays are numpy floats; scaler fits X_train.
        """
        # --- resolve path -----------------------------------------------
        if dataset_path_or_key in self._config_paths:
            filepath = self._config_paths[dataset_path_or_key]
        else:
            filepath = dataset_path_or_key

        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Dataset not found: {filepath}")

        # --- read -------------------------------------------------------
        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".csv":
            df = pd.read_csv(filepath)
        elif ext in (".parquet", ".pq"):
            df = pd.read_parquet(filepath, engine="pyarrow")
        else:
            raise ValueError(f"Unsupported format '{ext}'. Use .csv or .parquet.")

        # --- target column ----------------------------------------------
        if target_col is None:
            target_col = df.columns[-1]
        elif target_col not in df.columns:
            raise KeyError(f"Target column '{target_col}' not in dataset. "
                           f"Available columns: {list(df.columns)}")

        y = df[target_col].copy()
        X = df.drop(columns=[target_col])

        # --- keep only numeric features ---------------------------------
        X_num = X.select_dtypes(include=["number"])

        # --- impute inf / nan -------------------------------------------
        X_num = X_num.replace([np.inf, -np.inf], np.nan).fillna(0)

        # --- convert to plain numpy arrays (pandas 3.x Arrow compat) ----
        X_arr = np.asarray(X_num.to_numpy())
        y_arr = np.asarray(y)

        # --- stratified split -------------------------------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X_arr,
            y_arr,
            test_size=test_size,
            random_state=42,
            stratify=y_arr if len(np.unique(y_arr)) > 1 else None,
        )

        # --- scale ------------------------------------------------------
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        return X_train, X_test, y_train, y_test, scaler

    # ------------------------------------------------------------------
    def partition_for_clients(self, X_train, y_train, num_clients: int):
        """Split training data into *num_clients* equal contiguous chunks.

        Returns a list of ``(X_client_i, y_client_i)`` tuples.
        """
        n = len(X_train)
        chunk_size = n // num_clients
        remainder = n % num_clients
        partitions = []
        start = 0
        for i in range(num_clients):
            end = start + chunk_size + (1 if i < remainder else 0)
            partitions.append((X_train[start:end], y_train[start:end]))
            start = end
        return partitions
