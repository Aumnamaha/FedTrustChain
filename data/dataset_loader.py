"""Multi-format dataset loader with client partitioning for federated learning."""

from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class IoTDatasetLoader:
    """Load, preprocess, and partition IoT cybersecurity datasets.

    Supports CSV and Parquet formats, with optional config.yaml path resolution,
    missing-value imputation, scaling, equal-sized client chunking for simulated
    federated learning nodes, and stratified downsampling for ultra-large datasets.
    """

    def __init__(self, dataset_dir: str = None, config_path: str = "configs/config.yaml"):
        self.dataset_dir = dataset_dir
        self._config_paths: Dict[str, str] = {}
        if os.path.isfile(config_path):
            try:
                import yaml  # noqa: F811
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
        max_samples: int = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
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
        max_samples : int or None
            If total rows exceed this value, apply a stratified downsample to
            keep at most *max_samples* rows (default: no limit).

        Returns
        -------
        X_train, X_test, y_train, y_test, scaler
            All feature arrays are numpy floats; scaler fits on X_train only.
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
            raise KeyError(
                f"Target column '{target_col}' not in dataset. "
                f"Available columns: {list(df.columns)}"
            )

        y = df[target_col].copy()
        X = df.drop(columns=[target_col])

        # --- keep only numeric features ---------------------------------
        X_num = X.select_dtypes(include=["number"])

        # --- stratified downsampling for ultra-large datasets ----------
        if max_samples is not None and len(X_num) > max_samples:
            print(
                f"[Loader] {filepath}: {len(X_num)} rows exceeds limit "
                f"({max_samples}); applying stratified downsample."
            )
            # Build a combined DataFrame for stratified sampling
            df_combined = pd.DataFrame({**{c: X_num[c].values for c in X_num.columns},
                                        target_col: np.asarray(y)})

            # Single stratified sample to max_samples total rows
            n_total = len(df_combined)
            df_sampled, _ = train_test_split(
                df_combined,
                test_size=n_total - max_samples,
                random_state=42,
                stratify=df_combined[target_col]
                         if len(df_combined[target_col].unique()) > 1 else None,
            )

            # Now split the sampled data into train/test
            X_sampled = df_sampled.drop(columns=[target_col]).to_numpy(dtype=np.float64)
            y_sampled = np.asarray(df_sampled[target_col])

        else:
            # --- impute inf / nan -------------------------------------------
            X_num = X_num.replace([np.inf, -np.inf], np.nan).fillna(0)

            # --- convert to plain numpy arrays (pandas 3.x Arrow compat) ----
            X_sampled = np.asarray(X_num.to_numpy())
            y_sampled = np.asarray(y)

        # --- stratified train/test split --------------------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X_sampled,
            y_sampled,
            test_size=test_size,
            random_state=42,
            stratify=y_sampled if len(np.unique(y_sampled)) > 1 else None,
        )

        # --- scale ------------------------------------------------------
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        return X_train, X_test, y_train, y_test, scaler

    # ------------------------------------------------------------------
    def partition_for_clients(self, X_train: np.ndarray, y_train: np.ndarray,
                              num_clients: int):
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
