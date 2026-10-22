# Dataset Directory — FedTrustChain

## Overview

This directory contains the `dataset_loader.py` module and serves as the home for local dataset storage under `data/Datasets/`. **The `Datasets/` subdirectory is intentionally excluded from Git tracking** (see `.gitignore`) because the raw CSV and Parquet files are large (up to ~1.6 GB each) and should not be committed to the repository.

---

## Local Folder Structure

```
data/
├── Datasets/                     # ← LOCAL ONLY (ignored by Git via .gitignore)
│   ├── wustl-ehms-2020_with_attacks_categories.csv    # WUSTL-EHMS-2020 dataset
│   ├── train.csv                # CICIOT23 dataset (training split)
│   └── NF-ToN-IoT.parquet       # NF-ToN-IoT dataset (Parquet format)
├── dataset_loader.py            # Multi-format loader, preprocessor & client partitioner
└── README.md                    # This file — dataset placement instructions
```

> **Important:** `data/Datasets/` must be populated locally before running any benchmark or training script. The directory is tracked in `.gitignore` so no heavy binary files are pushed to GitHub.

---

## Target Datasets

### 1. WUSTL-EHMS-2020 (Healthcare IoT Intrusion Detection)

| Field | Description |
|---|---|
| **File name** | `wustl-ehms-2020_with_attacks_categories.csv` |
| **Format** | CSV (~3.9 MB on disk; ~13K rows after preprocessing) |
| **Features** | 36 numeric features + target label column |
| **Domain** | Healthcare IoT network traffic — benign vs. attack (DoS, Probe, R2L, U2R) classification |
| **Source** | [WUSTL-EHMS-2020 Dataset](https://github.com/WUSTLEHMS/EHMS-2020-Dataset) |

### 2. CICIOT23 (Modern Cyber-Attack Telemetry)

| Field | Description |
|---|---|
| **File name** | `train.csv` |
| **Format** | CSV (~1.6 GB on disk; ~5.5M rows, downsampled to 300K for benchmarks) |
| **Features** | 46 numeric features + target label column |
| **Domain** | CIC (Canadian Institute for Cybersecurity) IoT attack dataset — comprehensive telemetry with 70+ attack categories including Botnet, DDoS, Brute Force, and Malware traffic |
| **Source** | [CIC-IoT-2023 Dataset](https://www.unb.ca/cic/datasets/iot-2023.html) |

### 3. NF-ToN-IoT (Network Flow Telemetry for IoT)

| Field | Description |
|---|---|
| **File name** | `NF-ToN-IoT.parquet` |
| **Format** | Apache Parquet (~9.5 MB on disk; ~926K rows, no downsampling applied) |
| **Features** | 11 numeric features + target label column |
| **Domain** | Network flow-level IoT telemetry — captures normal and attack traffic across diverse IoT device types (cameras, routers, smart speakers, etc.) |
| **Source** | [NF-ToN-IoT Dataset](https://www.unb.ca/cic/datasets/toniot.html) |

---

## Setup Instructions

### Step 1: Download the Datasets

Place each file into `data/Datasets/` with the exact filename listed above:

```bash
# Create the local dataset directory (if not already present)
mkdir -p data/Datasets

# --- WUSTL-EHMS-2020 ---
wget -O data/Datasets/wustl-ehms-2020_with_attacks_categories.csv \
  "https://raw.githubusercontent.com/WUSTLEHMS/EHMS-2020-Dataset/main/wustl-ehms-2020_with_attacks_categories.csv"

# --- CICIOT23 (training split) ---
# Download from CIC website and place as train.csv:
mv ~/Downloads/train.csv data/Datasets/train.csv   # adjust path as needed

# --- NF-ToN-IoT ---
wget -O data/Datasets/NF-ToN-IoT.parquet \
  "https://raw.githubusercontent.com/WUSTLEHMS/EHMS-2020-Dataset/main/NF-ToN-IoT.parquet"
```

> **Note:** If the raw download URLs above are unavailable, download directly from the dataset source links in the table and rename/move files to match the expected names.

### Step 2: Verify Placement

```bash
ls -lh data/Datasets/
# Expected output:
# wustl-ehms-2020_with_attacks_categories.csv   (3.9 MB)
# train.csv                                    (~1.6 GB for CICIOT23)
# NF-ToN-IoT.parquet                           (9.5 MB)
```

### Step 3: Run Benchmarks

```bash
python scripts/benchmark_all.py
```

The benchmark script reads paths from `configs/config.yaml` and automatically applies stratified downsampling (`max_samples=300000`) for CICIOT23 to maintain memory stability during training.

---

## Configuration Management

All dataset file paths are centralized in [`configs/config.yaml`](../configs/config.yaml):

```yaml
paths:
  dataset_dir: "/home/aumnamaha/Documents/Proj/Capstone Proj/FedTrustChain/data/Datasets"
  wustl_ehms: "/home/aumnamaha/Documents/Proj/Capstone Proj/FedTrustChain/data/Datasets/wustl-ehms-2020_with_attacks_categories.csv"
  ciciot23:   "/home/aumnamaha/Documents/Proj/Capstone Proj/FedTrustChain/data/Datasets/train.csv"
  nf_ton_iot: "/home/aumnamaha/Documents/Proj/Capstone Proj/FedTrustChain/data/Datasets/NF-ToN-IoT.parquet"
```

The `IoTDatasetLoader` class resolves these paths at runtime and validates file existence before loading. If a dataset is missing, it raises `FileNotFoundError`.

---

## Why Is `data/Datasets/` Ignored by Git?

| Reason | Detail |
|---|---|
| **Size** | CICIOT23 alone is ~1.6 GB; the total dataset footprint exceeds 5 GB |
| **Immutability** | Raw datasets are reference data — they should not change between commits |
| **Reproducibility** | Download links and file names in this README ensure anyone can recreate the local setup |
| **GitHub limits** | GitHub has a 100 MB per-file soft limit; committing these files would break CI/CD workflows |

The `.gitignore` entry `data/Datasets/` ensures no CSV, Parquet, or other binary dataset files are ever accidentally staged or pushed to the remote repository.

---

## Troubleshooting

### "Dataset not found" error

If you see `FileNotFoundError: Dataset not found`, verify:
1. The file exists at the path specified in `configs/config.yaml`
2. The filename matches exactly (case-sensitive on Linux)
3. You have read permissions on the file (`chmod +r data/Datasets/train.csv`)

### Out of Memory during CICIOT23 benchmark

If training crashes with memory errors:
- Reduce `max_samples` in `scripts/benchmark_all.py` (default 300,000)
- Lower `n_estimators_per_client` in `configs/config.yaml` (default 50)
- Ensure at least 16 GB RAM is available

---

*This README is part of the FedTrustChain project — Zero-Trust Federated Learning Framework for Decentralized IoT Anomaly Detection.*
