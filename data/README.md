# Dataset Directory — FedTrustChain

## Overview

This directory contains the `dataset_loader.py` module and serves as the home for local dataset storage under **`data/Datasets/`**. 

> ⚠️ **ALL raw CSV and Parquet files MUST be placed exclusively inside `data/Datasets/`.** No other location is supported. The codebase (via `configs/config.yaml`) resolves all dataset paths relative to this directory.

The entire `Datasets/` subdirectory (except `.gitkeep`) is excluded from Git tracking via `.gitignore` because these files are large and should never be committed to the repository.

---

## Local Folder Structure

```
data/
├── Datasets/                     # ← PLACE ALL DATASETS HERE ONLY
│   ├── .gitkeep                  # Git placeholder (ensures folder exists on clone)
│   │                             # ⚠️ DO NOT DELETE — keeps the directory tracked by Git
│   ├── wustl-ehms-2020_with_attacks_categories.csv    # WUSTL-EHMS-2020 dataset
│   ├── train.csv                # CICIOT23 dataset (training split)
│   └── NF-ToN-IoT.parquet       # NF-ToN-IoT dataset (Parquet format)
├── __init__.py                  # Python package marker
├── dataset_loader.py            # Multi-format loader, preprocessor & client partitioner
└── README.md                    # This file — placement instructions & setup guide
```

### Placement Rules

| Rule | Detail |
|---|---|
| **Only one location** | All datasets go in `data/Datasets/` — nowhere else |
| **Exact filenames** | Filenames must match exactly (case-sensitive on Linux) |
| **No subdirectories** | Do not create nested folders inside `Datasets/` |
| **.gitkeep is required** | Never delete `.gitkeep` — it ensures the folder appears on GitHub after cloning |

---

## Target Datasets

### 1. WUSTL-EHMS-2020 (Healthcare IoT Intrusion Detection)

| Field | Description |
|---|---|
| **File name** | `wustl-ehms-2020_with_attacks_categories.csv` |
| **Format** | CSV (~3.9 MB on disk; ~13K rows after preprocessing) |
| **Features** | 36 numeric features + target label column |
| **Domain** | Healthcare IoT network traffic — benign vs. attack (DoS, Probe, R2L, U2R) classification |
| **Source** | [Kaggle: WUSTL-EHMS-2020 Dataset](https://www.kaggle.com/datasets/aghubaish/wustl-ehms-2020-dataset) |

### 2. CICIOT23 (Modern Cyber-Attack Telemetry)

| Field | Description |
|---|---|
| **File name** | `train.csv` |
| **Format** | CSV (~1.6 GB on disk; ~5.5M rows, downsampled to 300K for benchmarks) |
| **Features** | 46 numeric features + target label column |
| **Domain** | CIC (Canadian Institute for Cybersecurity) IoT attack dataset — comprehensive telemetry with 70+ attack categories including Botnet, DDoS, Brute Force, and Malware traffic |
| **Source** | [Kaggle: CICIOT2023 Dataset](https://www.kaggle.com/datasets/himadri07/ciciot2023) |

### 3. NF-ToN-IoT (Network Flow Telemetry for IoT)

| Field | Description |
|---|---|
| **File name** | `NF-ToN-IoT.parquet` |
| **Format** | Apache Parquet (~9.5 MB on disk; ~926K rows, no downsampling applied) |
| **Features** | 11 numeric features + target label column |
| **Domain** | Network flow-level IoT telemetry — captures normal and attack traffic across diverse IoT device types (cameras, routers, smart speakers, etc.) |
| **Source** | [Kaggle: NF-ToN-IoT Dataset](https://www.kaggle.com/datasets/dhoogla/nftoniot) |

---

## Setup Instructions

### Step 1: Clone the Repository and Create `Datasets/` Folder

```bash
git clone https://github.com/Aumnamaha/FedTrustChain.git
cd FedTrustChain
# The data/Datasets/ folder already exists with .gitkeep placeholder
ls -la data/Datasets/
```

### Step 2: Download Datasets into `data/Datasets/` ONLY

All datasets are hosted on Kaggle. You need a free [Kaggle account](https://www.kaggle.com/account/login) to download them.

#### Option A — Browser Download (Recommended)

1. Open each dataset page below and click **Download**:
   - [WUSTL-EHMS-2020 Dataset](https://www.kaggle.com/datasets/aghubaish/wustl-ehms-2020-dataset)
   - [CICIOT2023 Dataset](https://www.kaggle.com/datasets/himadri07/ciciot2023)
   - [NF-ToN-IoT Dataset](https://www.kaggle.com/datasets/dhoogla/nftoniot)
2. Extract the downloaded ZIP archives.
3. Place each file into `data/Datasets/` with the **exact filename** listed below:

| Expected filename | Source archive (may vary) |
|---|---|
| `wustl-ehms-2020_with_attacks_categories.csv` | WUSTL-EHMS-2020 ZIP → CSV file |
| `train.csv` | CICIOT2023 ZIP → training split CSV |
| `NF-ToN-IoT.parquet` | NF-ToN-IoT ZIP → Parquet file |

#### Option B — Kaggle CLI (for automation)

```bash
# Install the Kaggle CLI first:
pip install kagglehub
cd /home/aumnamaha/Documents/Proj/Capstone\ Proj/FedTrustChain/data/Datasets

# --- WUSTL-EHMS-2020 ---
kaggle datasets download -d aghubaish/wustl-ehms-2020-dataset -p .
unzip *.zip && rm *.zip
mv wustl-ehms-2020_with_attacks_categories.csv ./wustl-ehms-2020_with_attacks_categories.csv  # rename if needed

# --- CICIOT2023 ---
kaggle datasets download -d himadri07/ciciot2023 -p .
unzip *.zip && rm *.zip
mv train.csv ./train.csv   # ensure correct filename in Datasets/

# --- NF-ToN-IoT ---
kaggle datasets download -d hoogla/nftoniot -p .
unzip *.zip && rm *.zip
mv NF-ToN-IoT.parquet ./NF-ToN-IoT.parquet   # ensure correct filename in Datasets/
```

### Step 3: Verify Placement

```bash
ls -lh data/Datasets/
# Expected output (only .gitkeep + 3 dataset files):
# -rw-r--r-- 1 user user    0 Oct 22 .gitkeep
# -rw-r--r-- 1 user user 3.9M ... wustl-ehms-2020_with_attacks_categories.csv
# -rw-r--r-- 1 user user  1.6G ... train.csv
# -rw-r--r-- 1 user user 9.5M ... NF-ToN-IoT.parquet
```

### Step 4: Run Benchmarks

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
| **Size** | CICIOT23 alone is ~1.6 GB; total footprint exceeds 5 GB |
| **Immutability** | Raw datasets are reference data — they should not change between commits |
| **Reproducibility** | Download links and file names in this README ensure anyone can recreate the local setup |
| **GitHub limits** | GitHub has a 100 MB per-file soft limit; committing these files would break CI/CD workflows |

The `.gitignore` entry `data/Datasets/*` (with `!data/Datasets/.gitkeep` exception) ensures only the placeholder directory structure is tracked — no heavy binary files are ever pushed to GitHub.

---

## Troubleshooting

### "Dataset not found" error

If you see `FileNotFoundError: Dataset not found`, verify:
1. The file exists **inside `data/Datasets/`** with the exact filename from this README
2. The filename matches exactly (case-sensitive on Linux)
3. You have read permissions (`chmod +r data/Datasets/train.csv`)

### Out of Memory during CICIOT23 benchmark

If training crashes with memory errors:
- Reduce `max_samples` in `scripts/benchmark_all.py` (default 300,000)
- Lower `n_estimators_per_client` in `configs/config.yaml` (default 50)
- Ensure at least 16 GB RAM is available

---

*This README is part of the FedTrustChain project — Zero-Trust Federated Learning Framework for Decentralized IoT Anomaly Detection.*
