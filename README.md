# FedTrustChain

**Zero-Trust Federated Learning Framework for Decentralized IoT Anomaly Detection**

---

## Abstract

FedTrustChain is a production-grade federated learning (FL) system designed to detect cyber intrusions across distributed healthcare and IoT infrastructure without centralizing sensitive network traffic data. The framework employs **Federated Random Forest** aggregation — uniting locally-trained decision tree ensembles from multiple client nodes into a global model via estimator union (`F_global = ⋃ T_i`), eliminating the need for raw data sharing while preserving privacy by design.

Communication between clients and server uses **length-prefixed TCP socket streaming** (4-byte big-endian `struct.pack('>I', payload_length)`) to guarantee pickle-serialized model payloads are received atomically, avoiding TCP stream fragmentation entirely.

---

## Architecture & Algorithmic Principle

### Federated Random Forest Aggregation

Unlike gradient-based FL (FedAvg), tree ensembles cannot be averaged via weight averaging. Instead, FedTrustChain uses the **bag-of-trees** strategy:

```
F_global = ⋃_{i=1}^{K} T_i   where K = number of participating clients
```

Each client `i` trains a local `RandomForestClassifier(n_estimators, max_depth)` on its private dataset. The server receives each fitted model via TCP socket, extracts the list of individual decision trees (`estimators_`), and concatenates them into a single global forest. The resulting model retains the predictive power of all clients' data while never requiring raw samples to leave their local nodes.

### Network Protocol: Length-Prefixed Framing

To prevent TCP stream fragmentation with large pickle payloads (~20 MB for 50 trees × 36 features), every transmission is wrapped:

```
[4-byte big-endian unsigned int: payload_length][payload_bytes]
```

**Sender:** `sock.sendall(len(payload).to_bytes(4, 'big'))` → `sock.sendall(payload)`  
**Receiver:** Read 4 bytes → decode length → read exact N bytes → `pickle.loads()`

This guarantees atomic delivery regardless of network packet boundaries.

### System Components

```
┌─────────────────────────────────────────────────────┐
│                  Federated Server                    │
│  (Binds TCP socket, receives models, aggregates,    │
│   evaluates global model on held-out test data)      │
│                                                      │
│  ┌──────────────┐    ┌──────────────┐               │
│  │FederatedServer│    │aggregate_rf_ │               │
│  │(server_node.py)│   │models()      │               │
│  └──────────────┘    └──────────────┘               │
└─────────────────────────────────────────────────────┘
         ▲ TCP sockets (length-prefixed pickle)
         │
┌────────┴──────────┐  ┌────────┴──────────┐
│  Client Node 0    │  │  Client Node K-1  │
│  (client_node.py) │  │  (client_node.py) │
│                   │  │                   │
│  RandomForest     │  │  RandomForest     │
│  train_and_send() │  │  train_and_send() │
└───────────────────┘  └───────────────────┘
```

---

## Directory Structure

```
FedTrustChain/
├── configs/                     # YAML configuration (server, model, dataset paths)
│   └── config.yaml              # Host/port, tree hyperparameters, dataset locations
├── data/                        # Temporary data storage (gitignored for CSV/Parquet)
│   └── __init__.py
├── scripts/                     # Standalone execution scripts
│   ├── benchmark_all.py         # Multi-dataset FL benchmark orchestrator
│   └── __init__.py
├── src/                         # Core application modules (importable package)
│   ├── __init__.py
│   ├── client/                  # Federated learning client implementation
│   │   ├── __init__.py
│   │   └── client_node.py       # FederatedClient: trains + sends models over TCP
│   ├── server/                  # Federated learning server implementation
│   │   ├── __init__.py
│   │   └── server_node.py       # FederatedServer: receives, aggregates, evaluates
│   ├── models/                  # Model aggregation & architecture
│   │   ├── __init__.py
│   │   └── rf_federated.py      # aggregate_rf_models(): estimator union fusion
│   └── utils/                   # Shared utilities
│       ├── __init__.py
│       └── logger.py            # Structured logging helper (timestamp + level)
├── tests/                       # Unit and integration test suite
│   ├── __init__.py
│   ├── test_aggregation.py      # RF model aggregation edge cases (7 tests)
│   ├── test_loader.py           # Dataset loading & client partitioning (9 tests)
│   ├── test_networking.py       # TCP client-server pipeline (2 tests)
│   └── test_simulation.py       # Full benchmark round on synthetic data (2 tests)
├── results/                     # Benchmark output artifacts (gitignored except JSON)
│   └── benchmark_summary.json   # Structured metrics per dataset run
├── .gitignore                   # Ignore patterns for data, models, venvs, logs
├── requirements.txt             # Pinned Python dependencies
└── README.md                    # This file
```

---

## Installation & Setup

### Prerequisites

- **Python 3.10+** (tested on Fedora Linux with Python 3.14)
- **8 GB+ RAM** for NF-ToN-IoT; **16 GB+ recommended** for CICIOT23 (~9M samples)
- **~500 MB disk space** for datasets and model artifacts

### Step-by-Step Setup

```bash
# 1. Clone the repository
git clone https://github.com/Aumnamaha/FedTrustChain.git
cd FedTrustChain

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation (run all tests)
python -m unittest discover -s tests -p "test_*.py" -v
```

### Expected Dependencies

| Package | Minimum Version | Purpose |
|---|---|---|
| `numpy` | ≥1.24.0 | Numerical computing |
| `pandas` | ≥2.0.0 | Data loading & manipulation |
| `scikit-learn` | ≥1.3.0 | RandomForestClassifier, metrics |
| `pyyaml` | ≥6.0 | Config file parsing |
| `pyarrow` | ≥12.0.0 | Parquet dataset reading |
| `matplotlib` | ≥3.7.0 | Visualization (future) |
| `seaborn` | ≥0.12.0 | Statistical plots (future) |

---

## Execution Instructions

### 1. Run All Unit & Integration Tests

```bash
python -m unittest discover -s tests -p "test_*.py" -v
# Expected: 20 tests, all passing
```

### 2. Launch a Single Client-Server Pair (Interactive)

```python
# server.py — runs on the aggregation node
from src.server.server_node import FederatedServer
import numpy as np
from sklearn.datasets import make_classification

X_test, y_test = make_classification(n_samples=100, n_features=5, n_classes=3)
server = FederatedServer(host="127.0.0.1", port=5050, min_clients=1)
global_model, metrics = server.start(X_test, y_test)
print(metrics)

# client.py — runs on a distributed node (separate process)
from src.client.client_node import FederatedClient
client = FederatedClient(client_id=0, server_host="127.0.0.1", server_port=5050)
X_local, y_local = ...  # load local data
local_model = client.train_and_send(X_local, y_local)
```

### 3. Execute Multi-Dataset Benchmark Suite

```bash
python scripts/benchmark_all.py
```

This script:
1. Loads all three datasets from `configs/config.yaml`
2. For each dataset: partitions data across `min_clients`, spawns FL round via TCP sockets, aggregates global model, evaluates metrics
3. Exports structured results to `results/benchmark_summary.json`
4. Prints a formatted ASCII summary table

### 4. View Benchmark Results

```bash
cat results/benchmark_summary.json | python -m json.tool
```

---

## Empirical Benchmark Results

All benchmarks executed on Fedora Linux (RTX 5070 Ti 12GB, Core Ultra 9, 32 GB RAM).  
Configuration: `min_clients=2`, `n_estimators_per_client=50`, `max_depth=15`, `random_state=42`.

| Dataset | Samples | Clients | Total Trees | Accuracy | Weighted F1-Score | Precision | Recall | Time (s) |
|---|---|---|---|---|---|---|---|---|
| **WUSTL-EHMS-2020** | 13,054 | 2 | 100 | 94.06% | 93.28% | 94.31% | 94.06% | 0.47 |
| **CICIOT23** | 240,000* | 2 | 100 | 98.90% | 98.79% | 98.85% | 98.90% | 57.10 |
| **NF-ToN-IoT** | 926,395 | 2 | 100 | 68.91% | 59.99% | 55.63% | 68.91% | 28.14 |

> \* CICIOT23: Original dataset has ~5.5M rows; stratified downsampling to 300K total (per `max_samples=300000` in `scripts/benchmark_all.py`) was applied for memory-stable training and fast socket serialization (~12 MB/client). Each client received ~120K samples.

### Observations

- **CICIOT23**: Highest accuracy (98.9%) — stratified downsample to 300K preserved class distribution; the CIC suite's well-labeled traffic patterns yield excellent RF separation across attack categories
- **WUSTL-EHMS-2020**: Strong performance (94.1%) — clean healthcare network traffic with well-separated benign vs. attack classes; fastest benchmark at 0.47s total
- **NF-ToN-IoT**: Moderate performance (68.9% acc, 59.99% F1) — large-scale IoT dataset (~926K samples) with high class imbalance and diverse attack vectors reduces per-tree discriminative power; precision-recall gap indicates conservative classification bias

---

## Post-Review 2 Roadmap

The following enhancements are planned for the next development cycle:

### 🔗 Permissioned Blockchain Ledger Integration

- Integrate a **Hyperledger Fabric** or **Corda** permissioned ledger to immutably record every global model update
- Each aggregation round produces a blockchain transaction containing the global model hash, participating client IDs, and evaluation metrics
- Enables full audit trail for regulatory compliance (HIPAA, NIST 800-53)

### 📜 Smart Contract Model Verification

- Deploy **Solidity-based smart contracts** on a private Ethereum network to:
  - Verify model integrity via Merkle tree hashes before aggregation
  - Enforce client eligibility rules (e.g., minimum accuracy thresholds, reputation scores)
  - Automate reward distribution for high-quality contributors in incentive-aware FL settings

### 🏆 Dynamic Node Reputation Scoring

- Implement a **reputation engine** that tracks each client's contribution quality across rounds:
  - **Accuracy-weighted voting**: Clients with higher local model accuracy receive greater influence during aggregation
  - **Poisoning detection**: Statistical outlier detection on submitted model gradients/trees to identify adversarial clients
  - **Decay & recovery**: Reputation scores decay over time; honest retraining restores standing

### 📊 Expanded Benchmark Suite

- Add **XGBoost (CUDA)** and **PyTorch MLP** baselines for comparison against Random Forest
- Implement **non-IID data partitioning** strategies (Dirichlet, hierarchical) to evaluate robustness under realistic FL conditions
- Add **communication cost analysis**: measure bandwidth per round vs. accuracy trade-offs

---

## License

This project is released as open-source research software for academic and non-commercial use. See `LICENSE` file for details.

---

*Built for the Capstone Project — IoT Cybersecurity Intrusion Detection across Healthcare & IoT Infrastructure.*
