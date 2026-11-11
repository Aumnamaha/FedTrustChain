# FedTrustChain

**Zero-Trust Federated Learning Framework for Decentralized IoT Anomaly Detection**

---

## Abstract

FedTrustChain is a production-grade federated learning (FL) system designed to detect cyber intrusions across distributed healthcare and IoT infrastructure without centralizing sensitive network traffic data. The framework employs **Federated Random Forest** aggregation — uniting locally-trained decision tree ensembles from multiple client nodes into a global model via estimator union (`F_global = ⋃ T_i`), eliminating the need for raw data sharing while preserving privacy by design.

Communication between clients and server uses **chunked UDP datagram transport** with a custom 12-byte binary header (`struct.pack("!III", client_id, chunk_index, total_chunks)`) for low-latency wireless edge networking. Each serialized model payload is partitioned into independently routable datagrams of configurable size (default 1024 bytes), enabling efficient transmission across constrained IoT links.

---

## Architecture & Algorithmic Principle

### Federated Random Forest Aggregation

Unlike gradient-based FL (FedAvg), tree ensembles cannot be averaged via weight averaging. Instead, FedTrustChain uses the **bag-of-trees** strategy:

```
F_global = ⋃_{i=1}^{K} T_i   where K = number of participating clients
```

Each client `i` trains a local `RandomForestClassifier(n_estimators, max_depth)` on its private dataset. The server receives each fitted model over chunked UDP datagrams, extracts the list of individual decision trees (`estimators_`), and concatenates them into a single global forest. The resulting model retains the predictive power of all clients' data while never requiring raw samples to leave their local nodes.

### Network Protocol: Chunked UDP Datagram Transport

To minimize latency on wireless edge links, model payloads are split into small UDP datagrams with application-layer sequencing. Each datagram carries a 12-byte header followed by up to `chunk_size` bytes (default 1024) of payload:

```
[client_id: 4B][chunk_index: 4B][total_chunks: 4B] [payload_chunk_bytes]
```

**Sender:** Partition serialized model → prepend 12-byte header per chunk → `sock.sendto(datagram, (host, port))` for each chunk.  
**Receiver:** Bind UDP socket → unpack header → buffer chunks by `(client_id, chunk_index)` → reassemble when all chunks arrive → `pickle.loads(reassembled_payload)`.  

This approach tolerates individual datagram loss via timeout-based detection and trust-score penalties, making it well-suited for unreliable wireless IoT links where TCP's retransmission overhead would add unacceptable latency.

#### Wireless IoT Edge Node Considerations

The chunked UDP transport layer was specifically designed for resource-constrained wireless edge environments common in healthcare IoMT and industrial IoT deployments:

| Parameter | Default | Tuning Guidance |
|---|---|---|
| `chunk_size` (client/server) | 8192 bytes | Increase for high-bandwidth links; decrease to 1024–2048 on Wi‑Fi to avoid IP fragmentation |
| `reassembly_timeout` (server) | 5.0 s | Increase to 10–15 s for high-latency cellular links; decrease to 2 s for dense sensor meshes |
| Header overhead | 12 bytes per datagram | At default 8192-byte chunks: ~0.15 %; at 1024-byte chunks: ~1.2 % (use smaller chunks on Wi‑Fi to avoid IP fragmentation) |

**Why UDP over TCP for edge FL?**

- **Lower latency**: No handshakes, no retransmission stalls on lost packets — critical when client nodes are battery-powered IoT sensors with intermittent connectivity.
- **Smaller per-packet overhead**: 12-byte application header vs. TCP's minimum 20-byte IP + 20-byte TCP header + potential TLS (32+ bytes). At 8192-byte chunks this saves ~5–10 % of transport-layer bytes per packet.
- **Graceful degradation**: Dropped datagrams are detected at the application layer via timeout and trust-score penalties rather than stalling the entire connection waiting for TCP retransmission ACKs.

---

### System Components

```
┌──────────────────────────────────────────────────────────────────┐
│                  Federated Server                                 │
│  (Binds UDP socket, receives models, aggregates,                 │
│   evaluates global model on held-out test data)                  │
│                                                                   │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────────┐  │
│  │FederatedServer│    │aggregate_rf_ │    │Consensus Engine   │  │
│  │(server_node.py)│   │models()      │    │consensus.py       │  │
│  └──────────────┘    └──────────────┘    │• ProofOfWork       │  │
│     ▲                                     │• ValidatorRegistry │  │
│     │ UDP models                          │• ChainValidator    │  │
│     ▼                                     └───────────────────┘  │
│  ┌──────────────┐                                                  │
│  │DynamicTrust   │                                                  │
│  │Manager        │                                                  │
│  │(dynamic_trust)│                                                  │
│  └──────────────┘                                                  │
└──────────────────────────────────────────────────────────────────┘
         ▲ Chunked UDP datagrams (12-byte header + payload)
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

## Architecture Breakdown & Technical Conceptual Guide

### Server Role: Federated Orchestrator and Model Aggregator

The server **does not train on raw data**. Its responsibilities are:

1. **Bind a UDP socket** on a configurable host/port (default `127.0.0.1:5050`) and wait for incoming datagrams.
2. **Receive serialized models** from each participating client — one at a time, in the order they connect.
3. **Aggregate received estimators** into a single global forest using estimator union fusion (`aggregate_rf_models`), which extracts every `estimator_` (individual decision tree) from each client's `RandomForestClassifier` and concatenates them into one large ensemble.
4. **Evaluate the aggregated model** against a holdout test dataset to compute accuracy, F1-score, precision, and recall.
5. **Return** both the global model object and a metrics dictionary to the caller.

The server is stateless between rounds — it starts fresh each time `start()` is called.

### Client Role: Local Edge Trainer

Each client node operates independently on its private data slice:

1. **Load local training data** `(X_train, y_train)` from a local dataset file (CSV or Parquet).
2. **Partition the full dataset** into `num_clients` equal contiguous chunks using stratified chunking — each client receives one chunk.
3. **Train a `RandomForestClassifier`** strictly on its assigned chunk (default: 50 trees, max depth 15).
4. **Serialize the fitted model** via `pickle.dumps()` into a byte payload (~20 MB for 50 trees × 36 features).
5. **Transmit the payload over chunked UDP**: partition into 1024-byte chunks, prepend a 12-byte binary header per chunk (`client_id | chunk_index | total_chunks`), and `sendto()` each datagram to `(server_host, server_port)`.
6. **Return** the fitted local model to the caller.

Clients never communicate with each other; all communication flows through the central server.

### Data Privacy & Zero-Trust Boundary

```
┌───────────────────────────────────────────────────────┐
│                   TRUSTED ZONE                        │
│  (Federated Server — aggregation, evaluation)         │
│                                                       │
│  Receives: serialized tree estimators only            │
│  Never sees: raw data samples, feature values,        │
│              labels, or network traffic logs           │
└───────────────▲──────────────────▲────────────────────┘
                │ UDP datagram     │ UDP datagram
                ▼                  ▼
┌───────────────┴──────────┐ ┌────┴──────────────────────┐
│   Client Node 0          │ │   Client Node K-1         │
│                          │ │                           │
│  Raw data stays here     │ │  Raw data stays here      │
│  Only model params leave │ │  Only model params leave  │
└──────────────────────────┘ └───────────────────────────┘
```

**Key invariant:** Raw dataset logs never leave edge devices. The only artifacts transmitted across the wire are decision tree estimator objects — internal node split thresholds, feature indices, and leaf values. These reveal no individual training samples.

### Consensus Engine (Phase 3 Step 1)

FedTrustChain integrates a **Proof-of-Work consensus engine** (`src/blockchain/consensus.py`) that provides tamper-evident block creation and automatic fork resolution on top of the existing permissioned ledger:

```
Client Model Received -> DynamicTrustManager.evaluate_update()
    |
    +-- T_i >= 0.50 (trusted) --> ValidatorRegistry.verify_proposer(node_id)
    |   |                              |
    |   |                              +-- Authorized OK -> ProofOfWork.mine_block()
    |   |                              |                    |
    |   |                              |                    +-- Append transaction to block
    |   |                              |                    +-- Find nonce (3 leading zeros)
    |   |                              |                    +-- Broadcast mined block header
    |   |                              |
    |   |                              +-- Unauthorized FAIL -> Log warning, skip mining
    |   |
    +-- T_i < 0.50 (untrusted) --> Skip consensus mining, log REJECTED

ChainValidator.validate_chain() runs continuously:
    +-- Check 1: PoW hash verification (SHA-256 content match)
    +-- Check 2: Difficulty compliance (leading zeros in hash)
    +-- Check 3: Chain linkage (previous_hash integrity)

On fork detection -> resolve_fork() adopts longest valid chain.
```

**Components:**
| Component | Responsibility |
|---|---|
| `ProofOfWork` | Dynamic target difficulty mining (default: 3 leading zero hex characters). Mines blocks containing validated client transactions. |
| `ValidatorRegistry` | Tracks node signatures and trust scores. Only nodes with T_i >= 0.50 may propose or sign candidate blocks. |
| `ChainValidator` | Detects three tamper categories - payload modification, chain-link breach, invalid nonce - and triggers automatic fork resolution to the longest valid chain.'

### Multi-Node P2P Block Propagation (Phase 3 Step 2)

FedTrustChain extends the consensus engine with **asynchronous UDP block broadcasting**, enabling multi-node network convergence:

```
┌──────────┐    ┌──────────┐    ┌──────────────┐
│ Client A │    │ Client B │    │ Client C (Mal)│
│ T=0.95   │    │ T=0.90   │    │ T=0.35        │
│ Validator│    │ Listener │    │ Forged Block  │
└────▲─────┘    └────▲─────┘    └──────▲───────┘
     │               │                 │
     │ send_model()  │ send_model()    │ broadcast_forged()
     │──────────────>│────────────────>|
                     │ FederatedServer │
                     │ (UDP Server)    │
                     │                 │
                     │ ┌─────────────┐ │
                     │ │TrustManager │ │
                     │ │A: ACCEPTED  │ │
                     │ │B: ACCEPTED  │ │
                     │ │C: REJECTED  │ │
                     │ └─────────────┘ │
                     │                 │
                     │ ProofOfWork.mine│ │
                     │ ValidatorRegistry│
                     │                 │
                     │ <── broadcast_block_header() ──┐
                     │         │          │           │
                     │       UDP        UDP        UDP
                     │────────|──────────|──────────>|
                     │  verified     verified    REJECTED
                     │             (invalid prev_hash)
```

**Sequence Flow:**
1. Clients A & B train local models and send over chunked UDP to the server.
2. Server evaluates via `DynamicTrustManager` — both pass trust threshold (T_i >= 0.50).
3. `ValidatorRegistry.verify_proposer()` confirms authorization for both nodes.
4. `ProofOfWork.mine_block()` creates a consensus block with both transactions.
5. Server broadcasts the mined block header to all registered UDP client addresses.
6. Client C (malicious) simultaneously broadcasts a forged block with invalid previous hash.
7. All honest clients run `ChainValidator.validate_chain()` on received blocks:
   - Legitimate blocks pass all three checks (PoW hash, difficulty, chain linkage).
   - Forged block fails ChainLinkBreach check — rejected across all nodes.
8. Honest nodes converge on identical block height and ledger state.

**Network Resilience:**
- Nodes can disconnect mid-chain and rejoin via broadcast sync.
- Partial chains remain valid; full sync restores complete state.
- Fork resolution always selects the longest valid chain.

### High-Concurrency Throughput & Node Churn (Phase 2 Infrastructure)

FedTrustChain supports high-concurrency UDP traffic with dynamic node churn:

```
┌──────────┐ ┌──────────┐ ┌──────────┐    ┌──────────────┐
│ Worker 1 │ │ Worker 2 │ │ ...      │    │              │
│ T=0.95   │ │ T=0.90   │ │ Worker 7 │    │ Federated    │
│ ---------->│ ---------->|----------->|----|Server        │
│            │            │            |    │ (UDP)       │
│            │            │ DISCONNECT │    │              │
└──────────┘ └──────────┘ └────▲─────┘    │  ┌──────────┐ │
                               │          |--|TrustMgr   |-|
                    Reconnect◄─┘          |  └──────────┘ │
                           Workers 11,12  |              │
                                           └──────────────┘
```

**Churn Execution Flow:**
1. 10 concurrent worker nodes send model payloads simultaneously over chunked UDP.
2. After receiving models from 7 workers, nodes 8-10 are disconnected mid-round.
3. While aggregation/ledger logging is active, 2 new peer nodes (11, 12) join and send models.
4. Server completes aggregation with all available trusted models.
5. All active peers maintain identical block heights via consensus broadcast.

**Stress Test Results:**
| Metric | Value |
|---|---|
| Concurrent workers launched | 10 (+ 2 during churn) |
| Workers disconnected mid-round | 3 (nodes 8, 9, 10) |
| New peers attached during aggregation | 2 (nodes 11, 12) |
| Deadlock-free | YES — zero orphaned state |
| Convergence status | CONVERGED — all active nodes on identical block height |
| Audit log stability | VALID — transaction logging uninterrupted through churn  |

---

### Tree Estimator Union Aggregation

### Tree Estimator Union Aggregation

Unlike gradient-based FL (FedAvg), which averages model weights, tree ensembles cannot be averaged element-wise. FedTrustChain uses a **bag-of-trees** aggregation strategy:

$$F_{\text{global}} = \bigcup_{i=1}^{K} T_i$$

where:
- $K$ is the number of participating client nodes.
- $T_i$ represents the set of 50 decision tree estimators trained by client $i$.
- The union operator `∪` concatenates all individual trees into a single `RandomForestClassifier` with $50 \times K$ total estimators.

The resulting global forest retains the predictive power of every client's private dataset while never requiring raw samples to leave their local nodes. Inference on $F_{\text{global}}$ uses majority voting across all trees (both local and aggregated).

### Global Metric Evaluation Formulas

After aggregation, the server evaluates $F_{\text{global}}$ against a held-out test set:

**Prediction:**
$$\hat{y}_{\text{pred}} = F_{\text{global}}.\text{predict}(X_{\text{test}})$$

**Global Accuracy:**
$$\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Test Samples}} = \frac{1}{N}\sum_{j=1}^{N} \mathbb{I}(\hat{y}_j = y_j)$$

where $N$ is the number of test samples and $\mathbb{I}$ is the indicator function.

**Weighted F1-Score (per-class averaged across all classes):**
$$F1_{\text{weighted}} = \sum_{c=1}^{C} w_c \cdot \frac{2 \cdot \text{precision}_c \cdot \text{recall}_c}{\text{precision}_c + \text{recall}_c}$$

where $w_c$ is the support (proportion of samples) for class $c$, and $C$ is the total number of classes.

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
│   ├── simulate_consensus_network.py  # Phase 3: 3-node P2P consensus simulation
│   ├── stress_test_network.py   # Phase 2: High-concurrency UDP stress test & node churn
│   └── __init__.py
├── src/                         # Core application modules (importable package)
│   ├── __init__.py
│   ├── blockchain/              # Permissioned blockchain & consensus engine
│   │   ├── __init__.py
│   │   ├── ledger.py            # Phase 2: Block + PermissionedLedger (audit trail)
│   │   └── consensus.py         # Phase 3: ProofOfWork + ValidatorRegistry + ChainValidator
│   ├── client/                  # Federated learning client implementation
│   │   ├── __init__.py
│   │   └── client_node.py       # FederatedClient: trains + sends models over chunked UDP
│   ├── server/                  # Federated learning server implementation
│   │   ├── __init__.py
│   │   ├── dynamic_trust.py     # Dynamic trust scoring for client evaluation
│   │   └── server_node.py       # FederatedServer: receives, aggregates, evaluates + consensus mining
│   ├── models/                  # Model aggregation & architecture
│   │   ├── __init__.py
│   │   └── rf_federated.py      # aggregate_rf_models(): estimator union fusion
│   └── utils/                   # Shared utilities
│       ├── __init__.py
│       └── logger.py            # Structured logging helper (timestamp + level)
├── tests/                       # Unit and integration test suite
│   ├── __init__.py
│   ├── test_aggregation.py      # RF model aggregation edge cases (7 tests)
│   ├── test_adversarial_resilience.py  # Trust-filtered adversarial robustness (8 tests)
│   ├── test_blockchain_ledger.py     # Permissioned ledger integrity (9 tests)
│   ├── test_consensus.py           # PoW mining + tamper detection + fork resolution (24 tests)
│   ├── test_dynamic_trust.py       # Dynamic trust scoring (7 tests)
│   ├── test_loader.py              # Dataset loading & client partitioning (9 tests)
│   ├── test_network_churn.py       # Phase 2: Peer disconnect/reconnect, out-of-order packets, deadlock-free concurrency (5 tests)
│   ├── test_networking.py          # Chunked UDP client-server pipeline (2 tests)
│   ├── test_p2p_consensus.py       # Phase 3 Step 2: P2P broadcasting, fork resolution, sync recovery (8 tests)
│   └── test_simulation.py          # Full benchmark round on synthetic data (2 tests)
├── results/                     # Benchmark output artifacts (gitignored except JSON)
│   ├── benchmark_summary.json           # Structured metrics per dataset run
│   ├── blockchain_audit_log.json        # Phase 2+3: Chain state, block records, consensus metadata
│   ├── consensus_simulation.json        # Phase 3 Step 2: P2P simulation results (convergence, latency)
│   └── network_stress_results.json      # Phase 2: High-concurrency stress metrics & churn recovery
├── .gitignore                   # Ignore patterns for data, models, venvs, logs
├── requirements.txt             # Pinned Python dependencies
└── README.md                    # This file
```

---

## Installation & Setup

### Prerequisites

- **Python 3.10+** (tested on Fedora Linux with Python 3.14; also fully compatible with Windows 10/11 and macOS)
- **8 GB+ RAM** for NF-ToN-IoT; **16 GB+ recommended** for CICIOT23 (~9M samples)
- **~500 MB disk space** for datasets and model artifacts

### Cross-Platform Notes (Windows)

| Item | Detail |
|---|---|
| **Virtual env activation** | Use `.venv\Scripts\activate` instead of `source .venv/bin/activate` |
| **Path separators** | All Python code uses `os.path.join()` / `pathlib.Path`; no hard-coded `/` or `\` in path construction |
| **Socket binding** | Server binds to `127.0.0.1` with `SO_REUSEADDR=1` — works identically on Windows, Linux, macOS |
| **Line endings** | All `.py` files use LF; Python's standard library handles CRLF transparently for file I/O |

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
2. For each dataset: partitions data across `min_clients`, spawns FL round via chunked UDP datagrams, aggregates global model, evaluates metrics
3. Exports structured results to `results/benchmark_summary.json`
4. Prints a formatted ASCII summary table

### 4. View Benchmark Results

```bash
cat results/benchmark_summary.json | python -m json.tool
```

---

## Quick Interactive Tutorial (Multi-Terminal Live Demonstration)

Follow these exact steps to run a live federated learning round across four terminal windows. This demonstrates the full client-server pipeline end-to-end using the WUSTL-EHMS-2020 dataset.

> **Prerequisites:** Virtual environment activated (`source .venv/bin/activate`) and all dependencies installed via `pip install -r requirements.txt`.

### Step 1: Open Four Terminal Windows

---

### Terminal 1 — Server Orchestrator (Start First)

This terminal launches the federated server, which binds to port `5005` and waits for client connections:

```bash
python -c "
from data.dataset_loader import IoTDatasetLoader
from src.server.server_node import FederatedServer

loader = IoTDatasetLoader('data/Datasets')
X_tr, X_te, y_tr, y_te, _ = loader.load_and_preprocess('wustl_ehms')

server = FederatedServer('127.0.0.1', 5005, min_clients=2)
model, metrics = server.start(X_te, y_te)
print('\nGlobal Model Metrics:', metrics)
"
```

The server will print `Waiting for client models...` and block until both clients connect.

---

### Terminal 2 — Client Node 0 (Start Second)

This terminal trains a local Random Forest on the first partition of training data:

```bash
python -c "
from data.dataset_loader import IoTDatasetLoader
from src.client.client_node import FederatedClient

loader = IoTDatasetLoader('data/Datasets')
X_tr, X_te, y_tr, y_te, _ = loader.load_and_preprocess('wustl_ehms')
parts = loader.partition_for_clients(X_tr, y_tr, 2)

client = FederatedClient(client_id=0, server_host='127.0.0.1', server_port=5005)
model = client.train_and_send(parts[0][0], parts[0][1])
print(f'Client {client.client_id} sent model with {model.n_estimators} trees')
"
```

---

### Terminal 3 — Client Node 1 (Start Third)

This terminal trains a local Random Forest on the second partition of training data:

```bash
python -c "
from data.dataset_loader import IoTDatasetLoader
from src.client.client_node import FederatedClient

loader = IoTDatasetLoader('data/Datasets')
X_tr, X_te, y_tr, y_te, _ = loader.load_and_preprocess('wustl_ehms')
parts = loader.partition_for_clients(X_tr, y_tr, 2)

client = FederatedClient(client_id=1, server_host='127.0.0.1', server_port=5005)
model = client.train_and_send(parts[1][0], parts[1][1])
print(f'Client {client.client_id} sent model with {model.n_estimators} trees')
"
```

---

### Expected Output Sequence

1. **Terminal 1** prints `Starting server on 127.0.0.1:5005 (min_clients=2)` and waits.
2. **Terminal 2** connects, trains ~50 trees (~3 seconds), sends the pickle payload, then exits.
3. **Terminal 3** connects, trains ~50 trees (~3 seconds), sends the pickle payload, then exits.
4. **Terminal 1** receives both models, aggregates them into a global forest of 100 trees, evaluates on `X_test`, and prints:

```
Received model from client 1 (n_estimators=50)
Received model from client 2 (n_estimators=50)
Aggregating 2 client models ...
Global model: 100 total trees

Global Model Metrics: {'accuracy': 0.9406, 'f1_score': 0.9328, 'precision': 0.9431, 'recall': 0.9406}
```

---

### Terminal 4 — Automated Test Verification

In a separate terminal (or after stopping the server), run the full test suite:

```bash
cd /home/aumnamaha/Documents/Proj/Capstone\ Proj/FedTrustChain
python -m unittest discover -s tests -p "test_*.py" -v
```

Expected output: **20 tests, all passing**.

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

## Adversarial Resilience Benchmark Results

A controlled label-flipping poisoning attack was executed to evaluate FedTrustChain's dynamic trust scoring and permissioned blockchain ledger under adversarial conditions.

### Experimental Setup

| Parameter | Value |
|---|---|
| Dataset | WUSTL-EHMS-2020 (13,054 samples) |
| Clients | 3 (Client 1: clean, Client 2: clean, Client 3: adversarial) |
| Attack Strategy | 100% label flipping (`y_poisoned = 1 - y_clean`) on Client 3's partition |
| Trees per Client | 50 RandomForest estimators |
| Mode A Threshold | `min_trust_threshold = 0.0` (blind aggregation) |
| Mode B Threshold | `min_trust_threshold = 0.50` (FedTrustChain protected) |

### Comparison Matrix: Mode A vs Mode B

| Metric | Mode A (Unprotected) | Mode B (Protected) |
|---|---|---|
| **Global Accuracy** | 93.78% | **94.00%** |
| F1-Score (weighted) | 93.05% | **93.19%** |
| Precision (weighted) | 93.75% | **94.25%** |
| Recall (weighted) | 93.78% | **94.00%** |

### Client-Level Results

| Client | Role | Local Accuracy | Trust Score (Mode B) | Status |
|---|---|---|---|---|
| Client 1 | Benign | 93.69% | T₁ = 1.000 | ACCEPTED |
| Client 2 | Benign | 94.18% | T₂ = 1.000 | ACCEPTED |
| Client 3 | Adversarial (flipped labels) | 5.94% | **T₃ = 0.400** | **REJECTED** |

### Key Findings

- **Client 3 Detection**: The adversarial model achieved only 5.94% local accuracy and 4.8% validation accuracy on the holdout set, triggering a trust score penalty that dropped T₃ from 1.0 → 0.4 (below the 0.50 threshold).
- **Blockchain Audit Log**: Client 3's payload was recorded with SHA-256 hash `5b7fb6bc...` and status `REJECTED` in an immutable blockchain block, providing a tamper-proof audit trail.
- **Accuracy Preservation**: Mode B aggregated only the two trusted clients (100 total trees), maintaining accuracy at 94.0% — slightly higher than Mode A's 93.78% which included the poisoned model (150 total trees).
- **Trust Isolation**: The dynamic trust manager correctly identified and isolated Client 3 after a single adversarial round, demonstrating that FedTrustChain can detect label-flipping attacks in real-time without requiring multiple attack rounds.

---

## Post-Review 2 Roadmap

The following enhancements are planned for the next development cycle:

### 🔗 Permissioned Blockchain Ledger Integration **COMPLETED**

- **Phase 2 Step 2:** Implemented `PermissionedLedger` with SHA-256 block hashing and immutable audit trail (`src/blockchain/ledger.py`).
- Each aggregation round logs client model hashes, trust scores, and ACCEPTED/REJECTED decisions.
- JSON export to `results/blockchain_audit_log.json` for regulatory compliance (HIPAA, NIST 800-53).

### 🏗️ Consensus Engine **COMPLETED**

- **Phase 3 Step 1:** Implemented full consensus engine (`src/blockchain/consensus.py`):
  - `ProofOfWork`: Dynamic PoW mining with configurable difficulty (default = 3 leading zeros).
  - `ValidatorRegistry`: Node authorization enforced via trust threshold T_i >= 0.50.
  - `ChainValidator`: Tamper detection (payload, linkage, nonce) + automatic fork resolution to longest valid chain.
- Integrated into UDP aggregation server: trusted clients trigger consensus block mining per round.

### 🌐 P2P Block Propagation **COMPLETED**

- **Phase 3 Step 2:** Implemented multi-node asynchronous block broadcasting (`scripts/simulate_consensus_network.py`):
  - Server broadcasts mined block headers to all registered client UDP sockets via `broadcast_block_header()`.
  - Client nodes receive, parse, and verify blocks using `ChainValidator.validate_chain()` locally.
  - Malicious peer detection: forged blocks with invalid previous hash links are rejected across all honest nodes.
  - Network resilience: partial chains remain valid; rejoining nodes sync via broadcast stream.
- Integration tests (`tests/test_p2p_consensus.py`): UDP broadcasting, race conditions, fork resolution, disconnect recovery.

### 🚀 High-Concurrency Infrastructure **COMPLETED**

- **Phase 2 Milestone:** Implemented high-concurrency network stress testing and node churn resilience:
  - `scripts/stress_test_network.py`: Simulates 10 concurrent UDP worker nodes with dynamic churn (disconnect 3 mid-round, attach 2 new during aggregation).
  - `tests/test_network_churn.py`: Tests rapid peer disconnect/reconnect without payload loss, out-of-order UDP packet handling, ledger state recovery for re-joining nodes.
  - Zero orphaned state deadlocks confirmed; all active peers maintain identical block heights through churn events.
  - Audit log remains stable and uninterrupted during node turnover.

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
