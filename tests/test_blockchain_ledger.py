"""Unit tests for the Permissioned Blockchain Ledger.

Verifies genesis block creation, transaction logging (ACCEPTED / REJECTED),
chain tampering detection via is_chain_valid(), and JSON audit log export.

Run with:  python -m unittest discover -s tests -p "test_blockchain_ledger.py" -v
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.blockchain.ledger import Block, PermissionedLedger


# ------------------------------------------------------------------
class TestGenesisBlock(unittest.TestCase):
    """Verify genesis block creation and initial chain state."""

    def test_genesis_block_created_on_instantiation(self):
        """A new ledger must contain exactly one genesis block (index 0)."""
        ledger = PermissionedLedger(difficulty=1)
        self.assertEqual(len(ledger.chain), 1)
        genesis = ledger.chain[0]
        self.assertEqual(genesis.index, 0)
        self.assertEqual(genesis.previous_hash, "0" * 64)
        # Genesis must have at least one transaction
        self.assertGreater(len(genesis.transactions), 0)
        tx = genesis.transactions[0]
        self.assertEqual(tx["client_id"], 0)
        self.assertEqual(tx["status"], "GENESIS")

    def test_genesis_block_hash_is_valid(self):
        """The genesis block's computed hash must match its stored _hash."""
        ledger = PermissionedLedger(difficulty=1)
        genesis = ledger.chain[0]
        # The stored hash should be valid
        self.assertIsNotNone(genesis._hash)
        self.assertEqual(len(genesis._hash), 64)
        # Recomputing must yield the same value
        self.assertEqual(genesis.calculate_hash(), genesis._hash)

    def test_genesis_mines_with_difficulty(self):
        """Genesis block hash must start with `difficulty` leading zeros."""
        ledger = PermissionedLedger(difficulty=3)
        genesis = ledger.chain[0]
        self.assertTrue(genesis._hash.startswith("0" * 3))


# ------------------------------------------------------------------
class TestTransactionLogging(unittest.TestCase):
    """Verify transaction recording for accepted and rejected model updates."""

    def setUp(self):
        self.ledger = PermissionedLedger(difficulty=1)
        self.sample_hash = "a1b2c3d4e5f6" * 5  # 30-char hex string

    def test_accept_transaction_logged_correctly(self):
        """ACCEPTED transactions must record client_id, model_hash, trust_score."""
        self.ledger.record_transaction(
            client_id=1,
            model_hash=self.sample_hash,
            trust_score=0.95,
            status="ACCEPTED",
        )
        self.assertEqual(len(self.ledger.pending_transactions), 1)
        tx = self.ledger.pending_transactions[0]
        self.assertEqual(tx["client_id"], 1)
        self.assertEqual(tx["model_hash"], self.sample_hash)
        self.assertAlmostEqual(tx["trust_score"], 0.95, places=6)
        self.assertEqual(tx["status"], "ACCEPTED")

    def test_reject_transaction_logged_correctly(self):
        """REJECTED transactions must record status as REJECTED."""
        self.ledger.record_transaction(
            client_id=2,
            model_hash=self.sample_hash,
            trust_score=0.30,
            status="REJECTED",
        )
        tx = self.ledger.pending_transactions[0]
        self.assertEqual(tx["status"], "REJECTED")

    def test_mining_creates_new_block_with_transactions(self):
        """mine_pending_block must create a new block and clear pending pool."""
        for cid in range(3):
            self.ledger.record_transaction(
                client_id=cid,
                model_hash=f"hash_{cid}",
                trust_score=0.8 - cid * 0.1,
                status="ACCEPTED",
            )

        new_block = self.ledger.mine_pending_block()
        # New block should be index 1 (after genesis)
        self.assertEqual(new_block.index, 1)
        self.assertEqual(len(self.ledger.chain), 2)
        # Pending pool must be cleared
        self.assertEqual(len(self.ledger.pending_transactions), 0)
        # Block transactions must match what was pending
        self.assertEqual(
            len(new_block.transactions),
            sum(1 for _ in range(3)),
        )

    def test_mine_empty_pending_raises_error(self):
        """Mining with no pending transactions must raise ValueError."""
        self.ledger.pending_transactions.clear()  # genesis already mined
        with self.assertRaises(ValueError):
            self.ledger.mine_pending_block()


# ------------------------------------------------------------------
class TestChainTamperingDetection(unittest.TestCase):
    """Verify is_chain_valid detects tampering in block data or hashes."""

    def test_valid_chain_returns_true(self):
        """A freshly built chain must pass validation."""
        ledger = PermissionedLedger(difficulty=1)
        for i in range(3):
            ledger.record_transaction(
                client_id=i,
                model_hash=f"tx_{i}",
                trust_score=0.9,
                status="ACCEPTED",
            )
        ledger.mine_pending_block()
        self.assertTrue(ledger.is_chain_valid())

    def test_modified_block_data_detected(self):
        """Changing a transaction in an existing block must invalidate the chain."""
        ledger = PermissionedLedger(difficulty=1)
        for i in range(3):
            ledger.record_transaction(
                client_id=i,
                model_hash=f"tx_{i}",
                trust_score=0.9,
                status="ACCEPTED",
            )
        ledger.mine_pending_block()

        # Tamper: modify a transaction in block 1
        block_1 = ledger.chain[1]
        original_tx_count = len(block_1.transactions)
        if block_1.transactions:
            block_1.transactions[0]["client_id"] = 999  # tamper

        self.assertFalse(ledger.is_chain_valid())
        # Restore for other tests (if needed)
        if block_1.transactions:
            block_1.transactions[0]["client_id"] = original_tx_count

    def test_modified_block_hash_detected(self):
        """Corrupting a block's stored hash must invalidate the chain."""
        ledger = PermissionedLedger(difficulty=1)
        # Add a transaction so there's something to mine
        ledger.record_transaction(
            client_id=99,
            model_hash="fake_hash",
            trust_score=0.5,
            status="REJECTED",
        )
        ledger.mine_pending_block()  # creates block 1

        # Tamper: overwrite the stored hash of block 1
        ledger.chain[1]._hash = "0" * 64
        self.assertFalse(ledger.is_chain_valid())

    def test_broken_link_detected(self):
        """Changing a block's previous_hash to point to wrong parent must fail."""
        ledger = PermissionedLedger(difficulty=1)
        for i in range(3):
            ledger.record_transaction(
                client_id=i,
                model_hash=f"tx_{i}",
                trust_score=0.9,
                status="ACCEPTED",
            )
        ledger.mine_pending_block()

        # Tamper: change block 1's previous_hash to all zeros
        ledger.chain[1].previous_hash = "0" * 64
        self.assertFalse(ledger.is_chain_valid())


# ------------------------------------------------------------------
class TestAuditLogExport(unittest.TestCase):
    """Verify JSON audit log export to results/blockchain_audit_log.json."""

    def test_export_creates_json_file(self):
        """export_audit_log_json must write a valid JSON file at the target path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger = PermissionedLedger(
                difficulty=1,
                audit_log_path=os.path.join(tmpdir, "audit.json"),
            )
            ledger.record_transaction(
                client_id=1,
                model_hash="abc123",
                trust_score=0.85,
                status="ACCEPTED",
            )
            ledger.mine_pending_block()

            output_path = ledger.export_audit_log_json()
            self.assertTrue(output_path.exists())

    def test_export_contains_chain_data(self):
        """Exported JSON must contain chain_length, is_valid, and blocks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger = PermissionedLedger(
                difficulty=1,
                audit_log_path=os.path.join(tmpdir, "audit.json"),
            )
            for i in range(4):
                status = "ACCEPTED" if i % 2 == 0 else "REJECTED"
                ledger.record_transaction(
                    client_id=i + 1,
                    model_hash=f"hash_{i}",
                    trust_score=round(0.9 - i * 0.05, 4),
                    status=status,
                )
            ledger.mine_pending_block()

            output_path = ledger.export_audit_log_json()
            with open(output_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.assertEqual(data["chain_length"], 2)  # genesis + mined block
            self.assertTrue(data["is_valid"])
            self.assertIn("blocks", data)
            self.assertEqual(len(data["blocks"]), 2)

    def test_export_contains_block_details(self):
        """Each exported block must include hash, index, transactions, nonce."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger = PermissionedLedger(
                difficulty=1,
                audit_log_path=os.path.join(tmpdir, "audit.json"),
            )
            ledger.record_transaction(
                client_id=5,
                model_hash="test_hash",
                trust_score=0.75,
                status="REJECTED",
            )
            ledger.mine_pending_block()

            output_path = ledger.export_audit_log_json()
            with open(output_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            block_1 = data["blocks"][1]  # mined block (index 1)
            self.assertEqual(block_1["index"], 1)
            self.assertIn("hash", block_1)
            self.assertIn("transactions", block_1)
            self.assertIsInstance(block_1.get("nonce"), int)
            self.assertGreaterEqual(len(block_1["hash"]), 64)


# ------------------------------------------------------------------
class TestChainSummary(unittest.TestCase):
    """Verify get_chain_summary returns correct metadata."""

    def test_summary_contains_required_keys(self):
        """Summary must include chain_length, is_valid, latest_block_index."""
        ledger = PermissionedLedger(difficulty=1)
        summary = ledger.get_chain_summary()
        self.assertIn("chain_length", summary)
        self.assertIn("is_valid", summary)
        self.assertIn("latest_block_index", summary)
        self.assertIn("pending_transactions", summary)

    def test_summary_after_mining(self):
        """Summary must reflect chain state after mining a block."""
        ledger = PermissionedLedger(difficulty=1)
        for i in range(3):
            ledger.record_transaction(
                client_id=i,
                model_hash=f"tx_{i}",
                trust_score=0.9,
                status="ACCEPTED",
            )
        ledger.mine_pending_block()

        summary = ledger.get_chain_summary()
        self.assertEqual(summary["chain_length"], 2)
        self.assertEqual(summary["latest_block_index"], 1)
        self.assertTrue(summary["is_valid"])


if __name__ == "__main__":
    unittest.main()
