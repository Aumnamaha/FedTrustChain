"""Unit tests for the Consensus Engine (Phase 3 Step 1).

Tests:
- ProofOfWork mining speed and valid hash pattern matching.
- ValidatorRegistry node authorization and signature verification.
- ChainValidator tamper detection (modified transactions, invalid hashes, invalid nonces).
- Dynamic fork resolution when a longer valid chain is submitted.

Run with:  python -m unittest discover -s tests -p "test_consensus.py" -v
"""

import json
import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.blockchain.consensus import ProofOfWork, ValidatorRegistry, ChainValidator, NodeSignature


# ------------------------------------------------------------------
class TestProofOfWork(unittest.TestCase):
    """Verify PoW mining speed, hash patterns, and verification."""

    def test_mining_with_difficulty_1(self):
        """Mining at difficulty=1 should complete in < 5 seconds."""
        pox = ProofOfWork(default_difficulty=3)
        txs = [{"client_id": 1, "model_hash": "abc123", "trust_score": 0.9}]
        start = time.time()
        block = pox.mine_block(
            index=1, timestamp="2025-01-01T00:00:00+00:00",
            transactions=txs, previous_hash="0" * 64, difficulty=3,
        )
        elapsed = time.time() - start
        self.assertLess(elapsed, 5.0, f"Mining took {elapsed:.2f}s — too slow")

    def test_mined_block_hash_pattern(self):
        """Mined block hash must start with `difficulty` leading zeros."""
        pox = ProofOfWork(default_difficulty=3)
        txs = [{"client_id": 1, "model_hash": "abc", "trust_score": 0.9}]
        block = pox.mine_block(
            index=2, timestamp="2025-01-01T00:00:00+00:00",
            transactions=txs, previous_hash="deadbeef" * 8, difficulty=3,
        )
        self.assertTrue(block["hash"].startswith("000"))

    def test_verify_mined_block(self):
        """verify_mined_block must return True for a valid block."""
        pox = ProofOfWork(default_difficulty=3)
        txs = [{"client_id": 1, "model_hash": "abc", "trust_score": 0.9}]
        block = pox.mine_block(
            index=3, timestamp="2025-06-01T12:00:00+00:00",
            transactions=txs, previous_hash="cafe" * 8, difficulty=3,
        )
        self.assertTrue(ProofOfWork.verify_mined_block(block))

    def test_verify_invalid_nonce(self):
        """verify_mined_block must return False for a tampered nonce."""
        block = {
            "index": 10, "timestamp": "2025-01-01T00:00:00+00:00",
            "transactions": [], "previous_hash": "0" * 64,
            "nonce": 999999, "hash": "zzzz000xxx", "difficulty": 3,
        }
        self.assertFalse(ProofOfWork.verify_mined_block(block))

    def test_total_mined_blocks_counter(self):
        """total_mined_blocks should increment after each successful mine."""
        pox = ProofOfWork(default_difficulty=1)
        txs = [{"client_id": 1, "model_hash": "a", "trust_score": 0.9}]
        for i in range(5):
            pox.mine_block(i, "2025-01-01T00:00:00+00:00", txs, "0" * 64)
        self.assertEqual(pox.total_mined_blocks, 5)

    def test_default_difficulty_is_3(self):
        """Default mining difficulty must be 3 leading zeros."""
        pox = ProofOfWork()
        self.assertEqual(pox.default_difficulty, 3)


# ------------------------------------------------------------------
class TestValidatorRegistry(unittest.TestCase):
    """Verify node registration, trust scoring, and signature verification."""

    def setUp(self):
        self.registry = ValidatorRegistry(min_trust_threshold=0.50)

    def test_register_authorized_node(self):
        """Node with T_i >= 0.50 must be authorized."""
        sig = self.registry.register_node(1, "pk_hash_1", trust_score=0.9)
        self.assertTrue(sig.is_authorized)
        self.assertTrue(self.registry.verify_proposer(1))

    def test_register_unauthorized_node(self):
        """Node with T_i < 0.50 must NOT be authorized."""
        sig = self.registry.register_node(2, "pk_hash_2", trust_score=0.30)
        self.assertFalse(sig.is_authorized)
        self.assertFalse(self.registry.verify_proposer(2))

    def test_update_trust_score_above_threshold(self):
        """Updating T_i from below to above threshold must authorize the node."""
        self.registry.register_node(3, "pk_hash_3", trust_score=0.40)
        self.assertFalse(self.registry.verify_proposer(3))
        self.registry.update_trust_score(3, 0.60)
        self.assertTrue(self.registry.verify_proposer(3))

    def test_update_trust_score_below_threshold(self):
        """Updating T_i from above to below threshold must de-authorize the node."""
        self.registry.register_node(4, "pk_hash_4", trust_score=0.80)
        self.assertTrue(self.registry.verify_proposer(4))
        self.registry.update_trust_score(4, 0.30)
        self.assertFalse(self.registry.verify_proposer(4))

    def test_sign_block_rejects_unauthorized(self):
        """Unauthorized nodes must be rejected when signing a block."""
        self.registry.register_node(5, "pk_hash_5", trust_score=0.20)
        result = self.registry.sign_block(5, "block_hash_xyz")
        self.assertFalse(result)

    def test_sign_block_accepts_authorized(self):
        """Authorized nodes must successfully sign a block."""
        self.registry.register_node(6, "pk_hash_6", trust_score=0.75)
        result = self.registry.sign_block(6, "block_hash_xyz")
        self.assertTrue(result)

    def test_verify_block_signature(self):
        """verify_block_signature must match a previously signed block hash."""
        self.registry.register_node(7, "pk_hash_7", trust_score=0.90)
        bhash = "abc123def456"
        self.registry.sign_block(7, bhash)
        self.assertTrue(self.registry.verify_block_signature(7, bhash))

    def test_verify_nonexistent_signature(self):
        """verify_block_signature must return False for an unsigned block."""
        self.registry.register_node(8, "pk_hash_8", trust_score=0.90)
        self.assertFalse(self.registry.verify_block_signature(8, "no_such_block"))

    def test_get_authorized_nodes(self):
        """get_authorized_nodes must return only nodes with T_i >= threshold."""
        self.registry.register_node(10, "pk_10", trust_score=0.9)
        self.registry.register_node(11, "pk_11", trust_score=0.3)
        self.registry.register_node(12, "pk_12", trust_score=0.55)
        authorized = self.registry.get_authorized_nodes()
        ids = {s.node_id for s in authorized}
        self.assertEqual(ids, {10, 12})


# ------------------------------------------------------------------
class TestChainValidator(unittest.TestCase):
    """Verify chain integrity checks and tamper detection."""

    def _make_valid_block(self, index, txs=None, prev_hash="0" * 64, diff=3):
        """Helper: mine a valid block for testing."""
        pox = ProofOfWork(default_difficulty=diff)
        return pox.mine_block(
            index=index, timestamp=f"2025-01-{index:02d}T00:00:00+00:00",
            transactions=txs or [], previous_hash=prev_hash, difficulty=diff,
        )

    def test_valid_chain_passes(self):
        """A chain of properly linked blocks must validate as True."""
        b1 = self._make_valid_block(1)
        b2 = self._make_valid_block(2, prev_hash=b1["hash"])
        chain = [b1, b2]

        cv = ChainValidator(difficulty=3)
        valid, errors = cv.validate_chain(chain)
        self.assertTrue(valid)
        self.assertEqual(len(errors), 0)

    def test_tampered_transactions_detected(self):
        """Modifying a transaction payload must be detected as tamper."""
        b1 = self._make_valid_block(1)
        b2 = self._make_valid_block(2, prev_hash=b1["hash"])
        # Tamper: modify transactions in block 2
        b2_tampered = dict(b2)
        b2_tampered["transactions"] = [{"client_id": 999, "model_hash": "tampered"}]

        cv = ChainValidator(difficulty=3)
        valid, errors = cv.validate_chain([b1, b2_tampered])
        self.assertFalse(valid)
        self.assertGreater(len(errors), 0)
        tamper_events = cv.detect_tamper([b1, b2_tampered])
        self.assertTrue(any(e["event_type"] == "PAYLOAD_TAMPER" for e in tamper_events))

    def test_invalid_previous_hash_link(self):
        """A block with wrong previous_hash must be rejected."""
        b1 = self._make_valid_block(1)
        # Block 2 has an invalid previous hash (not matching b1's hash)
        bad_prev = "0" * 32 + "deadbeef" * 8
        chain = [b1, {"index": 2, "timestamp": "2025-01-02T00:00:00+00:00",
                       "transactions": [], "previous_hash": bad_prev,
                       "nonce": b1["nonce"], "hash": b1["hash"], "difficulty": 3}]

        cv = ChainValidator(difficulty=3)
        valid, errors = cv.validate_chain(chain)
        self.assertFalse(valid)
        self.assertTrue(any("invalid previous hash link" in e for e in errors))

    def test_invalid_nonce_detected(self):
        """A block with wrong nonce (hash doesn't start with required zeros) must fail."""
        b1 = self._make_valid_block(1, diff=3)
        # Create a fake block 2 with invalid PoW hash
        bad_block = {
            "index": 2, "timestamp": "2025-01-02T00:00:00+00:00",
            "transactions": [], "previous_hash": b1["hash"],
            "nonce": 42, "hash": "zzzz_non_valid" * 4, "difficulty": 3,
        }
        cv = ChainValidator(difficulty=3)
        valid, errors = cv.validate_chain([b1, bad_block])
        self.assertFalse(valid)
        # Should detect both hash mismatch and PoW failure

    def test_empty_chain(self):
        """An empty chain should be considered valid."""
        cv = ChainValidator(difficulty=3)
        valid, errors = cv.validate_chain([])
        self.assertTrue(valid)
        self.assertEqual(len(errors), 0)


# ------------------------------------------------------------------
class TestForkResolution(unittest.TestCase):
    """Verify automatic fork resolution to the longest valid chain."""

    def _build_chain(self, length, diff=3):
        """Build a list of `length` linked valid blocks."""
        pox = ProofOfWork(default_difficulty=diff)
        prev_hash = "0" * 64
        chain = []
        for i in range(1, length + 1):
            b = pox.mine_block(i, f"2025-01-{i:02d}T00:00:00+00:00", [], prev_hash, diff)
            chain.append(b)
            prev_hash = b["hash"]
        return chain

    def test_longer_valid_chain_replaces(self):
        """A longer valid candidate chain must replace the current shorter one."""
        cv = ChainValidator(difficulty=3)
        short_chain = self._build_chain(2)
        long_chain = self._build_chain(5)

        best, forked = cv.resolve_fork(short_chain, long_chain)
        self.assertTrue(forked)
        self.assertEqual(len(best), 5)
        self.assertIs(best, long_chain)

    def test_shorter_valid_chain_kept(self):
        """A shorter valid candidate must NOT replace a longer valid current chain."""
        cv = ChainValidator(difficulty=3)
        long_chain = self._build_chain(5)
        short_chain = self._build_chain(2)

        best, forked = cv.resolve_fork(long_chain, short_chain)
        self.assertFalse(forked)
        self.assertEqual(len(best), 5)

    def test_invalid_candidate_keeps_current(self):
        """An invalid candidate chain must not replace a valid current chain."""
        cv = ChainValidator(difficulty=3)
        valid_chain = self._build_chain(3)
        # Tamper with the candidate
        bad_block = dict(valid_chain[0])
        bad_block["hash"] = "tampered_hash" * 4
        invalid_candidate = [bad_block]

        best, forked = cv.resolve_fork(valid_chain, invalid_candidate)
        self.assertFalse(forked)

    def test_both_invalid_keeps_current(self):
        """When both chains are invalid, current chain is kept."""
        cv = ChainValidator(difficulty=3)
        valid_chain = self._build_chain(2)
        # Tamper with the candidate
        bad_block = dict(valid_chain[0])
        bad_block["hash"] = "bad" * 16 + "tampered" * 8
        invalid_candidate = [bad_block]

        best, forked = cv.resolve_fork(valid_chain, invalid_candidate)
        self.assertFalse(forked)


if __name__ == "__main__":
    unittest.main()
