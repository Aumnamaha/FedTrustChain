"""Permissioned Blockchain Ledger for FedTrustChain.

Implements a minimal proof-of-work blockchain to create an immutable audit trail
logging SHA-256 model payload hashes, node IDs, dynamic trust scores, and
aggregation decisions across federated learning rounds.

This module implements Phase 2 Step 2 of the FedTrustChain architecture:
permissioned ledger audit logging for model hashes and trust transactions.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# ------------------------------------------------------------------
@dataclass
class Block:
    """A single block in the FedTrustChain ledger.

    Attributes
    ----------
    index : int
        Block number in the chain (genesis = 0).
    timestamp : str
        ISO-8601 UTC timestamp of block creation.
    transactions : list[dict]
        List of client model evaluation records logged in this round.
    previous_hash : str
        SHA-256 hex digest of the preceding block's hash.
    nonce : int
        Proof-of-work counter incremented during mining.
    _hash : str | None
        Cached SHA-256 hash of this block (computed after mining).
    """

    index: int
    timestamp: str
    transactions: List[Dict[str, Any]] = field(default_factory=list)
    previous_hash: str = "0" * 64
    nonce: int = 0
    _hash: Optional[str] = None

    # ------------------------------------------------------------------
    def calculate_hash(self) -> str:
        """Compute SHA-256 hash over all block fields (deterministic).

        Hashes the JSON-serialized representation of index, timestamp,
        transactions, previous_hash, AND nonce — enabling proof-of-work mining.

        Returns
        -------
        str
            Hex-encoded SHA-256 digest (64 characters).
        """
        block_content = {
            "index": self.index,
            "timestamp": self.timestamp,
            "transactions": self.transactions,
            "previous_hash": self.previous_hash,
            "nonce": self.nonce,
        }
        content_str = json.dumps(block_content, sort_keys=True, default=str)
        return hashlib.sha256(content_str.encode("utf-8")).hexdigest()

    def mine(self, difficulty: int = 2) -> bool:
        """Mine this block by finding a nonce whose hash starts with *difficulty* zeros.

        Parameters
        ----------
        difficulty : int, default=2
            Number of leading zero hex characters required. Max practical limit ~6.

        Returns
        -------
        bool
            True if mining succeeded.

        Raises
        ------
        RuntimeError
            If no valid nonce is found within 1_000_000 attempts (indicates
            corrupted block data or impossibly high difficulty).
        """
        target = "0" * difficulty
        max_attempts = 1_000_000
        for _ in range(max_attempts):
            self._hash = self.calculate_hash()
            if self._hash.startswith(target):
                return True
            self.nonce += 1
        raise RuntimeError(
            f"Mining failed after {max_attempts} attempts (difficulty={difficulty}). "
            "This usually indicates corrupted block data or impossibly high difficulty."
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize block to a plain dict for JSON export."""
        d = asdict(self)
        # Rename _hash → hash in output
        if "_hash" in d:
            d["hash"] = d.pop("_hash")
        return d


# ------------------------------------------------------------------
class PermissionedLedger:
    """Minimal permissioned blockchain for audit logging.

    Parameters
    ----------
    difficulty : int, default=2
        Proof-of-work difficulty (leading zero hex characters).
    audit_log_path : str | Path, optional
        File path to export the full chain as JSON. Defaults to
        ``results/blockchain_audit_log.json`` relative to project root.

    Attributes
    ----------
    chain : list[Block]
        The complete blockchain (genesis block + mined blocks).
    pending_transactions : list[dict]
        Accumulated transactions awaiting inclusion in the next block.
    """

    def __init__(
        self,
        difficulty: int = 2,
        audit_log_path: Optional[str | Path] = None,
    ):
        self.difficulty = difficulty
        self.audit_log_path = (
            Path(audit_log_path) if audit_log_path else Path("results/blockchain_audit_log.json")
        )
        self.chain: List[Block] = []
        self.pending_transactions: List[Dict[str, Any]] = []

        # --- create genesis block -----------------------------------------
        self._create_genesis_block()

    # ------------------------------------------------------------------
    def _create_genesis_block(self):
        """Initialize the chain with a genesis block (index 0)."""
        genesis_tx = {
            "client_id": 0,
            "model_hash": "genesis",
            "trust_score": 1.0,
            "status": "GENESIS",
            "timestamp": self._now_iso(),
        }
        genesis = Block(
            index=0,
            timestamp=self._now_iso(),
            transactions=[genesis_tx],
            previous_hash="0" * 64,
            nonce=0,
        )
        genesis.mine(self.difficulty)
        self.chain.append(genesis)

    @staticmethod
    def _now_iso() -> str:
        """Return current UTC time as ISO-8601 string."""
        return datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    def record_transaction(
        self,
        client_id: int,
        model_hash: str,
        trust_score: float,
        status: str = "ACCEPTED",
    ):
        """Append a single client evaluation event to the pending pool.

        Parameters
        ----------
        client_id : int
            Unique identifier of the client node.
        model_hash : str
            SHA-256 hex digest of the serialized model payload.
        trust_score : float
            Current dynamic trust score T_i at time of evaluation.
        status : str, default="ACCEPTED"
            Either ``"ACCEPTED"`` or ``"REJECTED"``.
        """
        tx = {
            "client_id": client_id,
            "model_hash": model_hash,
            "trust_score": round(trust_score, 6),
            "status": status.upper(),
            "timestamp": self._now_iso(),
        }
        self.pending_transactions.append(tx)

    # ------------------------------------------------------------------
    def mine_pending_block(self) -> Block:
        """Mine a new block containing all pending transactions.

        Returns
        -------
        Block
            The newly mined block appended to the chain.
        """
        if not self.pending_transactions:
            raise ValueError("No pending transactions to mine.")

        prev_block = self.chain[-1]
        new_block = Block(
            index=prev_block.index + 1,
            timestamp=self._now_iso(),
            transactions=list(self.pending_transactions),
            previous_hash=prev_block._hash or prev_block.calculate_hash(),
            nonce=0,
        )
        new_block.mine(self.difficulty)

        self.chain.append(new_block)
        # Clear pending after mining
        self.pending_transactions.clear()
        return new_block

    # ------------------------------------------------------------------
    def is_chain_valid(self) -> bool:
        """Verify hash integrity across all blocks and link hashes.

        Returns
        -------
        bool
            True if every block's hash matches its calculated value AND each
            block's ``previous_hash`` points to the preceding block's hash.
        """
        for i in range(len(self.chain)):
            block = self.chain[i]

            # Verify this block's hash
            expected_hash = block.calculate_hash()
            if (block._hash is None or block._hash != expected_hash):
                return False

            # Verify link to previous block
            if i > 0:
                prev_block = self.chain[i - 1]
                prev_hash = prev_block._hash or prev_block.calculate_hash()
                if block.previous_hash != prev_hash:
                    return False

        return True

    # ------------------------------------------------------------------
    def export_audit_log_json(
        self,
        output_path: Optional[str | Path] = None,
    ) -> Path:
        """Save the full chain state to a JSON file.

        Parameters
        ----------
        output_path : str | Path, optional
            Destination path for the JSON audit log. Overrides the instance
            default if provided.

        Returns
        -------
        Path
            Absolute path of the written file.
        """
        target = Path(output_path) if output_path else self.audit_log_path
        # Ensure parent directory exists
        target.parent.mkdir(parents=True, exist_ok=True)

        chain_data = {
            "chain_length": len(self.chain),
            "is_valid": self.is_chain_valid(),
            "blocks": [block.to_dict() for block in self.chain],
            "pending_transactions_count": len(self.pending_transactions),
        }

        with open(target, "w", encoding="utf-8") as f:
            json.dump(chain_data, f, indent=2, default=str)

        return target.resolve()

    # ------------------------------------------------------------------
    def get_chain_summary(self) -> Dict[str, Any]:
        """Return a concise summary of the current chain state."""
        return {
            "chain_length": len(self.chain),
            "is_valid": self.is_chain_valid(),
            "latest_block_index": self.chain[-1].index if self.chain else 0,
            "latest_block_hash": (
                self.chain[-1]._hash or self.chain[-1].calculate_hash()
            ) if self.chain else None,
            "pending_transactions": len(self.pending_transactions),
        }


# ------------------------------------------------------------------
def create_genesis_block(difficulty: int = 2) -> Block:
    """Convenience function to create a standalone genesis block.

    Parameters
    ----------
    difficulty : int, default=2
        Proof-of-work mining difficulty.

    Returns
    -------
    Block
        A mined genesis block ready for chain initialization.
    """
    tx = {
        "client_id": 0,
        "model_hash": "genesis",
        "trust_score": 1.0,
        "status": "GENESIS",
        "timestamp": PermissionedLedger._now_iso(),
    }
    block = Block(
        index=0,
        timestamp=PermissionedLedger._now_iso(),
        transactions=[tx],
        previous_hash="0" * 64,
        nonce=0,
    )
    block.mine(difficulty)
    return block
