"""Consensus Engine for FedTrustChain.

Implements Phase 3 Step 1 of the FedTrustChain architecture:
- Proof-of-Work (PoW) mining with dynamic target difficulty (default = 3 leading zeros).
- ValidatorRegistry for node signature verification - only accepted client updates
  (T_i >= 0.50) may propose or sign candidate blocks.
- ChainValidator that detects tamper events (modified transactions, invalid previous
  hash links, invalid nonces) and triggers automatic fork resolution to the longest
  valid chain.

This module is designed as a drop-in companion to src.blockchain.ledger.Block and
PermissionedLedger, providing consensus-level semantics on top of the existing
permissioned ledger infrastructure.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple


# ------------------------------------------------------------------
@dataclass
class NodeSignature:
    """Cryptographic signature record for a participating node.

    Attributes
    ----------
    node_id : int
        Unique identifier of the client node.
    public_key_hash : str
        SHA-256 hash representing the node's public key (lightweight identity).
    trust_score : float
        Current dynamic trust score T_i in [0.0, 1.0].
    is_authorized : bool
        Whether the node may propose/sign blocks (T_i >= min_trust_threshold).
    block_signatures : list[str]
        Hashes of blocks this node has signed.
    """

    node_id: int
    public_key_hash: str
    trust_score: float = 1.0
    is_authorized: bool = True
    block_signatures: List[str] = field(default_factory=list)


# ------------------------------------------------------------------
class ProofOfWork:
    """Dynamic target difficulty mining for new block creation.

    Parameters
    ----------
    default_difficulty : int, default=3
        Number of leading zero hex characters required in the SHA-256 hash.

    Attributes
    ----------
    default_difficulty : int
        The configured default mining difficulty.
    total_mined_blocks : int
        Count of blocks successfully mined by this instance.
    """

    def __init__(self, default_difficulty: int = 3) -> None:
        self.default_difficulty = default_difficulty
        self.total_mined_blocks: int = 0

    # ------------------------------------------------------------------
    @staticmethod
    def _hash_block_content(block_data: Dict[str, Any]) -> str:
        """Compute SHA-256 hex digest over a deterministic block dict."""
        content_str = json.dumps(block_data, sort_keys=True, default=str)
        return hashlib.sha256(content_str.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    def mine_block(
        self,
        index: int,
        timestamp: str,
        transactions: List[Dict[str, Any]],
        previous_hash: str,
        difficulty: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Mine a new block by finding a nonce whose hash starts with *difficulty* zeros.

        Returns dict with keys: index, timestamp, transactions, previous_hash,
        nonce, hash, difficulty. Raises RuntimeError on failure after 1M attempts.
        """
        if difficulty is None:
            difficulty = self.default_difficulty

        target = "0" * difficulty
        max_attempts = 1_000_000
        nonce = 0

        while nonce < max_attempts:
            block_data = {
                "index": index,
                "timestamp": timestamp,
                "transactions": transactions,
                "previous_hash": previous_hash,
                "nonce": nonce,
            }
            block_hash = self._hash_block_content(block_data)
            if block_hash.startswith(target):
                self.total_mined_blocks += 1
                return {
                    "index": index,
                    "timestamp": timestamp,
                    "transactions": transactions,
                    "previous_hash": previous_hash,
                    "nonce": nonce,
                    "hash": block_hash,
                    "difficulty": difficulty,
                }
            nonce += 1

        raise RuntimeError(
            f"Mining failed after {max_attempts} attempts (difficulty={difficulty})."
        )

    # ------------------------------------------------------------------
    @staticmethod
    def verify_mined_block(block: Dict[str, Any], difficulty: Optional[int] = None) -> bool:
        """Verify that a mined block has a valid proof-of-work hash."""
        diff = difficulty or block.get("difficulty", 3)
        target = "0" * diff
        return block["hash"].startswith(target)


# ------------------------------------------------------------------
class ValidatorRegistry:
    """Node signature verification for block proposal and signing.

    Only nodes with trust score T_i >= min_trust_threshold (default 0.50) are
    authorized to propose or sign candidate blocks.
    """

    def __init__(self, min_trust_threshold: float = 0.50) -> None:
        self.min_trust_threshold = min_trust_threshold
        self._registry: Dict[int, NodeSignature] = {}

    # ------------------------------------------------------------------
    def register_node(
        self,
        node_id: int,
        public_key_hash: str,
        trust_score: float = 1.0,
    ) -> NodeSignature:
        """Register a new node in the validator registry."""
        is_authorized = trust_score >= self.min_trust_threshold
        sig = NodeSignature(
            node_id=node_id,
            public_key_hash=public_key_hash,
            trust_score=trust_score,
            is_authorized=is_authorized,
        )
        self._registry[node_id] = sig
        return sig

    # ------------------------------------------------------------------
    def update_trust_score(self, node_id: int, new_score: float) -> None:
        """Update a registered node's trust score and re-evaluate authorization."""
        if node_id not in self._registry:
            raise KeyError(f"Node {node_id} not registered.")
        sig = self._registry[node_id]
        sig.trust_score = new_score
        sig.is_authorized = new_score >= self.min_trust_threshold

    # ------------------------------------------------------------------
    def verify_proposer(self, node_id: int) -> bool:
        """Check whether a node is authorized to propose a block."""
        sig = self._registry.get(node_id)
        return sig is not None and sig.is_authorized

    # ------------------------------------------------------------------
    def sign_block(self, node_id: int, block_hash: str) -> bool:
        """Record a node's signature on a mined block. Returns True if authorized."""
        sig = self._registry.get(node_id)
        if sig is None or not sig.is_authorized:
            return False
        sig.block_signatures.append(block_hash)
        return True

    # ------------------------------------------------------------------
    def verify_block_signature(self, node_id: int, block_hash: str) -> bool:
        """Verify that a node has signed a specific block."""
        sig = self._registry.get(node_id)
        return sig is not None and block_hash in sig.block_signatures

    # ------------------------------------------------------------------
    def get_authorized_nodes(self) -> List[NodeSignature]:
        """Return all nodes currently authorized to propose/sign blocks."""
        return [s for s in self._registry.values() if s.is_authorized]


# ------------------------------------------------------------------
class ChainValidator:
    """Chain integrity validation and automatic fork resolution.

    Detects three categories of tamper events:
    1. Modified transaction payloads (block content hash mismatch).
    2. Invalid previous hash links (chain linkage broken).
    3. Invalid nonces (proof-of-work verification failure).
    """

    def __init__(self, difficulty: int = 3) -> None:
        self.difficulty = difficulty
        self._tamper_events: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    @staticmethod
    def _block_to_dict(block_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a block record to the canonical dict shape for hashing."""
        return {
            "index": block_dict["index"],
            "timestamp": block_dict["timestamp"],
            "transactions": block_dict.get("transactions", []),
            "previous_hash": block_dict["previous_hash"],
            "nonce": block_dict["nonce"],
        }

    # ------------------------------------------------------------------
    def validate_chain(
        self, chain: List[Dict[str, Any]]
    ) -> Tuple[bool, List[str]]:
        """Validate the integrity of an entire blockchain.

        Returns (valid, errors) where valid is True if every block passes all checks.
        """
        errors: List[str] = []

        for i, block in enumerate(chain):
            # --- Check 1: PoW hash verification ---------------------------
            recorded_hash = block.get("hash", "")
            expected_hash = ProofOfWork._hash_block_content(
                self._block_to_dict(block)
            )
            if recorded_hash != expected_hash:
                errors.append(
                    f"Block {i}: hash mismatch - possible transaction "
                    f"payload modification or nonce invalidation."
                )

            # --- Check 2: PoW difficulty check ----------------------------
            block_diff = block.get("difficulty", self.difficulty)
            if not ProofOfWork.verify_mined_block(block, block_diff):
                errors.append(
                    f"Block {i}: proof-of-work verification failed "
                    f"(difficulty={block_diff}). Invalid nonce."
                )

            # --- Check 3: Chain linkage (previous hash) -------------------
            if i > 0:
                prev_hash = chain[i - 1].get("hash", "")
                current_prev = block.get("previous_hash", "")
                if current_prev != prev_hash:
                    errors.append(
                        f"Block {i}: invalid previous hash link "
                        f"(expected={prev_hash[:16]}..., got="
                        f"{current_prev[:16]})"
                    )

        return len(errors) == 0, errors

    # ------------------------------------------------------------------
    def detect_tamper(
        self, chain: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Detect and record tamper events in the given chain."""
        valid, errors = self.validate_chain(chain)
        events: List[Dict[str, Any]] = []

        if not valid:
            ts = datetime.now(timezone.utc).isoformat()
            for i, error in enumerate(errors):
                if "hash mismatch" in error or "payload modification" in error:
                    event_type = "PAYLOAD_TAMPER"
                elif "previous hash link" in error:
                    event_type = "CHAIN_LINK_BREACH"
                elif "proof-of-work" in error or "nonce" in error:
                    event_type = "INVALID_NONCE"
                else:
                    event_type = "UNKNOWN_TAMPER"

                events.append({
                    "block_index": i,
                    "event_type": event_type,
                    "description": error,
                    "timestamp": ts,
                })

        self._tamper_events.extend(events)
        return events

    # ------------------------------------------------------------------
    def resolve_fork(
        self, current_chain: List[Dict[str, Any]],
        candidate_chain: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """Automatic fork resolution: adopt the longest valid chain.

        Returns (best_chain, forked) where forked is True if candidate replaced current.
        """
        curr_valid, _ = self.validate_chain(current_chain)
        cand_valid, _ = self.validate_chain(candidate_chain)

        if not curr_valid and not cand_valid:
            return current_chain, False
        if not cand_valid and curr_valid:
            return current_chain, False
        if cand_valid and len(candidate_chain) > len(current_chain):
            return candidate_chain, True
        return current_chain, False
