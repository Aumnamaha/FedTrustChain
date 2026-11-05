"""Permissioned Blockchain Ledger for FedTrustChain.

Provides an immutable audit trail logging model payload hashes, node IDs,
dynamic trust scores, and aggregation decisions across federated learning rounds.
"""

from src.blockchain.ledger import Block, PermissionedLedger

__all__ = ["Block", "PermissionedLedger"]
