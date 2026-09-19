"""SHA-256 Merkle commitments over leaderboard snapshots.

Encoding rules (shared by ``GET /leaderboard/commitment`` and
``GET /leaderboard/proof/{participant_id}``):

* Leaves are the snapshot's leaderboard entries in leaderboard order
  (total_points descending, participant_id ascending); rank 1 is leaf 0.
* Each entry is encoded as canonical JSON — keys sorted, no whitespace,
  non-ASCII escaped (``json.dumps(payload, sort_keys=True,
  separators=(",", ":"))``) — over exactly these fields::

      {"rank": int, "participant_id": str, "total_points": int,
       "event_ids": [str, ...]}

  where ``event_ids`` lists the entry's effective event ids in revision
  order (the same order as ``effective_events`` in ``GET /leaderboard``).
  The leaf digest is SHA-256 of the UTF-8 encoding.
* An internal node is SHA-256 of the two child digests concatenated as
  raw bytes (``left || right``). When a level holds an odd number of
  nodes, the last node is paired with itself.
* The root of an empty leaderboard is SHA-256 of the empty byte string.
"""

from __future__ import annotations

import hashlib
import json

from .schemas import CommitmentEntry, LeaderboardEntry, ProofSibling

EMPTY_ROOT: bytes = hashlib.sha256(b"").digest()
"""Fixed Merkle root of the empty leaderboard: SHA-256 of no bytes."""


def commitment_entry(entry: LeaderboardEntry) -> CommitmentEntry:
    """Project a leaderboard entry onto its deterministic leaf payload."""
    return CommitmentEntry(
        rank=entry.rank,
        participant_id=entry.participant_id,
        total_points=entry.total_points,
        event_ids=[ev.event_id for ev in entry.effective_events],
    )


def encode_entry(entry: CommitmentEntry) -> bytes:
    """Canonical JSON encoding of a leaf payload (deterministic)."""
    payload = {
        "rank": entry.rank,
        "participant_id": entry.participant_id,
        "total_points": entry.total_points,
        "event_ids": list(entry.event_ids),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def leaf_digest(entry: CommitmentEntry) -> bytes:
    """SHA-256 of the entry's canonical encoding."""
    return hashlib.sha256(encode_entry(entry)).digest()


def _pair_up(level: list[bytes]) -> list[bytes]:
    """One Merkle round: hash adjacent pairs; an odd last node pairs with itself."""
    parents: list[bytes] = []
    for i in range(0, len(level), 2):
        left = level[i]
        right = level[i + 1] if i + 1 < len(level) else left
        parents.append(hashlib.sha256(left + right).digest())
    return parents


def merkle_root(leaves: list[bytes]) -> bytes:
    """Root digest of the tree over ``leaves`` (empty list -> ``EMPTY_ROOT``)."""
    if not leaves:
        return EMPTY_ROOT
    level = list(leaves)
    while len(level) > 1:
        level = _pair_up(level)
    return level[0]


def merkle_proof(leaves: list[bytes], index: int) -> list[ProofSibling]:
    """Per-level siblings for ``leaves[index]``, from the leaf level up.

    ``position`` tells on which side of the running node the sibling
    hashes in: ``"left"`` means ``parent = SHA-256(sibling || node)``,
    ``"right"`` means ``parent = SHA-256(node || sibling)``. A duplicated
    odd-level last node appears as its own ``"right"`` sibling.
    """
    proof: list[ProofSibling] = []
    level = list(leaves)
    idx = index
    while len(level) > 1:
        if idx % 2 == 0:
            sibling = level[idx + 1] if idx + 1 < len(level) else level[idx]
            proof.append(ProofSibling(position="right", hash=sibling.hex()))
        else:
            proof.append(ProofSibling(position="left", hash=level[idx - 1].hex()))
        level = _pair_up(level)
        idx //= 2
    return proof


def fold_proof(leaf: bytes, siblings: list[ProofSibling]) -> bytes:
    """Recompute the root from a leaf digest and its proof (client algorithm)."""
    node = leaf
    for sib in siblings:
        digest = bytes.fromhex(sib.hash)
        if sib.position == "left":
            node = hashlib.sha256(digest + node).digest()
        else:
            node = hashlib.sha256(node + digest).digest()
    return node
