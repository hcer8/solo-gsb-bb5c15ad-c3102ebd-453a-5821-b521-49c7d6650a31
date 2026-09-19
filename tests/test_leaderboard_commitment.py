"""Tests for GET /leaderboard/commitment and GET /leaderboard/proof/{participant_id}."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

import app.db as db_module
from app.main import app

EMPTY_ROOT = hashlib.sha256(b"").hexdigest()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """TestClient backed by an isolated SQLite file."""
    monkeypatch.setattr(db_module, "DB_PATH", str(tmp_path / "commitment-test.db"))
    db_module.init_db()
    return TestClient(app)


def _post(client, event_id, participant_id, points, correction_of=None):
    payload = {
        "event_id": event_id,
        "participant_id": participant_id,
        "points": points,
    }
    if correction_of is not None:
        payload["correction_of"] = correction_of
    r = client.post("/events", json=payload)
    assert r.status_code == 201
    return r.json()


def _ref_leaf(rank, participant_id, total_points, event_ids):
    """Independent reference encoding of one leaf."""
    payload = {
        "rank": rank,
        "participant_id": participant_id,
        "total_points": total_points,
        "event_ids": event_ids,
    }
    enc = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(enc).digest()


def _ref_root(leaves):
    """Independent reference Merkle root (odd last node pairs with itself)."""
    if not leaves:
        return hashlib.sha256(b"").digest()
    level = list(leaves)
    while len(level) > 1:
        parents = []
        for i in range(0, len(level), 2):
            right = level[i + 1] if i + 1 < len(level) else level[i]
            parents.append(hashlib.sha256(level[i] + right).digest())
        level = parents
    return level[0]


def _ref_leaves(lb):
    """Reference leaf digests for a /leaderboard response body."""
    return [
        _ref_leaf(
            e["rank"],
            e["participant_id"],
            e["total_points"],
            [ev["event_id"] for ev in e["effective_events"]],
        )
        for e in lb["entries"]
    ]


def _fold(leaf_hash_hex, siblings):
    """Client-side root recomputation from a proof response."""
    node = bytes.fromhex(leaf_hash_hex)
    for sib in siblings:
        digest = bytes.fromhex(sib["hash"])
        if sib["position"] == "left":
            node = hashlib.sha256(digest + node).digest()
        else:
            node = hashlib.sha256(node + digest).digest()
    return node.hex()


def test_empty_ledger_commitment_is_sha256_of_empty_bytes(client):
    r = client.get("/leaderboard/commitment")
    assert r.status_code == 200
    assert r.json() == {"revision": 0, "leaf_count": 0, "root": EMPTY_ROOT}


def test_empty_ledger_proof_404(client):
    assert client.get("/leaderboard/proof/alice").status_code == 404


def test_commitment_matches_reference_tree(client):
    _post(client, "e1", "alice", 10)
    _post(client, "e2", "bob", 7)
    _post(client, "e3", "carol", 7)
    lb = client.get("/leaderboard").json()
    r = client.get("/leaderboard/commitment")
    assert r.status_code == 200
    body = r.json()
    assert body["revision"] == lb["revision"] == 3
    assert body["leaf_count"] == 3
    assert body["root"] == _ref_root(_ref_leaves(lb)).hex()


def test_single_leaf_root_is_leaf_digest_of_canonical_encoding(client):
    _post(client, "e1", "alice", 10)
    body = client.get("/leaderboard/commitment").json()
    assert body["leaf_count"] == 1
    assert body["root"] == _ref_leaf(1, "alice", 10, ["e1"]).hex()


def test_proof_single_participant_has_empty_siblings(client):
    _post(client, "e1", "alice", 10)
    r = client.get("/leaderboard/proof/alice")
    assert r.status_code == 200
    body = r.json()
    assert body["revision"] == 1
    assert body["leaf_count"] == 1
    assert body["leaf_index"] == 0
    assert body["entry"] == {
        "rank": 1,
        "participant_id": "alice",
        "total_points": 10,
        "event_ids": ["e1"],
    }
    assert body["leaf_hash"] == _ref_leaf(1, "alice", 10, ["e1"]).hex()
    assert body["siblings"] == []
    assert body["root"] == body["leaf_hash"]


def test_every_proof_recomputes_the_commitment_root(client):
    _post(client, "e1", "alice", 10)
    _post(client, "e2", "bob", 7)
    _post(client, "e3", "carol", 12)
    _post(client, "e4", "dave", 7)
    _post(client, "e5", "erin", 12)
    lb = client.get("/leaderboard").json()
    commitment = client.get("/leaderboard/commitment").json()
    assert commitment["leaf_count"] == 5
    for i, entry in enumerate(lb["entries"]):
        pid = entry["participant_id"]
        proof = client.get(f"/leaderboard/proof/{pid}").json()
        assert proof["revision"] == commitment["revision"]
        assert proof["leaf_index"] == i
        assert proof["leaf_count"] == commitment["leaf_count"]
        assert proof["root"] == commitment["root"]
        assert proof["entry"] == {
            "rank": entry["rank"],
            "participant_id": pid,
            "total_points": entry["total_points"],
            "event_ids": [ev["event_id"] for ev in entry["effective_events"]],
        }
        # client-side recomputation lands on the same root
        assert _fold(proof["leaf_hash"], proof["siblings"]) == commitment["root"]
        # the leaf hash is the SHA-256 of the entry's canonical encoding
        enc = json.dumps(proof["entry"], sort_keys=True, separators=(",", ":")).encode()
        assert hashlib.sha256(enc).hexdigest() == proof["leaf_hash"]


def test_odd_level_last_node_pairs_with_itself(client):
    # 3 leaves -> parents [H(L0|L1), H(L2|L2)]
    _post(client, "e1", "alice", 10)
    _post(client, "e2", "bob", 7)
    _post(client, "e3", "carol", 5)
    lb = client.get("/leaderboard").json()
    leaves = _ref_leaves(lb)
    expected_root = hashlib.sha256(
        hashlib.sha256(leaves[0] + leaves[1]).digest()
        + hashlib.sha256(leaves[2] + leaves[2]).digest()
    ).hexdigest()
    commitment = client.get("/leaderboard/commitment").json()
    assert commitment["root"] == expected_root

    # the last leaf's proof carries itself as its own "right" sibling
    proof = client.get("/leaderboard/proof/carol").json()
    assert proof["leaf_index"] == 2
    assert proof["siblings"] == [
        {"position": "right", "hash": leaves[2].hex()},
        {"position": "left", "hash": hashlib.sha256(leaves[0] + leaves[1]).hexdigest()},
    ]
    assert _fold(proof["leaf_hash"], proof["siblings"]) == commitment["root"]


def test_odd_duplication_at_multiple_levels(client):
    # 5 leaves: level 0 pairs [L0,L1] [L2,L3] [L4,L4]; level 1 pairs [P0,P1] [P2,P2]
    for i, pid in enumerate(["p0", "p1", "p2", "p3", "p4"]):
        _post(client, f"e-{pid}", pid, 10 - i)
    lb = client.get("/leaderboard").json()
    leaves = _ref_leaves(lb)
    assert len(leaves) == 5
    p0 = hashlib.sha256(leaves[0] + leaves[1]).digest()
    p1 = hashlib.sha256(leaves[2] + leaves[3]).digest()
    p2 = hashlib.sha256(leaves[4] + leaves[4]).digest()
    q0 = hashlib.sha256(p0 + p1).digest()
    q1 = hashlib.sha256(p2 + p2).digest()
    root = hashlib.sha256(q0 + q1).hexdigest()
    commitment = client.get("/leaderboard/commitment").json()
    assert commitment["root"] == root

    proof = client.get("/leaderboard/proof/p4").json()
    assert proof["leaf_index"] == 4
    assert proof["siblings"] == [
        {"position": "right", "hash": leaves[4].hex()},
        {"position": "right", "hash": p2.hex()},
        {"position": "left", "hash": q0.hex()},
    ]
    assert _fold(proof["leaf_hash"], proof["siblings"]) == root


def test_leaf_order_follows_leaderboard_order_with_ties(client):
    _post(client, "e1", "zeta", 10)
    _post(client, "e2", "alpha", 10)
    _post(client, "e3", "mid", 5)
    # ties broken by participant_id ascending: alpha, zeta, then mid
    alpha = client.get("/leaderboard/proof/alpha").json()
    zeta = client.get("/leaderboard/proof/zeta").json()
    mid = client.get("/leaderboard/proof/mid").json()
    assert (alpha["leaf_index"], zeta["leaf_index"], mid["leaf_index"]) == (0, 1, 2)
    assert (alpha["entry"]["rank"], zeta["entry"]["rank"], mid["entry"]["rank"]) == (1, 2, 3)


def test_correction_chain_leaf_carries_only_tip_event_ids(client):
    _post(client, "e1", "alice", 10)
    _post(client, "e2", "alice", 12, correction_of="e1")
    _post(client, "e3", "alice", 15, correction_of="e2")
    proof = client.get("/leaderboard/proof/alice").json()
    assert proof["entry"]["event_ids"] == ["e3"]
    assert proof["entry"]["total_points"] == 15
    # at revision 1 the original event is still the effective one
    proof1 = client.get("/leaderboard/proof/alice?revision=1").json()
    assert proof1["entry"]["event_ids"] == ["e1"]
    assert proof1["entry"]["total_points"] == 10
    assert proof1["root"] != proof["root"]


def test_event_ids_listed_in_revision_order(client):
    _post(client, "e1", "alice", 5)
    _post(client, "e2", "bob", 1)
    _post(client, "e3", "alice", 7)
    proof = client.get("/leaderboard/proof/alice").json()
    assert proof["entry"]["event_ids"] == ["e1", "e3"]
    assert proof["entry"]["total_points"] == 12


def test_corrected_away_participant_absent_404(client):
    _post(client, "e1", "alice", 10)
    _post(client, "e2", "bob", 10, correction_of="e1")  # re-attributes to bob
    assert client.get("/leaderboard/proof/alice").status_code == 404
    proof = client.get("/leaderboard/proof/bob").json()
    assert proof["entry"]["event_ids"] == ["e2"]
    # alice is still provable at the historical revision
    assert client.get("/leaderboard/proof/alice?revision=1").status_code == 200


def test_absent_participant_404(client):
    _post(client, "e1", "alice", 10)
    r = client.get("/leaderboard/proof/ghost")
    assert r.status_code == 404


def test_historical_snapshot_unaffected_by_later_writes(client):
    _post(client, "e1", "alice", 10)
    _post(client, "e2", "bob", 7)
    before_c = client.get("/leaderboard/commitment?revision=2").json()
    before_p = client.get("/leaderboard/proof/alice?revision=2").json()

    _post(client, "e3", "alice", 100)
    _post(client, "e4", "carol", 50)

    assert client.get("/leaderboard/commitment?revision=2").json() == before_c
    assert client.get("/leaderboard/proof/alice?revision=2").json() == before_p
    # ...while the latest snapshot moved on
    latest = client.get("/leaderboard/commitment").json()
    assert latest["revision"] == 4
    assert latest["leaf_count"] == 3
    assert latest["root"] != before_c["root"]


def test_default_revision_is_latest(client):
    _post(client, "e1", "alice", 10)
    _post(client, "e2", "bob", 7)
    assert (
        client.get("/leaderboard/commitment").json()
        == client.get("/leaderboard/commitment?revision=2").json()
    )
    assert (
        client.get("/leaderboard/proof/alice").json()
        == client.get("/leaderboard/proof/alice?revision=2").json()
    )


def test_revision_validation(client):
    _post(client, "e1", "alice", 10)
    for path in ("/leaderboard/commitment", "/leaderboard/proof/alice"):
        assert client.get(path, params={"revision": 0}).status_code == 404
        assert client.get(path, params={"revision": 2}).status_code == 404
        assert client.get(path, params={"revision": -1}).status_code == 422
        assert client.get(path, params={"revision": "abc"}).status_code == 422


def test_non_ascii_participant_id_encoding_is_deterministic(client):
    _post(client, "e1", "张三", 10)
    _post(client, "e2", "alice", 5)
    lb = client.get("/leaderboard").json()
    commitment = client.get("/leaderboard/commitment").json()
    assert commitment["root"] == _ref_root(_ref_leaves(lb)).hex()
    proof = client.get("/leaderboard/proof/张三").json()
    assert proof["entry"]["participant_id"] == "张三"
    assert _fold(proof["leaf_hash"], proof["siblings"]) == commitment["root"]


def test_negative_and_zero_totals_still_get_leaves(client):
    _post(client, "e1", "alice", -5)
    _post(client, "e2", "bob", 0)
    commitment = client.get("/leaderboard/commitment").json()
    assert commitment["leaf_count"] == 2
    for pid in ("alice", "bob"):
        proof = client.get(f"/leaderboard/proof/{pid}").json()
        assert _fold(proof["leaf_hash"], proof["siblings"]) == commitment["root"]


def test_pinned_revision_consistent_under_concurrent_writes(client):
    _post(client, "e1", "alice", 10)
    _post(client, "e2", "bob", 7)
    _post(client, "e3", "carol", 5)
    expected_c = client.get("/leaderboard/commitment?revision=3").json()
    expected_p = client.get("/leaderboard/proof/bob?revision=3").json()

    def write(i):
        return client.post("/events", json={
            "event_id": f"later-{i}", "participant_id": "alice", "points": 100,
        })

    def read(_):
        c = client.get("/leaderboard/commitment?revision=3")
        p = client.get("/leaderboard/proof/bob?revision=3")
        assert c.status_code == 200 and p.status_code == 200
        return c.json(), p.json()

    with ThreadPoolExecutor(max_workers=8) as pool:
        writes = [pool.submit(write, i) for i in range(10)]
        reads = [pool.submit(read, i) for i in range(10)]
        assert all(f.result().status_code == 201 for f in writes)
        bodies = [f.result() for f in reads]

    # every concurrent read saw the same pinned snapshot, and the two
    # endpoints agree with each other on every single response
    assert all(c == expected_c and p == expected_p for c, p in bodies)
    assert all(p["root"] == c["root"] for c, p in bodies)
    assert client.get("/leaderboard").json()["revision"] == 13
