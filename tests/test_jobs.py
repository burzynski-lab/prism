"""The file-based job store: creation, queueing after upload, exclusive claims, updates, cleanup."""

import time

from prism.jobs import JobStore


def test_store_roundtrip_and_exclusive_claim(tmp_path):
    a, b = JobStore(tmp_path), JobStore(tmp_path)          # two processes sharing one volume
    j1 = a.create("x", "first", "f1.hepmc", 10, {"sel": "A"})
    assert b.claim_next() is None                            # not visible before the upload completes
    a.enqueue(j1.id)
    time.sleep(0.01)
    j2 = a.create("x", "second", "f2.hepmc", 10); a.enqueue(j2.id)
    c1 = b.claim_next()
    assert c1.id == j1.id and c1.status == "running" and c1.started and c1.options == {"sel": "A"}
    assert a.claim_next().id == j2.id and a.claim_next() is None
    b.update(j1.id, progress="half", status="done", finished=time.time(), result={"sr_efficiency": 0.5})
    assert a.get(j1.id).result["sr_efficiency"] == 0.5 and a.get(j1.id).progress == "half"
    assert [j.id for j in a.list()] == [j2.id, j1.id]
    assert a.cleanup(ttl_hours=0) == 1 and a.get(j1.id) is None and a.get(j2.id).status == "running"
