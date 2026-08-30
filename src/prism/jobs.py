"""Job store shared by the web app and the workers: plain files on the
service volume, safe on any shared filesystem (NFS, CephFS, a PVC mounted by
pods on different nodes), where SQLite locking and WAL are not.

    <root>/jobs/<id>/job.json      the record, rewritten atomically (write to a temp file, rename)
    <root>/queue/<created>-<id>    one marker per queued job; claiming a job is renaming its
                                   marker into the job directory, which exactly one worker wins
"""

from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

STATUSES = ("queued", "running", "done", "failed")


def settings() -> dict:
    root = Path(os.environ.get("PRISM_SERVICE_DIR", "service_data")).resolve()
    return {
        "root": root,
        "max_upload_mb": float(os.environ.get("PRISM_MAX_UPLOAD_MB", "2000")),
        "max_events": int(os.environ.get("PRISM_MAX_EVENTS", "20000")),
        "ttl_hours": float(os.environ.get("PRISM_JOB_TTL_HOURS", "72")),
    }


@dataclass
class Job:
    id: str
    status: str
    created: float
    analysis: str
    label: str
    source: str
    max_events: int
    started: float | None = None
    finished: float | None = None
    result: dict | None = None
    error: str | None = None
    progress: str | None = None
    options: dict | None = None

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _write_atomic(path: Path, data: dict) -> None:
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}")
    tmp.write_text(json.dumps(data))
    os.replace(tmp, path)


class JobStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.jobs_dir = self.root / "jobs"
        self.queue_dir = self.root / "queue"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.queue_dir.mkdir(parents=True, exist_ok=True)

    def job_dir(self, job_id: str) -> Path:
        return self.jobs_dir / job_id

    def _record(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "job.json"

    def create(self, analysis: str, label: str, source: str, max_events: int, options: dict | None = None) -> Job:
        """Create the job directory and record. The job becomes visible to the
        workers only after ``enqueue`` (once its input is fully uploaded)."""
        job = Job(uuid.uuid4().hex[:12], "queued", time.time(), analysis, label, source, max_events, options=options or {})
        self.job_dir(job.id).mkdir(parents=True, exist_ok=True)
        _write_atomic(self._record(job.id), job.to_dict())
        return job

    def enqueue(self, job_id: str) -> None:
        job = self.get(job_id)
        (self.queue_dir / f"{job.created:017.6f}-{job_id}").write_text(job_id)

    def get(self, job_id: str) -> Job | None:
        p = self._record(job_id)
        for _ in range(3):                                   # a concurrent atomic rewrite is invisible, a partial read is retried
            try:
                return Job(**json.loads(p.read_text()))
            except FileNotFoundError:
                return None
            except json.JSONDecodeError:
                time.sleep(0.05)
        return None

    def list(self, limit: int = 50) -> list[Job]:
        jobs = [j for j in (self.get(p.name) for p in self.jobs_dir.iterdir() if p.is_dir()) if j is not None]
        jobs.sort(key=lambda j: j.created, reverse=True)
        return jobs[:limit]

    def claim_next(self) -> Job | None:
        """Move the oldest queued job to running. The rename of the queue
        marker is atomic, so with several workers exactly one succeeds."""
        for marker in sorted(self.queue_dir.iterdir()):
            job_id = marker.name.rsplit("-", 1)[-1]
            try:
                os.rename(marker, self.job_dir(job_id) / "claimed")
            except FileNotFoundError:                         # another worker got it first
                continue
            job = self.get(job_id)
            if job is None:
                continue
            self.update(job_id, status="running", started=time.time())
            return self.get(job_id)
        return None

    def update(self, job_id: str, **fields) -> None:
        job = self.get(job_id)
        if job is None:
            return
        d = job.to_dict(); d.update(fields)
        _write_atomic(self._record(job_id), d)

    def cleanup(self, ttl_hours: float) -> int:
        """Delete finished jobs (and their files) older than the TTL."""
        cutoff = time.time() - ttl_hours * 3600
        old = [j for j in self.list(limit=10**6) if j.status in ("done", "failed") and j.created < cutoff]
        for j in old:
            shutil.rmtree(self.job_dir(j.id), ignore_errors=True)
        return len(old)
