"""
storage.py — the job store.

Prototype implementation: an in-memory dict (thread-safe via a lock) plus a
JSON snapshot written to each job's own directory for inspectability. This
is intentionally the *only* place that knows jobs live in-process; the API
layer and pipeline only ever call `JobStore` methods, so replacing this
with Redis/Postgres + a real task queue later touches this one file plus
`main.py`'s background-task wiring, and nothing in the pipeline modules.
"""
from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from .models import JobStatus, RenderJob

DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data"
JOBS_ROOT = DATA_ROOT / "jobs"
JOBS_ROOT.mkdir(parents=True, exist_ok=True)


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, RenderJob] = {}
        self._uploaded_paths: dict[str, list[str]] = {}
        self._variant_counts: dict[str, int] = {}
        self._lock = threading.Lock()

    def job_dir(self, job_id: str) -> Path:
        d = JOBS_ROOT / job_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def create_job(self) -> RenderJob:
        job_id = uuid.uuid4().hex[:12]
        job = RenderJob(job_id=job_id, status=JobStatus.QUEUED, progress_message="Queued...")
        with self._lock:
            self._jobs[job_id] = job
            self._uploaded_paths[job_id] = []
            self._variant_counts[job_id] = 0
        self._persist(job)
        return job

    def get(self, job_id: str) -> Optional[RenderJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def set_uploaded_paths(self, job_id: str, paths: list[str]) -> None:
        with self._lock:
            self._uploaded_paths[job_id] = paths

    def get_uploaded_paths(self, job_id: str) -> list[str]:
        with self._lock:
            return list(self._uploaded_paths.get(job_id, []))

    def next_variant_seed(self, job_id: str) -> int:
        with self._lock:
            self._variant_counts[job_id] = self._variant_counts.get(job_id, 0) + 1
            return self._variant_counts[job_id]

    def update(self, job_id: str, **fields) -> RenderJob:
        with self._lock:
            job = self._jobs[job_id]
            updated = job.model_copy(update={**fields, "updated_at": datetime.utcnow()})
            self._jobs[job_id] = updated
        self._persist(updated)
        return updated

    def _persist(self, job: RenderJob) -> None:
        try:
            manifest_path = self.job_dir(job.job_id) / "job.json"
            manifest_path.write_text(job.model_dump_json(indent=2))
        except Exception:
            # Persistence is a nice-to-have for inspectability in this
            # prototype; never let it break the in-memory pipeline.
            pass


# Single process-wide store (see docstring re: swapping this for a real
# datastore later).
job_store = JobStore()
