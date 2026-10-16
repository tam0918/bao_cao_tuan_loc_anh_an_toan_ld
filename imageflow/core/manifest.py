"""Manifest: the source of truth of a job (spec sections 17, 27, 44).

Layout inside ``<job>/.imageflow/``:

* ``manifest.json``          compacted snapshot, replaced atomically
* ``manifest.journal.jsonl`` append-only log of full-record upserts, fsync'd per write

Replaying the journal on top of the snapshot is idempotent because every entry is a
full record, so a crash between snapshot replace and journal truncation is harmless.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .state_machine import ImageState, check_transition

SNAPSHOT_NAME = "manifest.json"
JOURNAL_NAME = "manifest.journal.jsonl"
SCHEMA_VERSION = 1
COMPACT_EVERY = 500  # journal entries before an automatic snapshot


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write via a temp file in the same directory, fsync, then os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


class ManifestStore:
    def __init__(self, system_dir: Path) -> None:
        self.system_dir = system_dir
        self._snapshot_path = system_dir / SNAPSHOT_NAME
        self._journal_path = system_dir / JOURNAL_NAME
        self._records: dict[str, dict[str, Any]] = {}
        self._journal_len = 0

    # ----- loading -------------------------------------------------------------
    def load(self) -> None:
        self._records = {}
        if self._snapshot_path.exists():
            snap = json.loads(self._snapshot_path.read_text(encoding="utf-8"))
            for rec in snap.get("records", []):
                self._records[rec["sha256"]] = rec
        self._journal_len = 0
        if self._journal_path.exists():
            with open(self._journal_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        # A torn final line from a crash mid-append. Nothing after it can
                        # be valid, so stop replaying here.
                        break
                    self._records[rec["sha256"]] = rec
                    self._journal_len += 1

    # ----- reads ---------------------------------------------------------------
    def get(self, sha256: str) -> dict[str, Any] | None:
        rec = self._records.get(sha256)
        return dict(rec) if rec is not None else None

    def all(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self._records.values()]

    def by_state(self, state: ImageState) -> list[dict[str, Any]]:
        return [dict(r) for r in self._records.values() if r["state"] == state.value]

    # ----- writes --------------------------------------------------------------
    def put(self, record: dict[str, Any]) -> None:
        """Persist a full record. Journal first, then in-memory, so a crash never loses a write."""
        record = dict(record)
        record["schema_version"] = SCHEMA_VERSION
        line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        self.system_dir.mkdir(parents=True, exist_ok=True)
        with open(self._journal_path, "a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())
        self._records[record["sha256"]] = record
        self._journal_len += 1
        if self._journal_len >= COMPACT_EVERY:
            try:
                self.compact()
            except OSError:
                # For example an antivirus scanner holding the snapshot open on Windows. The journal
                # already has every record, so nothing is lost; compaction is retried on the next write.
                pass

    def transition(self, sha256: str, target: ImageState, **fields: Any) -> dict[str, Any]:
        """Move a record to ``target`` (validated) and persist it together with ``fields``."""
        rec = self._records[sha256]
        current = ImageState(rec["state"])
        if current != target:
            check_transition(current, target)
        rec = dict(rec)
        rec["state"] = target.value
        rec.update(fields)
        self.put(rec)
        return rec

    def compact(self) -> None:
        payload = {"schema_version": SCHEMA_VERSION, "records": list(self._records.values())}
        atomic_write_bytes(
            self._snapshot_path,
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=1).encode("utf-8"),
        )
        # Snapshot is durable before the journal is dropped, so a crash here only leaves a
        # journal that is already reflected in the snapshot (replay is idempotent).
        if self._journal_path.exists():
            os.remove(self._journal_path)
        self._journal_len = 0
