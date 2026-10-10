"""Structured event log (spec section 40). One JSON object per line, never image bytes."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class EventLog:
    def __init__(self, path: Path, job_id: str) -> None:
        self._path = path
        self._job_id = job_id
        path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: str, **fields: Any) -> None:
        record = {"ts": round(time.time(), 3), "job_id": self._job_id, "event": event, **fields}
        with open(self._path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True, default=str) + "\n")
