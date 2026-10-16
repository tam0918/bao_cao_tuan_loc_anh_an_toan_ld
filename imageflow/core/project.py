"""Project registry: stable folder IDs and folder -> action mapping (spec sections 2.3, 2.4, 45 G).

Folder identity is a random ID stored in ``.imageflow/project.json``. A folder keeps its
ID across renames because the directory inode (dev, ino) is matched first. When identity
cannot be proven, the registered folder is marked MISSING and its actions are NOT moved to
another folder: a safe update instead of a silent guess.
"""

from __future__ import annotations

import json
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .manifest import atomic_write_bytes
from .workspace import FolderInfo, nfc

PROJECT_NAME = "project.json"
SCHEMA_VERSION = 1
ROUTING_MODES = frozenset({"move", "copy"})


@dataclass
class SyncEvent:
    kind: str  # "added" | "renamed" | "missing" | "restored"
    folder_id: str
    detail: str = ""


@dataclass
class ProjectStore:
    system_dir: Path
    known_action_types: frozenset[str]
    routing_mode: str = "move"
    folders: list[dict[str, Any]] = field(default_factory=list)
    # folder_id -> FolderInfo for folders found on disk by the last sync (current path, name, rule)
    live: dict[str, FolderInfo] = field(default_factory=dict)

    @property
    def path(self) -> Path:
        return self.system_dir / PROJECT_NAME

    # ----- persistence ---------------------------------------------------------
    @classmethod
    def load(cls, system_dir: Path, known_action_types: frozenset[str]) -> "ProjectStore":
        store = cls(system_dir=system_dir, known_action_types=known_action_types)
        if store.path.exists():
            data = json.loads(store.path.read_text(encoding="utf-8"))
            store.routing_mode = data.get("routing_mode", "move")
            store.folders = data.get("folders", [])
        return store

    def save(self) -> None:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "routing_mode": self.routing_mode,
            "folders": self.folders,
        }
        atomic_write_bytes(
            self.path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
        )

    # ----- sync with the disk --------------------------------------------------
    def sync(self, scanned: list[FolderInfo]) -> list[SyncEvent]:
        events: list[SyncEvent] = []
        matched_scan: set[int] = set()
        self.live = {}
        for reg in self.folders:
            idx = self._match(reg, scanned, matched_scan)
            if idx is None:
                if reg["status"] != "MISSING":
                    reg["status"] = "MISSING"
                    events.append(SyncEvent("missing", reg["folder_id"], reg["name"]))
                continue
            matched_scan.add(idx)
            found = scanned[idx]
            if reg["status"] == "MISSING":
                reg["status"] = "ACTIVE"
                events.append(SyncEvent("restored", reg["folder_id"], found.name))
            if reg["name"] != found.name:
                events.append(SyncEvent("renamed", reg["folder_id"], f"{reg['name']} -> {found.name}"))
                reg["name"] = found.name
            reg["dev"], reg["ino"] = found.dev, found.ino
            self.live[reg["folder_id"]] = found

        known_ids = {f["folder_id"] for f in self.folders}
        for idx, found in enumerate(scanned):
            if idx in matched_scan:
                continue
            fid = self._new_id(known_ids)
            known_ids.add(fid)
            self.folders.append(
                {
                    "folder_id": fid,
                    "name": found.name,
                    "dev": found.dev,
                    "ino": found.ino,
                    "status": "ACTIVE",
                    "actions": [],
                }
            )
            self.live[fid] = found
            events.append(SyncEvent("added", fid, found.name))

        if events:
            self.save()
        return events

    @staticmethod
    def _match(reg: dict[str, Any], scanned: list[FolderInfo], taken: set[int]) -> int | None:
        if reg["ino"]:
            for idx, found in enumerate(scanned):
                if idx not in taken and found.dev == reg["dev"] and found.ino == reg["ino"]:
                    return idx
            return None
        # Filesystems without inode numbers: fall back to the name, only when unambiguous.
        candidates = [i for i, f in enumerate(scanned) if i not in taken and f.name == nfc(reg["name"])]
        return candidates[0] if len(candidates) == 1 else None

    @staticmethod
    def _new_id(known: set[str]) -> str:
        while True:
            fid = "f_" + secrets.token_hex(3)
            if fid not in known:
                return fid

    # ----- user configuration --------------------------------------------------
    def set_actions(self, folder_id: str, actions: list[str]) -> None:
        unknown = set(actions) - self.known_action_types
        if unknown:
            raise ValueError(f"unknown action types: {sorted(unknown)}")
        reg = self._get(folder_id)
        wanted = sorted(set(actions))
        if reg["actions"] == wanted:
            return  # saving costs an fsync per call, so skip the no-op
        reg["actions"] = wanted
        self.save()

    def set_routing_mode(self, mode: str) -> None:
        if mode not in ROUTING_MODES:
            raise ValueError(f"routing_mode must be one of {sorted(ROUTING_MODES)}")
        self.routing_mode = mode
        self.save()

    def actions_for(self, folder_id: str) -> list[str]:
        return list(self._get(folder_id)["actions"])

    def name_of(self, folder_id: str) -> str:
        return self._get(folder_id)["name"]

    def active_ids(self) -> list[str]:
        return [f["folder_id"] for f in self.folders if f["status"] == "ACTIVE"]

    def _get(self, folder_id: str) -> dict[str, Any]:
        for reg in self.folders:
            if reg["folder_id"] == folder_id:
                return reg
        raise KeyError(f"unknown folder_id {folder_id}")
