"""Job pipeline: scan, normalize, classify, route, action (spec sections 8, 18, 25, 28, 44).

Ordering rules that keep the job crash-safe:

* the manifest records a decision before any file moves (classification -> routing intent -> move -> ROUTED)
* every filesystem step is idempotent, so resuming a half-done step is always safe
* the original is only ever moved into a category, never deleted, and never overwritten
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..actions.base import ActionContext, ActionRegistry
from ..classifiers.base import Choice
from ..errors import PermanentError
from ..preprocessing.image_loader import is_decodable, normalize_file
from .classification import ClassificationEngine, ClassifierUnavailable
from .events import EventLog
from .hashing import sha256_file
from .manifest import ManifestStore, atomic_write_bytes
from .metrics import JobMetrics
from .project import ProjectStore
from .routing import DestinationMissing, deterministic_name, plan_recovery, route_file
from .safety import SafetyScreenEngine
from .state_machine import RESUMABLE_STATES, ImageState
from .workspace import ImageFile, ScanResult, choice_description, nfc, scan_workspace

SYSTEM_DIR = ".imageflow"
SYSTEM_SUBDIRS = ("normalized", "staging", "outputs", "review", "logs", "temp")
MIME_EXTENSION = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}
# Action errors that will not change on retry. The action waits for a human instead of retrying forever.
PERMANENT_ACTION_CODES = frozenset({"UNSUPPORTED_IMAGE", "CORRUPTED_IMAGE", "INVALID_REQUEST", "INVALID_SCHEMA"})

_CLASSIFY_STATES = frozenset({ImageState.DISCOVERED, ImageState.NORMALIZED, ImageState.CLASSIFYING})
_ROUTE_STATES = frozenset({ImageState.CLASSIFIED, ImageState.ROUTING})
_ACTION_STATES = frozenset({ImageState.ROUTED, ImageState.ACTION_PENDING, ImageState.ACTION_RUNNING, ImageState.ACTION_FAILED})


class JobAborted(Exception):
    """A job-level failure that stops the whole run. ``code`` tells the caller which one."""

    def __init__(self, message: str, code: str = "ABORTED") -> None:
        super().__init__(message)
        self.code = code


class UnknownFolder(ValueError):
    """No active folder has the requested name."""


@dataclass(frozen=True)
class PipelineConfig:
    classify_concurrency: int = 4  # CLASSIFY_CONCURRENCY
    edit_concurrency: int = 2  # EDIT_CONCURRENCY
    # Stop after this many images in a row could not be classified because the service is unreachable,
    # instead of spending retries and minutes on every remaining image.
    abort_after_unavailable: int = 5


@dataclass(frozen=True)
class FolderRow:
    folder_id: str
    name: str
    actions: tuple[str, ...]


@dataclass
class JobReport:
    job_id: str
    images_found: int = 0
    skipped_duplicates: int = 0
    routed: int = 0
    actioned: int = 0
    failed: int = 0
    actions_failed: int = 0
    safety_review_required: int = 0
    deferred: int = 0
    internal_errors: int = 0  # unexpected exceptions, logged and isolated to one image
    unsupported: list[str] = field(default_factory=list)
    folders: list[str] = field(default_factory=list)
    totals: dict[str, int] = field(default_factory=dict)  # counts over the whole manifest
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def unfinished(self) -> int:
        """Images that another run could still finish: deferred, errored, or with a failed action."""
        return self.deferred + self.internal_errors + self.totals.get("action_failed", 0)


class ImageFlowJob:
    def __init__(
        self,
        root: Path,
        classifier: ClassificationEngine,
        actions: ActionRegistry,
        safety_screen: SafetyScreenEngine | None = None,
        safety_action_type: str | None = None,
        config: PipelineConfig = PipelineConfig(),
        job_id: str | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.system = self.root / SYSTEM_DIR
        self.classifier = classifier
        self.actions = actions
        self.safety_screen = safety_screen
        self.safety_action_type = safety_action_type
        self.config = config
        self.job_id = job_id or f"j_{uuid.uuid4().hex[:10]}"
        self.manifest = ManifestStore(self.system)
        self.project = ProjectStore.load(self.system, actions.known_types())
        self.metrics = JobMetrics()
        self.report = JobReport(job_id=self.job_id)
        self._dirs = {name: self.system / name for name in SYSTEM_SUBDIRS}
        self._scan: ScanResult | None = None
        self._log: EventLog | None = None
        self._unavailable_streak = 0
        # asyncio primitives belong to one event loop, so the limits are kept per loop (see _limit).
        self._limits: dict[str, tuple[asyncio.AbstractEventLoop, asyncio.Semaphore]] = {}

    def _limit(self, name: str) -> asyncio.Semaphore:
        """The shared concurrency limit ("classify" or "edit") for the running event loop.

        Every caller on one loop shares one semaphore, so the cap holds for run() and for
        assignments alike. A job is driven by one event loop at a time; it is not thread-safe.
        """
        loop = asyncio.get_running_loop()
        cached = self._limits.get(name)
        if cached is None or cached[0] is not loop:
            size = self.config.classify_concurrency if name == "classify" else self.config.edit_concurrency
            cached = (loop, asyncio.Semaphore(size))
            self._limits[name] = cached
        return cached[1]

    # ----- setup and read-only views -------------------------------------------
    def prepare(self) -> ScanResult:
        """Create system dirs, load the manifest, scan the workspace, sync folder identities."""
        if not self.root.is_dir():
            # Without this guard a mistyped path would get a stray .imageflow tree created under it.
            raise NotADirectoryError(f"workspace folder not found: {self.root}")
        for directory in self._dirs.values():
            directory.mkdir(parents=True, exist_ok=True)
        self._log = EventLog(self._dirs["logs"] / f"{self.job_id}.jsonl", self.job_id)
        self.manifest.load()
        self._scan = scan_workspace(self.root, self.actions.output_suffixes())
        for ev in self.project.sync(self._scan.folders):
            self._log.emit("folder_" + ev.kind, folder_id=ev.folder_id, detail=ev.detail)
        return self._scan

    def _fill_report_from_scan(self) -> None:
        assert self._scan is not None
        self.report.images_found = len(self._scan.images)
        self.report.unsupported = list(self._scan.unsupported)
        self.report.folders = [f.name for f in self._scan.folders]

    def choices(self) -> list[Choice]:
        """The classification options: one per active folder. Built at runtime, nothing hard-coded."""
        out: list[Choice] = []
        for fid in self.project.active_ids():
            folder = self.project.live.get(fid)
            if folder is not None:
                out.append(Choice(choice_id=fid, label=folder.name, description=choice_description(folder)))
        return sorted(out, key=lambda c: c.label)

    def folder_rows(self) -> list[FolderRow]:
        """The active folders with their assigned actions, for screens and command lines to list."""
        rows = [
            FolderRow(folder_id=fid, name=folder.name, actions=tuple(self.project.actions_for(fid)))
            for fid in self.project.active_ids()
            if (folder := self.project.live.get(fid)) is not None
        ]
        return sorted(rows, key=lambda r: r.name)

    def resolve_folder(self, name: str) -> str:
        """folder_id of the active folder called ``name``. A folder that no longer exists never matches."""
        wanted = nfc(name)
        for row in self.folder_rows():
            if row.name == wanted:
                return row.folder_id
        raise UnknownFolder(name)

    def review_items(self) -> list[dict[str, Any]]:
        """Images waiting for a human decision, with the top candidates for the review screen."""
        items = []
        for rec in self.manifest.by_state(ImageState.REVIEW_REQUIRED):
            candidates = [
                {"folder_id": c["folder_id"], "name": self._folder_name(c["folder_id"]), "probability": c["probability"]}
                for c in (rec.get("review") or {}).get("candidates", [])
            ]
            items.append(
                {
                    "image_id": rec["image_id"],  # the key to pass to assign(); a content hash may be shared by two photos
                    "sha256": rec["sha256"],
                    "name": rec["original_filename"],
                    "path": rec["current_rel"],
                    "candidates": candidates,
                }
            )
        return items

    def set_actions(self, folder_id: str, actions: list[str]) -> None:
        self.project.set_actions(folder_id, actions)

    def set_routing_mode(self, mode: str) -> None:
        self.project.set_routing_mode(mode)

    def _folder_name(self, folder_id: str) -> str | None:
        folder = self.project.live.get(folder_id)
        return folder.name if folder else None

    # ----- run -----------------------------------------------------------------
    async def run(self, *, refresh: bool = True) -> JobReport:
        """Process every pending image. ``refresh=False`` reuses the scan of a prepare() just made."""
        started = time.perf_counter()
        self.report = JobReport(job_id=self.job_id)  # the counters describe this run only
        self.metrics = JobMetrics()
        self._unavailable_streak = 0
        if refresh or self._scan is None:
            self.prepare()
        self._fill_report_from_scan()
        choices = self.choices()
        pending = self._pending()
        self._log.emit("job_started", pending=len(pending), folders=len(choices))
        try:
            async with asyncio.TaskGroup() as tg:
                for sha in pending:
                    tg.create_task(self._guarded(sha, choices))
        except* JobAborted as group:
            self._log.emit("job_aborted", reason=str(group.exceptions[0]))
            raise group.exceptions[0]
        self._finish(started)
        return self.report

    async def assign(self, image_id: str, folder_id: str) -> str | None:
        """The user picks the folder for an image in review. It is then routed and actioned like any other.

        Returns None when the image was routed and its actions finished. Otherwise it returns a
        problem code that the caller should tell the user about:

        * ``DESTINATION_MISSING``: the folder is gone. The image is back in the review list.
        * ``ROUTE_FAILED``: the file could not be moved yet (for example it is locked). The next run retries.
        * ``ACTION_FAILED``: the image was moved but an action failed. The next run retries it.
        * ``SOURCE_MISSING`` / ``LOST_SOURCE``: the original file is gone.
        """
        self.prepare()  # the folder may have been renamed or removed since the last run
        rec = self.manifest.get(image_id)
        if rec is None or rec["state"] != ImageState.REVIEW_REQUIRED.value:
            raise ValueError("image is not waiting for review")
        if folder_id not in self.project.active_ids():
            raise ValueError("folder is unknown or missing")
        classification = {
            "source": "manual",
            "provider": None,
            "model": None,
            "folder_id": folder_id,
            "confidence": None,
            "second_choice": None,
            "second_probability": None,
        }
        self.manifest.transition(image_id, ImageState.CLASSIFIED, classification=classification, last_error=None)
        if not self._route(image_id):
            problem = self.manifest.get(image_id)["last_error"] or "ROUTE_FAILED"
            if self._state(image_id) == ImageState.CLASSIFIED:
                # Nothing was moved, so the image goes back to the review list for another choice.
                self.manifest.transition(image_id, ImageState.REVIEW_REQUIRED, last_error=problem)
            return problem
        self._remove_review_copy(image_id)  # only after the original has safely left the root
        await self._run_actions(image_id)
        return "ACTION_FAILED" if self._state(image_id) == ImageState.ACTION_FAILED else None

    # ----- planning --------------------------------------------------------------
    def _pending(self) -> list[str]:
        assert self._scan is not None
        seen_instances: set[str] = set()
        pending: list[str] = []
        for img in sorted(self._scan.images, key=lambda i: i.name):
            content_sha = sha256_file(img.path)
            image_id = self._instance_id(img.rel, content_sha)
            seen_instances.add(image_id)
            rec = self.manifest.get(image_id)
            if rec is None:
                self.manifest.put(self._new_record(image_id, content_sha, img))
                pending.append(image_id)
                continue
            state = ImageState(rec["state"])
            if state == ImageState.COMPLETE:
                self.report.skipped_duplicates += 1  # this exact file instance was already processed
            elif state in RESUMABLE_STATES:
                pending.append(image_id)
            # REVIEW_REQUIRED waits for the user. FAILED needs a human fix. Neither is re-run here.
        # Images that already left the root (routed, mid-action) still need to finish.
        for rec in self.manifest.all():
            if rec["image_id"] not in seen_instances and ImageState(rec["state"]) in RESUMABLE_STATES:
                pending.append(rec["image_id"])
        return pending

    @staticmethod
    def _instance_id(source_rel: str, content_sha: str) -> str:
        """Stable identity for one file instance. Content SHA is only a cache/integrity key."""
        digest = hashlib.sha256((source_rel + "\0" + content_sha).encode("utf-8")).hexdigest()
        return "im_" + digest[:24]

    def _new_record(self, image_id: str, content_sha: str, img: ImageFile) -> dict[str, Any]:
        return {
            "image_id": image_id,
            "sha256": content_sha,
            "original_filename": img.name,
            "source_path": img.rel,
            "current_rel": img.rel,
            "state": ImageState.DISCOVERED.value,
            "normalized_rel": None,
            "classification": None,
            "safety": None,
            "attempts": [],
            "review": None,
            "review_copy_rel": None,
            "routing": None,
            "actions": [],
            "last_error": None,
        }

    async def _guarded(self, sha: str, choices: list[Choice]) -> None:
        try:
            await self._process(sha, choices)
        except JobAborted:
            raise
        except Exception as exc:  # one image's bug must not stop the other images
            self.report.internal_errors += 1
            self._log.emit("image_internal_error", sha256=sha, error=repr(exc))
            self._note_error(sha, "INTERNAL_ERROR")

    async def _process(self, sha: str, choices: list[Choice]) -> None:
        if self._state(sha) in _CLASSIFY_STATES:
            if not await self._classify(sha, choices):
                return
        if self._state(sha) in _ROUTE_STATES:
            if not self._route(sha):
                return
        if self._state(sha) in _ACTION_STATES:
            await self._run_actions(sha)

    # ----- classify --------------------------------------------------------------
    async def _classify(self, sha: str, choices: list[Choice]) -> bool:
        rec = self.manifest.get(sha)
        src = self._abs(rec["current_rel"])
        if not src.exists():
            self._fail(sha, "SOURCE_MISSING")
            return False
        try:
            encoded = normalize_file(src, self._dirs["normalized"], rec["sha256"])
        except PermanentError as exc:
            self._fail(sha, exc.code)
            return False
        if rec["state"] == ImageState.DISCOVERED.value:
            normalized_rel = self._rel(self._dirs["normalized"] / f"{rec['sha256']}.jpg")
            self.manifest.transition(sha, ImageState.NORMALIZED, normalized_rel=normalized_rel)
        if not choices:
            return False  # no category folder yet. The image waits in NORMALIZED.
        if self._unavailable_streak >= self.config.abort_after_unavailable:
            return False  # the job is already stopping because the service is down: leave this image for the next run
        self._set_state(sha, ImageState.CLASSIFYING)
        try:
            async with self._limit("classify"):
                decision = await self.classifier.decide(encoded, choices)
        except ClassifierUnavailable:
            self.manifest.transition(sha, ImageState.NORMALIZED, last_error="CLASSIFIER_UNAVAILABLE")
            self.report.deferred += 1
            self._log.emit("classify_deferred", sha256=sha, reason="CLASSIFIER_UNAVAILABLE")
            self._unavailable_streak += 1
            if self._unavailable_streak >= self.config.abort_after_unavailable:
                raise JobAborted(
                    f"the classification service is unreachable ({self._unavailable_streak} images in a row); "
                    "check the connection and run again",
                    code="SERVICE_UNAVAILABLE",
                )
            return False
        except PermanentError as exc:
            if exc.code == "AUTH_FAILED":
                raise JobAborted(str(exc), code="AUTH_FAILED") from exc
            self._fail(sha, exc.code)
            return False
        self._unavailable_streak = 0

        attempts = [a.as_dict() for a in decision.attempts]
        self.metrics.observe_decision(decision)
        self._log.emit(
            "classified",
            sha256=sha,
            image_id=rec["image_id"],
            filename=rec["original_filename"],
            decision=decision.kind,
            folder_id=decision.choice_id,
            confidence=round(decision.confidence, 6),
            attempts=[{k: a[k] for k in ("provider", "model", "outcome", "latency_ms", "retries", "error_code")} for a in attempts],
        )
        if decision.kind == "ROUTE":
            classification = {
                "source": "auto",
                "provider": decision.provider,
                "model": decision.model,
                "folder_id": decision.choice_id,
                "confidence": round(decision.confidence, 6),
                "second_choice": decision.second_choice,
                "second_probability": round(decision.second_probability, 6),
            }
            self.manifest.transition(sha, ImageState.CLASSIFIED, classification=classification, attempts=attempts, last_error=None)
            return True
        review = {
            "reason": "LOW_CONFIDENCE",
            "candidates": [{"folder_id": cid, "probability": round(p, 6)} for cid, p in decision.candidates],
        }
        self.manifest.transition(sha, ImageState.REVIEW_REQUIRED, review=review, attempts=attempts, last_error=None)
        self._make_review_copy(sha, src)
        return False

    def _make_review_copy(self, sha: str, src: Path) -> None:
        """A copy for the review folder. The original stays in the root until the user decides."""
        dst = self._dirs["review"] / f"{sha}{src.suffix.lower()}"
        if not dst.exists():
            tmp = dst.with_name(dst.name + ".tmp")
            shutil.copyfile(src, tmp)
            os.replace(tmp, dst)
        self.manifest.transition(sha, ImageState.REVIEW_REQUIRED, review_copy_rel=self._rel(dst))

    def _remove_review_copy(self, sha: str) -> None:
        rec = self.manifest.get(sha)
        rel = rec.get("review_copy_rel") if rec else None
        if rel:
            self._abs(rel).unlink(missing_ok=True)
            self.manifest.transition(sha, ImageState(rec["state"]), review_copy_rel=None)

    # ----- route -----------------------------------------------------------------
    def _route(self, sha: str) -> bool:
        rec = self.manifest.get(sha)
        state = ImageState(rec["state"])
        folder_id = rec["classification"]["folder_id"]
        src = self._abs(rec["current_rel"])
        mode = (rec.get("routing") or {}).get("mode") or self.project.routing_mode

        if state == ImageState.ROUTING:
            routing = rec.get("routing") or {}
            planned = self._abs(routing["destination_path"]) if routing.get("destination_path") else None
            plan = plan_recovery(src, planned, rec["sha256"], mode)
            self._log.emit("route_recovered", sha256=sha, plan=plan.action)
            if plan.action == "LOST_SOURCE":
                self._fail(sha, "LOST_SOURCE")
                return False
            if plan.action == "FINISH_MOVE":
                try:
                    os.unlink(src)  # the link already exists at the destination
                except OSError as exc:  # for example the file is open in another program on Windows
                    self._log.emit("route_retry", sha256=sha, error=repr(exc))
                    self._note_error(sha, "ROUTE_FAILED")
                    return False
            if plan.action in ("DONE", "FINISH_MOVE"):
                return self._mark_routed(sha, plan.destination, mode, folder_id)
            # REROUTE: fall through and place the file again with a fresh deterministic name.

        folder_dir = self._folder_dir(folder_id)
        if folder_dir is None:
            self._note_error(sha, "DESTINATION_MISSING")
            return False
        if not src.exists():
            self._fail(sha, "SOURCE_MISSING")
            return False

        planned = deterministic_name(folder_dir, src.name)
        routing = {"status": "PENDING", "mode": mode, "folder_id": folder_id, "destination_path": self._rel(planned)}
        self.manifest.transition(sha, ImageState.ROUTING, routing=routing, last_error=None)
        try:
            final = route_file(src, folder_dir, src.name, mode, dest=planned, temp_dir=self._dirs["temp"])
        except DestinationMissing:
            self._note_error(sha, "DESTINATION_MISSING")
            return False
        except OSError as exc:  # for example a file locked by another program on Windows. Retried next run.
            self._log.emit("route_retry", sha256=sha, error=repr(exc))
            self._note_error(sha, "ROUTE_FAILED")
            return False
        return self._mark_routed(sha, final, mode, folder_id)

    def _mark_routed(self, sha: str, final: Path, mode: str, folder_id: str) -> bool:
        final_rel = self._rel(final)
        routing = {"status": "ROUTED", "mode": mode, "folder_id": folder_id, "destination_path": final_rel}
        self.manifest.transition(sha, ImageState.ROUTED, current_rel=final_rel, routing=routing, last_error=None)
        self.metrics.observe_route()
        self.report.routed += 1
        self._log.emit("routed", sha256=sha, folder_id=folder_id, destination=final_rel, mode=mode)
        return True

    # ----- actions ---------------------------------------------------------------
    async def _run_actions(self, sha: str) -> None:
        rec = self.manifest.get(sha)
        folder_id = rec["routing"]["folder_id"]
        if folder_id not in self.project.active_ids():
            self._note_error(sha, "DESTINATION_MISSING")
            return

        if self.safety_screen is not None:
            status = await self._ensure_safety(sha)
            if status is None:
                return
            if status == "SAFE":
                self._set_state(sha, ImageState.COMPLETE)
                self._log.emit("complete", image_id=sha, folder_id=folder_id, actions=[], ppe_status="SAFE")
                return
            if status == "REVIEW":
                self.report.safety_review_required += 1
                self._set_state(sha, ImageState.COMPLETE)
                self._log.emit("complete", image_id=sha, folder_id=folder_id, actions=[], ppe_status="REVIEW")
                return
            action_types = [self.safety_action_type] if self.safety_action_type else []
        else:
            action_types = self.project.actions_for(folder_id)

        if not action_types:
            self._set_state(sha, ImageState.COMPLETE)
            self._log.emit("complete", image_id=sha, folder_id=folder_id, actions=[])
            return
        if self._state(sha) == ImageState.ROUTED:
            self._set_state(sha, ImageState.ACTION_PENDING)
        for atype in action_types:
            existing = _find_action(self.manifest.get(sha)["actions"], atype)
            if existing and existing["status"] == "SUCCESS":
                continue
            if existing and existing.get("permanent"):
                return
            if not await self._run_action(sha, atype, folder_id):
                return
        self._set_state(sha, ImageState.COMPLETE)
        self._log.emit("complete", image_id=sha, folder_id=folder_id, actions=action_types)

    async def _ensure_safety(self, sha: str) -> str | None:
        rec = self.manifest.get(sha)
        previous = rec.get("safety") or {}
        if previous.get("status") in {"SAFE", "NEEDS_EDIT", "REVIEW"}:
            return previous["status"]

        src = self._abs(rec["current_rel"])
        if not src.exists():
            self._fail(sha, "SOURCE_MISSING")
            return None
        try:
            encoded = normalize_file(src, self._dirs["normalized"], rec["sha256"])
        except PermanentError as exc:
            self._fail(sha, exc.code)
            return None

        try:
            async with self._limit("classify"):
                decision = await self.safety_screen.decide(encoded)
        except ClassifierUnavailable:
            self._note_error(sha, "SAFETY_CLASSIFIER_UNAVAILABLE")
            self.report.deferred += 1
            return None
        except PermanentError as exc:
            if exc.code == "AUTH_FAILED":
                raise JobAborted(str(exc), code="AUTH_FAILED") from exc
            self._fail(sha, exc.code)
            return None

        safety = {
            "status": decision.status,
            "provider": decision.provider,
            "model": decision.model,
            "confidence": round(decision.confidence, 6),
            "attempts": decision.attempts,
        }
        self.manifest.transition(sha, ImageState(rec["state"]), safety=safety, last_error=None)
        self._log.emit(
            "safety_screened",
            image_id=sha,
            content_sha256=rec["sha256"],
            folder_id=folder_id if (folder_id := rec["routing"]["folder_id"]) else None,
            status=decision.status,
            confidence=round(decision.confidence, 6),
        )
        return decision.status

    async def _run_action(self, sha: str, atype: str, folder_id: str) -> bool:
        self._set_state(sha, ImageState.ACTION_RUNNING)
        rec = self.manifest.get(sha)
        folder_dir = self._folder_dir(folder_id)
        if folder_dir is None:
            return self._action_failed(sha, atype, {}, "DESTINATION_MISSING", permanent=False)
        plugin = self.actions.get(atype)
        existing = _find_action(rec["actions"], atype) or {}

        # A crash after the output was staged or finalized must not pay for the provider twice.
        if existing.get("output_rel") is None and existing.get("staging_rel"):
            finalized = self._finalized_output(existing)
            if finalized is not None:
                return self._action_succeeded(sha, atype, existing, finalized)
            staged = self._abs(existing["staging_rel"])
            if staged.exists() and is_decodable(staged) and sha256_file(staged) == existing.get("output_sha256"):
                return self._finalize_output(sha, atype, existing, staged, folder_dir, plugin.output_suffix, rec["original_filename"])

        src = self._abs(rec["current_rel"])
        if not src.exists():
            return self._action_failed(sha, atype, {}, "SOURCE_MISSING", permanent=True)
        try:
            encoded = normalize_file(src, self._dirs["normalized"], rec["sha256"])
        except PermanentError as exc:
            return self._action_failed(sha, atype, {}, exc.code, permanent=True)

        context = ActionContext(job_id=self.job_id, image_sha256=rec["sha256"], folder_id=folder_id, original_name=rec["original_filename"])
        try:
            async with self._limit("edit"):
                result = await plugin.execute(encoded, context)
        except PermanentError as exc:
            if exc.code == "AUTH_FAILED":
                raise JobAborted(str(exc), code="AUTH_FAILED") from exc
            return self._action_failed(sha, atype, {}, exc.code, permanent=True)

        base = {"provider": result.provider, "model": result.model, "latency_ms": result.latency_ms, "retries": result.retries}
        if result.status != "SUCCESS" or result.output is None:
            code = result.error_code or "UNKNOWN"
            return self._action_failed(sha, atype, base, code, permanent=code in PERMANENT_ACTION_CODES)

        ext = MIME_EXTENSION.get(result.output.mime, ".png")
        staged = self._dirs["staging"] / f"{sha}__{atype}{ext}"
        atomic_write_bytes(staged, result.output.data)
        if not is_decodable(staged):
            staged.unlink(missing_ok=True)  # discard corrupt output. The original is untouched.
            return self._action_failed(sha, atype, base, "CORRUPTED_OUTPUT", permanent=False)

        record = dict(base, action_type=atype, staging_rel=self._rel(staged), output_sha256=sha256_file(staged), permanent=False)
        self._save_action(sha, record, status="RUNNING")
        return self._finalize_output(sha, atype, record, staged, folder_dir, plugin.output_suffix, rec["original_filename"])

    def _finalize_output(
        self, sha: str, atype: str, record: dict[str, Any], staged: Path, folder_dir: Path, suffix: str, original_name: str
    ) -> bool:
        ext = staged.suffix
        filename = f"{Path(original_name).stem}{suffix}{ext}"
        planned = deterministic_name(folder_dir, filename)
        record = dict(record, planned_output_rel=self._rel(planned))
        self._save_action(sha, record, status="RUNNING")
        try:
            final = route_file(staged, folder_dir, filename, "move", dest=planned)
        except (OSError, DestinationMissing) as exc:
            self._log.emit("finalize_retry", sha256=sha, error=repr(exc))
            return self._action_failed(sha, atype, record, "FINALIZE_FAILED", permanent=False)
        return self._action_succeeded(sha, atype, record, final)

    def _finalized_output(self, record: dict[str, Any]) -> Path | None:
        """Path of an output that was already moved into place, when its bytes still match."""
        planned_rel = record.get("planned_output_rel")
        if not planned_rel:
            return None
        planned = self._abs(planned_rel)
        if planned.exists() and sha256_file(planned) == record.get("output_sha256"):
            return planned
        return None

    def _action_succeeded(self, sha: str, atype: str, record: dict[str, Any], final: Path) -> bool:
        done = dict(record, action_type=atype, status="SUCCESS", output_rel=self._rel(final), error_code=None)
        self._save_action(sha, done, status=None)
        self.metrics.observe_action("SUCCESS", float(done.get("latency_ms") or 0.0))
        self.report.actioned += 1
        self._log.emit(
            "action_success",
            sha256=sha,
            action=atype,
            provider=done.get("provider"),
            model=done.get("model"),
            latency_ms=done.get("latency_ms"),
            retries=done.get("retries"),
            output=done["output_rel"],
        )
        return True

    def _action_failed(self, sha: str, atype: str, base: dict[str, Any], code: str, *, permanent: bool) -> bool:
        record = dict(base, action_type=atype, error_code=code, permanent=permanent)
        record["status"] = "FAILED"
        self._save_action(sha, record, status=None)
        self.manifest.transition(sha, ImageState.ACTION_FAILED, last_error=code)
        self.metrics.observe_action("FAILED", float(base.get("latency_ms") or 0.0))
        self.report.actions_failed += 1
        self._log.emit("action_failed", sha256=sha, action=atype, error_code=code, permanent=permanent)
        return False

    def _save_action(self, sha: str, record: dict[str, Any], status: str | None) -> None:
        """Persist one action record. ``status`` overrides the record's own status when given."""
        if status is not None:
            record = dict(record, status=status)
        rec = self.manifest.get(sha)
        actions = [a for a in rec["actions"] if a["action_type"] != record["action_type"]]
        actions.append(record)
        actions.sort(key=lambda a: a["action_type"])
        self.manifest.transition(sha, ImageState(rec["state"]), actions=actions)

    # ----- helpers ---------------------------------------------------------------
    def _state(self, sha: str) -> ImageState:
        return ImageState(self.manifest.get(sha)["state"])

    def _set_state(self, sha: str, target: ImageState) -> None:
        self.manifest.transition(sha, target)

    def _fail(self, sha: str, code: str) -> None:
        self.manifest.transition(sha, ImageState.FAILED, last_error=code)
        self.report.failed += 1
        self._log.emit("image_failed", sha256=sha, error_code=code)

    def _note_error(self, sha: str, code: str) -> None:
        self.manifest.transition(sha, self._state(sha), last_error=code)
        self._log.emit("image_waiting", sha256=sha, reason=code)

    def _folder_dir(self, folder_id: str) -> Path | None:
        if folder_id not in self.project.active_ids():
            return None
        folder = self.project.live.get(folder_id)
        if folder is None or not folder.path.is_dir():
            return None
        return folder.path

    def _rel(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def _abs(self, rel: str) -> Path:
        return self.root / rel

    def _finish(self, started: float) -> None:
        totals = {"complete": 0, "review_required": 0, "failed": 0, "action_failed": 0, "actions_success": 0}
        for rec in self.manifest.all():
            state = rec["state"]
            if state == ImageState.COMPLETE.value:
                totals["complete"] += 1
            elif state == ImageState.REVIEW_REQUIRED.value:
                totals["review_required"] += 1
            elif state == ImageState.FAILED.value:
                totals["failed"] += 1
            elif state == ImageState.ACTION_FAILED.value:
                totals["action_failed"] += 1
            totals["actions_success"] += sum(1 for a in rec["actions"] if a["status"] == "SUCCESS")
        self.report.totals = totals
        self.report.metrics = self.metrics.summary()
        self.report.metrics["total_job_duration_s"] = round(time.perf_counter() - started, 3)
        atomic_write_bytes(
            self._dirs["logs"] / f"{self.job_id}.metrics.json",
            _json_bytes(self.report.metrics),
        )
        self._log.emit("job_finished", totals=totals, routed=self.report.routed, actioned=self.report.actioned)


def _find_action(actions: list[dict[str, Any]], atype: str) -> dict[str, Any] | None:
    for record in actions:
        if record["action_type"] == atype:
            return record
    return None


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
