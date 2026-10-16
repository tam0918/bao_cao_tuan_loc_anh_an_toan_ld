"""Headless controller behind the desktop window (spec sections 29-31).

It owns the job and runs every long operation on one worker thread, one at a time. It imports no
tkinter, so what the window decides is covered by tests. The window only draws and calls in here.

One job is used by one event loop at a time, so operations are serialised: a double click on a
button gets ``Busy`` instead of racing the first click.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..actions.ppe_simulation import ACTION_TYPE as PPE_ACTION
from ..core.pipeline import FolderRow, ImageFlowJob, JobAborted, JobReport


class Busy(RuntimeError):
    """A task is still running, or its result has not been collected with poll() yet."""


@dataclass(frozen=True)
class OpenResult:
    images: int
    unsupported: int
    folders: tuple[FolderRow, ...]
    # True when every routed photo is screened for missing PPE automatically. A per-folder tick then
    # has no effect, so the window must not offer one.
    screening_is_global: bool = False


@dataclass(frozen=True)
class FolderChoice:
    name: str
    folder_id: str


@dataclass(frozen=True)
class TaskResult:
    kind: str  # "run" or "assign"
    report: JobReport | None = None
    problem: str | None = None  # assign: why the image was not fully handled. None when it was
    sha256: str | None = None  # assign: which image
    error: str | None = None  # the task stopped, with a message
    code: str | None = None  # JobAborted code, when that was the reason


def completion_numbers(report: JobReport) -> dict[str, int]:
    """The figures for the completion screen, from one place so the window and the tests agree."""
    totals = report.totals
    return {
        "classified": totals.get("complete", 0),
        "actioned": totals.get("actions_success", 0),
        "review": totals.get("review_required", 0),
        "pending": report.unfinished,
        "failed": totals.get("failed", 0),
        "safety_review": report.safety_review_required,
    }


class SessionController:
    def __init__(self, factory: Callable[[Path], ImageFlowJob]) -> None:
        self._factory = factory
        self._job: ImageFlowJob | None = None
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="imageflow")
        self._pending: Future[TaskResult] | None = None
        self._pending_kind = ""
        self.root: Path | None = None

    @property
    def busy(self) -> bool:
        return self._pending is not None

    # ----- opening a workspace ---------------------------------------------------------------------
    def open(self, root: Path) -> OpenResult:
        """Build the job for ``root`` and read its folders, including the actions saved earlier."""
        self._require_idle()
        job = self._factory(root)
        scan = job.prepare()
        self._job, self.root = job, root
        return OpenResult(
            images=len(scan.images),
            unsupported=len(scan.unsupported),
            folders=tuple(job.folder_rows()),
            screening_is_global=job.safety_screen is not None,
        )

    # ----- background work -------------------------------------------------------------------------
    def start(self, ppe: Mapping[str, bool]) -> None:
        """Apply the ticked boxes (folder_id -> wanted), then process the job in the background."""
        job = self._require_job()
        self._require_idle()
        self._submit("run", self._run, job, dict(ppe))

    def assign(self, sha256: str, folder_id: str) -> None:
        """Send one reviewed image to the chosen folder in the background."""
        job = self._require_job()
        self._require_idle()
        self._submit("assign", self._assign, job, sha256, folder_id)

    @staticmethod
    def _run(job: ImageFlowJob, ppe: dict[str, bool]) -> TaskResult:
        try:
            for folder_id, wanted in ppe.items():
                actions = set(job.project.actions_for(folder_id))  # keep any action this window does not offer
                actions = actions | {PPE_ACTION} if wanted else actions - {PPE_ACTION}
                job.set_actions(folder_id, sorted(actions))
            return TaskResult("run", report=asyncio.run(job.run()))
        except JobAborted as exc:
            return TaskResult("run", error=str(exc), code=exc.code)

    @staticmethod
    def _assign(job: ImageFlowJob, sha256: str, folder_id: str) -> TaskResult:
        try:
            problem = asyncio.run(job.assign(sha256, folder_id))
        except ValueError as exc:  # no longer in review, or the folder is gone
            return TaskResult("assign", sha256=sha256, error=str(exc))
        except JobAborted as exc:
            return TaskResult("assign", sha256=sha256, error=str(exc), code=exc.code)
        return TaskResult("assign", sha256=sha256, problem=problem)

    def _submit(self, kind: str, fn: Callable[..., TaskResult], *args: Any) -> None:
        self._pending_kind = kind
        self._pending = self._pool.submit(fn, *args)

    # ----- results -----------------------------------------------------------------------------------
    def poll(self) -> TaskResult | None:
        """The finished task's result, or None while it is still running. Collecting it frees the controller."""
        future = self._pending
        if future is None or not future.done():
            return None
        self._pending = None
        try:
            return future.result()
        except Exception as exc:  # a bug in the task: show it instead of waiting forever
            return TaskResult(self._pending_kind, error=f"{type(exc).__name__}: {exc}")

    def wait(self, timeout: float | None = None) -> TaskResult:
        """Block until the running task finishes. For tests and scripts; the window uses poll()."""
        future = self._pending
        if future is None:
            raise RuntimeError("no task is running")
        wait([future], timeout=timeout)
        result = self.poll()
        if result is None:
            raise TimeoutError("the task did not finish in time")
        return result

    # ----- review screen -----------------------------------------------------------------------------
    def review_items(self) -> list[dict[str, Any]]:
        return self._require_job().review_items()

    def review_choices(self) -> list[FolderChoice]:
        """Folders an image can be sent to: only live ones, so a deleted folder is never offered."""
        return [FolderChoice(name=c.label, folder_id=c.choice_id) for c in self._require_job().choices()]

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    def _require_job(self) -> ImageFlowJob:
        if self._job is None:
            raise RuntimeError("open a folder first")
        return self._job

    def _require_idle(self) -> None:
        if self._pending is not None:
            raise Busy("another task is still running")
