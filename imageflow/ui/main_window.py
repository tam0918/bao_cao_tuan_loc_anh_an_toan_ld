"""Desktop window (spec sections 29-31). tkinter ships with the Python installer on Windows.

The window only drives the same job API the CLI uses. Long work runs on a worker thread,
and results come back through a queue that the Tk thread polls, because Tk objects must
not be touched from other threads.
"""

from __future__ import annotations

import asyncio
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

from ..app import KNOWN_OUTPUT_SUFFIXES, ConfigError, build_job
from ..core.pipeline import ImageFlowJob
from ..core.workspace import scan_workspace
from . import texts

PPE_ACTION = "PPE_SIMULATION"
POLL_MS = 150


class ImageFlowWindow(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(texts.APP_TITLE)
        self.minsize(480, 420)
        self._events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._root_dir: Path | None = None
        self._job: ImageFlowJob | None = None
        self._action_vars: dict[str, tk.BooleanVar] = {}
        self._body = ttk.Frame(self, padding=24)
        self._body.pack(fill="both", expand=True)
        self.show_home()

    # ----- screens -------------------------------------------------------------
    def _clear(self) -> None:
        for widget in self._body.winfo_children():
            widget.destroy()

    def _label(self, text: str, **kwargs: Any) -> ttk.Label:
        label = ttk.Label(self._body, text=text, justify="center", wraplength=420, **kwargs)
        label.pack(pady=8)
        return label

    def show_home(self) -> None:
        self._clear()
        self._label(texts.HOME_TITLE, font=("Segoe UI", 16, "bold"))
        self._label(texts.HOME_HINT)
        ttk.Button(self._body, text=texts.CHOOSE_FOLDER, command=self.choose_folder).pack(pady=20)

    def choose_folder(self) -> None:
        chosen = filedialog.askdirectory(title=texts.CHOOSE_FOLDER)
        if chosen:
            self.open_folder(Path(chosen))

    def open_folder(self, root: Path) -> None:
        self._root_dir = root
        scan = scan_workspace(root, KNOWN_OUTPUT_SUFFIXES)
        self._clear()
        self._label(texts.scan_summary(len(scan.images), len(scan.folders), len(scan.unsupported)), font=("Segoe UI", 12))
        self._action_vars = {}
        if not scan.folders:
            self._label(texts.NO_FOLDERS)
        for folder in scan.folders:
            row = ttk.Frame(self._body)
            row.pack(fill="x", pady=4)
            ttk.Label(row, text=folder.name, width=28).pack(side="left")
            var = tk.BooleanVar(value=False)
            ttk.Checkbutton(row, text=texts.PPE_ACTION_LABEL, variable=var).pack(side="left")
            self._action_vars[folder.name] = var
        buttons = ttk.Frame(self._body)
        buttons.pack(pady=20)
        ttk.Button(buttons, text=texts.BACK, command=self.show_home).pack(side="left", padx=6)
        start = ttk.Button(buttons, text=texts.START, command=self.start)
        start.pack(side="left", padx=6)
        if not scan.folders:
            start.state(["disabled"])

    def start(self) -> None:
        if self._root_dir is None:
            return
        try:
            self._job = build_job(self._root_dir, os.environ)
        except ConfigError as exc:
            messagebox.showerror(texts.APP_TITLE, str(exc))
            return
        self._job.prepare()
        ids = {f["name"]: f["folder_id"] for f in self._job.project.folders}
        for name, var in self._action_vars.items():
            self._job.set_actions(ids[name], [PPE_ACTION] if var.get() else [])
        self._clear()
        self._label(texts.PROGRESS, font=("Segoe UI", 12))
        bar = ttk.Progressbar(self._body, mode="indeterminate", length=320)
        bar.pack(pady=20)
        bar.start(12)
        threading.Thread(target=self._run_job, daemon=True).start()
        self.after(POLL_MS, self._poll)

    def show_done(self, totals: dict[str, int], review: int) -> None:
        self._clear()
        self._label(texts.done_summary(totals.get("complete", 0), totals.get("actions_success", 0), review), font=("Segoe UI", 12))
        buttons = ttk.Frame(self._body)
        buttons.pack(pady=20)
        ttk.Button(buttons, text=texts.OPEN_FOLDER, command=lambda: _open_path(self._root_dir)).pack(side="left", padx=6)
        if review:
            ttk.Button(buttons, text=texts.REVIEW_BUTTON, command=self.show_review).pack(side="left", padx=6)
        ttk.Button(buttons, text=texts.BACK, command=self.show_home).pack(side="left", padx=6)

    def show_review(self) -> None:
        if self._job is None:
            return
        items = self._job.review_items()
        self._clear()
        self._label(texts.REVIEW_TITLE, font=("Segoe UI", 14, "bold"))
        if not items:
            self._label(texts.REVIEW_EMPTY)
        folder_names = sorted(f["name"] for f in self._job.project.folders if f["status"] == "ACTIVE")
        for item in items:
            row = ttk.Frame(self._body)
            row.pack(fill="x", pady=4)
            ttk.Label(row, text=texts.review_row(item["name"]), width=30).pack(side="left")
            combo = ttk.Combobox(row, values=folder_names, state="readonly", width=24)
            combo.pack(side="left", padx=4)
            ttk.Button(
                row,
                text=texts.MOVE_BUTTON,
                command=lambda sha=item["sha256"], c=combo: self._assign(sha, c.get()),
            ).pack(side="left")
        ttk.Button(self._body, text=texts.BACK, command=self.show_home).pack(pady=20)

    # ----- background work -----------------------------------------------------
    def _run_job(self) -> None:
        try:
            report = asyncio.run(self._job.run())
            self._events.put(("done", report))
        except Exception as exc:  # the job is aborted, for example by a revoked device token
            self._events.put(("error", str(exc)))

    def _assign(self, sha: str, folder_name: str) -> None:
        if not folder_name or self._job is None:
            return
        folder_id = next(f["folder_id"] for f in self._job.project.folders if f["name"] == folder_name)
        threading.Thread(target=self._assign_worker, args=(sha, folder_id), daemon=True).start()
        self.after(POLL_MS, self._poll)

    def _assign_worker(self, sha: str, folder_id: str) -> None:
        try:
            asyncio.run(self._job.assign(sha, folder_id))
            self._events.put(("reviewed", None))
        except Exception as exc:
            self._events.put(("error", str(exc)))

    def _poll(self) -> None:
        try:
            kind, payload = self._events.get_nowait()
        except queue.Empty:
            self.after(POLL_MS, self._poll)
            return
        if kind == "done":
            review = payload.totals.get("review_required", 0)
            self.show_done(payload.totals, review)
        elif kind == "reviewed":
            self.show_review()
        else:
            messagebox.showerror(texts.APP_TITLE, str(payload))
            self.show_home()


def _open_path(path: Path | None) -> None:
    if path is None:
        return
    if sys.platform.startswith("win"):
        os.startfile(path)  # type: ignore[attr-defined]  # Windows only
    elif sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=False)
    else:
        subprocess.run(["xdg-open", str(path)], check=False)


def launch() -> None:
    ImageFlowWindow().mainloop()
