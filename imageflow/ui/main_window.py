"""Desktop window (spec sections 29-31). tkinter ships with the Python installer on Windows.

The window only draws. What to do, and every long operation, lives in SessionController, which is
covered by headless tests. Results come back through poll() on the Tk thread, because Tk objects
must not be touched from other threads.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

from ..actions.ppe_simulation import ACTION_TYPE as PPE_ACTION
from ..app import ConfigError, build_job
from ..core.pipeline import JobReport
from . import texts
from .controller import Busy, SessionController, TaskResult, completion_numbers

POLL_MS = 150


class ImageFlowWindow(tk.Tk):
    def __init__(self, controller: SessionController | None = None) -> None:
        super().__init__()
        self.title(texts.APP_TITLE)
        self.minsize(520, 440)
        self._controller = controller or SessionController(lambda root: build_job(root, os.environ))
        self._body = ttk.Frame(self, padding=24)
        self._body.pack(fill="both", expand=True)
        self._root_dir: Path | None = None
        self._ppe_vars: dict[str, tk.BooleanVar] = {}  # folder_id -> ticked
        self._review_rows: dict[str, tuple[ttk.Frame, ttk.Button]] = {}  # sha256 -> (row, move button)
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.show_home()

    def report_callback_exception(self, exc: type[BaseException], val: BaseException, tb: Any) -> None:
        # Tk calls this when a callback raises. Without it a windowed app swallows the error silently.
        traceback.print_exception(exc, val, tb)
        messagebox.showerror(texts.APP_TITLE, texts.UNEXPECTED_ERROR.format(error=val))

    def _close(self) -> None:
        self._controller.close()
        self.destroy()

    # ----- screens -------------------------------------------------------------
    def _clear(self) -> None:
        for widget in self._body.winfo_children():
            widget.destroy()

    def _label(self, text: str, **kwargs: Any) -> ttk.Label:
        label = ttk.Label(self._body, text=text, justify="center", wraplength=440, **kwargs)
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
        try:
            result = self._controller.open(root)
        except Busy:
            return
        except ConfigError as exc:
            messagebox.showerror(texts.APP_TITLE, texts.CONFIG_ERROR.format(detail=exc))
            return
        except OSError as exc:  # the folder vanished or cannot be read
            messagebox.showerror(texts.APP_TITLE, texts.UNEXPECTED_ERROR.format(error=exc))
            return
        self._root_dir = root
        self._clear()
        self._label(texts.scan_summary(result.images, len(result.folders), result.unsupported), font=("Segoe UI", 12))
        self._ppe_vars = {}
        if not result.folders:
            self._label(texts.NO_FOLDERS)
        elif result.screening_is_global:
            self._label(texts.PPE_AUTOMATIC)  # nothing to choose per folder, so no boxes that would do nothing
        for row in result.folders:
            line = ttk.Frame(self._body)
            line.pack(fill="x", pady=4)
            ttk.Label(line, text=row.name, width=28).pack(side="left")
            if not result.screening_is_global:
                ticked = tk.BooleanVar(value=PPE_ACTION in row.actions)  # show what was saved, never reset it silently
                ttk.Checkbutton(line, text=texts.PPE_ACTION_LABEL, variable=ticked).pack(side="left")
                self._ppe_vars[row.folder_id] = ticked
        buttons = ttk.Frame(self._body)
        buttons.pack(pady=20)
        ttk.Button(buttons, text=texts.BACK, command=self.show_home).pack(side="left", padx=6)
        start = ttk.Button(buttons, text=texts.START, command=self.start)
        start.pack(side="left", padx=6)
        if not result.folders:
            start.state(["disabled"])

    def start(self) -> None:
        try:
            self._controller.start({folder_id: var.get() for folder_id, var in self._ppe_vars.items()})
        except Busy:
            return
        self._clear()
        self._label(texts.PROGRESS, font=("Segoe UI", 12))
        bar = ttk.Progressbar(self._body, mode="indeterminate", length=320)
        bar.pack(pady=20)
        bar.start(50)
        self.after(POLL_MS, self._poll)

    def show_done(self, report: JobReport) -> None:
        numbers = completion_numbers(report)
        self._clear()
        self._label(texts.done_summary(**numbers), font=("Segoe UI", 12))
        buttons = ttk.Frame(self._body)
        buttons.pack(pady=20)
        ttk.Button(buttons, text=texts.OPEN_FOLDER, command=lambda: _open_path(self._root_dir)).pack(side="left", padx=6)
        if numbers["review"]:
            ttk.Button(buttons, text=texts.REVIEW_BUTTON, command=self.show_review).pack(side="left", padx=6)
        ttk.Button(buttons, text=texts.BACK, command=self.show_home).pack(side="left", padx=6)

    def show_review(self) -> None:
        self._clear()
        self._review_rows = {}
        self._label(texts.REVIEW_TITLE, font=("Segoe UI", 14, "bold"))
        items = self._controller.review_items()
        by_name = {choice.name: choice.folder_id for choice in self._controller.review_choices()}
        if not items:
            self._label(texts.REVIEW_EMPTY)
        for item in items:
            line = ttk.Frame(self._body)
            line.pack(fill="x", pady=4)
            ttk.Label(line, text=texts.review_row(item["name"]), width=30).pack(side="left")
            box = ttk.Combobox(line, values=list(by_name), state="readonly", width=24)
            suggestion = next((c["name"] for c in item["candidates"] if c["name"] in by_name), None)
            if suggestion:
                box.set(suggestion)  # the model's best guess, for the person to confirm or change
            box.pack(side="left", padx=4)
            move = ttk.Button(
                line,
                text=texts.MOVE_BUTTON,
                command=lambda sha=item["sha256"], chooser=box: self._move(sha, by_name.get(chooser.get())),
            )
            move.pack(side="left")
            self._review_rows[item["sha256"]] = (line, move)
        ttk.Button(self._body, text=texts.BACK, command=self.show_home).pack(pady=20)

    def _move(self, sha256: str, folder_id: str | None) -> None:
        if folder_id is None:
            return  # nothing chosen yet
        try:
            self._controller.assign(sha256, folder_id)
        except Busy:
            return  # a double click: the first one is already running
        self._set_review_enabled(False)
        self.after(POLL_MS, self._poll)

    def _set_review_enabled(self, enabled: bool) -> None:
        for _, button in self._review_rows.values():
            button.state(["!disabled"] if enabled else ["disabled"])

    # ----- results from the worker thread --------------------------------------
    def _poll(self) -> None:
        result = self._controller.poll()
        if result is None:
            self.after(POLL_MS, self._poll)
            return
        if result.kind == "run":
            self._after_run(result)
        else:
            self._after_assign(result)

    def _after_run(self, result: TaskResult) -> None:
        if result.error is not None:
            messagebox.showerror(texts.APP_TITLE, texts.abort_message(result.code, result.error))
            self.show_home()
        else:
            assert result.report is not None
            self.show_done(result.report)

    def _after_assign(self, result: TaskResult) -> None:
        if result.error is not None:
            messagebox.showerror(texts.APP_TITLE, texts.abort_message(result.code, result.error))
            self.show_review()
            return
        if result.problem == "DESTINATION_MISSING":
            messagebox.showwarning(texts.APP_TITLE, texts.problem_message(result.problem))
            self.show_review()  # the image is back in the list, and the folder list is refreshed
            return
        row = self._review_rows.pop(result.sha256 or "", None)
        if row is not None:
            row[0].destroy()  # only the moved row goes; the choices made in the other rows stay
        self._set_review_enabled(True)
        if result.problem:
            messagebox.showinfo(texts.APP_TITLE, texts.problem_message(result.problem))
        if not self._review_rows:
            self.show_review()  # shows the empty message


def _open_path(path: Path | None) -> None:
    if path is None:
        return
    if sys.platform.startswith("win"):
        os.startfile(path)  # type: ignore[attr-defined]  # Windows only
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


def launch() -> None:
    ImageFlowWindow().mainloop()
