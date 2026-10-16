"""The real window code driven against a fake tkinter: clicks go in, the screen text comes out.

This checks the window's wiring (what it asks the controller, what it shows). It cannot check how
the window looks, which still needs a run on a machine with a display.
"""

from __future__ import annotations

import sys
import time
import types

import pytest

import fake_tk
from imageflow.errors import TransientError
from imageflow.ui import texts
from imageflow.ui.controller import SessionController

from imageflow.core.safety import PPE_NEEDS_EDIT, PPE_OK, PPE_UNCLEAR

from support import (
    BLUE,
    GREEN,
    RED,
    FailingClassifier,
    MockEditor,
    ScriptedClassifier,
    build_job,
    folder_files,
    nearest,
    peaked,
    routed_to,
    write_image,
)

UNSURE_A = lambda colour, choices: {c.choice_id: (0.6 if c.label == "A" else 0.4) for c in choices}  # noqa: E731


@pytest.fixture
def ui(monkeypatch):
    module, messagebox, filedialog = fake_tk.install(monkeypatch)
    windows = []

    def make(factory):
        window = module.ImageFlowWindow(controller=SessionController(factory))
        windows.append(window)
        return window

    yield types.SimpleNamespace(module=module, messagebox=messagebox, filedialog=filedialog, make=make)
    for window in windows:
        window._controller.close()
    sys.modules.pop("imageflow.ui.main_window", None)


def ppe_verdicts(by_colour):
    def decide(colour, choices):
        return peaked(choices, nearest(colour, by_colour))

    return decide


def screen_text(window) -> str:
    return "\n".join(w.text for w in window._body.walk() if isinstance(w, fake_tk.Label))


def button(window, text):
    return next(w for w in window._body.walk() if isinstance(w, fake_tk.Button) and w.text == text)


def pump(window, until, timeout=10.0):
    """Run the queued Tk callbacks until ``until()`` is true, like the real event loop would."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        window.run_after()
        if until():
            return
        time.sleep(0.01)
    raise AssertionError("the window never reached the expected screen:\n" + screen_text(window))


def workspace(tmp_path, folders=("Khu",), images=((RED, "red.jpg"),)):
    root = tmp_path / "job"
    for name in folders:
        (root / name).mkdir(parents=True)
    for colour, name in images:
        write_image(root / name, colour)
    return root


def run_to_the_end(window):
    button(window, texts.START).invoke()
    pump(window, lambda: "Hoàn tất" in screen_text(window) or "Chưa hoàn tất" in screen_text(window))


# ----- the folder screen -------------------------------------------------------------------------------------
def test_choosing_a_folder_shows_the_counts_and_one_box_per_folder(ui, tmp_path):
    root = workspace(tmp_path, folders=("Khu A", "Khu B"), images=((RED, "r.jpg"), (GREEN, "g.jpg")))
    window = ui.make(lambda r: build_job(r, ScriptedClassifier(UNSURE_A), editor=MockEditor()))
    ui.filedialog.answer = str(root)

    button(window, texts.CHOOSE_FOLDER).invoke()

    assert texts.scan_summary(2, 2, 0) in screen_text(window)
    boxes = [w for w in window._body.walk() if isinstance(w, fake_tk.Checkbutton)]
    assert len(boxes) == 2 and not any(b.options["variable"].get() for b in boxes)


def test_a_missing_configuration_is_explained_and_the_window_stays_usable(ui, tmp_path):
    from imageflow.app import ConfigError

    def no_gateway(root):
        raise ConfigError("set IMAGEFLOW_GATEWAY_URL and IMAGEFLOW_DEVICE_TOKEN before running a job")

    window = ui.make(no_gateway)
    window.open_folder(workspace(tmp_path))

    assert ui.messagebox.calls and ui.messagebox.calls[0][0] == "error"
    assert "IMAGEFLOW_GATEWAY_URL" in ui.messagebox.calls[0][1] and "người quản trị" in ui.messagebox.calls[0][1]
    assert texts.HOME_TITLE in screen_text(window)


# ----- saved choices are not silently erased -----------------------------------------------------------------
def test_the_saved_action_is_shown_ticked_and_pressing_start_keeps_it(ui, tmp_path):
    root = workspace(tmp_path)
    earlier = build_job(root, ScriptedClassifier(UNSURE_A), editor=MockEditor())
    earlier.prepare()
    earlier.set_actions(earlier.resolve_folder("Khu"), ["PPE_SIMULATION"])
    window = ui.make(lambda r: build_job(r, ScriptedClassifier(routed_to({RED: "Khu"})), editor=MockEditor()))

    window.open_folder(root)
    (box,) = [w for w in window._body.walk() if isinstance(w, fake_tk.Checkbutton)]
    assert box.options["variable"].get() is True  # it used to start unticked and then wipe the saved choice
    run_to_the_end(window)

    assert (root / "Khu" / "red_AI_PPE.png").exists()
    check = build_job(root, ScriptedClassifier(UNSURE_A), editor=MockEditor())
    check.prepare()
    assert check.project.actions_for(check.resolve_folder("Khu")) == ["PPE_SIMULATION"]


def test_unticking_the_box_turns_the_action_off(ui, tmp_path):
    root = workspace(tmp_path)
    earlier = build_job(root, ScriptedClassifier(UNSURE_A), editor=MockEditor())
    earlier.prepare()
    earlier.set_actions(earlier.resolve_folder("Khu"), ["PPE_SIMULATION"])
    window = ui.make(lambda r: build_job(r, ScriptedClassifier(routed_to({RED: "Khu"})), editor=MockEditor()))
    window.open_folder(root)
    (box,) = [w for w in window._body.walk() if isinstance(w, fake_tk.Checkbutton)]
    box.options["variable"].set(False)

    run_to_the_end(window)

    assert folder_files(root, "Khu") == ["red.jpg"]  # no illustration was made


def test_a_folder_renamed_after_the_window_opened_does_not_break_start(ui, tmp_path):
    root = workspace(tmp_path, folders=("Khu A",))
    window = ui.make(lambda r: build_job(r, ScriptedClassifier(routed_to({RED: "Khu B"})), editor=MockEditor()))
    window.open_folder(root)
    (root / "Khu A").rename(root / "Khu B")
    (box,) = [w for w in window._body.walk() if isinstance(w, fake_tk.Checkbutton)]
    box.options["variable"].set(True)

    run_to_the_end(window)  # it used to raise KeyError inside the Tk callback and show nothing

    assert ui.messagebox.calls == []
    assert (root / "Khu B" / "red_AI_PPE.png").exists()


# ----- automatic PPE screening -------------------------------------------------------------------------------
def test_with_automatic_screening_there_are_no_boxes_that_would_do_nothing(ui, tmp_path):
    root = workspace(tmp_path, images=((RED, "missing.jpg"), (GREEN, "fine.jpg"), (BLUE, "unclear.jpg")))
    folders = ScriptedClassifier(routed_to({RED: "Khu", GREEN: "Khu", BLUE: "Khu"}))
    screen = ScriptedClassifier(ppe_verdicts({RED: PPE_NEEDS_EDIT, GREEN: PPE_OK, BLUE: PPE_UNCLEAR}))
    window = ui.make(lambda r: build_job(r, folders, editor=MockEditor(), safety=screen))

    window.open_folder(root)

    assert not [w for w in window._body.walk() if isinstance(w, fake_tk.Checkbutton)]
    assert texts.PPE_AUTOMATIC in screen_text(window)
    run_to_the_end(window)
    assert "1 ảnh được tạo minh họa PPE" in screen_text(window)
    assert "1 ảnh chưa xác định được PPE, nên xem lại" in screen_text(window)  # not silently treated as fine
    assert (root / "Khu" / "missing_AI_PPE.png").exists()


# ----- the completion screen is honest -----------------------------------------------------------------------
def test_a_successful_run_says_it_is_done_and_offers_the_folder(ui, tmp_path, monkeypatch):
    root = workspace(tmp_path)
    window = ui.make(lambda r: build_job(r, ScriptedClassifier(routed_to({RED: "Khu"})), editor=MockEditor()))
    opened = []
    monkeypatch.setattr(ui.module, "_open_path", opened.append)
    window.open_folder(root)

    run_to_the_end(window)

    assert "Hoàn tất" in screen_text(window) and "1 ảnh đã phân loại" in screen_text(window)
    button(window, texts.OPEN_FOLDER).invoke()
    assert opened == [root]


def test_work_left_for_another_run_is_not_reported_as_done(ui, tmp_path):
    root = workspace(tmp_path, images=((RED, "r.jpg"), (GREEN, "g.jpg")))
    window = ui.make(lambda r: build_job(r, FailingClassifier(TransientError("UPSTREAM_UNAVAILABLE")), editor=MockEditor()))
    window.open_folder(root)

    run_to_the_end(window)

    assert "Chưa hoàn tất" in screen_text(window)
    assert "2 ảnh chưa xử lý xong" in screen_text(window)  # it used to say "Hoàn tất" with all zeros


def test_an_unreachable_service_ends_with_a_clear_message_and_the_first_screen(ui, tmp_path):
    root = workspace(tmp_path, images=[((30 * i, 40, 90), f"img{i}.jpg") for i in range(6)])
    window = ui.make(lambda r: build_job(r, FailingClassifier(TransientError("NETWORK")), editor=MockEditor()))
    window.open_folder(root)

    button(window, texts.START).invoke()
    pump(window, lambda: bool(ui.messagebox.calls))

    assert ui.messagebox.calls == [("error", texts.abort_message("SERVICE_UNAVAILABLE", ""))]
    assert texts.HOME_TITLE in screen_text(window)
    assert len(list(root.glob("*.jpg"))) == 6  # nothing was moved


# ----- the review screen -------------------------------------------------------------------------------------
def open_review(ui, tmp_path):
    root = workspace(tmp_path, folders=("A", "B"), images=((RED, "red.jpg"), (GREEN, "green.jpg")))
    window = ui.make(lambda r: build_job(r, ScriptedClassifier(UNSURE_A), fallback=ScriptedClassifier(UNSURE_A), editor=MockEditor()))
    window.open_folder(root)
    run_to_the_end(window)
    button(window, texts.REVIEW_BUTTON).invoke()
    return root, window


def test_review_preselects_the_models_best_guess_and_lists_live_folders(ui, tmp_path):
    root, window = open_review(ui, tmp_path)
    boxes = [w for w in window._body.walk() if isinstance(w, fake_tk.Combobox)]
    assert len(boxes) == 2
    assert all(b.get() == "A" for b in boxes)
    assert all(b.options["values"] == ["A", "B"] for b in boxes)


def test_moving_one_image_removes_only_its_row_and_keeps_the_other_choices(ui, tmp_path):
    root, window = open_review(ui, tmp_path)
    rows = {w.options["text"]: w for w in window._body.walk() if isinstance(w, fake_tk.Label) and w.options.get("width") == 30}
    red_row = next(row for text, row in rows.items() if text.startswith("red.jpg"))
    green_row = next(row for text, row in rows.items() if text.startswith("green.jpg"))
    combos = {row: next(c for c in row.parent.children if isinstance(c, fake_tk.Combobox)) for row in (red_row, green_row)}
    combos[green_row].set("B")  # a choice made, but not yet moved
    next(b for b in red_row.parent.children if isinstance(b, fake_tk.Button)).invoke()
    pump(window, lambda: folder_files(root, "A") == ["red.jpg"])
    pump(window, lambda: red_row.parent.destroyed)

    assert combos[green_row].get() == "B"  # the screen was not rebuilt, so this survived
    assert not green_row.parent.destroyed


def test_a_double_click_on_move_neither_errors_nor_moves_twice(ui, tmp_path):
    root, window = open_review(ui, tmp_path)
    move = next(w for w in window._body.walk() if isinstance(w, fake_tk.Button) and w.text == texts.MOVE_BUTTON)

    move.invoke()
    move.invoke()  # the second click of a double click
    pump(window, lambda: len(folder_files(root, "A")) == 1 and not window._controller.busy)  # idle: nothing late to come

    assert ui.messagebox.calls == []  # it used to show "image is not waiting for review" and jump home
    assert len(folder_files(root, "A")) == 1


def test_moving_the_last_image_shows_that_nothing_is_left(ui, tmp_path):
    root, window = open_review(ui, tmp_path)
    for _ in range(2):
        move = next(w for w in window._body.walk() if isinstance(w, fake_tk.Button) and w.text == texts.MOVE_BUTTON)
        before = len(folder_files(root, "A"))
        move.invoke()
        pump(window, lambda: len(folder_files(root, "A")) == before + 1)
        pump(window, lambda: not window._controller.busy)
    assert texts.REVIEW_EMPTY in screen_text(window)


# ----- anything unexpected is shown, not swallowed ----------------------------------------------------------
def test_an_unexpected_error_in_a_callback_becomes_a_dialog(ui, tmp_path):
    window = ui.make(lambda r: build_job(r, ScriptedClassifier(UNSURE_A), editor=MockEditor()))
    window.report_callback_exception(ValueError, ValueError("boom"), None)
    assert ui.messagebox.calls == [("error", texts.UNEXPECTED_ERROR.format(error="boom"))]
