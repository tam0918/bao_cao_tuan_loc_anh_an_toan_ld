"""The window's decisions, tested without a display (spec sections 29-31)."""

from __future__ import annotations

import asyncio

import pytest

from imageflow.actions.base import ActionPlugin, ActionRegistry
from imageflow.actions.ppe_simulation import PpeSimulationAction
from imageflow.core.pipeline import ImageFlowJob, JobReport, PipelineConfig
from imageflow.core.state_machine import ImageState
from imageflow.errors import TransientError
from imageflow.ui.controller import Busy, SessionController, completion_numbers

from support import (
    GREEN,
    RED,
    FailingClassifier,
    MockEditor,
    ScriptedClassifier,
    build_engine,
    build_job,
    folder_files,
    no_sleep,
    routed_to,
    sha_of,
    write_image,
)

UNSURE = lambda colour, choices: {c.choice_id: 0.5 for c in choices}  # noqa: E731


class OtherAction(ActionPlugin):
    """A second action the window does not offer, to prove it is left alone."""

    action_type = "OTHER_ACTION"
    output_suffix = "_AI_OTHER"

    async def execute(self, source_image, context):
        raise AssertionError("never run in these tests")


def job_with_two_actions(root, classifier, fallback=None):
    registry = ActionRegistry()
    registry.register(PpeSimulationAction(MockEditor(), sleep=no_sleep))
    registry.register(OtherAction())
    return ImageFlowJob(root, build_engine(classifier, fallback), registry, config=PipelineConfig(), job_id="j_ctl")


def controller_for(classifier, *, fallback=None, editor=None):
    return SessionController(lambda root: build_job(root, classifier, fallback=fallback, editor=editor or MockEditor()))


# ----- opening ----------------------------------------------------------------------------------------------
def test_open_lists_the_folders_with_the_actions_saved_earlier(tmp_path):
    root = tmp_path / "job"
    (root / "Khu A").mkdir(parents=True)
    (root / "Khu B").mkdir()
    write_image(root / "red.jpg", RED)
    (root / "note.heic").write_bytes(b"x")
    earlier = build_job(root, ScriptedClassifier(UNSURE), editor=MockEditor())
    earlier.prepare()
    earlier.set_actions(earlier.resolve_folder("Khu B"), ["PPE_SIMULATION"])

    opened = controller_for(ScriptedClassifier(UNSURE)).open(root)

    assert (opened.images, opened.unsupported) == (1, 1)
    assert [(r.name, r.actions) for r in opened.folders] == [("Khu A", ()), ("Khu B", ("PPE_SIMULATION",))]


# ----- start ------------------------------------------------------------------------------------------------
def test_start_applies_the_ticks_and_keeps_actions_the_window_does_not_offer(tmp_path):
    root = tmp_path / "job"
    (root / "Khu").mkdir(parents=True)
    job = job_with_two_actions(root, ScriptedClassifier(UNSURE))
    job.prepare()
    folder_id = job.resolve_folder("Khu")
    job.set_actions(folder_id, ["OTHER_ACTION", "PPE_SIMULATION"])
    ctl = SessionController(lambda _: job)
    ctl.open(root)

    ctl.start({folder_id: False})
    ctl.wait(timeout=10)
    assert job.project.actions_for(folder_id) == ["OTHER_ACTION"]  # PPE unticked, the other one untouched

    ctl.start({folder_id: True})
    ctl.wait(timeout=10)
    assert job.project.actions_for(folder_id) == ["OTHER_ACTION", "PPE_SIMULATION"]


def test_start_survives_a_folder_renamed_after_the_window_opened(tmp_path):
    root = tmp_path / "job"
    (root / "Khu A").mkdir(parents=True)
    write_image(root / "red.jpg", RED)
    ctl = controller_for(ScriptedClassifier(routed_to({RED: "Khu B"})))
    (row,) = ctl.open(root).folders
    (root / "Khu A").rename(root / "Khu B")

    ctl.start({row.folder_id: True})
    result = ctl.wait(timeout=10)

    assert result.error is None
    assert (root / "Khu B" / "red.jpg").exists() and (root / "Khu B" / "red_AI_PPE.png").exists()


def test_a_second_task_is_refused_until_the_first_result_is_collected(tmp_path):
    root = tmp_path / "job"
    (root / "Khu").mkdir(parents=True)
    ctl = controller_for(ScriptedClassifier(UNSURE))
    ctl.open(root)

    ctl.start({})
    assert ctl.busy
    with pytest.raises(Busy):
        ctl.start({})  # a double click
    with pytest.raises(Busy):
        ctl.open(root)
    ctl.wait(timeout=10)

    assert not ctl.busy
    ctl.open(root)  # free again


def test_poll_is_empty_while_the_task_runs(tmp_path):
    root = tmp_path / "job"
    (root / "Khu").mkdir(parents=True)
    write_image(root / "red.jpg", RED)

    class Slow(ScriptedClassifier):
        async def classify(self, image, choices):
            await asyncio.sleep(0.3)
            return await super().classify(image, choices)

    ctl = controller_for(Slow(routed_to({RED: "Khu"})))
    ctl.open(root)
    ctl.start({})
    assert ctl.poll() is None
    assert ctl.wait(timeout=10).report.routed == 1


def test_an_outage_comes_back_as_a_message_with_a_code_not_an_exception(tmp_path):
    root = tmp_path / "job"
    (root / "Khu").mkdir(parents=True)
    for i in range(6):
        write_image(root / f"img{i}.jpg", (30 * i, 40, 90))
    ctl = controller_for(FailingClassifier(TransientError("NETWORK")))
    ctl.open(root)
    ctl.start({})

    result = ctl.wait(timeout=10)

    assert result.report is None and result.code == "SERVICE_UNAVAILABLE" and "unreachable" in result.error


def test_a_bug_inside_a_task_is_reported_instead_of_hanging_the_window(tmp_path, monkeypatch):
    root = tmp_path / "job"
    (root / "Khu").mkdir(parents=True)
    job = build_job(root, ScriptedClassifier(UNSURE))

    async def boom():
        raise RuntimeError("boom")

    monkeypatch.setattr(job, "run", boom)
    ctl = SessionController(lambda _: job)
    ctl.open(root)
    ctl.start({})

    result = ctl.wait(timeout=10)

    assert result.kind == "run" and result.error == "RuntimeError: boom"
    assert not ctl.busy


# ----- review -----------------------------------------------------------------------------------------------
def make_review(tmp_path, folders=("A", "B")):
    root = tmp_path / "job"
    for name in folders:
        (root / name).mkdir(parents=True)
    write_image(root / "red.jpg", RED)
    write_image(root / "green.jpg", GREEN)
    unsure = lambda colour, choices: {c.choice_id: (0.6 if c.label == folders[0] else 0.4) for c in choices}  # noqa: E731
    ctl = controller_for(ScriptedClassifier(unsure), fallback=ScriptedClassifier(unsure))
    ctl.open(root)
    ctl.start({})
    ctl.wait(timeout=10)
    return root, ctl


def test_review_offers_the_models_guess_first_and_only_live_folders(tmp_path):
    root, ctl = make_review(tmp_path)
    items = ctl.review_items()
    assert {i["name"] for i in items} == {"red.jpg", "green.jpg"}
    assert [c["name"] for c in items[0]["candidates"]][:2] == ["A", "B"]  # best guess first
    assert [c.name for c in ctl.review_choices()] == ["A", "B"]


def test_assigning_a_reviewed_image_moves_it_and_leaves_the_rest_waiting(tmp_path):
    root, ctl = make_review(tmp_path)
    folder_a = next(c.folder_id for c in ctl.review_choices() if c.name == "A")
    sha = sha_of(root, "red.jpg")

    ctl.assign(sha, folder_a)
    result = ctl.wait(timeout=10)

    assert (result.kind, result.sha256, result.problem, result.error) == ("assign", sha, None, None)
    assert folder_files(root, "A") == ["red.jpg"]
    assert [i["name"] for i in ctl.review_items()] == ["green.jpg"]


def test_a_second_click_on_move_is_refused_instead_of_raising_an_error_dialog(tmp_path):
    root, ctl = make_review(tmp_path)
    folder_a = ctl.review_choices()[0].folder_id
    sha = sha_of(root, "red.jpg")

    ctl.assign(sha, folder_a)
    with pytest.raises(Busy):
        ctl.assign(sha, folder_a)
    ctl.wait(timeout=10)

    assert folder_files(root, "A") == ["red.jpg"]  # moved exactly once


def test_assigning_an_image_that_is_no_longer_in_review_is_a_clear_error(tmp_path):
    root, ctl = make_review(tmp_path)
    folder_a = ctl.review_choices()[0].folder_id
    sha = sha_of(root, "red.jpg")
    ctl.assign(sha, folder_a)
    ctl.wait(timeout=10)

    ctl.assign(sha, folder_a)
    result = ctl.wait(timeout=10)

    assert result.problem is None and "not waiting for review" in result.error


def test_a_move_that_could_not_happen_is_reported_with_its_reason(tmp_path, monkeypatch):
    root, ctl = make_review(tmp_path)
    job = ctl._job
    folder_a = ctl.review_choices()[0].folder_id
    sha = sha_of(root, "red.jpg")
    monkeypatch.setattr(job, "_folder_dir", lambda folder_id: None)  # the folder vanished mid-way

    ctl.assign(sha, folder_a)
    result = ctl.wait(timeout=10)

    assert result.problem == "DESTINATION_MISSING"
    assert job.manifest.get(sha)["state"] == ImageState.REVIEW_REQUIRED.value
    assert sha in {i["sha256"] for i in ctl.review_items()}


def test_review_offers_the_recreated_folder_not_its_stale_twin(tmp_path):
    root, ctl = make_review(tmp_path)
    old = {c.name: c.folder_id for c in ctl.review_choices()}
    job = ctl._job
    for record in job.project.folders:
        record["ino"] += 10_000_000  # Windows gives a recreated folder a new file id
    job.project.save()
    ctl2 = SessionController(lambda r: build_job(r, ScriptedClassifier(UNSURE), editor=MockEditor()))
    ctl2.open(root)

    now = {c.name: c.folder_id for c in ctl2.review_choices()}

    assert set(now) == {"A", "B"} and now["A"] != old["A"]
    sha = sha_of(root, "red.jpg")
    ctl2.assign(sha, now["A"])
    assert ctl2.wait(timeout=10).problem is None  # it works, where the stale id raised an error before
    assert folder_files(root, "A") == ["red.jpg"]


# ----- completion screen figures ------------------------------------------------------------------------------
def test_completion_numbers_count_unfinished_work_that_a_rerun_could_finish():
    report = JobReport(job_id="j", deferred=2, internal_errors=1)
    report.totals = {"complete": 9, "actions_success": 4, "review_required": 3, "failed": 1, "action_failed": 5}
    assert completion_numbers(report) == {"classified": 9, "actioned": 4, "review": 3, "pending": 8, "failed": 1}


def test_a_clean_run_has_nothing_pending():
    report = JobReport(job_id="j")
    report.totals = {"complete": 3}
    assert completion_numbers(report)["pending"] == 0

