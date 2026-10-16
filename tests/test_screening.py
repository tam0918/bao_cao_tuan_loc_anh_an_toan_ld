"""The second stage: every routed photo is screened for missing PPE, and only those get an illustration.

The stage was unreachable for a while because a leftover copy of _run_actions shadowed it, and nothing
tested it below the end-to-end level, so each outcome has its own test here.
"""

from __future__ import annotations

import ast
import asyncio
from pathlib import Path

import pytest

from imageflow.core.pipeline import JobAborted
from imageflow.core.safety import PPE_NEEDS_EDIT, PPE_OK, PPE_UNCLEAR
from imageflow.core.state_machine import ImageState
from imageflow.errors import PermanentError, TransientError

from support import (
    BLUE,
    GREEN,
    RED,
    FailingClassifier,
    MockEditor,
    ScriptedClassifier,
    build_job,
    folder_files,
    job_state,
    nearest,
    peaked,
    routed_to,
    run,
    write_image,
)

PKG = Path(__file__).resolve().parents[1] / "imageflow"


def ppe_verdicts(by_colour):
    """A screening model that judges the photo by its colour."""

    def decide(colour, choices):
        return peaked(choices, nearest(colour, by_colour))

    return decide


def three_photos(tmp_path):
    root = tmp_path / "job"
    (root / "Khu").mkdir(parents=True)
    write_image(root / "missing.jpg", RED)
    write_image(root / "fine.jpg", GREEN)
    write_image(root / "unclear.jpg", BLUE)
    folders = ScriptedClassifier(routed_to({RED: "Khu", GREEN: "Khu", BLUE: "Khu"}))
    screen = ScriptedClassifier(ppe_verdicts({RED: PPE_NEEDS_EDIT, GREEN: PPE_OK, BLUE: PPE_UNCLEAR}))
    return root, folders, screen


def test_only_photos_that_need_it_get_an_illustration(tmp_path):
    root, folders, screen = three_photos(tmp_path)
    editor = MockEditor()

    report = run(build_job(root, folders, editor=editor, safety=screen))

    assert folder_files(root, "Khu") == ["fine.jpg", "missing.jpg", "missing_AI_PPE.png", "unclear.jpg"]
    assert len(editor.calls) == 1  # only the photo with a worker missing PPE reaches the paid model
    assert report.safety_review_required == 1  # the unclear one is counted for a human, never guessed
    assert report.actioned == 1


def test_every_photo_ends_complete_and_is_screened_exactly_once(tmp_path):
    root, folders, screen = three_photos(tmp_path)
    editor = MockEditor()
    job = build_job(root, folders, editor=editor, safety=screen)
    run(job)
    assert all(r["state"] == ImageState.COMPLETE.value for r in job.manifest.all())
    assert [r["safety"]["status"] for r in sorted(job.manifest.all(), key=lambda r: r["original_filename"])] == ["SAFE", "NEEDS_EDIT", "REVIEW"]

    run(build_job(root, folders, editor=editor, safety=screen, job_id="j_again"))

    assert screen.count == 3 and len(editor.calls) == 1  # a rerun neither re-screens nor re-edits


def test_screening_does_not_depend_on_what_the_user_assigned_to_the_folder(tmp_path):
    root, folders, screen = three_photos(tmp_path)
    job = build_job(root, folders, editor=MockEditor(), safety=screen)
    job.prepare()
    assert job.project.actions_for(job.resolve_folder("Khu")) == []  # nothing assigned

    run(job)

    assert (root / "Khu" / "missing_AI_PPE.png").exists()


def test_an_outage_during_screening_keeps_the_photos_where_they_are_and_a_rerun_finishes(tmp_path):
    root, folders, screen = three_photos(tmp_path)
    editor = MockEditor()
    job = build_job(root, folders, editor=editor, safety=FailingClassifier(TransientError("UPSTREAM_UNAVAILABLE")))

    report = run(job)

    assert report.deferred == 3 and report.unfinished == 3  # the screens could not be made, and that is not 'done'
    assert folder_files(root, "Khu") == ["fine.jpg", "missing.jpg", "unclear.jpg"]  # already filed, nothing lost
    assert {r["state"] for r in job.manifest.all()} == {ImageState.ROUTED.value}
    assert editor.calls == []

    run(build_job(root, folders, editor=editor, safety=screen, job_id="j_back"))

    assert (root / "Khu" / "missing_AI_PPE.png").exists() and len(editor.calls) == 1


def test_a_revoked_token_during_screening_aborts_with_its_code(tmp_path):
    root, folders, _ = three_photos(tmp_path)
    job = build_job(root, folders, editor=MockEditor(), safety=FailingClassifier(PermanentError("AUTH_FAILED")))
    with pytest.raises(JobAborted) as info:
        asyncio.run(job.run())
    assert info.value.code == "AUTH_FAILED"


# ----- photos are file instances, not contents ---------------------------------------------------------------
def test_two_identical_photos_in_review_are_two_items_that_can_be_moved_separately(tmp_path):
    root = tmp_path / "job"
    (root / "A").mkdir(parents=True)
    write_image(root / "one.jpg", RED)
    write_image(root / "two.jpg", RED)  # the same bytes under another name
    unsure = lambda colour, choices: {c.choice_id: 0.5 for c in choices}  # noqa: E731
    job = build_job(root, ScriptedClassifier(unsure), fallback=ScriptedClassifier(unsure))
    run(job)

    items = job.review_items()
    assert sorted(i["name"] for i in items) == ["one.jpg", "two.jpg"]
    assert len({i["image_id"] for i in items}) == 2 and len({i["sha256"] for i in items}) == 1
    review_copies = sorted(p.name for p in (root / ".imageflow" / "review").iterdir())
    assert len(review_copies) == 2  # each photo has its own review copy

    first = next(i for i in items if i["name"] == "one.jpg")
    assert asyncio.run(job.assign(first["image_id"], job.resolve_folder("A"))) is None

    assert folder_files(root, "A") == ["one.jpg"]
    assert [i["name"] for i in job.review_items()] == ["two.jpg"]
    assert len(list((root / ".imageflow" / "review").iterdir())) == 1  # the other photo's copy survives
    second = next(i for i in items if i["name"] == "two.jpg")
    assert (root / "two.jpg").exists()
    assert job_state(job, second["image_id"]) == ImageState.REVIEW_REQUIRED.value
    assert job.manifest.get(first["sha256"]) is None  # a content hash shared by two photos names neither of them


# ----- source hygiene ------------------------------------------------------------------------------------------
def duplicate_methods():
    found = []
    for path in sorted(PKG.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
            seen: dict[str, int] = {}
            for node in cls.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name in seen and not node.decorator_list:
                        found.append(f"{path.relative_to(PKG.parent)}:{node.lineno} {cls.name}.{node.name} (first at line {seen[node.name]})")
                    seen.setdefault(node.name, node.lineno)
    return found


def test_no_class_defines_the_same_method_twice():
    # Python silently keeps the last definition, which once hid the whole PPE screening stage.
    assert duplicate_methods() == []
