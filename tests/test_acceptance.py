"""Acceptance criteria A-I (spec section 45) plus the scenarios the spec names explicitly."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from imageflow.core.hashing import sha256_file
from imageflow.core.pipeline import JobAborted
from imageflow.core.state_machine import ImageState
from imageflow.errors import PermanentError, TransientError
from imageflow.preprocessing.image_loader import is_decodable
from imageflow.classifiers.base import Choice, ClassificationResult, ClassifierProvider
from imageflow.preprocessing.image_loader import EncodedImage
from imageflow.image_editors.base import ImageEditProvider, ImageEditResult
from imageflow.core.routing import route_file

from support import (
    BLUE,
    GREEN,
    PURPLE,
    RED,
    YELLOW,
    FailingClassifier,
    MockEditor,
    ScriptedClassifier,
    build_job,
    folder_files,
    folder_id,
    job_state,
    peaked,
    routed_to,
    run,
    sha_of,
    write_image,
)

PKG = Path(__file__).resolve().parents[1] / "imageflow"


# ----- A: dynamic taxonomy ---------------------------------------------------
def test_a_classifies_by_whatever_folders_the_user_created(tmp_path):
    root = tmp_path / "job"
    for name in ("A", "B", "C"):
        (root / name).mkdir(parents=True)
    write_image(root / "red.jpg", RED)
    write_image(root / "green.jpg", GREEN)
    write_image(root / "blue.jpg", BLUE)
    clf = ScriptedClassifier(routed_to({RED: "A", GREEN: "B", BLUE: "C"}))

    run(build_job(root, clf))

    assert clf.labels_seen[0] == ["A", "B", "C"]
    assert folder_files(root, "A") == ["red.jpg"]
    assert folder_files(root, "B") == ["green.jpg"]
    assert folder_files(root, "C") == ["blue.jpg"]


def test_a_same_code_classifies_against_new_folders_without_changes(tmp_path):
    root = tmp_path / "job_xy"
    (root / "X").mkdir(parents=True)
    (root / "Y").mkdir(parents=True)
    write_image(root / "red.jpg", RED)
    write_image(root / "green.jpg", GREEN)
    clf = ScriptedClassifier(routed_to({RED: "X", GREEN: "Y"}))

    run(build_job(root, clf))

    assert clf.labels_seen[0] == ["X", "Y"]
    assert folder_files(root, "X") == ["red.jpg"]
    assert folder_files(root, "Y") == ["green.jpg"]


# ----- B: no hard-coded categories -------------------------------------------
BUSINESS_TOKENS = [
    "An toàn", "an toàn", "Tiến độ", "Chất lượng", "Vật tư", "Thiết bị", "Nghiệm thu",
    "Thi công móng", "Kết cấu thép", "Mất an toàn", "Uncertain", "Không rõ",
]


def test_b_source_has_no_fixed_business_categories():
    offenders = []
    for py in PKG.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        offenders += [(str(py), token) for token in BUSINESS_TOKENS if token in text]
    assert offenders == []


def test_b_core_never_imports_vendor_modules():
    forbidden = ("imageflow.classifiers.clef", "imageflow.image_editors.seedream", "infrastructure.gateway", "cloudflare", "seedream")
    offenders = []
    for py in (PKG / "core").rglob("*.py"):
        text = py.read_text(encoding="utf-8").lower()
        offenders += [(str(py), token) for token in forbidden if token in text]
    assert offenders == []


# ----- C: action mapping is by assignment, never by folder name --------------
@pytest.mark.parametrize("assigned", ["Mất an toàn lao động", "Khu vực 7"])
def test_c_action_follows_the_assignment_not_the_name(tmp_path, assigned):
    root = tmp_path / "job"
    (root / "Mất an toàn lao động").mkdir(parents=True)
    (root / "Khu vực 7").mkdir(parents=True)
    write_image(root / "red.jpg", RED)
    write_image(root / "green.jpg", GREEN)
    clf = ScriptedClassifier(routed_to({RED: "Mất an toàn lao động", GREEN: "Khu vực 7"}))
    editor = MockEditor()
    job = build_job(root, clf, editor=editor)
    job.prepare()
    job.set_actions(folder_id(job, assigned), ["PPE_SIMULATION"])

    run(job)

    # RED is classified into the name-matching folder, GREEN into "Khu vực 7".
    expected_output = {"Mất an toàn lao động": "red_AI_PPE.png", "Khu vực 7": "green_AI_PPE.png"}[assigned]
    other = "Khu vực 7" if assigned == "Mất an toàn lao động" else "Mất an toàn lao động"
    assert len(editor.calls) == 1
    assert editor.calls[0]["folder_id"] == folder_id(job, assigned)
    assert (root / assigned / expected_output).exists()
    assert [n for n in folder_files(root, other) if n.endswith("_AI_PPE.png")] == []


# ----- D: original preservation -----------------------------------------------
def test_d_original_hash_is_unchanged_after_generative_edit(tmp_path):
    root = tmp_path / "job"
    (root / "Khu").mkdir(parents=True)
    write_image(root / "green.jpg", GREEN)
    before = sha_of(root, "green.jpg")
    clf = ScriptedClassifier(routed_to({GREEN: "Khu"}))
    editor = MockEditor()
    job = build_job(root, clf, editor=editor)
    job.prepare()
    job.set_actions(folder_id(job, "Khu"), ["PPE_SIMULATION"])

    run(job)

    assert sha_of(root, "Khu/green.jpg") == before  # original bytes untouched
    output = root / "Khu" / "green_AI_PPE.png"
    assert output.exists()
    assert sha256_file(output) != before  # the AI output is a separate, derived file
    assert is_decodable(output)
    rec = job.manifest.get(before)
    assert rec["state"] == ImageState.COMPLETE.value
    assert rec["actions"][0]["output_rel"] == "Khu/green_AI_PPE.png"


# ----- E: low confidence is never auto-routed --------------------------------
def test_e_low_confidence_stays_in_root_and_goes_to_review(tmp_path):
    root = tmp_path / "job"
    (root / "A").mkdir(parents=True)
    (root / "B").mkdir(parents=True)
    write_image(root / "red.jpg", RED)
    unsure = lambda colour, choices: {c.choice_id: (0.6 if c.label == "A" else 0.4) for c in choices}
    primary = ScriptedClassifier(unsure, model="clef-flash")
    fallback = ScriptedClassifier(unsure, model="clef")

    report = run(build_job(root, primary, fallback=fallback))

    assert report.routed == 0
    assert (root / "red.jpg").exists()  # the original is not moved
    assert folder_files(root, "A") == [] and folder_files(root, "B") == []
    sha = sha_of(root, "red.jpg")
    reloaded = build_job(root, primary, fallback=fallback, job_id="j_reload")
    reloaded.prepare()
    assert job_state(reloaded, sha) == ImageState.REVIEW_REQUIRED.value
    assert (root / ".imageflow" / "review" / f"{sha}.jpg").exists()  # review copy, never the only copy
    assert fallback.count == 1  # the fallback model was consulted before giving up


def test_e_manual_assignment_routes_the_reviewed_image(tmp_path):
    root = tmp_path / "job"
    (root / "A").mkdir(parents=True)
    write_image(root / "red.jpg", RED)
    unsure = lambda colour, choices: {c.choice_id: 0.5 for c in choices}
    job = build_job(root, ScriptedClassifier(unsure), fallback=ScriptedClassifier(unsure))
    run(job)
    sha = sha_of(root, "red.jpg")
    assert job_state(job, sha) == ImageState.REVIEW_REQUIRED.value

    import asyncio

    asyncio.run(job.assign(sha, folder_id(job, "A")))

    assert folder_files(root, "A") == ["red.jpg"]
    assert not (root / "red.jpg").exists()
    assert not (root / ".imageflow" / "review" / f"{sha}.jpg").exists()  # copy cleaned up after routing
    assert job_state(job, sha) == ImageState.COMPLETE.value


# ----- F: crash recovery -----------------------------------------------------
class SimulatedCrash(BaseException):
    """BaseException, so no ``except Exception`` in the pipeline can swallow it."""


@pytest.mark.parametrize("stage", ["before_link", "after_link", "after_move"])
def test_f_crash_during_routing_resumes_without_duplicates(tmp_path, monkeypatch, stage):
    import imageflow.core.pipeline as pipeline_mod

    root = tmp_path / "job"
    (root / "Khu").mkdir(parents=True)
    write_image(root / "red.jpg", RED)
    write_image(root / "green.jpg", GREEN)
    write_image(root / "blue.jpg", BLUE)
    clf = ScriptedClassifier(routed_to({RED: "Khu", GREEN: "Khu", BLUE: "Khu"}))
    real_route = pipeline_mod.route_file
    state = {"calls": 0}

    def crashing_route(src, folder, filename, mode, *, dest=None, temp_dir=None):
        state["calls"] += 1
        if state["calls"] != 2:
            return real_route(src, folder, filename, mode, dest=dest, temp_dir=temp_dir)
        if stage == "before_link":
            raise SimulatedCrash()
        if stage == "after_link":
            os.link(src, dest)  # the destination exists, the source is not yet removed
            raise SimulatedCrash()
        real_route(src, folder, filename, mode, dest=dest, temp_dir=temp_dir)  # fully moved, manifest not yet updated
        raise SimulatedCrash()

    monkeypatch.setattr(pipeline_mod, "route_file", crashing_route)
    with pytest.raises(BaseException) as info:
        run(build_job(root, clf, job_id="j_crash1"))
    raised = getattr(info.value, "exceptions", (info.value,))
    assert any(isinstance(e, SimulatedCrash) for e in raised), "the simulated crash must actually happen"

    monkeypatch.setattr(pipeline_mod, "route_file", real_route)
    run(build_job(root, clf, job_id="j_crash2"))

    expected_plan = {"before_link": "REROUTE", "after_link": "FINISH_MOVE", "after_move": "DONE"}[stage]
    events = [json.loads(line) for line in (root / ".imageflow" / "logs" / "j_crash2.jsonl").read_text().splitlines()]
    assert any(e["event"] == "route_recovered" and e["plan"] == expected_plan for e in events), (
        f"the interrupted move must be recovered with {expected_plan}"
    )
    assert folder_files(root, "Khu") == ["blue.jpg", "green.jpg", "red.jpg"]  # no duplicates, no __2 copies
    assert not any(p.is_file() for p in root.iterdir())  # root holds no leftover original
    job = build_job(root, clf, job_id="j_check")
    job.prepare()
    assert len(job.manifest.all()) == 3
    assert all(r["state"] == ImageState.COMPLETE.value for r in job.manifest.all())


def test_f_rerunning_a_finished_job_changes_nothing(tmp_path):
    root = tmp_path / "job"
    (root / "Khu").mkdir(parents=True)
    write_image(root / "red.jpg", RED)
    clf = ScriptedClassifier(routed_to({RED: "Khu"}))
    editor = MockEditor()
    job = build_job(root, clf, editor=editor)
    job.prepare()
    job.set_actions(folder_id(job, "Khu"), ["PPE_SIMULATION"])
    run(job)
    snapshot = sorted(p.name for p in (root / "Khu").iterdir())

    second = build_job(root, clf, editor=editor, job_id="j_again")
    second.prepare()
    second.set_actions(folder_id(second, "Khu"), ["PPE_SIMULATION"])
    run(second)

    assert clf.count == 1  # no second classification
    assert len(editor.calls) == 1  # no second generative edit
    assert sorted(p.name for p in (root / "Khu").iterdir()) == snapshot


# ----- G: rename -------------------------------------------------------------
def test_g_rename_keeps_folder_id_actions_and_routes_into_the_renamed_folder(tmp_path):
    root = tmp_path / "job"
    (root / "Nghiệm thu").mkdir(parents=True)
    job1 = build_job(root, ScriptedClassifier(routed_to({})), editor=MockEditor())
    job1.prepare()
    fid = folder_id(job1, "Nghiệm thu")
    job1.set_actions(fid, ["PPE_SIMULATION"])

    os.rename(root / "Nghiệm thu", root / "Ảnh cần xử lý")
    write_image(root / "green.jpg", GREEN)
    clf = ScriptedClassifier(routed_to({GREEN: "Ảnh cần xử lý"}))
    editor = MockEditor()
    job2 = build_job(root, clf, editor=editor, job_id="j_renamed")
    run(job2)

    assert folder_id(job2, "Ảnh cần xử lý") == fid  # same identity after the rename
    assert job2.project.actions_for(fid) == ["PPE_SIMULATION"]
    assert len(editor.calls) == 1
    assert (root / "Ảnh cần xử lý" / "green_AI_PPE.png").exists()
    events = [json.loads(line) for line in (root / ".imageflow" / "logs" / "j_renamed.jsonl").read_text().splitlines()]
    assert any(e["event"] == "folder_renamed" and e["folder_id"] == fid for e in events)


def test_g_a_removed_folder_is_marked_missing_and_keeps_its_actions(tmp_path):
    root = tmp_path / "job"
    (root / "Khu").mkdir(parents=True)
    job = build_job(root, ScriptedClassifier(routed_to({})), editor=MockEditor())
    job.prepare()
    fid = folder_id(job, "Khu")
    job.set_actions(fid, ["PPE_SIMULATION"])

    os.rmdir(root / "Khu")
    job2 = build_job(root, ScriptedClassifier(routed_to({})), editor=MockEditor(), job_id="j_missing")
    job2.prepare()

    entry = next(f for f in job2.project.folders if f["folder_id"] == fid)
    assert entry["status"] == "MISSING"
    assert entry["actions"] == ["PPE_SIMULATION"]  # never silently moved to another folder
    assert fid not in job2.project.active_ids()


# ----- H: any classifier provider can replace the default --------------------
class AltDecisionModel(ClassifierProvider):
    name = "alt"

    def __init__(self, model: str) -> None:
        self.model = model
        self.count = 0

    async def classify(self, image: EncodedImage, choices: list[Choice]) -> ClassificationResult:
        self.count += 1
        winner = choices[0].choice_id  # always the first folder, deliberately simple
        return ClassificationResult(choice_id=winner, confidence=0.99, probabilities=peaked(choices, winner, 0.99), model=self.model)


def test_h_pipeline_runs_unchanged_with_a_replacement_classifier(tmp_path):
    root = tmp_path / "job"
    (root / "A").mkdir(parents=True)
    (root / "B").mkdir(parents=True)
    write_image(root / "red.jpg", RED)
    alt = AltDecisionModel("replacement-1")

    run(build_job(root, alt))

    assert alt.count == 1
    assert folder_files(root, "A") == ["red.jpg"]


# ----- I: any editor can replace Seedream ------------------------------------
class AltEditor(ImageEditProvider):
    name = "alt-editor"

    def __init__(self) -> None:
        self.model = "alt-edit-1"
        self.count = 0

    async def edit(self, image: EncodedImage, instruction: str, metadata: dict) -> ImageEditResult:
        self.count += 1
        return ImageEditResult(image=EncodedImage(data=image.data, mime="image/jpeg"), model=self.model)


def test_i_ppe_action_runs_with_a_replacement_editor(tmp_path):
    root = tmp_path / "job"
    (root / "Khu").mkdir(parents=True)
    write_image(root / "purple.jpg", PURPLE)
    clf = ScriptedClassifier(routed_to({PURPLE: "Khu"}))
    editor = AltEditor()
    job = build_job(root, clf, editor=editor)
    job.prepare()
    job.set_actions(folder_id(job, "Khu"), ["PPE_SIMULATION"])

    run(job)

    assert editor.count == 1
    assert (root / "Khu" / "purple_AI_PPE.jpg").exists()  # mime decides the extension


# ----- edge cases the spec calls out -----------------------------------------
def test_nested_folders_are_not_categories(tmp_path):
    root = tmp_path / "job"
    (root / "Mất an toàn lao động" / "minh_hoa").mkdir(parents=True)
    job = build_job(root, ScriptedClassifier(routed_to({})))
    job.prepare()
    assert [c.label for c in job.choices()] == ["Mất an toàn lao động"]


def test_rule_file_becomes_part_of_the_choice_description(tmp_path):
    root = tmp_path / "job"
    (root / "Mất an toàn lao động").mkdir(parents=True)
    (root / "Mất an toàn lao động" / ".rule.txt").write_text("Người không đủ PPE.", encoding="utf-8")
    job = build_job(root, ScriptedClassifier(routed_to({})))
    job.prepare()
    (choice,) = job.choices()
    assert choice.description == "Mất an toàn lao động: Người không đủ PPE."


def test_unsupported_formats_are_reported_and_left_alone(tmp_path):
    root = tmp_path / "job"
    (root / "Khu").mkdir(parents=True)
    (root / "IMG.heic").write_bytes(b"not really heic")
    write_image(root / "red.jpg", RED)
    report = run(build_job(root, ScriptedClassifier(routed_to({RED: "Khu"}))))
    assert report.unsupported == ["IMG.heic"]
    assert (root / "IMG.heic").exists()


def test_corrupt_image_fails_without_classification(tmp_path):
    root = tmp_path / "job"
    (root / "Khu").mkdir(parents=True)
    (root / "broken.jpg").write_bytes(b"\x00\x01 definitely not a jpeg")
    clf = ScriptedClassifier(routed_to({}))
    job = build_job(root, clf)
    run(job)
    assert clf.count == 0
    assert (root / "broken.jpg").exists()
    states = [r["state"] for r in job.manifest.all()]
    assert states == [ImageState.FAILED.value]


def test_name_collision_gets_a_deterministic_suffix(tmp_path):
    root = tmp_path / "job"
    (root / "Khu").mkdir(parents=True)
    write_image(root / "Khu" / "red.jpg", YELLOW)  # an existing, different file with the same name
    existing = sha_of(root, "Khu/red.jpg")
    write_image(root / "red.jpg", RED)
    run(build_job(root, ScriptedClassifier(routed_to({RED: "Khu"}))))
    assert sha_of(root, "Khu/red.jpg") == existing  # the existing file is never overwritten
    assert folder_files(root, "Khu") == ["red.jpg", "red__2.jpg"]


def test_duplicate_bytes_are_distinct_file_instances(tmp_path):
    root = tmp_path / "job"
    (root / "Khu").mkdir(parents=True)
    write_image(root / "a.jpg", RED)
    write_image(root / "b.jpg", RED)  # identical bytes
    clf = ScriptedClassifier(routed_to({RED: "Khu"}))
    report = run(build_job(root, clf))
    assert clf.count == 2
    assert report.skipped_duplicates == 0
    assert folder_files(root, "Khu") == ["a.jpg", "b.jpg"]


def test_classifier_outage_moves_nothing_and_next_run_completes(tmp_path):
    root = tmp_path / "job"
    (root / "Khu").mkdir(parents=True)
    write_image(root / "red.jpg", RED)
    down = FailingClassifier(TransientError("UPSTREAM_UNAVAILABLE"))
    report = run(build_job(root, down))
    assert down.count == 3  # one call plus MAX_RETRIES = 2
    assert report.deferred == 1
    assert (root / "red.jpg").exists() and folder_files(root, "Khu") == []

    healthy = ScriptedClassifier(routed_to({RED: "Khu"}))
    run(build_job(root, healthy, job_id="j_recovered"))
    assert folder_files(root, "Khu") == ["red.jpg"]


def test_auth_failure_aborts_the_job_and_moves_nothing(tmp_path):
    root = tmp_path / "job"
    (root / "Khu").mkdir(parents=True)
    write_image(root / "red.jpg", RED)
    with pytest.raises(JobAborted):
        run(build_job(root, FailingClassifier(PermanentError("AUTH_FAILED"))))
    assert (root / "red.jpg").exists()
    assert folder_files(root, "Khu") == []


def test_no_folders_means_nothing_is_classified_yet(tmp_path):
    root = tmp_path / "job"
    root.mkdir(parents=True)
    write_image(root / "red.jpg", RED)
    clf = ScriptedClassifier(routed_to({}))
    run(build_job(root, clf))
    assert clf.count == 0
    assert (root / "red.jpg").exists()


def test_failed_editor_call_keeps_the_routed_original_and_retries_next_run(tmp_path):
    root = tmp_path / "job"
    (root / "Khu").mkdir(parents=True)
    write_image(root / "green.jpg", GREEN)
    clf = ScriptedClassifier(routed_to({GREEN: "Khu"}))
    bad = MockEditor(error=TransientError("UPSTREAM_UNAVAILABLE"))
    job = build_job(root, clf, editor=bad)
    job.prepare()
    job.set_actions(folder_id(job, "Khu"), ["PPE_SIMULATION"])
    run(job)
    sha = sha_of(root, "Khu/green.jpg")
    assert job_state(job, sha) == ImageState.ACTION_FAILED.value
    assert (root / "Khu" / "green.jpg").exists()  # original still intact

    good = MockEditor()
    job2 = build_job(root, clf, editor=good, job_id="j_retry")
    job2.prepare()
    job2.set_actions(folder_id(job2, "Khu"), ["PPE_SIMULATION"])
    run(job2)
    assert job_state(job2, sha) == ImageState.COMPLETE.value
    assert (root / "Khu" / "green_AI_PPE.png").exists()
    assert clf.count == 1  # the routed image was not classified again


def test_route_file_never_overwrites(tmp_path):
    src = write_image(tmp_path / "src" / "x.jpg", RED)
    dst_dir = tmp_path / "dst"
    dst_dir.mkdir()
    write_image(dst_dir / "x.jpg", GREEN)
    existing_before = sha256_file(dst_dir / "x.jpg")
    src_sha = sha256_file(src)

    final = route_file(src, dst_dir, "x.jpg", "copy")

    assert final.name == "x__2.jpg"
    assert sha256_file(dst_dir / "x.jpg") == existing_before  # the existing file is untouched
    assert sha256_file(final) == src_sha
    assert src.exists()  # copy mode keeps the source
