"""Concurrency bound, metrics contract, and permanent action errors (spec sections 25, 26, 41)."""

from __future__ import annotations

import asyncio
import json

from imageflow.classifiers.base import Choice, ClassificationResult, ClassifierProvider
from imageflow.core.state_machine import ImageState
from imageflow.errors import PermanentError
from imageflow.preprocessing.image_loader import EncodedImage

from support import (
    GREEN,
    RED,
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


class TrackingClassifier(ClassifierProvider):
    name = "tracking"
    model = "tracking-1"

    def __init__(self) -> None:
        self.in_flight = 0
        self.peak = 0

    async def classify(self, image: EncodedImage, choices: list[Choice]) -> ClassificationResult:
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            await asyncio.sleep(0.02)  # give the scheduler a chance to overlap calls
            winner = choices[0].choice_id
            return ClassificationResult(winner, 0.99, peaked(choices, winner, 0.99), self.model)
        finally:
            self.in_flight -= 1


def test_classification_concurrency_is_bounded_by_configuration(tmp_path):
    root = tmp_path / "job"
    (root / "A").mkdir(parents=True)
    for i in range(10):
        write_image(root / f"img_{i:02d}.jpg", (20 * i, 40, 200 - 10 * i))
    clf = TrackingClassifier()

    run(build_job(root, clf))  # build_job sets CLASSIFY_CONCURRENCY = 3

    assert clf.peak == 3  # parallel, and never above the configured bound
    assert len(folder_files(root, "A")) == 10


REQUIRED_METRICS = {
    "classification_count",
    "classification_latency_p50_ms",
    "classification_latency_p95_ms",
    "clef_flash_accept_rate",
    "clef_fallback_rate",
    "manual_review_rate",
    "images_routed",
    "images_actioned",
    "edit_success_rate",
    "edit_latency_p50_ms",
    "edit_latency_p95_ms",
    "total_job_duration_s",
}


def test_metrics_cover_the_required_set_and_are_written_to_disk(tmp_path):
    root = tmp_path / "job"
    (root / "Khu").mkdir(parents=True)
    write_image(root / "red.jpg", RED)
    job = build_job(root, ScriptedClassifier(routed_to({RED: "Khu"})), editor=MockEditor(), job_id="j_metrics")
    job.prepare()
    job.set_actions(folder_id(job, "Khu"), ["PPE_SIMULATION"])

    report = run(job)

    assert REQUIRED_METRICS <= set(report.metrics)
    assert report.metrics["images_routed"] == 1
    assert report.metrics["images_actioned"] == 1
    assert report.metrics["edit_success_rate"] == 1.0
    on_disk = json.loads((root / ".imageflow" / "logs" / "j_metrics.metrics.json").read_text(encoding="utf-8"))
    assert REQUIRED_METRICS <= set(on_disk)


def test_permanent_action_error_waits_for_a_human_instead_of_retrying(tmp_path):
    root = tmp_path / "job"
    (root / "Khu").mkdir(parents=True)
    write_image(root / "green.jpg", GREEN)
    clf = ScriptedClassifier(routed_to({GREEN: "Khu"}))
    editor = MockEditor(error=PermanentError("UNSUPPORTED_IMAGE"))
    job = build_job(root, clf, editor=editor)
    job.prepare()
    job.set_actions(folder_id(job, "Khu"), ["PPE_SIMULATION"])
    run(job)
    sha = sha_of(root, "Khu/green.jpg")
    assert job_state(job, sha) == ImageState.ACTION_FAILED.value

    again = build_job(root, clf, editor=editor, job_id="j_again")
    again.prepare()
    run(again)

    assert len(editor.calls) == 1  # a permanent error is not paid for again on every run
    rec = again.manifest.get(sha)
    assert rec["actions"][0]["permanent"] is True
    assert rec["actions"][0]["error_code"] == "UNSUPPORTED_IMAGE"
