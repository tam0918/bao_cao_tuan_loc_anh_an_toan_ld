"""Regression tests for defects found when reviewing the CLI and window commit."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import http.client
import os
import socket
import threading
from http.server import BaseHTTPRequestHandler
from io import BytesIO

import pytest
from PIL import Image

import imageflow.core.manifest as manifest_mod
import imageflow.core.pipeline as pipeline_mod
import imageflow.core.workspace as workspace_mod
from imageflow.core.manifest import ManifestStore
from imageflow.core.pipeline import JobAborted, PipelineConfig, UnknownFolder
from imageflow.core.project import ProjectStore
from imageflow.core.state_machine import ImageState, check_transition
from imageflow.core.workspace import scan_workspace
from imageflow.errors import PermanentError, TransientError
from imageflow.image_editors.base import ImageEditProvider, ImageEditResult
from imageflow.infrastructure.gateway import GatewayClient
from imageflow.preprocessing.image_loader import EncodedImage, is_decodable, normalize_file

from support import (
    GREEN,
    RED,
    FailingClassifier,
    ScriptedClassifier,
    build_job,
    folder_files,
    job_state,
    json_handler,
    noisy_jpeg_bytes,
    routed_to,
    run,
    serve,
    sha_of,
    write_image,
)

UNSURE = lambda colour, choices: {c.choice_id: 0.5 for c in choices}  # noqa: E731


def cut(blob: bytes) -> bytes:
    return blob[: len(blob) * 6 // 10]


# ----- truncated images: an interrupted copy is the common field failure -------------------------------
def test_normalize_rejects_a_truncated_jpeg_as_corrupt(tmp_path):
    damaged = tmp_path / "cut.jpg"
    damaged.write_bytes(cut(noisy_jpeg_bytes()))
    with pytest.raises(PermanentError) as info:
        normalize_file(damaged, tmp_path / "cache", "sha-cut")
    assert info.value.code == "CORRUPTED_IMAGE"


def test_a_truncated_photo_fails_once_instead_of_erroring_on_every_run(tmp_path):
    root = tmp_path / "job"
    (root / "Khu").mkdir(parents=True)
    (root / "cut.jpg").write_bytes(cut(noisy_jpeg_bytes()))
    clf = ScriptedClassifier(routed_to({}))
    job = build_job(root, clf)

    run(job)  # support.run fails the test if any exception was swallowed as an internal error

    (rec,) = job.manifest.all()
    assert rec["state"] == ImageState.FAILED.value and rec["last_error"] == "CORRUPTED_IMAGE"
    assert clf.count == 0 and (root / "cut.jpg").exists()


def test_is_decodable_rejects_truncated_images(tmp_path):
    jpeg = noisy_jpeg_bytes()
    as_png = BytesIO()
    Image.open(BytesIO(jpeg)).save(as_png, format="PNG")
    png = as_png.getvalue()
    for name, blob, expected in (
        ("whole.jpg", jpeg, True),
        ("cut.jpg", cut(jpeg), False),
        ("whole.png", png, True),
        ("cut.png", cut(png), False),
    ):
        path = tmp_path / name
        path.write_bytes(blob)
        assert is_decodable(path) is expected, name


class TruncatingEditor(ImageEditProvider):
    name, model = "truncating", "truncating-1"

    async def edit(self, image, instruction, metadata):
        return ImageEditResult(EncodedImage(data=cut(noisy_jpeg_bytes()), mime="image/jpeg"), model=self.model)


def test_a_truncated_ai_output_is_discarded_and_the_original_is_untouched(tmp_path):
    root = tmp_path / "job"
    (root / "Khu").mkdir(parents=True)
    write_image(root / "green.jpg", GREEN)
    before = sha_of(root, "green.jpg")
    job = build_job(root, ScriptedClassifier(routed_to({GREEN: "Khu"})), editor=TruncatingEditor())
    job.prepare()
    job.set_actions(job.resolve_folder("Khu"), ["PPE_SIMULATION"])

    run(job)

    assert job_state(job, before) == ImageState.ACTION_FAILED.value
    assert job.manifest.get(before)["actions"][0]["error_code"] == "CORRUPTED_OUTPUT"
    assert folder_files(root, "Khu") == ["green.jpg"]  # the corrupt picture never reached the folder
    assert list((root / ".imageflow" / "staging").iterdir()) == []
    assert sha_of(root, "Khu/green.jpg") == before


# ----- gateway client ---------------------------------------------------------------------------------------
@contextlib.contextmanager
def truncating_server():
    """Promises 5000 bytes of body, sends a few, and hangs up: a flaky mobile connection."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(5)

    def loop():
        while True:
            try:
                conn, _ = listener.accept()
            except OSError:
                return
            with conn:
                buf = b""
                while b"\r\n\r\n" not in buf:
                    chunk = conn.recv(65536)
                    if not chunk:
                        break
                    buf += chunk
                head, _, rest = buf.partition(b"\r\n\r\n")
                length = next((int(h.split(b":")[1]) for h in head.split(b"\r\n") if h.lower().startswith(b"content-length")), 0)
                while len(rest) < length:
                    chunk = conn.recv(65536)
                    if not chunk:
                        break
                    rest += chunk
                conn.sendall(b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 5000\r\n\r\n{"choice_id": "f_')

    threading.Thread(target=loop, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{listener.getsockname()[1]}"
    finally:
        listener.close()


def _post(base: str):
    return asyncio.run(GatewayClient(base, "device-token").post_json("/v1/x", {"x": 1}))


def test_a_reply_cut_off_mid_body_is_a_retryable_network_error():
    with truncating_server() as base:
        with pytest.raises(TransientError) as info:
            _post(base)
    assert info.value.code == "NETWORK"
    assert isinstance(info.value.__cause__, http.client.IncompleteRead)


def test_gateway_408_is_a_retryable_timeout():
    with serve(json_handler({"/v1/x": (408, {}, b"{}")})) as base:
        with pytest.raises(TransientError) as info:
            _post(base)
    assert info.value.code == "TIMEOUT"


def test_gateway_body_that_is_not_utf8_is_an_invalid_schema():
    with serve(json_handler({"/v1/x": (200, {}, b'{"choice_id": "\xff\xff"}')})) as base:  # invalid UTF-8, valid shape
        with pytest.raises(PermanentError) as info:
            _post(base)
    assert info.value.code == "INVALID_SCHEMA"


def test_gateway_does_not_follow_a_redirect_so_the_token_cannot_leak():
    seen: list[str | None] = []

    class Elsewhere(BaseHTTPRequestHandler):
        def do_POST(self):
            seen.append(self.headers.get("Authorization"))
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *args):
            return

    with serve(Elsewhere) as elsewhere:
        with serve(json_handler({"/v1/x": (302, {"Location": elsewhere + "/v1/x"}, b"")})) as gateway:
            with pytest.raises(PermanentError) as info:
                _post(gateway)
    assert info.value.code == "INVALID_REQUEST"
    assert seen == []  # the second server never saw a request, let alone the Authorization header


# ----- workspace --------------------------------------------------------------------------------------------
def test_rule_files_saved_by_other_editors_do_not_break_the_scan(tmp_path):
    root = tmp_path / "job"
    for name in ("utf8", "utf16", "ansi", "garbage"):
        (root / name).mkdir(parents=True)
    (root / "utf8" / ".rule.txt").write_bytes("Người không đủ PPE.".encode("utf-8-sig"))
    (root / "utf16" / ".rule.txt").write_bytes("Người không đủ PPE.".encode("utf-16"))  # "Unicode" in old Notepad
    (root / "ansi" / ".rule.txt").write_bytes(b"Ng\xddi \xf0\xe3 PPE")  # legacy Vietnamese code page, invalid UTF-8
    (root / "garbage" / ".rule.txt").write_bytes(b"\x00\xff\xfe\xfd")

    folders = {f.name: f for f in scan_workspace(root).folders}

    assert set(folders) == {"utf8", "utf16", "ansi", "garbage"}
    assert folders["utf8"].rule == "Người không đủ PPE."
    assert folders["utf16"].rule == "Người không đủ PPE."
    assert "PPE" in folders["ansi"].rule


def test_folder_identity_comes_from_os_stat_not_the_directory_entry(tmp_path, monkeypatch):
    # DirEntry.stat() reports (0, 0) on Windows, which would disable rename tracking there.
    (tmp_path / "job" / "Khu").mkdir(parents=True)
    real = os.stat(tmp_path / "job" / "Khu")
    assert workspace_mod._identity(tmp_path / "job" / "Khu") == (real.st_dev, real.st_ino)

    monkeypatch.setattr(workspace_mod, "_identity", lambda path: (7, 4242))
    (folder,) = scan_workspace(tmp_path / "job").folders
    assert (folder.dev, folder.ino) == (7, 4242)


# ----- manifest, state machine, project ---------------------------------------------------------------------
def test_a_failed_compaction_does_not_fail_the_write_that_triggered_it(tmp_path, monkeypatch):
    def held_open(*args, **kwargs):
        raise PermissionError("held open by another program")

    monkeypatch.setattr(manifest_mod, "COMPACT_EVERY", 3)
    monkeypatch.setattr(manifest_mod, "atomic_write_bytes", held_open)
    store = ManifestStore(tmp_path)
    store.load()
    for i in range(5):
        store.put({"sha256": str(i) * 64, "state": ImageState.DISCOVERED.value, "actions": []})

    again = ManifestStore(tmp_path)
    again.load()
    assert len(again.all()) == 5  # the journal kept every record


def test_a_crash_between_routed_and_action_running_can_still_complete():
    check_transition(ImageState.ACTION_PENDING, ImageState.COMPLETE)


def test_assigning_the_same_actions_again_does_not_rewrite_project_json(tmp_path, monkeypatch):
    (tmp_path / ".imageflow").mkdir()
    store = ProjectStore.load(tmp_path / ".imageflow", frozenset({"PPE_SIMULATION"}))
    store.folders = [{"folder_id": "f_1", "name": "x", "dev": 0, "ino": 0, "status": "ACTIVE", "actions": []}]
    saves = []
    monkeypatch.setattr(store, "save", lambda: saves.append(1))

    store.set_actions("f_1", ["PPE_SIMULATION"])
    store.set_actions("f_1", ["PPE_SIMULATION"])
    store.set_actions("f_1", [])

    assert len(saves) == 2  # one write for each real change, none for the repeat


# ----- pipeline ---------------------------------------------------------------------------------------------
def test_pipeline_config_cannot_be_mutated_through_a_shared_default():
    with pytest.raises(dataclasses.FrozenInstanceError):
        PipelineConfig().edit_concurrency = 99


def test_a_second_run_reports_only_its_own_work(tmp_path):
    root = tmp_path / "job"
    (root / "Khu").mkdir(parents=True)
    write_image(root / "red.jpg", RED)
    job = build_job(root, ScriptedClassifier(routed_to({RED: "Khu", GREEN: "Khu"})))
    assert run(job).routed == 1

    write_image(root / "green.jpg", GREEN)
    second = run(job)

    assert second.routed == 1  # not 2: counters describe this run, not the lifetime of the object


def test_a_mistyped_workspace_path_is_refused_without_creating_anything(tmp_path):
    job = build_job(tmp_path / "typo" / "job", ScriptedClassifier(routed_to({})))
    with pytest.raises(NotADirectoryError):
        job.prepare()
    assert not (tmp_path / "typo").exists()


def _recreate_folder_with_a_new_identity(job):
    """Model Windows: a deleted and recreated folder gets a different file id, so its old record goes MISSING."""
    for record in job.project.folders:
        record["ino"] += 10_000_000
    job.project.save()


def test_lookups_by_name_ignore_missing_records_and_normalise_unicode(tmp_path):
    root = tmp_path / "job"
    (root / "Nghiệm thu").mkdir(parents=True)
    job = build_job(root, ScriptedClassifier(routed_to({})))
    job.prepare()
    old_id = job.resolve_folder("Nghiệm thu")
    _recreate_folder_with_a_new_identity(job)

    fresh = build_job(root, ScriptedClassifier(routed_to({})), job_id="j_fresh")
    fresh.prepare()

    new_id = fresh.resolve_folder("Nghiệm thu")
    assert new_id != old_id
    assert [r.folder_id for r in fresh.folder_rows()] == [new_id]  # the MISSING twin is not listed
    import unicodedata

    assert fresh.resolve_folder(unicodedata.normalize("NFD", "Nghiệm thu")) == new_id
    with pytest.raises(UnknownFolder):
        fresh.resolve_folder("Không có")


def test_a_reviewed_image_can_go_to_a_folder_that_was_recreated(tmp_path):
    root = tmp_path / "job"
    (root / "A").mkdir(parents=True)
    write_image(root / "red.jpg", RED)
    job = build_job(root, ScriptedClassifier(UNSURE), fallback=ScriptedClassifier(UNSURE))
    run(job)
    sha = sha_of(root, "red.jpg")
    _recreate_folder_with_a_new_identity(job)

    fresh = build_job(root, ScriptedClassifier(UNSURE), job_id="j_fresh")
    fresh.prepare()
    problem = asyncio.run(fresh.assign(sha, fresh.resolve_folder("A")))

    assert problem is None
    assert folder_files(root, "A") == ["red.jpg"]


def test_assign_hands_the_image_back_to_review_when_the_folder_vanishes_mid_way(tmp_path, monkeypatch):
    root = tmp_path / "job"
    (root / "A").mkdir(parents=True)
    write_image(root / "red.jpg", RED)
    job = build_job(root, ScriptedClassifier(UNSURE), fallback=ScriptedClassifier(UNSURE))
    run(job)
    sha = sha_of(root, "red.jpg")
    fid = job.resolve_folder("A")
    monkeypatch.setattr(job, "_folder_dir", lambda folder_id: None)  # removed between the check and the move

    problem = asyncio.run(job.assign(sha, fid))

    assert problem == "DESTINATION_MISSING"
    assert job_state(job, sha) == ImageState.REVIEW_REQUIRED.value
    assert [item["sha256"] for item in job.review_items()] == [sha]  # still listed, never silently dropped
    assert (root / "red.jpg").exists()


def test_assign_reports_a_locked_file_and_the_next_run_finishes_the_move(tmp_path, monkeypatch):
    root = tmp_path / "job"
    (root / "A").mkdir(parents=True)
    write_image(root / "red.jpg", RED)
    job = build_job(root, ScriptedClassifier(UNSURE), fallback=ScriptedClassifier(UNSURE))
    run(job)
    sha = sha_of(root, "red.jpg")
    real_route = pipeline_mod.route_file

    def locked(*args, **kwargs):
        raise PermissionError("the file is open in another program")

    monkeypatch.setattr(pipeline_mod, "route_file", locked)
    problem = asyncio.run(job.assign(sha, job.resolve_folder("A")))
    assert problem == "ROUTE_FAILED"
    assert job_state(job, sha) == ImageState.ROUTING.value  # resumable, not lost
    assert (root / "red.jpg").exists()

    monkeypatch.setattr(pipeline_mod, "route_file", real_route)
    run(build_job(root, ScriptedClassifier(UNSURE), job_id="j_retry"))
    assert folder_files(root, "A") == ["red.jpg"]


def test_a_move_that_cannot_remove_its_source_is_retried_not_an_internal_error(tmp_path, monkeypatch):
    root = tmp_path / "job"
    (root / "Khu").mkdir(parents=True)
    write_image(root / "red.jpg", RED)
    clf = ScriptedClassifier(routed_to({RED: "Khu"}))
    real_route = pipeline_mod.route_file

    class Crash(BaseException):
        pass

    def link_then_crash(src, folder, filename, mode, *, dest=None, temp_dir=None):
        os.link(src, dest)  # the destination exists, the source has not been removed yet
        raise Crash()

    monkeypatch.setattr(pipeline_mod, "route_file", link_then_crash)
    with pytest.raises(BaseException):
        run(build_job(root, clf, job_id="j_crash"))
    monkeypatch.setattr(pipeline_mod, "route_file", real_route)

    real_unlink = os.unlink

    def source_is_open(path, *args, **kwargs):
        if os.fspath(path).endswith(os.path.join("job", "red.jpg")):
            raise PermissionError("open in another program")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", source_is_open)
    job = build_job(root, clf, job_id="j_blocked")
    run(job)  # fails the test if the PermissionError escaped as an internal error
    sha = job.manifest.all()[0]["sha256"]
    assert job_state(job, sha) == ImageState.ROUTING.value
    assert job.manifest.get(sha)["last_error"] == "ROUTE_FAILED"

    monkeypatch.setattr(os, "unlink", real_unlink)
    run(build_job(root, clf, job_id="j_free"))
    assert folder_files(root, "Khu") == ["red.jpg"] and not (root / "red.jpg").exists()


class SlowEditor(ImageEditProvider):
    name, model = "slow", "slow-1"

    def __init__(self):
        self.in_flight = self.peak = 0

    async def edit(self, image, instruction, metadata):
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            await asyncio.sleep(0.03)
            return ImageEditResult(EncodedImage(image.data, "image/jpeg"), self.model)
        finally:
            self.in_flight -= 1


def test_edit_concurrency_also_caps_review_assignments_that_overlap(tmp_path):
    root = tmp_path / "job"
    (root / "Khu").mkdir(parents=True)
    for i in range(6):
        write_image(root / f"img{i}.jpg", (40 * i, 10, 200 - 30 * i))
    editor = SlowEditor()
    job = build_job(root, ScriptedClassifier(UNSURE), fallback=ScriptedClassifier(UNSURE), editor=editor)
    job.prepare()
    folder_id = job.resolve_folder("Khu")
    job.set_actions(folder_id, ["PPE_SIMULATION"])
    run(job)
    shas = [item["sha256"] for item in job.review_items()]
    assert len(shas) == 6

    async def assign_all():
        return await asyncio.gather(*(job.assign(sha, folder_id) for sha in shas))

    assert asyncio.run(assign_all()) == [None] * 6
    assert editor.peak == job.config.edit_concurrency == 2  # parallel, and never above the configured cap


def test_an_unreachable_service_stops_the_job_instead_of_retrying_every_image(tmp_path):
    root = tmp_path / "job"
    (root / "Khu").mkdir(parents=True)
    for i in range(12):
        write_image(root / f"img{i:02d}.jpg", (20 * i, 40, 200 - 10 * i))
    down = FailingClassifier(TransientError("NETWORK"))
    job = build_job(root, down)

    with pytest.raises(JobAborted) as info:
        asyncio.run(job.run())

    assert info.value.code == "SERVICE_UNAVAILABLE"
    assert down.count == job.config.abort_after_unavailable * 3  # five images tried (one call plus two retries each)
    assert down.count < 12 * 3  # and it did not grind through all twelve
    assert len(list(root.glob("*.jpg"))) == 12  # nothing moved

    healthy = ScriptedClassifier(lambda colour, choices: {c.choice_id: 0.99 for c in choices[:1]} | {c.choice_id: 0.0 for c in choices[1:]})
    run(build_job(root, healthy, job_id="j_back"))
    assert len(folder_files(root, "Khu")) == 12  # the next run picks everything up


def test_a_few_failures_below_the_limit_are_deferred_not_fatal(tmp_path):
    root = tmp_path / "job"
    (root / "Khu").mkdir(parents=True)
    for i in range(3):
        write_image(root / f"img{i}.jpg", (50 * i, 40, 100))
    report = run(build_job(root, FailingClassifier(TransientError("NETWORK"))))
    assert report.deferred == 3


def test_auth_failure_aborts_with_its_own_code(tmp_path):
    root = tmp_path / "job"
    (root / "Khu").mkdir(parents=True)
    write_image(root / "red.jpg", RED)
    with pytest.raises(JobAborted) as info:
        asyncio.run(build_job(root, FailingClassifier(PermanentError("AUTH_FAILED"))).run())
    assert info.value.code == "AUTH_FAILED"

