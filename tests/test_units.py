"""Unit-level behaviour: retries, gateway mapping, manifest journal, thresholds, routing, normalisation."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO

import pytest
from PIL import Image

from imageflow.classifiers.base import Choice, ClassificationResult, ClassifierProvider, validate_probabilities
from imageflow.core.classification import ClassificationEngine, ClassifierUnavailable, passes_thresholds, rank
from imageflow.core.manifest import ManifestStore
from imageflow.core.metrics import percentile
from imageflow.core.project import ProjectStore
from imageflow.core.routing import DestinationMissing, deterministic_name, plan_recovery, route_file
from imageflow.core.state_machine import ImageState, IllegalTransition, check_transition
from imageflow.core.workspace import choice_description, scan_workspace
from imageflow.errors import PermanentError, TransientError
from imageflow.infrastructure.gateway import GatewayClient
from imageflow.infrastructure.retries import RetryPolicy, backoff_delay, retry_async
from imageflow.preprocessing.image_loader import EncodedImage, normalize_file

from support import RED, GREEN, no_sleep, write_image


# ----- retries ---------------------------------------------------------------
def test_transient_errors_are_retried_then_succeed():
    attempts, sleeps = [], []

    async def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise TransientError("RATE_LIMITED")
        return "ok"

    async def record_sleep(d):
        sleeps.append(d)

    value, retries = asyncio.run(retry_async(flaky, RetryPolicy(), sleep=record_sleep, rand=lambda: 0.0))
    assert (value, retries) == ("ok", 2)
    assert len(sleeps) == 2


def test_transient_errors_stop_after_max_retries():
    calls = []

    async def always_down():
        calls.append(1)
        raise TransientError("UPSTREAM_UNAVAILABLE")

    with pytest.raises(TransientError):
        asyncio.run(retry_async(always_down, RetryPolicy(max_retries=2), sleep=no_sleep))
    assert len(calls) == 3  # first try plus two retries, never unbounded


def test_permanent_errors_fail_fast_without_retry():
    calls = []

    async def rejected():
        calls.append(1)
        raise PermanentError("UNSUPPORTED_IMAGE")

    with pytest.raises(PermanentError):
        asyncio.run(retry_async(rejected, RetryPolicy(), sleep=no_sleep))
    assert len(calls) == 1


def test_retry_after_header_wins_over_own_backoff():
    policy = RetryPolicy(max_delay=8.0)
    assert backoff_delay(policy, 0, retry_after=3.0, rand=lambda: 0.0) == 3.0
    assert backoff_delay(policy, 0, retry_after=60.0, rand=lambda: 0.0) == 8.0  # capped


def test_backoff_is_exponential_with_equal_jitter_and_a_cap():
    policy = RetryPolicy(base_delay=0.5, max_delay=8.0)
    assert backoff_delay(policy, 0, None, rand=lambda: 0.0) == pytest.approx(0.25)
    assert backoff_delay(policy, 0, None, rand=lambda: 1.0) == pytest.approx(0.5)
    assert backoff_delay(policy, 1, None, rand=lambda: 1.0) == pytest.approx(1.0)
    assert backoff_delay(policy, 10, None, rand=lambda: 1.0) == pytest.approx(8.0)


# ----- gateway ---------------------------------------------------------------
@pytest.fixture
def gateway_server():
    responses: dict[str, tuple[int, dict, bytes]] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            status, headers, body = responses[self.path]
            self.send_response(status)
            for key, value in headers.items():
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield responses, f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def _post(base: str, path: str = "/v1/classify") -> dict:
    return asyncio.run(GatewayClient(base, "device-token").post_json(path, {"x": 1}))


def test_gateway_success_returns_the_json_object(gateway_server):
    responses, base = gateway_server
    responses["/v1/classify"] = (200, {}, json.dumps({"choice_id": "f_1"}).encode())
    assert _post(base) == {"choice_id": "f_1"}


def test_gateway_429_is_transient_and_keeps_retry_after(gateway_server):
    responses, base = gateway_server
    responses["/v1/classify"] = (429, {"Retry-After": "3"}, b"{}")
    with pytest.raises(TransientError) as info:
        _post(base)
    assert info.value.code == "RATE_LIMITED"
    assert info.value.retry_after == 3.0


def test_gateway_5xx_is_transient(gateway_server):
    responses, base = gateway_server
    responses["/v1/classify"] = (503, {}, b"{}")
    with pytest.raises(TransientError) as info:
        _post(base)
    assert info.value.code == "UPSTREAM_UNAVAILABLE"


def test_gateway_401_is_a_permanent_auth_failure(gateway_server):
    responses, base = gateway_server
    responses["/v1/classify"] = (401, {}, b"{}")
    with pytest.raises(PermanentError) as info:
        _post(base)
    assert info.value.code == "AUTH_FAILED"


def test_gateway_passes_through_a_known_image_error_code(gateway_server):
    responses, base = gateway_server
    responses["/v1/classify"] = (422, {}, json.dumps({"error": {"code": "UNSUPPORTED_IMAGE"}}).encode())
    with pytest.raises(PermanentError) as info:
        _post(base)
    assert info.value.code == "UNSUPPORTED_IMAGE"


def test_gateway_non_json_body_is_an_invalid_schema(gateway_server):
    responses, base = gateway_server
    responses["/v1/classify"] = (200, {}, b"<html>oops</html>")
    with pytest.raises(PermanentError) as info:
        _post(base)
    assert info.value.code == "INVALID_SCHEMA"


def test_gateway_network_failure_is_transient():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]  # nothing listens here once the probe closes
    with pytest.raises(TransientError) as info:
        _post(f"http://127.0.0.1:{port}")
    assert info.value.code == "NETWORK"


def test_gateway_refuses_plain_http_for_remote_hosts_and_missing_tokens():
    with pytest.raises(ValueError):
        GatewayClient("http://api.example.com", "token")
    with pytest.raises(ValueError):
        GatewayClient("https://api.example.com", "")
    GatewayClient("http://localhost:8787", "token")  # local development is allowed


# ----- state machine ---------------------------------------------------------
def test_illegal_transitions_are_rejected():
    with pytest.raises(IllegalTransition):
        check_transition(ImageState.DISCOVERED, ImageState.ROUTED)
    with pytest.raises(IllegalTransition):
        check_transition(ImageState.REVIEW_REQUIRED, ImageState.ROUTING)


def test_terminal_states_do_not_move_and_failed_is_reachable_before_them():
    with pytest.raises(IllegalTransition):
        check_transition(ImageState.COMPLETE, ImageState.FAILED)
    check_transition(ImageState.ROUTING, ImageState.FAILED)  # a permanent error at any stage
    check_transition(ImageState.ACTION_FAILED, ImageState.COMPLETE)  # the user removed the action


# ----- manifest journal ------------------------------------------------------
def _rec(sha: str, state: ImageState = ImageState.DISCOVERED) -> dict:
    return {"sha256": sha, "image_id": "im_" + sha[:6], "state": state.value, "actions": []}


def test_manifest_reload_restores_every_record(tmp_path):
    store = ManifestStore(tmp_path)
    store.load()
    store.put(_rec("aa" * 32))
    store.transition("aa" * 32, ImageState.NORMALIZED, normalized_rel="x.jpg")
    store.put(_rec("bb" * 32))

    again = ManifestStore(tmp_path)
    again.load()
    assert again.get("aa" * 32)["state"] == ImageState.NORMALIZED.value
    assert again.get("aa" * 32)["normalized_rel"] == "x.jpg"
    assert len(again.all()) == 2


def test_a_torn_final_journal_line_is_ignored(tmp_path):
    store = ManifestStore(tmp_path)
    store.load()
    store.put(_rec("cc" * 32))
    with open(tmp_path / "manifest.journal.jsonl", "a", encoding="utf-8") as fh:
        fh.write('{"sha256": "dd')  # crash in the middle of a write

    again = ManifestStore(tmp_path)
    again.load()
    assert [r["sha256"] for r in again.all()] == ["cc" * 32]


def test_compaction_keeps_the_same_records_and_empties_the_journal(tmp_path):
    store = ManifestStore(tmp_path)
    store.load()
    store.put(_rec("ee" * 32))
    store.transition("ee" * 32, ImageState.NORMALIZED)
    store.compact()
    assert not (tmp_path / "manifest.journal.jsonl").exists()
    again = ManifestStore(tmp_path)
    again.load()
    assert again.get("ee" * 32)["state"] == ImageState.NORMALIZED.value
    snapshot = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert snapshot["records"][0]["sha256"] == "ee" * 32


# ----- thresholds and validation --------------------------------------------
@pytest.mark.parametrize(
    "ranked, expected",
    [
        ([("A", 0.85), ("B", 0.0)], True),  # boundary: both thresholds inclusive
        ([("A", 0.849), ("B", 0.0)], False),
        ([("A", 0.9), ("B", 0.76)], False),  # margin 0.14
        ([("A", 0.9), ("B", 0.75)], True),  # margin 0.15
        ([("A", 0.9)], True),  # a single folder has no competitor
    ],
)
def test_route_thresholds(ranked, expected):
    assert passes_thresholds(ranked, 0.85, 0.15) is expected


def test_validation_rejects_unknown_ids_disagreement_and_bad_numbers():
    choices = [Choice("A", "a", "a"), Choice("B", "b", "b")]
    good = ClassificationResult("A", 0.9, {"A": 0.9, "B": 0.1}, "m")
    assert validate_probabilities(good, choices) == {"A": 0.9, "B": 0.1}
    assert validate_probabilities(ClassificationResult("Z", 0.9, {"Z": 0.9}, "m"), choices) is None
    assert validate_probabilities(ClassificationResult("A", 0.9, {"A": 0.1, "B": 0.9}, "m"), choices) is None
    assert validate_probabilities(ClassificationResult("A", 0.9, {"A": float("nan"), "B": 0.1}, "m"), choices) is None


class Scripted(ClassifierProvider):
    """Returns queued outcomes: a ClassificationResult or an exception to raise."""

    name = "scripted"

    def __init__(self, model: str, outcomes: list) -> None:
        self.model = model
        self.outcomes = list(outcomes)
        self.count = 0

    async def classify(self, image: EncodedImage, choices: list[Choice]) -> ClassificationResult:
        self.count += 1
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _result(model: str, a: float, b: float) -> ClassificationResult:
    return ClassificationResult("A" if a >= b else "B", max(a, b), {"A": a, "B": b}, model)


CHOICES = [Choice("A", "a", "a"), Choice("B", "b", "b")]
IMAGE = EncodedImage(data=b"x", mime="image/jpeg")


def _engine(primary, fallback=None):
    return ClassificationEngine(primary, fallback, sleep=no_sleep)


def test_primary_acceptance_never_calls_the_fallback():
    primary = Scripted("clef-flash", [_result("clef-flash", 0.95, 0.05)])
    fallback = Scripted("clef", [_result("clef", 0.95, 0.05)])
    decision = asyncio.run(_engine(primary, fallback).decide(IMAGE, CHOICES))
    assert decision.kind == "ROUTE" and decision.model == "clef-flash"
    assert fallback.count == 0


def test_uncertain_primary_falls_back_to_the_stronger_model():
    primary = Scripted("clef-flash", [_result("clef-flash", 0.6, 0.4)])
    fallback = Scripted("clef", [_result("clef", 0.92, 0.08)])
    decision = asyncio.run(_engine(primary, fallback).decide(IMAGE, CHOICES))
    assert decision.kind == "ROUTE" and decision.model == "clef"
    assert [a.outcome for a in decision.attempts] == ["rejected", "accepted"]


def test_primary_outage_uses_the_fallback():
    primary = Scripted("clef-flash", [TransientError("UPSTREAM_UNAVAILABLE")])
    fallback = Scripted("clef", [_result("clef", 0.95, 0.05)])
    decision = asyncio.run(_engine(primary, fallback).decide(IMAGE, CHOICES))
    assert decision.kind == "ROUTE" and decision.model == "clef"


def test_every_model_down_is_unavailable_not_review():
    primary = Scripted("clef-flash", [TransientError("NETWORK")])
    fallback = Scripted("clef", [TransientError("NETWORK")])
    with pytest.raises(ClassifierUnavailable):
        asyncio.run(_engine(primary, fallback).decide(IMAGE, CHOICES))


def test_outage_on_one_model_and_low_confidence_on_the_other_means_review():
    primary = Scripted("clef-flash", [TransientError("NETWORK")])
    fallback = Scripted("clef", [_result("clef", 0.55, 0.45)])
    decision = asyncio.run(_engine(primary, fallback).decide(IMAGE, CHOICES))
    assert decision.kind == "REVIEW"
    assert decision.candidates[0][0] == "A"


def test_invalid_model_output_is_no_decision_and_never_a_guess():
    bad = ClassificationResult("A", 0.99, {"A": 0.1, "B": 0.9}, "clef-flash")  # argmax disagrees
    primary = Scripted("clef-flash", [bad])
    fallback = Scripted("clef", [_result("clef", 0.95, 0.05)])
    decision = asyncio.run(_engine(primary, fallback).decide(IMAGE, CHOICES))
    assert decision.model == "clef"  # the bad answer was not trusted


def test_auth_failures_propagate_to_the_pipeline():
    primary = Scripted("clef-flash", [PermanentError("AUTH_FAILED")])
    with pytest.raises(PermanentError):
        asyncio.run(_engine(primary).decide(IMAGE, CHOICES))


def test_rank_breaks_ties_the_same_way_every_run():
    assert [k for k, _ in rank({"B": 0.5, "A": 0.5, "C": 0.1})] == ["A", "B", "C"]


# ----- routing ---------------------------------------------------------------
def test_deterministic_names_skip_taken_slots(tmp_path):
    folder = tmp_path / "f"
    folder.mkdir()
    (folder / "x.jpg").write_bytes(b"1")
    (folder / "x__2.jpg").write_bytes(b"2")
    assert deterministic_name(folder, "x.jpg").name == "x__3.jpg"
    assert deterministic_name(folder, "y.jpg").name == "y.jpg"
    assert deterministic_name(folder, "x.jpg", taken={"x__3.jpg"}).name == "x__4.jpg"


def test_route_into_a_missing_folder_is_an_error_not_a_silent_mkdir(tmp_path):
    src = write_image(tmp_path / "x.jpg", RED)
    with pytest.raises(DestinationMissing):
        route_file(src, tmp_path / "gone", "x.jpg", "move")
    assert src.exists()


def test_move_removes_the_source_and_copy_keeps_it(tmp_path):
    folder = tmp_path / "dst"
    folder.mkdir()
    src = write_image(tmp_path / "a.jpg", GREEN)
    copied = route_file(src, folder, "a.jpg", "copy", temp_dir=tmp_path / "temp")
    assert src.exists() and copied.exists()
    assert not os.path.samefile(src, copied)  # an independent file, not a hard link to the original
    moved = route_file(src, folder, "b.jpg", "move")
    assert not src.exists() and moved.exists()
    assert not any(p.name.endswith(".part") for p in (tmp_path / "temp").iterdir())  # staging cleaned up


def test_unknown_routing_mode_is_refused(tmp_path):
    src = write_image(tmp_path / "a.jpg", GREEN)
    with pytest.raises(ValueError):
        route_file(src, tmp_path, "a.jpg", "shred")


def test_recovery_plans_cover_every_interrupted_state(tmp_path):
    src = write_image(tmp_path / "src.jpg", RED)
    from imageflow.core.hashing import sha256_file

    sha = sha256_file(src)
    dst = tmp_path / "dst.jpg"
    assert plan_recovery(src, dst, sha, "move").action == "REROUTE"  # nothing happened yet
    os.link(src, dst)
    assert plan_recovery(src, dst, sha, "move").action == "FINISH_MOVE"  # linked, source not yet removed
    assert plan_recovery(src, dst, sha, "copy").action == "DONE"  # copy mode keeps the source
    os.unlink(src)
    assert plan_recovery(src, dst, sha, "move").action == "DONE"  # move fully completed
    dst.unlink()
    assert plan_recovery(src, dst, sha, "move").action == "LOST_SOURCE"


# ----- normalisation ---------------------------------------------------------
def test_exif_orientation_is_applied_before_inference(tmp_path):
    path = tmp_path / "rotated.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees on display
    Image.new("RGB", (40, 20), RED).save(path, format="JPEG", exif=exif.tobytes())
    encoded = normalize_file(path, tmp_path / "cache", "sha-rotated")
    with Image.open(BytesIO(encoded.data)) as out:
        assert out.size == (20, 40)


def test_large_images_are_downscaled_for_inference(tmp_path):
    path = write_image(tmp_path / "big.jpg", RED, size=(2000, 1000))
    encoded = normalize_file(path, tmp_path / "cache", "sha-big")
    with Image.open(BytesIO(encoded.data)) as out:
        assert max(out.size) == 1024


def test_transparent_png_is_flattened_to_jpeg(tmp_path):
    path = tmp_path / "alpha.png"
    Image.new("RGBA", (32, 32), (255, 0, 0, 0)).save(path)
    encoded = normalize_file(path, tmp_path / "cache", "sha-alpha")
    assert encoded.mime == "image/jpeg"


def test_corrupt_and_unsupported_inputs_fail_permanently(tmp_path):
    corrupt = tmp_path / "corrupt.jpg"
    corrupt.write_bytes(b"not an image at all")
    with pytest.raises(PermanentError) as info:
        normalize_file(corrupt, tmp_path / "cache", "sha-corrupt")
    assert info.value.code == "CORRUPTED_IMAGE"

    bmp = tmp_path / "bitmap.jpg"  # extension lies, content is BMP
    Image.new("RGB", (8, 8), RED).save(bmp, format="BMP")
    with pytest.raises(PermanentError) as info:
        normalize_file(bmp, tmp_path / "cache", "sha-bmp")
    assert info.value.code == "UNSUPPORTED_IMAGE"


# ----- project registry and scanner -----------------------------------------
def test_renaming_a_folder_keeps_its_id_and_reports_the_rename(tmp_path):
    root = tmp_path / "job"
    (root / "Khu").mkdir(parents=True)
    store = ProjectStore.load(root / ".imageflow", frozenset({"PPE_SIMULATION"}))
    store.system_dir.mkdir(parents=True)
    first = store.sync(scan_workspace(root).folders)
    fid = store.folders[0]["folder_id"]
    assert [e.kind for e in first] == ["added"]

    os.rename(root / "Khu", root / "Vị trí mới")
    events = store.sync(scan_workspace(root).folders)

    assert [(e.kind, e.folder_id) for e in events] == [("renamed", fid)]
    assert store.folders[0]["folder_id"] == fid and store.folders[0]["name"] == "Vị trí mới"


def test_unknown_actions_and_routing_modes_are_refused(tmp_path):
    (tmp_path / ".imageflow").mkdir()
    store = ProjectStore.load(tmp_path / ".imageflow", frozenset({"PPE_SIMULATION"}))
    store.folders = [{"folder_id": "f_1", "name": "x", "dev": 0, "ino": 0, "status": "ACTIVE", "actions": []}]
    with pytest.raises(ValueError):
        store.set_actions("f_1", ["SOMETHING_ELSE"])
    with pytest.raises(ValueError):
        store.set_routing_mode("zip")


def test_scanner_follows_the_workspace_contract(tmp_path):
    root = tmp_path / "job"
    (root / "Khu" / "minh_hoa").mkdir(parents=True)
    (root / "Khu" / "minh_hoa" / "inner.jpg").write_bytes(b"x")
    (root / ".hidden").mkdir()
    (root / "$RECYCLE.BIN").mkdir()
    (root / ".imageflow").mkdir()
    write_image(root / "IMG_001.jpg", RED)
    write_image(root / "IMG_001_AI_PPE.png", RED)  # an action output moved back to the root
    write_image(root / ".secret.jpg", RED)
    (root / "Thumbs.db").write_bytes(b"x")
    (root / "IMG.heic").write_bytes(b"x")
    (root / "Khu" / ".rule.txt").write_text("  Người không đủ PPE.  \n", encoding="utf-8")

    scan = scan_workspace(root, frozenset({"_AI_PPE"}))

    assert [i.name for i in scan.images] == ["IMG_001.jpg"]
    assert [f.name for f in scan.folders] == ["Khu"]
    assert scan.unsupported == ["IMG.heic"]
    assert choice_description(scan.folders[0]) == "Khu: Người không đủ PPE."


def test_folder_names_are_normalised_to_nfc(tmp_path):
    import unicodedata

    root = tmp_path / "job"
    decomposed = unicodedata.normalize("NFD", "Nghiệm thu")
    (root / decomposed).mkdir(parents=True)
    scan = scan_workspace(root)
    assert scan.folders[0].name == "Nghiệm thu"


def test_percentiles_use_nearest_rank():
    values = [float(v) for v in range(1, 11)]
    assert percentile(values, 50) == 5.0
    assert percentile(values, 95) == 10.0
    assert percentile([], 50) is None
