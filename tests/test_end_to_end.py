"""End to end through the real composition root, gateway client, and provider classes.

A local HTTP stub plays the gateway. It checks the device token, then hands each request to the
same fakes the unit tests use (ScriptedClassifier, MockEditor), so the JSON contract is exercised
for real and the fakes cannot drift apart. No vendor is contacted.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
from http.server import BaseHTTPRequestHandler

import pytest

from imageflow.app import ENV_DEVICE_TOKEN, ENV_GATEWAY_URL, main
from imageflow.classifiers.base import Choice
from imageflow.core.hashing import sha256_file
from imageflow.preprocessing.image_loader import EncodedImage, is_decodable

from support import GREEN, RED, MockEditor, ScriptedClassifier, folder_files, routed_to, serve, write_image

TOKEN = "dev-token-123"


class GatewayStub:
    def __init__(self, palette: dict[tuple[int, int, int], str]) -> None:
        self.classifier = ScriptedClassifier(routed_to(palette))  # decides by colour, like the unit tests
        self.editor = MockEditor()
        self.classify_models: list[str] = []
        self.edit_models: list[str] = []
        self.rejected_auth = 0

    def handler(self):
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                if self.headers.get("Authorization") != f"Bearer {TOKEN}":
                    stub.rejected_auth += 1
                    return self._send(401, {"error": {"code": "AUTH_FAILED"}})
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                image = EncodedImage(data=base64.b64decode(payload["image"]["data_b64"]), mime=payload["image"]["mime"])
                if self.path == "/v1/classify":
                    stub.classify_models.append(payload["model"])
                    choices = [Choice(c["id"], c["label"], c["description"]) for c in payload["choices"]]
                    result = asyncio.run(stub.classifier.classify(image, choices))
                    return self._send(
                        200,
                        {"choice_id": result.choice_id, "confidence": result.confidence, "probabilities": result.probabilities, "model": payload["model"]},
                    )
                if self.path == "/v1/edit":
                    stub.edit_models.append(payload["model"])
                    result = asyncio.run(stub.editor.edit(image, payload["instruction"], payload["metadata"]))
                    edited = {"mime": result.image.mime, "data_b64": base64.b64encode(result.image.data).decode()}
                    return self._send(200, {"image": edited, "model": payload["model"]})
                return self._send(404, {"error": {"code": "INVALID_REQUEST"}})

            def _send(self, status, body):
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                return

        return Handler


@pytest.fixture
def gateway(monkeypatch):
    with contextlib.ExitStack() as servers:

        def start(palette):
            stub = GatewayStub(palette)
            base = servers.enter_context(serve(stub.handler()))
            monkeypatch.setenv(ENV_GATEWAY_URL, base)
            monkeypatch.setenv(ENV_DEVICE_TOKEN, TOKEN)
            return stub

        yield start


def test_run_routes_and_acts_through_the_gateway(tmp_path, gateway, capsys):
    root = tmp_path / "job"
    (root / "Mất an toàn lao động").mkdir(parents=True)
    (root / "Nghiệm thu").mkdir(parents=True)
    write_image(root / "red.jpg", RED)
    write_image(root / "green.jpg", GREEN)
    stub = gateway({RED: "Mất an toàn lao động", GREEN: "Nghiệm thu"})

    code = main(["run", str(root), "--action", "Mất an toàn lao động=PPE_SIMULATION"])

    assert code == 0
    assert stub.rejected_auth == 0  # the device token was sent and accepted
    assert stub.classify_models == ["clef-flash", "clef-flash"]  # confident, so no fallback call
    assert stub.edit_models == ["seedream-5.0-flash"]  # only the image in the action folder
    assert (root / "Mất an toàn lao động" / "red.jpg").exists()
    assert (root / "Nghiệm thu" / "green.jpg").exists()
    output = root / "Mất an toàn lao động" / "red_AI_PPE.png"
    assert is_decodable(output)
    assert sha256_file(output) != sha256_file(root / "Mất an toàn lao động" / "red.jpg")
    report = json.loads(capsys.readouterr().out)
    assert report["routed"] == 2
    assert report["actioned"] == 1
    assert report["totals"]["complete"] == 2


def test_wrong_device_token_stops_the_job_without_moving_anything(tmp_path, gateway, monkeypatch, capsys):
    root = tmp_path / "job"
    (root / "Khu").mkdir(parents=True)
    write_image(root / "red.jpg", RED)
    stub = gateway({RED: "Khu"})
    monkeypatch.setenv(ENV_DEVICE_TOKEN, "revoked-token")

    code = main(["run", str(root)])

    assert code == 2  # the job is aborted with a clear error, not a traceback
    assert stub.rejected_auth >= 1
    assert (root / "red.jpg").exists()
    assert folder_files(root, "Khu") == []


def test_missing_configuration_is_reported_not_crashed(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv(ENV_GATEWAY_URL, raising=False)
    monkeypatch.delenv(ENV_DEVICE_TOKEN, raising=False)
    (tmp_path / "job").mkdir()
    assert main(["run", str(tmp_path / "job")]) == 2
    assert ENV_GATEWAY_URL in capsys.readouterr().err


def test_scan_lists_folders_and_unsupported_files_without_configuration(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv(ENV_GATEWAY_URL, raising=False)
    root = tmp_path / "job"
    (root / "Nghiệm thu").mkdir(parents=True)
    write_image(root / "a.jpg", RED)
    (root / "b.heic").write_bytes(b"x")

    assert main(["scan", str(root)]) == 0

    summary = json.loads(capsys.readouterr().out)
    assert summary == {"images": 1, "folders": ["Nghiệm thu"], "unsupported": ["b.heic"]}


def test_assignment_for_an_unknown_folder_is_refused(tmp_path, gateway, capsys):
    root = tmp_path / "job"
    root.mkdir()
    gateway({})
    assert main(["run", str(root), "--action", "Không có=PPE_SIMULATION"]) == 2
    assert "no folder named" in capsys.readouterr().err
