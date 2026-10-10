"""End to end through the real composition root, gateway client, and provider classes.

A local HTTP stub plays the gateway. It checks the device token, decides by colour, and
returns an edited PNG, so the JSON contract is exercised for real. No vendor is contacted.
"""

from __future__ import annotations

import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO

import pytest
from PIL import Image, ImageDraw

from imageflow.app import ENV_DEVICE_TOKEN, ENV_GATEWAY_URL, main
from imageflow.core.hashing import sha256_file
from imageflow.preprocessing.image_loader import is_decodable
from support import GREEN, RED, nearest, write_image

TOKEN = "dev-token-123"


class GatewayStub:
    def __init__(self, palette: dict[tuple[int, int, int], str]) -> None:
        self.palette = palette
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
                raw = base64.b64decode(payload["image"]["data_b64"])
                with Image.open(BytesIO(raw)) as img:
                    colour = img.convert("RGB").getpixel((img.width // 2, img.height // 2))
                if self.path == "/v1/classify":
                    stub.classify_models.append(payload["model"])
                    label = nearest(colour, stub.palette)  # the palette maps colour -> folder name
                    probs = {c["id"]: (0.95 if c["label"] == label else 0.025) for c in payload["choices"]}
                    winner = max(probs, key=probs.__getitem__)
                    return self._send(200, {"choice_id": winner, "confidence": probs[winner], "probabilities": probs, "model": payload["model"]})
                if self.path == "/v1/edit":
                    stub.edit_models.append(payload["model"])
                    img = Image.open(BytesIO(raw)).convert("RGB")
                    ImageDraw.Draw(img).rectangle([4, 4, 20, 20], fill=(255, 255, 0))
                    out = BytesIO()
                    img.save(out, format="PNG")
                    data = {"image": {"mime": "image/png", "data_b64": base64.b64encode(out.getvalue()).decode()}, "model": payload["model"]}
                    return self._send(200, data)
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
    holder: dict[str, GatewayStub] = {}

    def start(palette):
        stub = GatewayStub(palette)
        server = ThreadingHTTPServer(("127.0.0.1", 0), stub.handler())
        threading.Thread(target=server.serve_forever, daemon=True).start()
        monkeypatch.setenv(ENV_GATEWAY_URL, f"http://127.0.0.1:{server.server_address[1]}")
        monkeypatch.setenv(ENV_DEVICE_TOKEN, TOKEN)
        holder["server"] = server
        holder["stub"] = stub
        return stub

    yield start
    if "server" in holder:
        holder["server"].shutdown()
        holder["server"].server_close()


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
    assert (root / "Khu").exists() and not any((root / "Khu").iterdir())


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
