"""Shared test doubles and builders. No network: every model is a scripted fake."""

from __future__ import annotations

import asyncio
import contextlib
import random
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageDraw

from imageflow.actions.base import ActionRegistry
from imageflow.actions.ppe_simulation import PpeSimulationAction
from imageflow.classifiers.base import Choice, ClassificationResult, ClassifierProvider
from imageflow.core.classification import ClassificationEngine
from imageflow.core.pipeline import ImageFlowJob, PipelineConfig
from imageflow.core.safety import SafetyScreenEngine
from imageflow.image_editors.base import ImageEditProvider, ImageEditResult
from imageflow.preprocessing.image_loader import EncodedImage

# Solid colours identify an image's "truth" in tests without needing names inside the image.
RED = (220, 30, 30)
GREEN = (30, 200, 60)
BLUE = (30, 60, 220)
YELLOW = (230, 220, 40)
PURPLE = (150, 40, 190)


async def no_sleep(_: float) -> None:
    return None


def write_image(path: Path, rgb: tuple[int, int, int], size: tuple[int, int] = (64, 48), fmt: str = "JPEG") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, rgb).save(path, format=fmt, quality=95)
    return path


def dominant_colour(data: bytes) -> tuple[int, int, int]:
    with Image.open(BytesIO(data)) as img:
        return img.convert("RGB").getpixel((img.width // 2, img.height // 2))


def nearest(colour: tuple[int, int, int], palette: dict[tuple[int, int, int], str]) -> str:
    def dist(key: tuple[int, int, int]) -> int:
        return sum((a - b) ** 2 for a, b in zip(colour, key))

    return palette[min(palette, key=dist)]


def peaked(choices: list[Choice], winner: str, top: float = 0.95) -> dict[str, float]:
    """Probabilities where ``winner`` has ``top`` and the rest share the remainder evenly."""
    others = [c.choice_id for c in choices if c.choice_id != winner]
    rest = (1.0 - top) / len(others) if others else 0.0
    probs = {c.choice_id: rest for c in choices}
    probs[winner] = top if others else 1.0
    return probs


class ScriptedClassifier(ClassifierProvider):
    """Decides from the image's colour through ``decide(colour, choices) -> probabilities``."""

    name = "scripted"

    def __init__(self, decide: Callable[[tuple[int, int, int], list[Choice]], dict[str, float]], model: str = "scripted-1") -> None:
        self.model = model
        self._decide = decide
        self.calls: list[list[str]] = []  # choice ids seen on each call
        self.labels_seen: list[list[str]] = []  # choice labels seen on each call
        self.count = 0

    async def classify(self, image: EncodedImage, choices: list[Choice]) -> ClassificationResult:
        self.count += 1
        self.calls.append([c.choice_id for c in choices])
        self.labels_seen.append(sorted(c.label for c in choices))
        probs = self._decide(dominant_colour(image.data), choices)
        winner = max(probs, key=probs.__getitem__)
        return ClassificationResult(choice_id=winner, confidence=probs[winner], probabilities=probs, model=self.model)


class FailingClassifier(ClassifierProvider):
    """Raises the given exception on every call. Used for outage and auth scenarios."""

    name = "failing"

    def __init__(self, exc: Exception, model: str = "failing-1") -> None:
        self.model = model
        self._exc = exc
        self.count = 0

    async def classify(self, image: EncodedImage, choices: list[Choice]) -> ClassificationResult:
        self.count += 1
        raise self._exc


class MockEditor(ImageEditProvider):
    """Draws a visible mark so the output bytes differ from the input. Records every call."""

    name = "mock-editor"

    def __init__(self, model: str = "mock-edit-1", error: Exception | None = None) -> None:
        self.model = model
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def edit(self, image: EncodedImage, instruction: str, metadata: dict[str, Any]) -> ImageEditResult:
        self.calls.append({"instruction": instruction, **metadata})
        if self.error is not None:
            raise self.error
        img = Image.open(BytesIO(image.data)).convert("RGB")
        ImageDraw.Draw(img).rectangle([4, 4, 20, 20], fill=(255, 255, 0))
        out = BytesIO()
        img.save(out, format="PNG")
        return ImageEditResult(image=EncodedImage(data=out.getvalue(), mime="image/png"), model=self.model)


def build_engine(primary: ClassifierProvider, fallback: ClassifierProvider | None = None) -> ClassificationEngine:
    return ClassificationEngine(primary, fallback, sleep=no_sleep)


def build_registry(editor: ImageEditProvider | None) -> ActionRegistry:
    registry = ActionRegistry()
    if editor is not None:
        registry.register(PpeSimulationAction(editor, sleep=no_sleep))
    return registry


def build_job(
    root: Path,
    classifier: ClassifierProvider,
    *,
    fallback: ClassifierProvider | None = None,
    editor: ImageEditProvider | None = None,
    safety: ClassifierProvider | None = None,
    job_id: str = "j_test",
) -> ImageFlowJob:
    """``safety`` switches on the second stage: every routed photo is screened for missing PPE."""
    screen = SafetyScreenEngine(build_engine(safety)) if safety is not None else None
    return ImageFlowJob(
        root,
        build_engine(classifier, fallback),
        build_registry(editor),
        safety_screen=screen,
        safety_action_type=PpeSimulationAction.action_type if screen is not None else None,
        config=PipelineConfig(classify_concurrency=3, edit_concurrency=2),
        job_id=job_id,
    )


def run(job: ImageFlowJob):
    """Run a job. Any isolated internal error fails the test, so bugs cannot hide behind the per-image guard."""
    report = asyncio.run(job.run())
    assert report.internal_errors == 0, "an unexpected exception was isolated to one image"
    return report


def folder_files(root: Path, folder: str) -> list[str]:
    return sorted(p.name for p in (root / folder).iterdir() if p.is_file())


def folder_id(job: ImageFlowJob, name: str) -> str:
    return next(f["folder_id"] for f in job.project.folders if f["name"] == name)


def by_label(choices: list[Choice], label: str) -> str:
    return next(c.choice_id for c in choices if c.label == label)


def routed_to(palette: dict[tuple[int, int, int], str]):
    """Decision function: pick the choice whose label matches the colour's palette entry."""

    def decide(colour, choices):
        return peaked(choices, by_label(choices, nearest(colour, palette)))

    return decide


def job_state(job: ImageFlowJob, sha: str) -> str:
    return job.manifest.get(sha)["state"]


def sha_of(root: Path, rel: str) -> str:
    from imageflow.core.hashing import sha256_file

    return sha256_file(root / rel)


def noisy_jpeg_bytes(size: tuple[int, int] = (200, 150), seed: int = 1) -> bytes:
    """A JPEG of random pixels. It does not compress well, so cutting it short really breaks decoding."""
    rng = random.Random(seed)
    img = Image.new("RGB", size)
    img.putdata([(rng.randrange(256), rng.randrange(256), rng.randrange(256)) for _ in range(size[0] * size[1])])
    out = BytesIO()
    img.save(out, format="JPEG", quality=90)
    return out.getvalue()


@contextlib.contextmanager
def serve(handler_class):
    """Run a local HTTP server for the duration of the block and yield its base URL.

    The short poll interval keeps shutdown instant; the default 0.5 s would add half a second per test.
    """
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_class)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def json_handler(responses: dict[str, tuple[int, dict[str, str], bytes]]):
    """A request handler that answers each POST path with a canned (status, headers, body)."""

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

    return Handler
