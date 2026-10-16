"""Composition root and command line (spec sections 8, 33, 35).

app.py decides which providers and models are wired together. Vendor-specific code lives only in
classifiers/clef.py, image_editors/seedream.py and infrastructure/gateway.py. The pipeline depends
on the provider interfaces alone, so a model can be swapped here without touching it.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path

from .actions.base import ActionRegistry
from .actions.ppe_simulation import PpeSimulationAction
from .classifiers.clef import FALLBACK_MODEL, ClefClassifierProvider
from .core.classification import ClassificationEngine
from .core.pipeline import ImageFlowJob, JobAborted, JobReport, PipelineConfig, UnknownFolder
from .core.workspace import scan_workspace
from .image_editors.seedream import Seedream5FlashProvider
from .infrastructure.gateway import GatewayClient

ENV_GATEWAY_URL = "IMAGEFLOW_GATEWAY_URL"
ENV_DEVICE_TOKEN = "IMAGEFLOW_DEVICE_TOKEN"
ENV_CLASSIFY_CONCURRENCY = "IMAGEFLOW_CLASSIFY_CONCURRENCY"
ENV_EDIT_CONCURRENCY = "IMAGEFLOW_EDIT_CONCURRENCY"

# Suffixes of derived outputs, so `scan` agrees with the pipeline about what counts as an unclassified
# image. A test checks this against the registry that build_job assembles.
ACTION_CLASSES = (PpeSimulationAction,)
KNOWN_OUTPUT_SUFFIXES = frozenset(cls.output_suffix for cls in ACTION_CLASSES)

EXIT_OK = 0
EXIT_INCOMPLETE = 1  # the run finished but some images still need another run
EXIT_ERROR = 2  # the run could not start, or was stopped


class ConfigError(RuntimeError):
    pass


def gateway_from_env(env: Mapping[str, str]) -> GatewayClient:
    url = env.get(ENV_GATEWAY_URL, "").strip()
    token = env.get(ENV_DEVICE_TOKEN, "").strip()
    if not url or not token:
        raise ConfigError(f"set {ENV_GATEWAY_URL} and {ENV_DEVICE_TOKEN} before running a job")
    try:
        return GatewayClient(url, token)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc


def _positive_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ConfigError(f"{name} must be a whole number, got {raw!r}") from None
    if value < 1:
        raise ConfigError(f"{name} must be at least 1")
    return value


def build_job(root: Path, env: Mapping[str, str] = os.environ, job_id: str | None = None) -> ImageFlowJob:
    gateway = gateway_from_env(env)
    classifier = ClassificationEngine(
        ClefClassifierProvider(gateway),  # each provider module owns its default model name
        ClefClassifierProvider(gateway, FALLBACK_MODEL),
    )
    actions = ActionRegistry()
    actions.register(PpeSimulationAction(Seedream5FlashProvider(gateway)))
    defaults = PipelineConfig()
    config = dataclasses.replace(
        defaults,
        classify_concurrency=_positive_int(env, ENV_CLASSIFY_CONCURRENCY, defaults.classify_concurrency),
        edit_concurrency=_positive_int(env, ENV_EDIT_CONCURRENCY, defaults.edit_concurrency),
    )
    return ImageFlowJob(root, classifier, actions, config=config, job_id=job_id)


def require_directory(folder: Path) -> Path:
    """The resolved workspace path, or a ConfigError. Checked before anything is created under it."""
    resolved = folder.expanduser().resolve()
    if not resolved.is_dir():
        raise ConfigError(f"not a folder: {folder}")
    return resolved


def exit_code_for(report: JobReport) -> int:
    """EXIT_INCOMPLETE when images were left for a retry, so a script can tell a clean run from a partial one."""
    return EXIT_INCOMPLETE if report.unfinished else EXIT_OK


def _cmd_scan(folder: Path) -> int:
    scan = scan_workspace(require_directory(folder), KNOWN_OUTPUT_SUFFIXES)
    summary = {
        "images": len(scan.images),
        "folders": [f.name for f in scan.folders],
        "unsupported": scan.unsupported,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return EXIT_OK


def _parse_assignment(text: str) -> tuple[str, list[str]]:
    """``"Folder name=PPE_SIMULATION,OTHER"`` -> ``("Folder name", ["PPE_SIMULATION", "OTHER"])``."""
    name, sep, actions = text.rpartition("=")
    if not sep or not name:
        raise ConfigError(f"expected FOLDER=ACTION[,ACTION], got {text!r}")
    return name, [a.strip() for a in actions.split(",") if a.strip()]


def _cmd_run(folder: Path, assignments: list[str]) -> int:
    root = require_directory(folder)
    parsed = [_parse_assignment(text) for text in assignments]
    job = build_job(root, os.environ)
    job.prepare()
    known = job.actions.known_types()
    for name, actions in parsed:
        unknown = sorted(set(actions) - known)
        if unknown:
            raise ConfigError(f"unknown action {unknown}; available: {sorted(known)}")
        try:
            folder_id = job.resolve_folder(name)
        except UnknownFolder:
            raise ConfigError(f"no folder named {name!r} in the workspace") from None
        job.set_actions(folder_id, actions)
    report = asyncio.run(job.run(refresh=False))  # prepare() just ran, so skip scanning twice
    print(json.dumps(dataclasses.asdict(report), ensure_ascii=False, indent=2, default=str))
    code = exit_code_for(report)
    if code != EXIT_OK:
        print(f"imageflow: {report.unfinished} image(s) could not be processed yet; run again to retry", file=sys.stderr)
    return code


def _cmd_ui() -> int:
    try:
        from .ui.main_window import launch
    except ImportError as exc:  # tkinter is not installed on some Python builds
        raise ConfigError(f"the desktop window needs tkinter: {exc}") from exc
    launch()
    return EXIT_OK


def _use_utf8_output() -> None:
    """Folder names are Vietnamese. A redirected Windows console would otherwise raise on print."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass


def main(argv: list[str] | None = None) -> int:
    _use_utf8_output()
    parser = argparse.ArgumentParser(prog="imageflow", description="Classify job photos into your folders, then run folder actions.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_scan = sub.add_parser("scan", help="list the folders and unclassified images in a job folder")
    p_scan.add_argument("folder", type=Path)

    p_run = sub.add_parser("run", help="classify and route every unclassified image in a job folder")
    p_run.add_argument("folder", type=Path)
    p_run.add_argument(
        "--action",
        action="append",
        default=[],
        metavar="FOLDER=ACTION[,ACTION]",
        help="assign actions to a folder. The assignment is saved with the job and stays until changed; "
        'use "<folder name>=" to clear it',
    )

    sub.add_parser("ui", help="open the desktop window")

    args = parser.parse_args(argv)
    try:
        if args.command == "scan":
            return _cmd_scan(args.folder)
        if args.command == "run":
            return _cmd_run(args.folder, args.action)
        return _cmd_ui()
    except (ConfigError, JobAborted) as exc:
        print(f"imageflow: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
