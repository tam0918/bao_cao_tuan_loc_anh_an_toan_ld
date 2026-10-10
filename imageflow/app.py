"""Composition root and command line (spec sections 8, 33, 35).

This is the only module that names concrete vendor models. Everything else depends on the
provider interfaces, so a model can be swapped here without touching the pipeline.
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
from .classifiers.clef import ClefClassifierProvider
from .core.classification import ClassificationEngine
from .core.pipeline import ImageFlowJob, JobAborted, PipelineConfig
from .core.workspace import scan_workspace
from .image_editors.seedream import Seedream5FlashProvider
from .infrastructure.gateway import GatewayClient

ENV_GATEWAY_URL = "IMAGEFLOW_GATEWAY_URL"
ENV_DEVICE_TOKEN = "IMAGEFLOW_DEVICE_TOKEN"
ENV_CLASSIFY_CONCURRENCY = "IMAGEFLOW_CLASSIFY_CONCURRENCY"
ENV_EDIT_CONCURRENCY = "IMAGEFLOW_EDIT_CONCURRENCY"

PRIMARY_CLASSIFIER_MODEL = "clef-flash"
FALLBACK_CLASSIFIER_MODEL = "clef"
EDIT_MODEL = "seedream-5.0-flash"

# Suffixes of derived outputs. The scanner never treats these as unclassified images.
KNOWN_OUTPUT_SUFFIXES = frozenset({PpeSimulationAction.output_suffix})


class ConfigError(RuntimeError):
    pass


def gateway_from_env(env: Mapping[str, str]) -> GatewayClient:
    url = env.get(ENV_GATEWAY_URL, "").strip()
    token = env.get(ENV_DEVICE_TOKEN, "").strip()
    if not url or not token:
        raise ConfigError(f"set {ENV_GATEWAY_URL} and {ENV_DEVICE_TOKEN} before running a job")
    return GatewayClient(url, token)


def _positive_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    value = int(raw)
    if value < 1:
        raise ConfigError(f"{name} must be at least 1")
    return value


def build_job(root: Path, env: Mapping[str, str] = os.environ, job_id: str | None = None) -> ImageFlowJob:
    gateway = gateway_from_env(env)
    classifier = ClassificationEngine(
        ClefClassifierProvider(gateway, PRIMARY_CLASSIFIER_MODEL),
        ClefClassifierProvider(gateway, FALLBACK_CLASSIFIER_MODEL),
    )
    actions = ActionRegistry()
    actions.register(PpeSimulationAction(Seedream5FlashProvider(gateway, EDIT_MODEL)))
    config = PipelineConfig(
        classify_concurrency=_positive_int(env, ENV_CLASSIFY_CONCURRENCY, 4),
        edit_concurrency=_positive_int(env, ENV_EDIT_CONCURRENCY, 2),
    )
    return ImageFlowJob(root, classifier, actions, config=config, job_id=job_id)


def _cmd_scan(folder: Path) -> int:
    scan = scan_workspace(folder.resolve(), KNOWN_OUTPUT_SUFFIXES)
    summary = {
        "images": len(scan.images),
        "folders": [f.name for f in scan.folders],
        "unsupported": scan.unsupported,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _parse_assignment(text: str) -> tuple[str, list[str]]:
    """``"Folder name=PPE_SIMULATION,OTHER"`` -> ``("Folder name", ["PPE_SIMULATION", "OTHER"])``."""
    name, sep, actions = text.rpartition("=")
    if not sep or not name:
        raise ConfigError(f"expected FOLDER=ACTION[,ACTION], got {text!r}")
    return name, [a.strip() for a in actions.split(",") if a.strip()]


def _cmd_run(folder: Path, assignments: list[str]) -> int:
    job = build_job(folder.resolve(), os.environ)
    job.prepare()
    by_name = {f["name"]: f["folder_id"] for f in job.project.folders}
    for text in assignments:
        name, actions = _parse_assignment(text)
        if name not in by_name:
            raise ConfigError(f"no folder named {name!r} in the workspace")
        job.set_actions(by_name[name], actions)
    report = asyncio.run(job.run())
    print(json.dumps(dataclasses.asdict(report), ensure_ascii=False, indent=2, default=str))
    return 0


def _cmd_ui() -> int:
    try:
        from .ui.main_window import launch
    except ImportError as exc:  # tkinter is not installed on some Python builds
        raise ConfigError(f"the desktop window needs tkinter: {exc}") from exc
    launch()
    return 0


def main(argv: list[str] | None = None) -> int:
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
        help='assign actions to a folder for this run, e.g. "<folder name>=PPE_SIMULATION"',
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
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
