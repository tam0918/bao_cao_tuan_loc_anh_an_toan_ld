"""Command-line behaviour: clean errors, exit codes, and no surprises on a mistyped path."""

from __future__ import annotations

import io
import json
import sys
import unicodedata

import pytest

import imageflow.app as app_mod
import imageflow.core.pipeline as pipeline_mod
from imageflow.app import ENV_CLASSIFY_CONCURRENCY, ENV_DEVICE_TOKEN, ENV_EDIT_CONCURRENCY, ENV_GATEWAY_URL, KNOWN_OUTPUT_SUFFIXES, build_job, main
from imageflow.core.classification import ClassificationEngine

from support import RED, json_handler, no_sleep, serve, write_image


@pytest.fixture
def configured(monkeypatch):
    """A syntactically valid configuration that points nowhere. Tests with no images never call it."""
    monkeypatch.setenv(ENV_GATEWAY_URL, "http://127.0.0.1:9")
    monkeypatch.setenv(ENV_DEVICE_TOKEN, "token")
    monkeypatch.delenv(ENV_CLASSIFY_CONCURRENCY, raising=False)
    monkeypatch.delenv(ENV_EDIT_CONCURRENCY, raising=False)


def make_workspace(tmp_path, *folders):
    root = tmp_path / "job"
    for name in folders:
        (root / name).mkdir(parents=True)
    return root


# ----- configuration problems are messages, not tracebacks ----------------------------------------------------
def test_an_unknown_action_name_lists_the_available_ones(configured, tmp_path, capsys):
    root = make_workspace(tmp_path, "Khu")
    assert main(["run", str(root), "--action", "Khu=NOPE"]) == 2
    err = capsys.readouterr().err
    assert "unknown action" in err and "PPE_SIMULATION" in err


@pytest.mark.parametrize("value, fragment", [("two", "whole number"), ("0", "at least 1"), ("2.5", "whole number")])
def test_a_bad_concurrency_setting_names_the_variable(configured, tmp_path, capsys, monkeypatch, value, fragment):
    monkeypatch.setenv(ENV_EDIT_CONCURRENCY, value)
    assert main(["run", str(make_workspace(tmp_path, "Khu"))]) == 2
    err = capsys.readouterr().err
    assert ENV_EDIT_CONCURRENCY in err and fragment in err


def test_a_plain_http_gateway_address_is_refused_with_a_message(configured, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv(ENV_GATEWAY_URL, "http://gateway.example.com")
    assert main(["run", str(make_workspace(tmp_path, "Khu"))]) == 2
    assert "https" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["scan", "run"])
def test_a_mistyped_folder_is_refused_and_nothing_is_created(configured, tmp_path, capsys, command):
    typo = tmp_path / "does" / "not" / "exist"
    assert main([command, str(typo)]) == 2
    assert "not a folder" in capsys.readouterr().err
    assert not (tmp_path / "does").exists()  # run used to create .imageflow/ under the typo and exit 0


def test_a_file_given_instead_of_a_folder_is_refused(configured, tmp_path, capsys):
    a_file = tmp_path / "photo.jpg"
    write_image(a_file, RED)
    assert main(["run", str(a_file)]) == 2
    assert "not a folder" in capsys.readouterr().err


# ----- folder names -----------------------------------------------------------------------------------------
def test_a_folder_name_typed_in_decomposed_unicode_is_found(configured, tmp_path, capsys):
    root = make_workspace(tmp_path, "Nghiệm thu")
    decomposed = unicodedata.normalize("NFD", "Nghiệm thu")
    assert main(["run", str(root), "--action", f"{decomposed}=PPE_SIMULATION"]) == 0
    saved = json.loads((root / ".imageflow" / "project.json").read_text(encoding="utf-8"))
    assert saved["folders"][0]["actions"] == ["PPE_SIMULATION"]


def test_an_assignment_for_a_folder_that_was_removed_is_refused(configured, tmp_path, capsys):
    root = make_workspace(tmp_path, "A", "B")
    assert main(["run", str(root)]) == 0  # registers the folders
    (root / "A").rmdir()

    assert main(["run", str(root), "--action", "A=PPE_SIMULATION"]) == 2
    assert "no folder named 'A'" in capsys.readouterr().err  # it used to write onto the missing record and exit 0


def test_an_assignment_is_saved_and_an_empty_one_clears_it(configured, tmp_path):
    root = make_workspace(tmp_path, "Khu")
    saved = lambda: json.loads((root / ".imageflow" / "project.json").read_text(encoding="utf-8"))["folders"][0]["actions"]  # noqa: E731
    assert main(["run", str(root), "--action", "Khu=PPE_SIMULATION"]) == 0
    assert saved() == ["PPE_SIMULATION"]
    assert main(["run", str(root), "--action", "Khu="]) == 0
    assert saved() == []


def test_an_action_flag_warns_that_ppe_screening_is_automatic(configured, tmp_path, capsys):
    root = make_workspace(tmp_path, "Khu")
    assert main(["run", str(root), "--action", "Khu=PPE_SIMULATION"]) == 0
    assert "PPE screening is automatic" in capsys.readouterr().err


# ----- exit codes ---------------------------------------------------------------------------------------------
def test_images_left_for_another_run_give_exit_code_1(tmp_path, capsys, monkeypatch):
    root = make_workspace(tmp_path, "Khu")
    write_image(root / "red.jpg", RED)
    monkeypatch.setenv(ENV_DEVICE_TOKEN, "token")
    monkeypatch.setattr(app_mod, "ClassificationEngine", lambda primary, fallback: ClassificationEngine(primary, fallback, sleep=no_sleep))

    with serve(json_handler({"/v1/classify": (503, {}, b"{}")})) as gateway:
        monkeypatch.setenv(ENV_GATEWAY_URL, gateway)
        code = main(["run", str(root)])

    captured = capsys.readouterr()
    assert code == 1
    assert json.loads(captured.out)["deferred"] == 1
    assert "1 image(s) could not be processed yet" in captured.err
    assert (root / "red.jpg").exists()


# ----- consistency and efficiency -----------------------------------------------------------------------------
def test_scan_and_the_job_agree_about_which_files_are_outputs(configured, tmp_path):
    job = build_job(make_workspace(tmp_path, "Khu"))
    assert job.actions.output_suffixes() == KNOWN_OUTPUT_SUFFIXES


def test_run_scans_the_workspace_once(configured, tmp_path, monkeypatch):
    scans = []
    real = pipeline_mod.scan_workspace
    monkeypatch.setattr(pipeline_mod, "scan_workspace", lambda *a, **k: scans.append(1) or real(*a, **k))
    assert main(["run", str(make_workspace(tmp_path, "Khu"))]) == 0
    assert len(scans) == 1  # it used to scan twice: prepare() and then run() prepared again


def test_output_survives_a_console_that_cannot_encode_vietnamese(configured, tmp_path, monkeypatch):
    root = make_workspace(tmp_path, "Nghiệm thu")
    console = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")
    monkeypatch.setattr(sys, "stdout", console)

    assert main(["scan", str(root)]) == 0  # it used to raise UnicodeEncodeError on a redirected Windows console

    console.flush()
    assert "Nghiệm thu" in console.buffer.getvalue().decode("utf-8")
