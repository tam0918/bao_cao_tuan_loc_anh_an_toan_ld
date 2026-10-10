"""COPY routing mode (spec section 6): the root original stays, the category gets its own file."""

from __future__ import annotations

import os

from support import RED, ScriptedClassifier, build_job, folder_files, routed_to, run, write_image


def test_copy_mode_keeps_the_root_original_and_a_rerun_never_duplicates(tmp_path):
    root = tmp_path / "job"
    (root / "Khu").mkdir(parents=True)
    write_image(root / "red.jpg", RED)
    clf = ScriptedClassifier(routed_to({RED: "Khu"}))

    first = build_job(root, clf, job_id="j_copy1")
    first.prepare()
    first.set_routing_mode("copy")
    run(first)

    assert (root / "red.jpg").exists()  # the source root is kept
    assert folder_files(root, "Khu") == ["red.jpg"]
    assert not os.path.samefile(root / "red.jpg", root / "Khu" / "red.jpg")  # independent bytes

    run(build_job(root, clf, job_id="j_copy2"))

    assert clf.count == 1  # the root copy is recognised as already processed
    assert folder_files(root, "Khu") == ["red.jpg"]  # and nothing new was routed
