"""The UI's wording is pure Python, so it is tested without a display."""

from imageflow.ui import texts


def test_scan_summary_counts_images_and_folders():
    assert texts.scan_summary(127, 5, 0) == "Tìm thấy:\n\n127 ảnh chưa phân loại\n5 thư mục"


def test_scan_summary_mentions_skipped_unsupported_files_only_when_present():
    assert "2 tệp không hỗ trợ" in texts.scan_summary(1, 1, 2)
    assert "không hỗ trợ" not in texts.scan_summary(1, 1, 0)


def test_done_summary_matches_the_completion_screen():
    summary = texts.done_summary(127, 23, 4)
    assert summary.splitlines()[0] == "Hoàn tất"
    assert "127 ảnh đã phân loại" in summary
    assert "23 ảnh được tạo minh họa PPE" in summary
    assert "4 ảnh cần kiểm tra" in summary


def test_user_facing_text_never_names_models_or_internal_settings():
    shown = " ".join(
        [texts.HOME_HINT, texts.PROGRESS, texts.PPE_ACTION_LABEL, texts.done_summary(1, 1, 1), texts.scan_summary(1, 1, 1)]
    ).lower()
    for internal in ("seedream", "clef", "prompt", "confidence", "concurrency", "api", "threshold"):
        assert internal not in shown
