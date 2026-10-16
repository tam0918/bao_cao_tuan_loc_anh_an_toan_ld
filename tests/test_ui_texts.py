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
        [
            texts.HOME_HINT,
            texts.PROGRESS,
            texts.PPE_ACTION_LABEL,
            texts.NO_FOLDERS,
            texts.done_summary(1, 1, 1, pending=1, failed=1),
            texts.scan_summary(1, 1, 1),
            *texts._ABORTED.values(),
            *texts._PROBLEMS.values(),
        ]
    ).lower()
    for internal in ("seedream", "clef", "prompt", "confidence", "concurrency", "api", "threshold"):
        assert internal not in shown


def test_work_left_for_another_run_is_never_called_done():
    summary = texts.done_summary(3, 1, 0, pending=2)
    assert summary.splitlines()[0] == "Chưa hoàn tất"
    assert "2 ảnh chưa xử lý xong" in summary
    assert "Hoàn tất" not in summary.splitlines()[0]


def test_unreadable_images_are_counted_separately_from_work_to_retry():
    summary = texts.done_summary(5, 0, 0, failed=2)
    assert summary.splitlines()[0] == "Hoàn tất"  # a corrupt photo is not something a rerun can fix
    assert "2 ảnh bị lỗi hoặc không đọc được" in summary


def test_a_stopped_run_explains_itself_in_vietnamese_and_falls_back_to_the_detail():
    assert "Không kết nối được máy chủ" in texts.abort_message("SERVICE_UNAVAILABLE", "english detail")
    assert "quyền" in texts.abort_message("AUTH_FAILED", "english detail")
    assert texts.abort_message(None, "plain detail") == "plain detail"
    assert texts.abort_message("SOMETHING_NEW", "plain detail") == "plain detail"


def test_every_problem_a_move_can_report_has_wording():
    for code in ("DESTINATION_MISSING", "ROUTE_FAILED", "ACTION_FAILED", "SOURCE_MISSING", "LOST_SOURCE"):
        assert "(" not in texts.problem_message(code)  # not the generic fallback
    assert "SOMETHING_NEW" in texts.problem_message("SOMETHING_NEW")
