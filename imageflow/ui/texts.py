"""User-facing strings and summary formatting (spec sections 29-32). Pure Python, no tkinter.

Field users never see model names, prompts, thresholds or concurrency. Those stay internal.
"""

from __future__ import annotations

APP_TITLE = "ImageFlow"
HOME_TITLE = "Chọn thư mục công việc"
HOME_HINT = (
    "Tạo các thư mục theo nhu cầu, copy ảnh vào thư mục gốc, rồi chọn thư mục đó. "
    "Ảnh sẽ tự được phân loại vào đúng thư mục."
)
CHOOSE_FOLDER = "Chọn thư mục"
BACK = "Quay lại"
START = "Bắt đầu"
PPE_ACTION_LABEL = "Tạo ảnh minh họa PPE"
PROGRESS = "Đang xử lý, vui lòng chờ..."
OPEN_FOLDER = "Mở thư mục"
REVIEW_BUTTON = "Xem ảnh cần kiểm tra"
MOVE_BUTTON = "Chuyển"
REVIEW_TITLE = "Ảnh cần kiểm tra"
REVIEW_EMPTY = "Không còn ảnh nào cần kiểm tra."
NO_FOLDERS = "Chưa có thư mục nào. Hãy tạo thư mục trong thư mục công việc trước khi bắt đầu."
CONFIG_ERROR = "Chưa cấu hình được ứng dụng. Hãy liên hệ người quản trị.\n\n{detail}"
UNEXPECTED_ERROR = "Có lỗi không mong muốn:\n{error}"

_ABORTED = {
    "SERVICE_UNAVAILABLE": "Không kết nối được máy chủ. Hãy kiểm tra mạng rồi chạy lại. Ảnh chưa xử lý vẫn còn nguyên.",
    "AUTH_FAILED": "Máy tính này chưa được cấp quyền hoặc quyền đã bị thu hồi. Hãy liên hệ người quản trị.",
}

_PROBLEMS = {
    "DESTINATION_MISSING": "Thư mục đích không còn nữa. Ảnh vẫn nằm trong danh sách, hãy chọn thư mục khác.",
    "ROUTE_FAILED": "Chưa chuyển được ảnh, có thể tệp đang được chương trình khác dùng. Ảnh sẽ được thử lại ở lần chạy sau.",
    "ACTION_FAILED": "Ảnh đã được chuyển nhưng chưa tạo được ảnh minh họa. Hãy chạy lại để thử tiếp.",
    "SOURCE_MISSING": "Không tìm thấy ảnh gốc nữa.",
    "LOST_SOURCE": "Không tìm thấy ảnh gốc nữa.",
}


def scan_summary(images: int, folders: int, unsupported: int) -> str:
    lines = ["Tìm thấy:", "", f"{images} ảnh chưa phân loại", f"{folders} thư mục"]
    if unsupported:
        lines += ["", f"{unsupported} tệp không hỗ trợ (sẽ bỏ qua)"]
    return "\n".join(lines)


def done_summary(classified: int, actioned: int, review: int, pending: int = 0, failed: int = 0) -> str:
    """The completion screen. ``pending`` images are waiting for another run, so it must not say 'done'."""
    lines = [
        "Hoàn tất" if not pending else "Chưa hoàn tất",
        "",
        f"{classified} ảnh đã phân loại",
        f"{actioned} ảnh được tạo minh họa PPE",
        f"{review} ảnh cần kiểm tra",
    ]
    if pending:
        lines += ["", f"{pending} ảnh chưa xử lý xong, hãy chạy lại để thử tiếp"]
    if failed:
        lines += [f"{failed} ảnh bị lỗi hoặc không đọc được"]
    return "\n".join(lines)


def review_row(name: str) -> str:
    return f"{name}: chọn thư mục"


def abort_message(code: str | None, detail: str) -> str:
    """Why a run or an assignment stopped. Falls back to the raw detail for codes without wording."""
    return _ABORTED.get(code or "", detail)


def problem_message(code: str) -> str:
    return _PROBLEMS.get(code, f"Chưa xử lý được ảnh này ({code}).")
