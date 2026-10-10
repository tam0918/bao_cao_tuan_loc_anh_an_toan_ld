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


def scan_summary(images: int, folders: int, unsupported: int) -> str:
    lines = ["Tìm thấy:", "", f"{images} ảnh chưa phân loại", f"{folders} thư mục"]
    if unsupported:
        lines += ["", f"{unsupported} tệp không hỗ trợ (sẽ bỏ qua)"]
    return "\n".join(lines)


def done_summary(classified: int, actioned: int, review: int) -> str:
    return "\n".join(
        [
            "Hoàn tất",
            "",
            f"{classified} ảnh đã phân loại",
            f"{actioned} ảnh được tạo minh họa PPE",
            f"{review} ảnh cần kiểm tra",
        ]
    )


def review_row(name: str) -> str:
    return f"{name}: chọn thư mục"
