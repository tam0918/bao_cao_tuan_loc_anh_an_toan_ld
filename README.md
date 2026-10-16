# ImageFlow

Phân loại ảnh hiện trường theo **chính hệ thống thư mục do bạn tự tạo**, sau đó chỉ chạy
action AI cho những thư mục được gán action. Taxonomy không nằm trong code: đổi tên hoặc
thêm thư mục là xong, không cần sửa mã.

Đặc tả kiến trúc gốc nằm trong yêu cầu của dự án. Tài liệu này mô tả phần đã được cài đặt.

## Cấu trúc thư mục công việc

```
JOB_FOLDER/
├── IMG_001.jpg            ảnh gốc chưa phân loại (nằm ở root)
├── Thi công móng/         thư mục cấp 1 = một lựa chọn phân loại
├── Mất an toàn lao động/
│   └── .rule.txt          tùy chọn: mô tả thêm cho model
└── .imageflow/            dữ liệu nội bộ, do ImageFlow tạo
```

- Chỉ thư mục **cấp 1** là category. Thư mục lồng bên trong không được tính.
- `.imageflow/`, thư mục ẩn và thư mục hệ thống bị bỏ qua.
- Định dạng hỗ trợ: `.jpg .jpeg .png .webp`. Định dạng khác được báo cáo, không tự convert.
- Ảnh gốc không bao giờ bị xóa hoặc ghi đè. Ảnh AI luôn là file mới, ví dụ
  `IMG_018_AI_PPE.png`, cùng thư mục với ảnh gốc.

## Chạy thử

Yêu cầu Python 3.11 trở lên.

```bash
python -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

export IMAGEFLOW_GATEWAY_URL=https://<gateway-của-bạn>
export IMAGEFLOW_DEVICE_TOKEN=<token-thiết-bị>   # có thể thu hồi từ phía gateway

imageflow scan  "D:/Công trình/Ngày 10"
imageflow run   "D:/Công trình/Ngày 10" --action "Mất an toàn lao động=PPE_SIMULATION"
imageflow ui                     # cửa sổ desktop
```

Các biến tùy chọn: `IMAGEFLOW_CLASSIFY_CONCURRENCY` (mặc định 4),
`IMAGEFLOW_EDIT_CONCURRENCY` (mặc định 2). Giá trị sai (không phải số nguyên, nhỏ hơn 1) hoặc
địa chỉ gateway dùng `http://` từ xa sẽ báo lỗi rõ ràng và thoát với mã 2, không in traceback.

`--action "<tên thư mục>=PPE_SIMULATION"` được **lưu cùng job** và giữ nguyên cho các lần chạy
sau. Muốn tắt, dùng `--action "<tên thư mục>="`. Tên thư mục không tồn tại, hoặc thư mục đã bị
xóa, bị từ chối thay vì lặng lẽ ghi vào một bản ghi cũ.

Mã thoát của `imageflow run`: `0` xong hẳn; `1` đã chạy xong nhưng còn ảnh cần chạy lại
(mất kết nối, ảnh minh họa lỗi); `2` không chạy được hoặc bị dừng (sai cấu hình, thư mục không
tồn tại, token bị thu hồi, máy chủ không kết nối được).

Khóa API của nhà cung cấp **không** nằm trong ứng dụng. Ứng dụng chỉ có token thiết bị
và gọi qua gateway.

## Quy trình xử lý

```
scan → normalize → classify (clef-flash, fallback clef) → route → action → complete
                         │
                         └─ không đủ tự tin → REVIEW_REQUIRED → người dùng chọn thư mục
```

- Tự động route khi `top1 ≥ 0.85` **và** `top1 − top2 ≥ 0.15`. Không đạt thì thử model
  fallback, vẫn không đạt thì đưa vào review. Hệ thống không đoán.
- Mỗi ảnh chỉ vào **một** thư mục (single-label).
- Action chỉ chạy trên thư mục mà người dùng đã gán. Tên thư mục không quyết định gì.
- Đổi tên thư mục giữ nguyên ID và action. Thư mục biến mất được đánh dấu `MISSING`,
  action không bị chuyển sang thư mục khác.
- Ảnh review: bản sao nằm ở `.imageflow/review/`, ảnh gốc vẫn ở root cho đến khi người
  dùng chọn thư mục. Nếu thư mục đích biến mất đúng lúc chuyển, ảnh quay lại danh sách review;
  nếu tệp đang bị chương trình khác khóa, lần chạy sau sẽ thử lại.
- Nếu máy chủ không kết nối được cho 5 ảnh liên tiếp, job dừng ngay thay vì thử lại từng ảnh.
  Lỗi mạng không gọi model dự phòng vì cùng đi qua một gateway. Ảnh chưa xử lý vẫn nguyên.
- Ảnh bị cắt dở (copy gián đoạn) được đánh dấu lỗi một lần. Ảnh AI bị cắt dở bị loại bỏ,
  không bao giờ vào thư mục của người dùng.

## An toàn khi mất điện hoặc crash

Manifest (`.imageflow/manifest.json` + journal) là nguồn sự thật. Mỗi quyết định phân
loại được ghi trước khi file di chuyển. Việc move dùng hard link để không bao giờ ghi đè
file trùng tên, sau đó mới xóa nguồn. Chạy lại job sẽ tiếp tục từ trạng thái đã ghi, không
phân loại lại và không tạo bản trùng.

## Gateway contract

Ứng dụng gọi hai route. Gateway phải kiểm tra `Authorization: Bearer <token>` và trả
JSON. Gateway giữ khóa Cloudflare và Seedream, cũng như tên model thật.

`POST /v1/classify`

```json
{
  "model": "clef-flash",
  "image": {"mime": "image/jpeg", "data_b64": "..."},
  "choices": [{"id": "f_8a31", "label": "Mất an toàn lao động", "description": "..."}],
  "question": "Trong các thư mục được cung cấp, ảnh này phù hợp nhất với thư mục nào?"
}
```

Phản hồi: `{"choice_id": "f_8a31", "confidence": 0.94, "probabilities": {"f_8a31": 0.94, ...}, "model": "clef-flash"}`

`POST /v1/edit`

```json
{"model": "seedream-5.0-flash", "image": {...}, "instruction": "...", "metadata": {...}}
```

Phản hồi: `{"image": {"mime": "image/png", "data_b64": "..."}, "model": "seedream-5.0-flash"}`

Lỗi: `429` (kèm `Retry-After`) và `5xx` là lỗi tạm thời, ứng dụng sẽ thử lại tối đa 2 lần.
`401/403` là `AUTH_FAILED` và dừng job. `400/404/409/413/415/422` là lỗi vĩnh viễn, có thể
kèm `{"error": {"code": "UNSUPPORTED_IMAGE" | "CORRUPTED_IMAGE" | "INVALID_REQUEST"}}`.

## Kiểm thử

```bash
pytest
```

`tests/test_acceptance.py` ánh xạ tiêu chí A–I của đặc tả: taxonomy động, không hard-code
category, action theo gán thủ công, bảo toàn ảnh gốc, ảnh confidence thấp không được
route, phục hồi sau crash, rename thư mục, thay model, thay editor.
`tests/test_end_to_end.py` chạy toàn bộ luồng qua HTTP với một gateway giả lập cục bộ.
`tests/test_review_fixes.py` giữ các lỗi đã tìm thấy khi review (ảnh cắt dở, gateway trả lời
đứt đoạn, chuyển hướng, nhận diện thư mục, giới hạn đồng thời, dừng khi mất kết nối).
`tests/test_cli.py` kiểm tra thông báo lỗi và mã thoát. `tests/test_controller.py` và
`tests/test_window_logic.py` kiểm tra logic cửa sổ không cần màn hình (xem bên dưới).
Không có lệnh gọi mạng ra nhà cung cấp nào trong bộ test.

## Chưa làm / cần lưu ý

- **Gateway (Cloudflare Worker) chưa có trong repo.** Cần triển khai theo hợp đồng ở
  trên. Đối chiếu tên model `clef-flash`, `clef`, `seedream-5.0-flash` với nhà cung cấp
  thực tế trước khi dùng production.
- **Đóng gói Windows (.exe/installer) chưa làm.** Không kiểm chứng được trong môi trường
  hiện tại. Cần build trên Windows.
- **Giao diện Tk chưa được chạy thử trên màn hình thật.** Môi trường không có tkinter.
  Mọi quyết định của cửa sổ nằm trong `imageflow/ui/controller.py` (có test). Phần vẽ được
  kiểm tra bằng một bản tkinter giả (`tests/fake_tk.py`): bấm nút, chạy vòng lặp, đọc chữ trên
  màn hình. Cách này không kiểm tra được bố cục, cỡ chữ hay hành vi riêng của Tk trên Windows,
  nên vẫn cần chạy thử bằng mắt.
- **Nhận diện đổi tên thư mục trên Windows chưa kiểm chứng.** Mã lấy device/file index bằng
  `os.stat` (không dùng `DirEntry.stat`, vốn trả về 0 trên Windows). Việc đó dựa trên tài liệu
  của Python, chưa chạy trên Windows.
- Chưa có kéo-thả thư mục vào cửa sổ. Hiện chỉ có nút chọn thư mục.
- Chưa có khóa đa tiến trình. Không chạy hai job (hai cửa sổ, hoặc cửa sổ và dòng lệnh) trên
  cùng một thư mục cùng lúc. Trong một cửa sổ, các thao tác được xếp hàng tuần tự nên bấm đúp
  không gây lỗi.
- Nhận diện đổi tên dựa trên inode/file index. Trên một số hệ thống file tái sử dụng ID
  ngay sau khi xóa, một thư mục xóa rồi tạo lại cùng tên có thể kế thừa mapping cũ.
- Thư mục `.imageflow/outputs/` đã được tạo nhưng chưa dùng.
