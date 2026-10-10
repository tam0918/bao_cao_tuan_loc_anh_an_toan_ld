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
`IMAGEFLOW_EDIT_CONCURRENCY` (mặc định 2).

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
  dùng chọn thư mục.

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
Không có lệnh gọi mạng ra nhà cung cấp nào trong bộ test.

## Chưa làm / cần lưu ý

- **Gateway (Cloudflare Worker) chưa có trong repo.** Cần triển khai theo hợp đồng ở
  trên. Đối chiếu tên model `clef-flash`, `clef`, `seedream-5.0-flash` với nhà cung cấp
  thực tế trước khi dùng production.
- **Đóng gói Windows (.exe/installer) chưa làm.** Không kiểm chứng được trong môi trường
  hiện tại. Cần build trên Windows.
- **Giao diện Tk chưa được chạy thử.** Môi trường không có tkinter. Chỉ kiểm tra biên dịch
  và phần chữ (`imageflow/ui/texts.py`) bằng test. Cần chạy thử thủ công trên Windows.
- Chưa có kéo-thả thư mục vào cửa sổ. Hiện chỉ có nút chọn thư mục.
- Chưa có khóa đa tiến trình. Không chạy hai job trên cùng một thư mục cùng lúc.
- Nhận diện đổi tên dựa trên inode/file index. Trên một số hệ thống file tái sử dụng ID
  ngay sau khi xóa, một thư mục xóa rồi tạo lại cùng tên có thể kế thừa mapping cũ.
- Thư mục `.imageflow/outputs/` đã được tạo nhưng chưa dùng.
