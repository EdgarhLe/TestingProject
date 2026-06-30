# API Contract v1

**Trạng thái:** Finalized — đã chốt sau review của FE1, Tuần 2
**Phiên bản trước:** `api_contract_v0.md`

> Tài liệu này chỉ định nghĩa **hình dạng request/response** giữa frontend và backend.
> Các quyết định về framework, hosting, authentication, và infrastructure sẽ được thảo luận riêng.

## Thay Đổi So Với v0

| # | Thay đổi | Lý do |
|---|---|---|
| 1 | Thêm `description` vào mỗi event trong Trake response | FE cần text sự kiện để hiển thị trên timeline, tránh phải tự map lại theo index |
| 2 | Thêm `timestamp_seconds` (giữ cả `video_fps`) vào mọi response | FE không cần tự tính `frame_index / video_fps` ở mọi component |
| 3 | Thêm `total_results` và `has_next` vào mọi response | Chuẩn bị cho Load More / Infinite Scroll nếu làm Grid view sau này |
| 4 | Field names giữ nguyên (`score`, `event_index`, `answer`) | FE cập nhật `types.ts` để khớp theo contract, không đổi theo FE |

**Quyết định cho 3 câu hỏi mở của v0:**
- **Q1 (`top_k` defaults):** Giữ `5` cho KIS, `3` cho Trake/Q&A — FE1 xác nhận phù hợp với UI hiện tại.
- **Q2 (Trake `found: false`):** FE sẽ style mờ hoặc icon cảnh báo cho event chưa tự tin — không cần BE thay đổi response.
- **Q3 (Q&A `answer_source`):** Hiển thị lên UI dưới dạng badge `"OCR"` hoặc `"LLM"` để báo độ tin cậy cho người dùng.

---

## Endpoints

### Single-query (Manual Verification Mode)

| Endpoint | Nhiệm vụ |
|---|---|
| `POST /query/kis` | Known-Item Search |
| `POST /query/trake` | Temporal Retrieval of Keyframes |
| `POST /query/qa` | Question Answering |

### Batch (Competition Mode)

| Endpoint | Nhiệm vụ |
|---|---|
| `POST /query/kis/batch` | Batch KIS — array of queries |
| `POST /query/trake/batch` | Batch Trake — array of queries |
| `POST /query/qa/batch` | Batch Q&A — array of queries |

Batch endpoints accept an array of request objects (same shape as single-query, plus `query_id`) and return an array of response objects with `query_id` preserved for result matching. The frontend file parser constructs the request bodies and calls these endpoints after importing the organiser's `.txt` file pack.

---

## Video Path Resolution

Video không được truyền qua mạng. Backend trả về `video_id`, frontend tự dựng đường dẫn đến file local.

**Cấu trúc dataset:**
```
{group}/video/{video_id}.mp4
# ví dụ: L01/video/L01_V001.mp4
```

**Quy tắc:**
- `video_id` có dạng `L01_V001`
- `group` = phần đầu của `video_id` tách theo `_` → `L01`
- Backend resolve: `{VIDEO_BASE_PATH}/L01/video/L01_V001.mp4`
- Frontend resolve: `{FRONTEND_VIDEO_BASE_URL}/L01/video/L01_V001.mp4`

Cả hai giá trị base được cấu hình qua `.env` trên từng máy — không hardcode trong code.

**FPS:**
FPS mặc định được lưu trong `configs/dataset.yaml`. Tuy nhiên, để an toàn, mỗi response đều trả về `video_fps` cho từng video cụ thể. Frontend dùng để convert frame sang timestamp:

```js
const timestamp = frame_index / video_fps  // ví dụ: 1042 / 25 = 41.68s
videoPlayer.seek(timestamp)
```

---

## 1. KIS — Known-Item Search

### Request

```json
POST /query/kis
{
  "query_id": "p1-1",
  "query": "Cảnh quay bằng flycam một cây cầu ở TP Hồ Chí Minh...",
  "top_k": 5
}
```

| Trường | Kiểu | Bắt buộc | Mô tả |
|---|---|---|---|
| `query_id` | string | — | ID từ filename (e.g. `p1-1`); bỏ trống nếu gọi thủ công từ UI |
| `query` | string | ✓ | Mô tả cảnh cần tìm (tiếng Việt hoặc tiếng Anh) |
| `top_k` | integer | — | Số kết quả trả về; mặc định `5` |

### Batch Request

```json
POST /query/kis/batch
{
  "queries": [
    { "query_id": "p1-1", "query": "Cảnh quay bằng flycam...", "top_k": 5 },
    { "query_id": "p1-2", "query": "Một người đàn ông đang trả lời phỏng vấn...", "top_k": 5 }
  ]
}
```

### Batch Response

```json
{
  "results": [
    {
      "query_id": "p1-1",
      "results": [ { "video_id": "L01_V001", "video_fps": 25, "frame_index": 1042, "start_frame": 1020, "end_frame": 1080, "timestamp_seconds": 41.68, "score": 0.87 } ],
      "total_results": 1,
      "has_next": false
    },
    {
      "query_id": "p1-2",
      "results": [ { "video_id": "L02_V003", "video_fps": 25, "frame_index": 530, "start_frame": 510, "end_frame": 560, "timestamp_seconds": 21.2, "score": 0.81 } ],
      "total_results": 1,
      "has_next": false
    }
  ],
  "total_query_time_ms": 1240
}
```

### Response

```json
{
  "results": [
    {
      "video_id": "L01_V001",
      "video_fps": 25,
      "frame_index": 1042,
      "start_frame": 1020,
      "end_frame": 1080,
      "timestamp_seconds": 41.68,
      "score": 0.87
    },
    {
      "video_id": "L02_V003",
      "video_fps": 25,
      "frame_index": 530,
      "start_frame": 510,
      "end_frame": 560,
      "timestamp_seconds": 21.2,
      "score": 0.81
    }
  ],
  "total_results": 2,
  "has_next": false,
  "query_time_ms": 340
}
```

| Trường | Kiểu | Mô tả |
|---|---|---|
| `results` | array | Danh sách kết quả, sắp xếp theo `score` giảm dần |
| `results[].video_id` | string | ID video trong dataset |
| `results[].video_fps` | float | FPS của video — giữ lại cho component cần raw fps |
| `results[].frame_index` | integer | Keyframe tốt nhất trong segment |
| `results[].start_frame` | integer | Frame đầu của segment |
| `results[].end_frame` | integer | Frame cuối của segment |
| `results[].timestamp_seconds` | float | `frame_index / video_fps` — tính sẵn phía backend |
| `results[].score` | float | Cosine similarity với query embedding (0–1) |
| `total_results` | integer | Tổng số kết quả tìm được (có thể nhiều hơn `top_k`) |
| `has_next` | boolean | `true` nếu còn kết quả chưa trả về — dùng cho Load More |
| `query_time_ms` | integer | Thời gian xử lý toàn bộ query (ms) |

---

## 2. Trake — Temporal Retrieval of Keyframes

### Request

```json
POST /query/trake
{
  "query_id": "p1-5",
  "events": [
    "Người đàn ông mặc áo đỏ bước lên sân khấu",
    "Cô gái cầm micro bắt đầu hát",
    "Toàn bộ khán giả đứng dậy vỗ tay"
  ],
  "top_k": 3
}
```

| Trường | Kiểu | Bắt buộc | Mô tả |
|---|---|---|---|
| `query_id` | string | — | ID từ filename (e.g. `p1-5`); bỏ trống nếu gọi thủ công |
| `events` | string[] | ✓ | Danh sách mô tả sự kiện theo thứ tự thời gian (E1, E2, ..., En) |
| `top_k` | integer | — | Số video candidate trả về; mặc định `3` |

### Response

```json
{
  "results": [
    {
      "video_id": "L01_V001",
      "video_fps": 25,
      "overall_score": 0.87,
      "events": [
        {
          "event_index": 0,
          "description": "Người đàn ông mặc áo đỏ bước lên sân khấu",
          "frame_index": 512,
          "start_frame": 490,
          "end_frame": 540,
          "timestamp_seconds": 20.48,
          "score": 0.91,
          "found": true
        },
        {
          "event_index": 1,
          "description": "Cô gái cầm micro bắt đầu hát",
          "frame_index": 870,
          "start_frame": 850,
          "end_frame": 910,
          "timestamp_seconds": 34.8,
          "score": 0.84,
          "found": true
        },
        {
          "event_index": 2,
          "description": "Toàn bộ khán giả đứng dậy vỗ tay",
          "frame_index": 1203,
          "start_frame": 1190,
          "end_frame": 1240,
          "timestamp_seconds": 48.12,
          "score": 0.31,
          "found": false
        }
      ]
    }
  ],
  "total_results": 1,
  "has_next": false,
  "query_time_ms": 610
}
```

| Trường | Kiểu | Mô tả |
|---|---|---|
| `results` | array | Danh sách video candidate, sắp xếp theo `overall_score` giảm dần |
| `results[].video_id` | string | ID video |
| `results[].video_fps` | float | FPS của video |
| `results[].overall_score` | float | Score tổng hợp của toàn bộ chuỗi sự kiện trong video này |
| `results[].events` | array | Luôn trả về đủ N phần tử — khớp 1-1 với mảng `events` trong request |
| `events[].event_index` | integer | Index 0-based, khớp với vị trí trong request `events[]` |
| `events[].description` | string | Text sự kiện gốc từ request — echo lại để FE hiển thị trên timeline không cần tự map |
| `events[].frame_index` | integer | Frame tốt nhất cho sự kiện này |
| `events[].start_frame` | integer | Frame đầu của segment |
| `events[].end_frame` | integer | Frame cuối của segment |
| `events[].timestamp_seconds` | float | `frame_index / video_fps` — tính sẵn phía backend |
| `events[].score` | float | Cosine similarity của sub-query với segment |
| `events[].found` | boolean | `true` nếu kết quả tự tin; `false` nếu là best-effort (score thấp) |
| `total_results` | integer | Tổng số video candidate tìm được |
| `has_next` | boolean | `true` nếu còn candidate chưa trả về — dùng cho Load More |
| `query_time_ms` | integer | Thời gian xử lý (ms) |

**Lưu ý:**
- Thứ tự thời gian được đảm bảo: `frame_index(En) > frame_index(En-1)` trong cùng một video.
- Kết quả partial (có `found: false`) được highlight trên UI để team kiểm tra thủ công trước khi submit.

### Batch Request

```json
POST /query/trake/batch
{
  "queries": [
    { "query_id": "p1-5", "events": ["...", "...", "..."], "top_k": 3 },
    { "query_id": "p1-6", "events": ["...", "...", "..."], "top_k": 3 }
  ]
}
```

### Batch Response

```json
{
  "results": [
    {
      "query_id": "p1-5",
      "results": [
        {
          "video_id": "L01_V001", "video_fps": 25, "overall_score": 0.87,
          "events": [
            { "event_index": 0, "description": "Người đầu bếp cho cá vào một tô màu trắng...", "frame_index": 512, "start_frame": 490, "end_frame": 540, "timestamp_seconds": 20.48, "score": 0.91, "found": true },
            { "event_index": 1, "description": "Người đầu bếp đổ bột vào một tô cá để chiên...", "frame_index": 870, "start_frame": 850, "end_frame": 910, "timestamp_seconds": 34.8, "score": 0.84, "found": true },
            { "event_index": 2, "description": "Người đầu bếp dùng đũa để kiểm tra độ nóng của dầu...", "frame_index": 1203, "start_frame": 1190, "end_frame": 1240, "timestamp_seconds": 48.12, "score": 0.31, "found": false }
          ]
        }
      ],
      "total_results": 1,
      "has_next": false
    }
  ],
  "total_query_time_ms": 1850
}
```


---

## 3. Q&A — Question Answering

### Request

```json
POST /query/qa
{
  "query_id": "p1-3",
  "scene": "Cảnh quay tại một hội nghị lớn, có bảng tên và logo công ty phía sau diễn giả",
  "question": "Tên của diễn giả được ghi trên bảng tên là gì?",
  "top_k": 3
}
```

| Trường | Kiểu | Bắt buộc | Mô tả |
|---|---|---|---|
| `query_id` | string | — | ID từ filename (e.g. `p1-3`); bỏ trống nếu gọi thủ công |
| `scene` | string | ✓ | Mô tả cảnh — dùng cho Stage 1 retrieval (embedding + BM25) |
| `question` | string | ✓ | Câu hỏi cụ thể — dùng cho Stage 2 frame selection + OCR/LLM |
| `top_k` | integer | — | Số kết quả trả về; mặc định `3` |

### Batch Request

```json
POST /query/qa/batch
{
  "queries": [
    { "query_id": "p1-3", "scene": "Cảnh quay tại một hội nghị lớn...", "question": "Tên của diễn giả là gì?", "top_k": 3 },
    { "query_id": "p1-4", "scene": "Một bác sĩ tóc bạc đeo kính...", "question": "Họ và tên đầy đủ của bác sĩ là gì?", "top_k": 3 }
  ]
}
```

### Batch Response

```json
{
  "results": [
    {
      "query_id": "p1-3",
      "results": [
        { "video_id": "L01_V001", "video_fps": 25, "frame_index": 2310, "start_frame": 2290, "end_frame": 2350, "timestamp_seconds": 92.4, "answer": "Nguyễn Văn An", "answer_source": "ocr", "score": 0.79 }
      ],
      "total_results": 1,
      "has_next": false
    }
  ],
  "total_query_time_ms": 1640
}
```

### Response

```json
{
  "results": [
    {
      "video_id": "L01_V001",
      "video_fps": 25,
      "frame_index": 2310,
      "start_frame": 2290,
      "end_frame": 2350,
      "timestamp_seconds": 92.4,
      "answer": "Nguyễn Văn An",
      "answer_source": "ocr",
      "score": 0.79
    },
    {
      "video_id": "L02_V004",
      "video_fps": 30,
      "frame_index": 870,
      "start_frame": 850,
      "end_frame": 910,
      "timestamp_seconds": 29.0,
      "answer": "Trần Minh Khoa",
      "answer_source": "llm",
      "score": 0.61
    }
  ],
  "total_results": 2,
  "has_next": false,
  "query_time_ms": 820
}
```

| Trường | Kiểu | Mô tả |
|---|---|---|
| `results` | array | Danh sách kết quả, sắp xếp theo `score` giảm dần |
| `results[].video_id` | string | ID video |
| `results[].video_fps` | float | FPS của video |
| `results[].frame_index` | integer | Frame tốt nhất chứa câu trả lời |
| `results[].start_frame` | integer | Frame đầu của segment |
| `results[].end_frame` | integer | Frame cuối của segment |
| `results[].timestamp_seconds` | float | `frame_index / video_fps` — tính sẵn phía backend |
| `results[].answer` | string | Câu trả lời đã trích xuất |
| `results[].answer_source` | string | `"ocr"` — đọc trực tiếp từ màn hình; `"llm"` — cần suy luận thêm. Hiển thị lên UI dưới dạng badge `"OCR"` / `"LLM"` |
| `results[].score` | float | Score retrieval tổng hợp (0–1) |
| `total_results` | integer | Tổng số kết quả tìm được |
| `has_next` | boolean | `true` nếu còn kết quả chưa trả về — dùng cho Load More |
| `query_time_ms` | integer | Thời gian xử lý (ms) |

---

## Lỗi

Cùng một format trên cả 3 endpoint:

```json
{
  "code": "NO_RESULTS_FOUND",
  "message": "Không tìm thấy kết quả phù hợp với truy vấn."
}
```

| Code | HTTP Status | Ý nghĩa |
|---|---|---|
| `NO_RESULTS_FOUND` | 404 | Không có kết quả nào vượt ngưỡng score tối thiểu |
| `INVALID_REQUEST` | 400 | Thiếu trường bắt buộc hoặc sai kiểu dữ liệu |
| `INTERNAL_ERROR` | 500 | Lỗi server không xác định |

---

## Submission CSV Format

Sau khi chạy batch, frontend export một file CSV duy nhất cho tất cả task types:

| Column | KIS | Trake | Q&A |
|---|---|---|---|
| `query_id` | ✓ | ✓ | ✓ |
| `video_id` | ✓ | ✓ | ✓ |
| `frame_index` | ✓ | ✓ | ✓ |
| `event_index` | — | ✓ (một dòng per event) | — |
| `answer` | — | — | ✓ |

Ví dụ:
```csv
query_id,video_id,frame_index,event_index,answer
p1-1,L01_V001,1042,,
p1-3,L02_V003,2310,,Nguyễn Văn An
p1-5,L01_V001,512,0,
p1-5,L01_V001,870,1,
p1-5,L01_V001,1203,2,
```

**Lưu ý:** Format này là giả định — phải xác nhận với ban tổ chức khi có đề bài chính thức. Các điểm cần confirm:
- KIS/Q&A submit top-1 hay top-k?
- Trake một dòng per event hay một dòng per query?
- Tên cột chính xác?


---

## Trạng Thái Sign-off

- [x] FE1 — reviewed, feedback đã áp dụng (xem "Thay Đổi So Với v0" ở đầu file)
- [ ] FE2 — chờ review

Sau khi FE2 sign-off, contract chính thức "frozen" cho Tuần 2. Thay đổi sau điểm này cần thông báo cho cả nhóm và bump version lên `v2`.
