# API Contract v0

**Trạng thái:** Draft — chờ FE1 + FE2 review, chốt cuối Tuần 1
**Phiên bản tiếp theo:** `api_contract_v1.md` (đầu Tuần 2)

> Tài liệu này chỉ định nghĩa **hình dạng request/response** giữa frontend và backend.
> Các quyết định về framework, hosting, authentication, và infrastructure sẽ được thảo luận riêng.

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
      "results": [ { "video_id": "L01_V001", "video_fps": 25, "frame_index": 1042, "start_frame": 1020, "end_frame": 1080, "score": 0.87 } ]
    },
    {
      "query_id": "p1-2",
      "results": [ { "video_id": "L02_V003", "video_fps": 25, "frame_index": 530, "start_frame": 510, "end_frame": 560, "score": 0.81 } ]
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
      "score": 0.87
    },
    {
      "video_id": "L02_V003",
      "video_fps": 25,
      "frame_index": 530,
      "start_frame": 510,
      "end_frame": 560,
      "score": 0.81
    }
  ],
  "query_time_ms": 340
}
```

| Trường | Kiểu | Mô tả |
|---|---|---|
| `results` | array | Danh sách kết quả, sắp xếp theo `score` giảm dần |
| `results[].video_id` | string | ID video trong dataset |
| `results[].video_fps` | float | FPS của video — dùng để convert frame sang timestamp |
| `results[].frame_index` | integer | Keyframe tốt nhất trong segment |
| `results[].start_frame` | integer | Frame đầu của segment |
| `results[].end_frame` | integer | Frame cuối của segment |
| `results[].score` | float | Cosine similarity với query embedding (0–1) |
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
          "frame_index": 512,
          "start_frame": 490,
          "end_frame": 540,
          "score": 0.91,
          "found": true
        },
        {
          "event_index": 1,
          "frame_index": 870,
          "start_frame": 850,
          "end_frame": 910,
          "score": 0.84,
          "found": true
        },
        {
          "event_index": 2,
          "frame_index": 1203,
          "start_frame": 1190,
          "end_frame": 1240,
          "score": 0.31,
          "found": false
        }
      ]
    }
  ],
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
| `events[].frame_index` | integer | Frame tốt nhất cho sự kiện này |
| `events[].start_frame` | integer | Frame đầu của segment |
| `events[].end_frame` | integer | Frame cuối của segment |
| `events[].score` | float | Cosine similarity của sub-query với segment |
| `events[].found` | boolean | `true` nếu kết quả tự tin; `false` nếu là best-effort (score thấp) |
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
            { "event_index": 0, "frame_index": 512, "start_frame": 490, "end_frame": 540, "score": 0.91, "found": true },
            { "event_index": 1, "frame_index": 870, "start_frame": 850, "end_frame": 910, "score": 0.84, "found": true },
            { "event_index": 2, "frame_index": 1203, "start_frame": 1190, "end_frame": 1240, "score": 0.31, "found": false }
          ]
        }
      ]
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
        { "video_id": "L01_V001", "video_fps": 25, "frame_index": 2310, "start_frame": 2290, "end_frame": 2350, "answer": "Nguyễn Văn An", "answer_source": "ocr", "score": 0.79 }
      ]
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
      "answer": "Trần Minh Khoa",
      "answer_source": "llm",
      "score": 0.61
    }
  ],
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
| `results[].answer` | string | Câu trả lời đã trích xuất |
| `results[].answer_source` | string | `"ocr"` — đọc trực tiếp từ màn hình; `"llm"` — cần suy luận thêm |
| `results[].score` | float | Score retrieval tổng hợp (0–1) |
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

## Câu Hỏi Mở — FE1 + FE2 Phản Hồi Trước Thứ Sáu

1. `top_k` mặc định `5` cho KIS, `3` cho Trake và Q&A — phù hợp với UI không?
2. Trake `found: false` — UI highlight thế nào? Màu khác, icon cảnh báo, hay ẩn đi?
3. Q&A `answer_source` — có cần hiển thị lên UI không, hay chỉ dùng để debug nội bộ?
