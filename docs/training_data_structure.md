# Training Data Pipeline — Cấu Trúc `training/data/`

Tài liệu này giải thích vai trò từng file trong pipeline dữ liệu Phase 1
(issue #48, #49) trên nhánh `data`. Không lặp lại `README.md` gốc — chỉ đi
sâu vào riêng `training/data/` và các file liên quan trực tiếp.

---

## Luồng dữ liệu tổng quan

```
Raw metadata (CC3M TSV/stream, Panda-70M JSONL)
        │
        ▼
adapters/{cc3m,panda70m}.py     — parse raw format -> TrainingSample
        │
        ▼
validator.py                    — loại sample lỗi (path tuyệt đối, thiếu field...)
        │
        ▼
metadata/writer.py               — ghi phase1_combined.jsonl (schema 6 field)
        │
        ▼
downloader/cache_manager.py      — cache-miss thì tải (qua url_index)
        │
        ▼
unified_dataset.py               — __getitem__() -> (frames, caption)      ← #48 dừng ở đây
        │
        ▼
prompt_ensemble_wrapper.py       — __getitem__() -> {video_frames,          ← #49
                                     caption_event, caption_content}
        │
        ▼
build_phase1_loader()            — DataLoader -> batch cho training loop (#47)
```

---

## `training/data/` — theo thư mục con

### Gốc

| File | Vai trò |
|---|---|
| `schema.py` | `TrainingSample` — 6-field contract của #48: `data_path, caption, media_type, num_frames, duration, source`. |
| `validator.py` | Kiểm tra 1 `TrainingSample` hợp lệ (path phải tương đối, `media_type`/`source` đúng whitelist, `num_frames > 0`...). |
| `unified_dataset.py` | `UnifiedDataset` — đọc `phase1_combined.jsonl`, gọi cache manager, decode ảnh/video, trả `(frames, caption)`. `.T` là attribute mutable điều khiển curriculum (`dataset.T = 4`). |
| `prompt_ensemble_wrapper.py` | `PromptEnsembleDataset` (#49) — bọc `UnifiedDataset`, sinh `caption_event`/`caption_content` từ 2 prompt cố định, `video_frames` pass-through nguyên vẹn. `build_phase1_loader()` là entrypoint DataLoader duy nhất. |
| `curriculum.py` | Đọc `configs/training.yaml` — số frame mỗi curriculum stage. |

### `adapters/` — parse từng nguồn raw thành `TrainingSample`

| File | Vai trò |
|---|---|
| `cc3m.py` | Đọc stream CC3M (`pixparse/cc3m-wds`) -> `TrainingSample` (ảnh). |
| `panda70m.py` | Đọc metadata JSONL đã tiền xử lý (do `downloader/prepare_panda70m.py` sinh ra) -> `TrainingSample` (video). Thay `yfcc100m.py` cũ — xem mục "Ghi chú" bên dưới. |
| `registry.py` | Map tên `source` (string trong `dataset.yaml`) -> hàm adapter tương ứng; dùng chung bởi `metadata/writer.py` và `downloader/cache_manager.py` nên không thể lệch nhau. |

### `downloader/` — tải & cache

| File | Vai trò |
|---|---|
| `downloader.py` | `download_file()` — HTTP GET có retry/backoff/ghi file atomic, dùng cho nguồn có URL tải trực tiếp (CC3M). |
| `prepare_panda70m.py` | Script CLI độc lập (không phải adapter): tải video gốc YouTube (`yt-dlp`) + cắt sub-clip theo đúng timestamp CSV (`ffmpeg`) + đo `num_frames`/`duration` thật (`cv2`) -> ghi metadata JSONL cho `adapters/panda70m.py` đọc. Chạy trực tiếp: `python -m training.data.downloader.prepare_panda70m --limit N --data-root ...`. |
| `cache_manager.py` | `ensure_cached()` — nếu file chưa có ở local thì tra URL qua `url_index` rồi tải. `build_url_indexes()` build index này từ adapters trước khi training bắt đầu. |
| `url_index.py` | SQLite key-value đơn giản: `data_path -> url`, dùng cho bước cache-miss ở trên. |

### `loaders/` — decode file thành tensor

| File | Vai trò |
|---|---|
| `image_loader.py` | Decode 1 ảnh -> tensor `(3, H, W)`. |
| `video_loader.py` | Decode video, sample đều `T` frame -> tensor `(T, 3, H, W)`. |

### `metadata/` — đọc/ghi manifest

| File | Vai trò |
|---|---|
| `writer.py` | `write_train_jsonl()` — chạy adapter -> validator -> ghi `phase1_combined.jsonl`. |
| `reader.py` | `read_train_jsonl()` — đọc lại file đó thành `list[TrainingSample]`. |

### `transforms/`, `utils/`

| File | Vai trò |
|---|---|
| `transforms/resize.py` | Resize tensor về đúng kích thước ảnh model (đọc từ `configs/model.yaml`). |
| `utils/config.py` | Đọc `configs/dataset.yaml` -> `DatasetConfig` (data_root, cache_dir, sources, train_jsonl). |
| `utils/logger.py` | Logger dùng chung cho toàn bộ pipeline. |

---

## Config liên quan (`configs/`)

| File | Vai trò |
|---|---|
| `dataset.yaml` | Danh sách `sources` (`cc3m`, `panda70m`), thư mục cache, tên file output `phase1_combined.jsonl`. |
| `training.yaml` | Curriculum stage -> số frame mỗi clip. |
| `model.yaml` | Kích thước ảnh input model (dùng bởi `transforms/resize.py`). |

---

## Test

- `tests/unit/` — 1 file test cho mỗi module production ở trên (`test_<tên module>.py`), không cần mạng, dùng fixture giả lập.
- `tests/integration/live_phase1_pipeline.py` — **không phải pytest**, chạy dữ liệu THẬT qua mạng (tải ảnh CC3M + video Panda-70M thật), dùng để tự kiểm tra thủ công trước khi báo cáo cho L. Chạy trực tiếp: `python tests/integration/live_phase1_pipeline.py`.

---

## Ghi chú

- `yfcc100m.py` (adapter cũ cho YFCC-100M) đã bị xoá hẳn khỏi nhánh này — L
  reject vì caption chỉ là placeholder (nguồn caption chính thức duy nhất,
  Yahoo Webscope, đã ngừng hoạt động). Panda-70M thay thế hoàn toàn theo
  #48-addendum, giữ nguyên schema 6 field không đổi.
- `data_path` trong mọi `TrainingSample` luôn **tương đối** với `DATA_ROOT` —
  không bao giờ tuyệt đối — để `phase1_combined.jsonl` dùng được trên nhiều
  máy khác nhau (Máy 1 / Máy 2).
