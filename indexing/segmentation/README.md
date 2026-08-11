# indexing/segmentation

Bước 1 của indexing pipeline: cắt raw video thành các segment có ý nghĩa
ngữ nghĩa, output feed vào FAISS index builder (#70).

Không dùng Shot Boundary Detection. Segment boundary xuất hiện từ cấu trúc
ngữ nghĩa trong embedding stream: X-Encoder (frozen, #21) + Predictor (#44)
chạy qua video theo sliding window → agglomerative clustering (Ward
linkage) với temporal connectivity constraint.

## Files

| File | Vai trò |
|---|---|
| `sliding_window.py` | Bước 1: chạy X-Encoder + Predictor theo stride cố định → embedding stream, rồi denoise bằng local average pooling. |
| `clustering.py` | Bước 2: agglomerative clustering (`sklearn.cluster.AgglomerativeClustering`, `linkage="ward"`, `metric="euclidean"`) với temporal connectivity constraint (chain graph — chỉ merge window liền kề), cắt dendrogram bằng `distance_threshold`. |
| `schema.py` | `SegmentMetadata` dataclass đúng schema section 6.2: `{video_id, start_frame, end_frame, midpoint_frame, embedding: float32[1536], level}`, + đọc/ghi jsonl. |
| `pipeline.py` | `segment_video()` — nối 3 bước lại, output `list[SegmentMetadata]`. |
| `tests/` | Unit tests, dùng `DummyXEncoder`/`DummyPredictor` (không tải model thật, không cần mạng). |

## Vì sao dùng `sklearn.cluster.AgglomerativeClustering` thay vì `scipy.cluster.hierarchy` trực tiếp

`scipy.cluster.hierarchy.linkage()` không hỗ trợ connectivity constraint —
nó luôn coi mọi cặp điểm là có thể merge. Để enforce "chỉ merge window liền
kề", cần một connectivity graph, và `sklearn.cluster.AgglomerativeClustering`
hỗ trợ đúng việc này (tham số `connectivity`) trong khi vẫn dùng cùng tiêu
chí Ward/Euclidean mà spec yêu cầu. Đây là pattern chuẩn của scikit-learn
cho constrained clustering (tương tự cách dùng cho image segmentation với
pixel-adjacency graph).

## Query-free inference

Predictor được gọi ở chế độ "visual only" — `predictor(visual_embeds=...)`,
không có text query — giống hệt cách `training/phase1.py`'s
`run_validation()` / `run_validation_retrieval()` đã gọi. Điều này khớp với
Phase 1 training (`query_conditioned: false`): checkpoint được train để tạo
`S_hat_Y` có ý nghĩa từ visual_embeds một mình, nên đây là code path đúng
để dùng ở index-build time.

## Segment boundary placement

Vì sliding window overlap (stride ~5-10 frame, window_size lớn hơn nhiều),
không thể tính `end_frame` của segment từ "window cuối cùng kết thúc ở
đâu" — điều đó có thể tạo ra các segment chồng lấn. Thay vào đó, boundary
giữa 2 segment được đặt tại `start_frame` của window đầu tiên thuộc segment
tiếp theo — tức là điểm mà nhãn cluster đổi khi quét qua window theo thời
gian. Cách này đảm bảo segment không có khoảng trống, không chồng lấn, và
phủ đúng `[0, num_frames)`.

## Chạy trên real videos — CHƯA làm ở Tuần 4

Model đang pretraining ở Máy 1; checkpoint đầu tiên chỉ sync về ở Tuần 5
(`training/phase1.py`'s `sync_checkpoints_to_machine2()`). Chạy pipeline
này với checkpoint chưa train sẽ tạo ra vector chất lượng ngẫu nhiên, phải
rebuild lại toàn bộ. Tuần này chỉ implement + test bằng dummy
video/tensor — xem `tests/dummy_components.py`.

Khi có checkpoint thật (Tuần 5+), dùng thật:

```python
from model.x_encoder.vjepa2 import XEncoder
from model.predictor.gwen2_5_1_5b import build_model as build_predictor
from indexing.segmentation.pipeline import segment_video

x_encoder = XEncoder(device="cuda")   # frozen, #21
predictor, _ = build_predictor(device="cuda", vision_dim=x_encoder.vision_dim)  # #44
# load trained predictor weights from the synced deploy/ checkpoint here

video_frames = x_encoder.load_video_frames("path/to/video.mp4")
segments = segment_video(x_encoder, predictor, video_id="video_001", video_frames=video_frames, device="cuda")
```

`clustering.DEFAULT_WARD_DISTANCE_THRESHOLD` (hiện là placeholder `15.0`)
cần tune lại dựa trên phân bố khoảng cách thật của embedding từ checkpoint
đã train — giá trị này phụ thuộc hoàn toàn vào scale của `S_hat_Y` thật,
chưa biết được từ một checkpoint chưa train.

## Test trên 3-5 video thật (Tuần 5, sau khi có checkpoint)

Chạy `segment_video()` trên video ngắn (~2-5 phút), sau đó:
- Visualize segment boundary trên timeline / xuất `midpoint_frame` làm
  keyframe để check bằng mắt.
- Check boundary rơi vào chỗ đổi cảnh/nội dung, không phải giữa câu nói.
- Check số segment/video hợp lý (không quá nhiều mảnh vụn, không quá ít).
- Nếu quá nhiều mảnh vụn quanh transition (giống hiện tượng thấy trong
  `tests/test_pipeline.py`'s hard-cut dummy tests), cân nhắc tăng
  `denoise_kernel` hoặc `distance_threshold`.

## Chạy test

```bash
pip install torch scipy scikit-learn numpy pytest
pytest indexing/segmentation/tests/ -v
```
