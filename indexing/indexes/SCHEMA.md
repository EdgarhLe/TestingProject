# Segment Metadata Store — Schema (for DE)

**File:** `indexing/indexes/segment_store.py`
**Storage:** SQLite, single file (e.g. `segments.db`), table `segments`.
**Purpose:** this is the lookup table behind the FAISS index (#70). FAISS only returns an integer vector ID — this table maps that ID back to real video/frame info, and is where you attach OCR and ASR text.

## Table: `segments`

| Column | Type | Notes |
|---|---|---|
| `segment_id` | `INTEGER PRIMARY KEY AUTOINCREMENT` | **This is the FAISS vector ID.** Unique, stable, never reused (even if a row is ever deleted) — safe to use as the join key everywhere. |
| `video_id` | `TEXT NOT NULL` | Matches the video ID used elsewhere in the pipeline. |
| `start_frame` | `INTEGER NOT NULL` | Segment start, inclusive. |
| `end_frame` | `INTEGER NOT NULL` | Segment end, exclusive. |
| `midpoint_frame` | `INTEGER NOT NULL` | Keyframe used for KIS/Q&A retrieval display. |
| `embedding` | `BLOB NOT NULL` | `float32[1536]`, packed via `np.ndarray.tobytes()`. Not something you need to touch — decoded automatically by `SegmentStore.get()` / `get_by_video()`. |
| `ocr_text` | `TEXT`, nullable | **You fill this in** via `update_ocr()`. `NULL` until then. |
| `asr_text` | `TEXT`, nullable | **You fill this in** via `update_asr()`. `NULL` until then. |
| `level` | `TEXT NOT NULL` | `'coarse'` or `'fine'`. Today, every row is `'coarse'` (single-pass segmentation). |

Constraints enforced by the DB itself (inserts/updates that violate these will raise, not silently corrupt data):
- `start_frame < end_frame`
- `start_frame <= midpoint_frame < end_frame`
- `level IN ('coarse', 'fine')`
- Index on `video_id`, so per-video lookups are fast.

## How to align OCR/ASR text to a segment

You do **not** need to compute or choose `segment_id` yourself — it's assigned by the indexer when a segment is inserted (Week 5, when the pipeline runs on real videos). Your job is: given a `video_id` you're processing, find its segments, run OCR/ASR on the frame range each segment covers, and write the result back using that segment's `segment_id`.

```python
from indexing.indexes.segment_store import SegmentStore

with SegmentStore("segments.db") as store:
    # 1. Get every segment belonging to the video you're processing,
    #    already ordered by start_frame.
    segments = store.get_by_video("video_001")

    for seg in segments:
        # seg is a dict: segment_id, video_id, start_frame, end_frame,
        # midpoint_frame, embedding, ocr_text, asr_text, level

        # 2. Run your OCR pipeline over frames [seg["start_frame"], seg["end_frame"]),
        #    e.g. sampling seg["midpoint_frame"] as the representative frame for OCR.
        ocr_result = run_ocr(video_id="video_001", start_frame=seg["start_frame"], end_frame=seg["end_frame"])

        # 3. Run your ASR pipeline over the same frame range (converted to a time range).
        asr_result = run_asr(video_id="video_001", start_frame=seg["start_frame"], end_frame=seg["end_frame"])

        # 4. Write back using segment_id -- this is the ONLY key you need.
        store.update_ocr(seg["segment_id"], ocr_result)
        store.update_asr(seg["segment_id"], asr_result)
```

Notes:
- `update_ocr()` / `update_asr()` raise `KeyError` if you pass a `segment_id` that doesn't exist — treat that as a bug (e.g. stale ID from a rebuilt index), not something to silently swallow.
- `update_ocr()` and `update_asr()` are independent — updating one never touches the other.
- If you need to re-run OCR/ASR later (e.g. a better model), just call `update_ocr`/`update_asr` again with the new `segment_id` — it overwrites the previous value.
- If you're iterating by frame range instead of by video (e.g. a batch OCR job scanning a whole dataset), use `get_by_video(video_id)` per video rather than scanning the whole table — it's index-backed and returns segments already sorted by `start_frame`.

## Direct SQL (if you need something `SegmentStore` doesn't expose)

The DB is a plain SQLite file — you can also query it directly if needed, e.g.:

```sql
-- Segments still missing OCR for a given video
SELECT segment_id, start_frame, end_frame
FROM segments
WHERE video_id = 'video_001' AND ocr_text IS NULL
ORDER BY start_frame;
```

If you write to the table outside of `SegmentStore` (raw SQL), please still go through `update_ocr`/`update_asr` where possible — the Python layer is the place we'll add validation/logging later, and keeps everyone using the same code path.

## Status this week

The store is implemented and unit-tested (dummy data only) — **no real segments are inserted yet**. It gets populated when the indexer (#68) runs on real videos in Week 5. This doc is meant to let DE build/test their OCR/ASR alignment logic against the schema now, ahead of that.
