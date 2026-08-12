# indexing/indexes

Two pieces live here:

| File | Role |
|---|---|
| `segment_store.py` | SQLite metadata lookup table (`segment_id` → video/frame/OCR/ASR info). See `SCHEMA.md` for the doc shared with DE. |
| `faiss_index.py` | `SegmentFaissIndex` — FAISS HNSW wrapper that produces `coarse.faiss` / `fine.faiss`. |

The builder script that ties `#68` (segmentation) + this module + `segment_store.py` together lives at `scripts/build_index.py`, with a companion `scripts/benchmark_search.py`.

## `SegmentFaissIndex`

- **Index type:** `faiss.IndexHNSWFlat(1536, M=32)`, per system design. This is the *un-quantized* HNSW variant (full float32 vectors) — the highest-accuracy option FAISS offers for HNSW, strictly above the scalar-quantized `HNSW+SQ` variant.
- **Metric:** inner product on L2-normalized vectors (cosine similarity), not raw L2 — see "Why cosine similarity" below.
- **Custom IDs:** `segment_id` (from `SegmentStore.add()`) is the FAISS vector ID. Rather than using `faiss.IndexIDMap2`, this class keeps its own simple `id_map` list (row *i* → `segment_id` added at row *i*) — see the module docstring for why: it sidesteps a GPU/cuVS-index-conversion edge case (wrapping an already-populated converted index in `IndexIDMap2`) that couldn't be verified without real GPU hardware, at the cost of a few extra lines of very easy-to-test code.
- **Two separate index files, not one:** `coarse.faiss` and `fine.faiss` are built as two independent `SegmentFaissIndex` instances (independent `segment_id` spaces, independent `SegmentStore` databases by default) — one per sliding-window granularity from `#68`.

### Is HNSWFlat accurate enough? (measure, don't assume)

FAISS's own published SIFT1M benchmark (`facebookresearch/faiss` wiki, "Indexing 1M vectors") shows HNSWFlat recall@1 as a direct, tunable function of `efSearch`:

| efSearch | R@1 |
|---|---|
| 16 | 0.874 |
| 32 | 0.949 |
| **64 (this class's default)** | **0.978** |
| 128 | 0.989 |
| 256 | 0.992 |

That's on 128-dim SIFT vectors, not our 1536-dim VL-JEPA embeddings — recall is sensitive to a dataset's *intrinsic* dimensionality, not just ambient dimension, and that's unknown for real VL-JEPA output until Week 5. **Don't take the SIFT1M number as a promise for this system.** Measure it directly instead:

```python
from indexing.indexes.faiss_index import recall_at_k
recall_at_k(my_index, vectors, ids, k=10)   # vs. an exact IndexFlat baseline, same metric
```

`tests/test_faiss_index.py` includes a recall@10 benchmark (`test_recall_at_k_benchmark_1000_vectors`) and `scripts/benchmark_search.py` reports `recall_at_k` by default alongside latency — **on random dummy vectors this reads as ~1.0, which is not representative of real embedding recall** (random Gaussian data has none of the cluster structure that makes ANN search actually hard). Re-run both against real embeddings in Week 5 for a number that means something, and drop `efSearch`/`M` up if it's lower than expected — that trade-off is exactly what the table above shows.

### GPU / cuVS acceleration — read before running the real build

`use_gpu="auto"` (the default) uses NVIDIA cuVS's `GpuIndexCagra` to build the graph on GPU, then converts it to a CPU-searchable `IndexHNSWCagra` — this is the actual answer to "advanced faiss usage" here: plain `faiss-cpu` has **no GPU path for HNSW at all** (HNSW's graph traversal has no classic-GPU-faiss equivalent), so cuVS's CAGRA is the mechanism that makes GPU-accelerated *building* possible, while still producing a normal, portable, GPU-free-at-search-time index file. This directly targets the spec's "build sẽ mất nhiều giờ khi chạy full dataset" concern.

**This was written from the official docs, not from memory.** The first version of this code was written without consulting the real faiss/cuVS docs and had two real bugs, found by actually fetching [the official wiki](https://github.com/facebookresearch/faiss/wiki/GPU-Faiss-with-cuVS-usage):
- `GpuIndexCagraConfig.use_cuvs` was never set to `True` — the docs are explicit this flag gates whether cuVS is actually dispatched.
- The build call was `.add(vectors)`; the official recipe uses `.train(n, xb)`, then explicit `IndexHNSWCagra` construction + `.copyTo()` (not a bare `index_gpu_to_cpu()`).

`_build_gpu_cagra_index` now tries the official recipe first (`.train()` → `IndexHNSWCagra` + `.copyTo()`, `base_level_only=True` since this class never adds vectors after a build), and falls back to the `.add()`/`index_gpu_to_cpu()` variant (seen in a newer third-party source, possibly a convenience path added in a later faiss version) if that raises. **Still unverified against real hardware** — this dev environment only has plain `faiss-cpu`. Please smoke-test on the real machine (the one with `faiss-gpu-cuvs` in `conda list`) before relying on it for the Week 5 build:

```python
from indexing.indexes.faiss_index import SegmentFaissIndex
import numpy as np

idx = SegmentFaissIndex(use_gpu=True)   # raises immediately if cuVS isn't actually available here
idx.add(np.arange(1000), np.random.randn(1000, 1536).astype("float32"))
idx.build()
print(idx.backend, len(idx))            # should print "gpu_cagra" (if the GPU path succeeded) and 1000
```

If that raises or silently falls back to `cpu_hnsw`, check the exception logged by `build()` (it now logs which of the two GPU recipes failed) — the CPU path (fully tested in this repo) will still work either way, just without the GPU build speed-up.

Everything else — `add()`, `search()`, `save()`/`load()`, `recall_at_k()`, and all the correctness tests — is identical code regardless of which backend actually built the index, and is fully tested against the CPU backend in `tests/test_faiss_index.py`.

### Why cosine similarity, precisely

The system design's training objective is **bidirectional InfoNCE with uniformity regularization** ("alignment + uniformity regularization; prevents representation collapse") — this is *not* CLIP's loss (CLIP: two independent encoders, symmetric InfoNCE, no uniformity term; here: a Predictor conditioned on visual+query trained against a separate Y-Encoder target, plus an explicit uniformity term). The uniformity term (Wang & Isola-style alignment/uniformity) is defined **on the unit hypersphere** — training directly optimizes normalized embeddings for good angular separation. Separately, `model/vl_jepa.py`'s own logit computation is `scale * (P_norm @ T_norm^T)` — normalized dot product with a learnable scale (its comment calls this "like CLIP", describing just that computation, not the whole objective). Both point at the same conclusion: retrieval should use cosine similarity on normalized embeddings, not raw L2 — which is what `normalize=True` (default) does automatically.


## `scripts/build_index.py`

```bash
python scripts/build_index.py \
  --level coarse \
  --checkpoint path/to/checkpoint.deploy.pt \
  --videos path/to/video_list.txt \
  --output $INDEX_ROOT/coarse.faiss
```

Run once per `--level` (`coarse` / `fine`) — each run produces its own `.faiss` file and its own `SegmentStore` database (default: `<output>` with the extension swapped for `.segments.db`, override with `--segment-store`).

What it does, per video in `--videos`:
1. Decode + run `#68`'s `segment_video()` (query-free X-Encoder + Predictor from `--checkpoint`).
2. `SegmentStore.add()` each resulting segment → get back its `segment_id`.
3. Buffer `(segment_id, embedding)` and periodically `SegmentFaissIndex.add()` them (`--flush-every`, default 2000).
4. On completion: `build()` + `save()` the FAISS index.

A video that fails segmentation is logged (with traceback) and skipped — one bad video doesn't abort an hours-long full-dataset run. Progress is shown via `tqdm` (`--no-progress-bar` to disable) and optionally mirrored to `--log-file`.

**`--level`-specific sliding-window defaults are frame-based placeholders**, not the literal 5–10s / 0.5–1s from system design — `#68`'s `sliding_window.py` works in frames, and converting seconds→frames needs a real fps assumption we don't have yet (`ASSUMED_FPS = 25` in the script, clearly marked as a placeholder). Override with `--window-size`/`--stride` once real video fps/content is known in Week 5.

**Do not run this against real videos yet** — the checkpoint it needs doesn't exist (Phase 1 is still training). This week it's implemented and covered by smoke tests (`scripts/tests/test_build_index.py`) that check CLI parsing, video-list loading, and level defaults without needing a real checkpoint, GPU, or network access.

## `scripts/benchmark_search.py`

```bash
python scripts/benchmark_search.py --num-vectors 1000 --output benchmarks/bench_1000_vectors_cpu.json
```

Builds a `SegmentFaissIndex` from N random vectors and reports build time + search latency (single-query and batched). Rerun with a larger `--num-vectors` (e.g. 100,000 / 1,000,000) once bigger dummy runs or real data are available, to see how latency actually scales toward the real few-million-segment target, rather than guessing from the N=1000 number alone.

**This week's baseline** (`benchmarks/bench_1000_vectors_cpu.json`, CPU `IndexHNSWFlat`, N=1000, k=5, default M=32/efSearch=64):

| Metric | Value |
|---|---|
| Build time | ~0.4–0.5 s |
| Single-query avg latency | ~0.4–0.5 ms |
| Batched (100 queries) avg latency | ~0.3–0.4 ms/query |
| Recall@k vs. exact search | ~1.0 on random dummy vectors — **not representative of real embeddings, see the recall caveat above** |

(Small run-to-run variance is expected — these are from a shared sandbox CPU, not the real serving hardware. Re-run on the actual machine for numbers that matter.)

## Tests

```bash
pytest indexing/indexes/tests/ scripts/tests/ -v
```

- `test_faiss_index.py` (14 tests) — covers the three dummy-data checks from the spec plus edge cases:
  1. **1,000 random vectors, self-search returns correct IDs** (`test_self_search_recovers_correct_ids_cpu_backend`), including a variant with sparse/non-sequential IDs to stress the custom id-map logic.
  2. **Roundtrip: add segment → `SegmentStore` → FAISS search → metadata lookup matches** (`test_faiss_and_segment_store_roundtrip`, plus an all-segments variant).
  3. **Search latency benchmark at N=1000** (`test_search_latency_benchmark_1000_vectors`), printed and softly asserted (not a strict perf gate).
  4. **Recall@10 benchmark at N=1000** (`test_recall_at_k_benchmark_1000_vectors`) against an exact brute-force baseline, plus a monotonicity check (`test_recall_at_k_improves_with_higher_ef_search`) that recall doesn't decrease as `efSearch` increases — a sanity check on the measurement itself, not just the index.
  - Plus: incremental multi-batch `add()`, cosine-similarity scale invariance, save/load round trip, empty-index and shape-mismatch error handling, and `use_gpu=True` correctly raising in this cuVS-less environment.
- `test_segment_store.py` (20 tests) — from `#69`.
- `scripts/tests/` (6 tests) — CLI/config smoke tests for both scripts, no real checkpoint/GPU needed.

All 71 tests (across `indexing/` + `scripts/`) pass.
