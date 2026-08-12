import time

import numpy as np
import pytest

from indexing.indexes.faiss_index import SegmentFaissIndex, EMBEDDING_DIM, recall_at_k
from indexing.indexes.segment_store import SegmentStore


def _random_vectors(n, seed=0, dim=EMBEDDING_DIM):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).astype(np.float32)


# ------------------------------------------------------------------
# Basic correctness: 1,000 random vectors, self-search recovers the
# correct IDs (spec's dummy-data test #1)
# ------------------------------------------------------------------
def test_self_search_recovers_correct_ids_cpu_backend():
    n = 1000
    ids = np.arange(n, dtype=np.int64)
    vectors = _random_vectors(n, seed=0)

    index = SegmentFaissIndex(use_gpu=False)
    index.add(ids, vectors)
    index.build()
    assert len(index) == n

    distances, result_ids = index.search(vectors, k=5)

    assert distances.shape == (n, 5)
    assert result_ids.shape == (n, 5)
    # every vector's own (normalized) self is its own nearest neighbor
    assert np.array_equal(result_ids[:, 0], ids)


def test_search_with_custom_nonsequential_ids():
    """
    segment_id values coming from SegmentStore are AUTOINCREMENT primary
    keys -- they won't generally be 0..N-1. Use large, non-contiguous IDs
    here to make sure the id-translation logic (self._segment_ids) isn't
    accidentally relying on ids matching row position.
    """
    n = 200
    ids = np.arange(500000, 500000 + n, dtype=np.int64) * 7   # sparse, offset, non-trivial ids
    vectors = _random_vectors(n, seed=1)

    index = SegmentFaissIndex(use_gpu=False)
    index.add(ids, vectors)
    index.build()

    _, result_ids = index.search(vectors, k=1)
    assert np.array_equal(result_ids[:, 0], ids)


def test_add_can_be_called_incrementally_across_multiple_batches():
    """Mirrors the real build script's usage: add() called once per video."""
    index = SegmentFaissIndex(use_gpu=False)
    all_ids, all_vecs = [], []
    for batch in range(5):
        ids = np.arange(batch * 50, batch * 50 + 50, dtype=np.int64)
        vecs = _random_vectors(50, seed=batch)
        index.add(ids, vecs)
        all_ids.append(ids)
        all_vecs.append(vecs)

    index.build()
    assert len(index) == 250

    query = np.concatenate(all_vecs)
    _, result_ids = index.search(query, k=1)
    assert np.array_equal(result_ids[:, 0], np.concatenate(all_ids))


def test_add_rejects_mismatched_ids_and_vectors_length():
    index = SegmentFaissIndex(use_gpu=False)
    with pytest.raises(ValueError):
        index.add(np.arange(5), _random_vectors(3))


def test_search_on_empty_index_raises():
    index = SegmentFaissIndex(use_gpu=False)
    with pytest.raises(RuntimeError):
        index.search(_random_vectors(1), k=5)


def test_normalize_makes_cosine_similarity_scale_invariant():
    """A vector and a scaled copy of itself should retrieve identically
    under metric='ip' + normalize=True (cosine similarity), since scaling
    doesn't change direction."""
    n = 50
    ids = np.arange(n, dtype=np.int64)
    vectors = _random_vectors(n, seed=2)

    index = SegmentFaissIndex(use_gpu=False, metric="ip", normalize=True)
    index.add(ids, vectors)
    index.build()

    scaled_query = vectors * 37.0
    _, result_ids = index.search(scaled_query, k=1)
    assert np.array_equal(result_ids[:, 0], ids)


def test_use_gpu_true_raises_when_gpu_cuvs_unavailable():
    """This environment only has plain faiss-cpu -- forcing GPU must fail
    loudly rather than silently falling back."""
    with pytest.raises(RuntimeError):
        SegmentFaissIndex(use_gpu=True)


def test_use_gpu_auto_falls_back_to_cpu_when_unavailable():
    index = SegmentFaissIndex(use_gpu="auto")
    assert index.backend == "cpu_hnsw"


# ------------------------------------------------------------------
# Persistence
# ------------------------------------------------------------------
def test_save_and_load_round_trip(tmp_path):
    n = 300
    ids = np.arange(1000, 1000 + n, dtype=np.int64)
    vectors = _random_vectors(n, seed=3)

    index = SegmentFaissIndex(use_gpu=False)
    index.add(ids, vectors)
    index.build()

    out_path = tmp_path / "test.faiss"
    index.save(out_path)
    assert out_path.exists()
    assert (tmp_path / "test.faiss.ids.npy").exists()

    loaded = SegmentFaissIndex.load(out_path)
    assert len(loaded) == n

    _, result_ids = loaded.search(vectors, k=1)
    assert np.array_equal(result_ids[:, 0], ids)


# ------------------------------------------------------------------
# Roundtrip test: add dummy segment -> SegmentStore (#69) -> FAISS search
# -> lookup metadata -> must match the original segment (spec's dummy-data
# test #2)
# ------------------------------------------------------------------
def test_faiss_and_segment_store_roundtrip(tmp_path):
    n = 100
    vectors = _random_vectors(n, seed=4)

    faiss_index = SegmentFaissIndex(use_gpu=False)
    db_path = tmp_path / "segments.db"

    with SegmentStore(db_path) as store:
        segment_ids = []
        for i in range(n):
            record = {
                "video_id": f"video_{i % 10}",
                "start_frame": i * 20,
                "end_frame": i * 20 + 20,
                "midpoint_frame": i * 20 + 10,
                "embedding": vectors[i],
                "level": "coarse",
            }
            segment_id = store.add(record)
            segment_ids.append(segment_id)

        faiss_index.add(np.array(segment_ids, dtype=np.int64), vectors)
        faiss_index.build()

        # Pick one segment, search FAISS for its (exact) embedding, and
        # confirm the returned segment_id looks up to the SAME original
        # segment in the store -- this is the actual end-to-end path a
        # real query will take: FAISS gives an ID, the store resolves it.
        query_idx = 42
        query_vector = vectors[query_idx][None, :]
        expected_segment_id = segment_ids[query_idx]

        _, result_ids = faiss_index.search(query_vector, k=1)
        found_segment_id = int(result_ids[0, 0])
        assert found_segment_id == expected_segment_id

        looked_up = store.get(found_segment_id)
        assert looked_up is not None
        assert looked_up["video_id"] == f"video_{query_idx % 10}"
        assert looked_up["start_frame"] == query_idx * 20
        assert looked_up["end_frame"] == query_idx * 20 + 20
        assert looked_up["midpoint_frame"] == query_idx * 20 + 10
        np.testing.assert_allclose(looked_up["embedding"], vectors[query_idx], rtol=1e-5, atol=1e-6)


def test_faiss_and_segment_store_roundtrip_all_segments():
    """Same as above but checks every segment, not just one, to catch any
    off-by-one in the id-translation logic across the whole batch."""
    n = 500
    vectors = _random_vectors(n, seed=5)

    faiss_index = SegmentFaissIndex(use_gpu=False)
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        with SegmentStore(f"{tmp}/segments.db") as store:
            segment_ids = []
            for i in range(n):
                record = {
                    "video_id": "video_only",
                    "start_frame": i * 20,
                    "end_frame": i * 20 + 20,
                    "midpoint_frame": i * 20 + 10,
                    "embedding": vectors[i],
                    "level": "fine",
                }
                segment_ids.append(store.add(record))

            faiss_index.add(np.array(segment_ids, dtype=np.int64), vectors)
            faiss_index.build()

            _, result_ids = faiss_index.search(vectors, k=1)
            for i in range(n):
                found_segment_id = int(result_ids[i, 0])
                assert found_segment_id == segment_ids[i]
                looked_up = store.get(found_segment_id)
                assert looked_up["start_frame"] == i * 20


# ------------------------------------------------------------------
# Latency benchmark (spec's dummy-data test #3) -- soft assertion, mainly
# to establish and print a baseline number at N=1000 before scaling up.
# ------------------------------------------------------------------
def test_search_latency_benchmark_1000_vectors(capsys):
    n = 1000
    ids = np.arange(n, dtype=np.int64)
    vectors = _random_vectors(n, seed=6)

    index = SegmentFaissIndex(use_gpu=False)
    index.add(ids, vectors)
    index.build()

    num_queries = 100
    queries = vectors[:num_queries]

    index.search(queries[:1], k=5)   # warm-up, exclude first-call overhead from the measurement

    t0 = time.perf_counter()
    for i in range(num_queries):
        index.search(queries[i:i + 1], k=5)
    elapsed = time.perf_counter() - t0
    avg_latency_ms = (elapsed / num_queries) * 1000

    with capsys.disabled():
        print(f"\n[benchmark] N={n} vectors, backend=cpu_hnsw, "
              f"single-query avg latency: {avg_latency_ms:.3f} ms "
              f"({num_queries} queries, k=5)")

    # Generous soft bound -- this is a baseline measurement, not a strict
    # perf gate (real hardware/index size at Week 5 will differ a lot).
    # Mainly catches the search path being catastrophically broken (e.g.
    # accidentally doing a full rebuild per query).
    assert avg_latency_ms < 50.0


# ------------------------------------------------------------------
# Recall@k benchmark -- distinct from the latency benchmark above.
# Answers "how close to exact search is this actually getting", which
# latency alone says nothing about.
# ------------------------------------------------------------------
def test_recall_at_k_benchmark_1000_vectors(capsys):
    n = 1000
    ids = np.arange(n, dtype=np.int64)
    vectors = _random_vectors(n, seed=7)

    index = SegmentFaissIndex(use_gpu=False)   # default M=32, efConstruction=200, efSearch=64
    index.add(ids, vectors)
    index.build()

    recall = recall_at_k(index, vectors, ids, k=10, num_queries=200, seed=8)

    with capsys.disabled():
        print(f"\n[benchmark] N={n} vectors, backend=cpu_hnsw, M={index.M}, "
              f"efSearch={index.ef_search}, recall@10 vs. brute force: {recall:.4f}")

    # NOTE: this is measured on random Gaussian dummy vectors, which do NOT
    # have the cluster/intrinsic-dimensionality structure of real VL-JEPA
    # embeddings -- see recall_at_k's docstring. This asserts the recall
    # methodology works and default params aren't pathologically bad, not
    # that recall will be this good on real data. Re-measure on real
    # embeddings once available (Week 5+).
    assert 0.0 <= recall <= 1.0
    assert recall > 0.85


def test_recall_at_k_improves_with_higher_ef_search():
    """
    Sanity check on the recall_at_k utility itself: recall should be
    monotonically non-decreasing as efSearch increases (matching FAISS's
    own published SIFT1M behavior -- see recall_at_k's docstring) -- if this
    ever fails, something is wrong with the measurement, not just the index.
    """
    n = 800
    ids = np.arange(n, dtype=np.int64)
    vectors = _random_vectors(n, seed=9)

    low_ef = SegmentFaissIndex(use_gpu=False, ef_search=8)
    low_ef.add(ids, vectors)
    low_ef.build()
    recall_low = recall_at_k(low_ef, vectors, ids, k=10, num_queries=200, seed=10)

    high_ef = SegmentFaissIndex(use_gpu=False, ef_search=256)
    high_ef.add(ids, vectors)
    high_ef.build()
    recall_high = recall_at_k(high_ef, vectors, ids, k=10, num_queries=200, seed=10)

    assert recall_high >= recall_low
