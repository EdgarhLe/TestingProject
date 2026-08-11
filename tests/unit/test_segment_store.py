import numpy as np
import pytest

from indexing.indexes.segment_store import SegmentStore, EMBEDDING_DIM
from indexing.segmentation.schema import SegmentMetadata


def _dummy_record(video_id="video_001", start_frame=0, end_frame=10, midpoint_frame=5,
                   level="coarse", ocr_text=None, asr_text=None, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "video_id": video_id,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "midpoint_frame": midpoint_frame,
        "embedding": rng.standard_normal(EMBEDDING_DIM).astype(np.float32),
        "level": level,
        "ocr_text": ocr_text,
        "asr_text": asr_text,
    }


@pytest.fixture
def store(tmp_path):
    db_path = tmp_path / "segments.db"
    s = SegmentStore(db_path)
    yield s
    s.close()


# ------------------------------------------------------------------
# add / get
# ------------------------------------------------------------------
def test_add_returns_int_segment_id(store):
    segment_id = store.add(_dummy_record())
    assert isinstance(segment_id, int)


def test_add_then_get_round_trips_all_fields(store):
    record = _dummy_record(video_id="video_042", start_frame=100, end_frame=140, midpoint_frame=120)
    segment_id = store.add(record)

    fetched = store.get(segment_id)
    assert fetched["segment_id"] == segment_id
    assert fetched["video_id"] == "video_042"
    assert fetched["start_frame"] == 100
    assert fetched["end_frame"] == 140
    assert fetched["midpoint_frame"] == 120
    assert fetched["level"] == "coarse"
    assert fetched["ocr_text"] is None
    assert fetched["asr_text"] is None
    np.testing.assert_array_equal(fetched["embedding"], record["embedding"])
    assert fetched["embedding"].dtype == np.float32


def test_get_missing_id_returns_none(store):
    assert store.get(999999) is None


def test_segment_ids_are_unique_and_monotonically_increasing(store):
    ids = [store.add(_dummy_record(video_id=f"video_{i}", seed=i)) for i in range(20)]
    assert len(set(ids)) == len(ids)
    assert ids == sorted(ids)


def test_add_accepts_segment_metadata_instance(store):
    """Integration with #68's output type directly, not just a dict."""
    seg = SegmentMetadata(
        video_id="video_from_pipeline",
        start_frame=0,
        end_frame=20,
        midpoint_frame=10,
        embedding=np.ones(EMBEDDING_DIM, dtype=np.float32),
        level="coarse",
    )
    segment_id = store.add(seg)
    fetched = store.get(segment_id)
    assert fetched["video_id"] == "video_from_pipeline"
    np.testing.assert_array_equal(fetched["embedding"], seg.embedding)
    assert fetched["ocr_text"] is None and fetched["asr_text"] is None


def test_add_rejects_wrong_embedding_shape(store):
    record = _dummy_record()
    record["embedding"] = np.zeros(128, dtype=np.float32)
    with pytest.raises(ValueError):
        store.add(record)


def test_add_rejects_invalid_level(store):
    with pytest.raises(ValueError):
        store.add(_dummy_record(level="medium"))


def test_add_rejects_missing_required_field(store):
    record = _dummy_record()
    del record["video_id"]
    with pytest.raises(ValueError):
        store.add(record)


def test_add_rejects_invalid_frame_ordering(store):
    with pytest.raises(Exception):   # sqlite3.IntegrityError from the CHECK constraint
        store.add(_dummy_record(start_frame=10, end_frame=5, midpoint_frame=7))


# ------------------------------------------------------------------
# get_by_video
# ------------------------------------------------------------------
def test_get_by_video_returns_only_that_videos_segments_in_order(store):
    store.add(_dummy_record(video_id="video_a", start_frame=20, end_frame=30, midpoint_frame=25, seed=1))
    store.add(_dummy_record(video_id="video_a", start_frame=0, end_frame=20, midpoint_frame=10, seed=2))
    store.add(_dummy_record(video_id="video_b", start_frame=0, end_frame=10, midpoint_frame=5, seed=3))

    results = store.get_by_video("video_a")
    assert len(results) == 2
    assert [r["start_frame"] for r in results] == [0, 20]   # ordered by start_frame
    assert all(r["video_id"] == "video_a" for r in results)


def test_get_by_video_with_no_segments_returns_empty_list(store):
    assert store.get_by_video("nonexistent_video") == []


# ------------------------------------------------------------------
# update_ocr / update_asr
# ------------------------------------------------------------------
def test_update_ocr_sets_text(store):
    segment_id = store.add(_dummy_record())
    store.update_ocr(segment_id, "some ocr text")
    assert store.get(segment_id)["ocr_text"] == "some ocr text"


def test_update_asr_sets_text(store):
    segment_id = store.add(_dummy_record())
    store.update_asr(segment_id, "some asr text")
    assert store.get(segment_id)["asr_text"] == "some asr text"


def test_update_ocr_and_asr_are_independent(store):
    segment_id = store.add(_dummy_record())
    store.update_ocr(segment_id, "ocr only")
    fetched = store.get(segment_id)
    assert fetched["ocr_text"] == "ocr only"
    assert fetched["asr_text"] is None

    store.update_asr(segment_id, "asr only")
    fetched = store.get(segment_id)
    assert fetched["ocr_text"] == "ocr only"   # unchanged by the asr update
    assert fetched["asr_text"] == "asr only"


def test_update_ocr_on_missing_id_raises(store):
    with pytest.raises(KeyError):
        store.update_ocr(999999, "text")


def test_update_asr_on_missing_id_raises(store):
    with pytest.raises(KeyError):
        store.update_asr(999999, "text")


def test_update_ocr_can_overwrite_previous_value(store):
    segment_id = store.add(_dummy_record())
    store.update_ocr(segment_id, "first")
    store.update_ocr(segment_id, "second")
    assert store.get(segment_id)["ocr_text"] == "second"


# ------------------------------------------------------------------
# Persistence / lifecycle
# ------------------------------------------------------------------
def test_data_persists_across_reopening_the_same_db_file(tmp_path):
    db_path = tmp_path / "segments.db"

    with SegmentStore(db_path) as store:
        segment_id = store.add(_dummy_record(video_id="persisted_video"))

    with SegmentStore(db_path) as reopened:
        fetched = reopened.get(segment_id)
        assert fetched is not None
        assert fetched["video_id"] == "persisted_video"


def test_len_reflects_number_of_inserted_segments(store):
    assert len(store) == 0
    for i in range(5):
        store.add(_dummy_record(video_id=f"video_{i}", seed=i))
    assert len(store) == 5


def test_context_manager_closes_connection(tmp_path):
    db_path = tmp_path / "segments.db"
    with SegmentStore(db_path) as store:
        store.add(_dummy_record())
    with pytest.raises(Exception):
        store._conn.execute("SELECT 1")   # connection should be closed after the `with` block
