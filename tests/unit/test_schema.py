import numpy as np
import pytest

from indexing.segmentation.schema import SegmentMetadata, EMBEDDING_DIM


def _valid_kwargs(**overrides):
    kwargs = dict(
        video_id="video_001",
        start_frame=0,
        end_frame=10,
        midpoint_frame=5,
        embedding=np.zeros(EMBEDDING_DIM, dtype=np.float32),
        level="coarse",
    )
    kwargs.update(overrides)
    return kwargs


def test_segment_metadata_accepts_valid_input():
    seg = SegmentMetadata(**_valid_kwargs())
    assert seg.embedding.shape == (EMBEDDING_DIM,)
    assert seg.embedding.dtype == np.float32


def test_segment_metadata_casts_list_embedding_to_float32_array():
    seg = SegmentMetadata(**_valid_kwargs(embedding=[0.0] * EMBEDDING_DIM))
    assert isinstance(seg.embedding, np.ndarray)
    assert seg.embedding.dtype == np.float32


def test_segment_metadata_rejects_wrong_embedding_dim():
    with pytest.raises(ValueError):
        SegmentMetadata(**_valid_kwargs(embedding=np.zeros(128, dtype=np.float32)))


def test_segment_metadata_rejects_start_frame_not_less_than_end_frame():
    with pytest.raises(ValueError):
        SegmentMetadata(**_valid_kwargs(start_frame=10, end_frame=10))
    with pytest.raises(ValueError):
        SegmentMetadata(**_valid_kwargs(start_frame=11, end_frame=10))


def test_segment_metadata_rejects_midpoint_outside_range():
    with pytest.raises(ValueError):
        SegmentMetadata(**_valid_kwargs(start_frame=0, end_frame=10, midpoint_frame=10))
    with pytest.raises(ValueError):
        SegmentMetadata(**_valid_kwargs(start_frame=0, end_frame=10, midpoint_frame=-1))


def test_segment_metadata_json_roundtrip():
    seg = SegmentMetadata(**_valid_kwargs(embedding=np.arange(EMBEDDING_DIM, dtype=np.float32)))
    restored = SegmentMetadata.from_json_dict(seg.to_json_dict())
    assert restored.video_id == seg.video_id
    np.testing.assert_allclose(restored.embedding, seg.embedding)
