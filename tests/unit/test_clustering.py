import numpy as np
import pytest
import torch

from indexing.segmentation.clustering import (
    build_temporal_connectivity,
    cluster_embedding_stream,
    group_into_segments,
)


def test_build_temporal_connectivity_only_links_adjacent_windows():
    conn = build_temporal_connectivity(5)
    dense = conn.toarray()
    expected = np.array([
        [0, 1, 0, 0, 0],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [0, 0, 0, 1, 0],
    ])
    assert np.array_equal(dense, expected)


def test_build_temporal_connectivity_single_window_returns_none():
    assert build_temporal_connectivity(1) is None
    assert build_temporal_connectivity(0) is None


def test_cluster_embedding_stream_two_well_separated_blocks():
    block_a = torch.zeros(10, 16)
    block_b = torch.full((10, 16), 1000.0)
    embeddings = torch.cat([block_a, block_b], dim=0)

    labels = cluster_embedding_stream(embeddings, distance_threshold=10.0)

    assert len(set(labels[:10].tolist())) == 1
    assert len(set(labels[10:].tolist())) == 1
    assert labels[0] != labels[-1]


def test_cluster_embedding_stream_single_window():
    embeddings = torch.randn(1, 16)
    labels = cluster_embedding_stream(embeddings)
    assert list(labels) == [0]


def test_cluster_embedding_stream_does_not_merge_distant_similar_blocks():
    """
    Regression test for the temporal connectivity constraint: scene 1 and
    scene 3 have IDENTICAL embeddings but are separated in time by scene 2 --
    a correct temporally-constrained clustering must never assign them the
    same label, even though an unconstrained Ward clustering would (they are
    literally the closest pair of points in the whole set).
    """
    scene_1 = torch.zeros(8, 16)
    scene_2 = torch.full((8, 16), 500.0)
    scene_3 = torch.zeros(8, 16)   # identical to scene_1
    embeddings = torch.cat([scene_1, scene_2, scene_3], dim=0)

    labels = cluster_embedding_stream(embeddings, distance_threshold=10.0)

    label_scene_1 = labels[0]
    label_scene_3 = labels[-1]
    assert label_scene_1 != label_scene_3, (
        "scene 1 and scene 3 were merged despite being non-adjacent in time -- "
        "temporal connectivity constraint was not enforced"
    )


def test_group_into_segments_covers_video_with_no_gaps_or_overlaps():
    # 3 windows -> labels split as [0, 0, 1]
    labels = np.array([0, 0, 1])
    window_starts = [0, 5, 10]
    num_frames = 18

    segments = group_into_segments(labels, window_starts, num_frames)

    assert len(segments) == 2
    assert segments[0]["start_frame"] == 0
    assert segments[0]["end_frame"] == segments[1]["start_frame"]   # no gap
    assert segments[-1]["end_frame"] == num_frames                  # covers to the end
    assert segments[0]["window_idx_start"] == 0 and segments[0]["window_idx_end"] == 1
    assert segments[1]["window_idx_start"] == 2 and segments[1]["window_idx_end"] == 2


def test_group_into_segments_raises_on_non_contiguous_label_reappearance():
    # Simulates what would happen if the temporal connectivity constraint
    # were NOT enforced upstream: label 0 reappears after label 1.
    labels = np.array([0, 0, 1, 1, 0])
    window_starts = [0, 5, 10, 15, 20]
    num_frames = 28

    with pytest.raises(ValueError):
        group_into_segments(labels, window_starts, num_frames)


def test_group_into_segments_single_segment():
    labels = np.array([0, 0, 0])
    window_starts = [0, 5, 10]
    num_frames = 18

    segments = group_into_segments(labels, window_starts, num_frames)
    assert len(segments) == 1
    assert segments[0]["start_frame"] == 0
    assert segments[0]["end_frame"] == 18
