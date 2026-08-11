import numpy as np
import torch

from indexing.segmentation.pipeline import segment_video
from indexing.segmentation.schema import SegmentMetadata, EMBEDDING_DIM, write_segments_jsonl, read_segments_jsonl
from tests.utils.dummy_components import (
    DummyXEncoder,
    DummyPredictor,
    make_two_scene_video,
    make_three_scene_video,
)


def _run_pipeline(video, video_id="dummy_video_001", distance_threshold=25.0):
    x_encoder = DummyXEncoder(vision_dim=8)
    predictor = DummyPredictor(vision_dim=8, shared_dim=EMBEDDING_DIM, seed=0)
    return segment_video(
        x_encoder, predictor, video_id, video,
        device="cpu", window_size=20, stride=5,
        distance_threshold=distance_threshold, batch_size=4,
    )


def test_segment_video_returns_valid_segment_metadata():
    video = make_two_scene_video(frames_per_scene=40)  # 80 frames
    segments = _run_pipeline(video)

    assert len(segments) >= 1
    for seg in segments:
        assert isinstance(seg, SegmentMetadata)
        assert seg.video_id == "dummy_video_001"
        assert seg.embedding.shape == (EMBEDDING_DIM,)
        assert seg.embedding.dtype == np.float32
        assert seg.level == "coarse"
        assert seg.start_frame < seg.end_frame
        assert seg.start_frame <= seg.midpoint_frame < seg.end_frame


def test_segment_video_covers_whole_video_with_no_gaps_or_overlaps():
    video = make_two_scene_video(frames_per_scene=40)  # 80 frames
    segments = _run_pipeline(video)

    assert segments[0].start_frame == 0
    assert segments[-1].end_frame == video.shape[0]
    for prev_seg, next_seg in zip(segments, segments[1:]):
        assert prev_seg.end_frame == next_seg.start_frame


def _segment_containing(segments, frame):
    for seg in segments:
        if seg.start_frame <= frame < seg.end_frame:
            return seg
    raise AssertionError(f"no segment contains frame {frame}")


def test_segment_video_finds_two_segments_for_a_two_scene_video():
    """
    With overlapping sliding windows (stride < window_size), a handful of
    windows straddle the hard cut between the two scenes and blend both
    scenes' content -- those windows legitimately form their own short
    "transition" segments (this mirrors what happens at a real hard cut,
    just exaggerated here since the dummy scenes are two flat, maximally
    different colors with zero within-scene noise). So we don't assert an
    exact segment count; instead we check the two dominant, pure-scene
    regions come out as their own segments that don't bleed into each
    other, and that the transition doesn't blow up into one segment per
    window.
    """
    video = make_two_scene_video(frames_per_scene=60)  # 120 frames, two visually distinct halves
    segments = _run_pipeline(video, distance_threshold=25.0)

    num_windows = len(list(range(0, 120 - 20 + 1, 5))) + 1   # matches make_window_starts's math
    assert len(segments) < num_windows   # not one segment per window

    scene_1_segment = _segment_containing(segments, frame=10)   # deep inside scene 1
    scene_2_segment = _segment_containing(segments, frame=110)  # deep inside scene 2
    assert scene_1_segment is not scene_2_segment
    assert scene_1_segment.start_frame == 0
    assert scene_2_segment.end_frame == 120
    # the pure-scene segments should each cover a large, contiguous majority
    # of their half of the video, not just a sliver next to the transition
    assert scene_1_segment.end_frame - scene_1_segment.start_frame >= 30
    assert scene_2_segment.end_frame - scene_2_segment.start_frame >= 30


def test_segment_video_does_not_merge_non_adjacent_similar_scenes():
    """
    End-to-end version of the temporal-connectivity regression test: scene 1
    and scene 3 look identical, but must never end up sharing a single
    segment (that would require a non-contiguous [start_frame, end_frame]
    range, which SegmentMetadata can't even represent) -- the frame deep in
    scene 1 and the frame deep in scene 3 must fall in two different,
    non-adjacent segments, with scene 2's segment(s) between them.
    """
    video = make_three_scene_video(frames_per_scene=50)  # 150 frames
    segments = _run_pipeline(video, distance_threshold=25.0)

    # every segment's start/end must be contiguous, no gaps or overlaps
    for prev_seg, next_seg in zip(segments, segments[1:]):
        assert prev_seg.end_frame == next_seg.start_frame

    scene_1_segment = _segment_containing(segments, frame=10)    # deep in scene 1 (dark)
    scene_3_segment = _segment_containing(segments, frame=140)   # deep in scene 3 (dark, same as scene 1)
    assert scene_1_segment is not scene_3_segment
    assert scene_1_segment.end_frame <= scene_3_segment.start_frame


def test_segment_video_reasonable_segment_count_not_one_per_window():
    """
    Sanity check against the "too many tiny segments" failure mode: a
    single, visually uniform video (one scene) should collapse to exactly
    one segment, not one segment per sliding window.
    """
    video = make_two_scene_video(frames_per_scene=40)
    video[:] = video[0]   # make it visually uniform throughout (one scene only)
    segments = _run_pipeline(video, distance_threshold=25.0)

    assert len(segments) == 1
    assert segments[0].start_frame == 0
    assert segments[0].end_frame == video.shape[0]


def test_segment_video_jsonl_roundtrip(tmp_path):
    video = make_two_scene_video(frames_per_scene=40)
    segments = _run_pipeline(video)

    out_path = tmp_path / "segments.jsonl"
    write_segments_jsonl(segments, out_path)
    loaded = read_segments_jsonl(out_path)

    assert len(loaded) == len(segments)
    for original, restored in zip(segments, loaded):
        assert original.video_id == restored.video_id
        assert original.start_frame == restored.start_frame
        assert original.end_frame == restored.end_frame
        assert original.midpoint_frame == restored.midpoint_frame
        assert original.level == restored.level
        np.testing.assert_allclose(original.embedding, restored.embedding, rtol=1e-5, atol=1e-6)
