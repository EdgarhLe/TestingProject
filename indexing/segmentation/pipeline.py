"""
indexing/segmentation/pipeline.py

Orchestrates steps 1-3 of the semantic segmentation pipeline (#69):
sliding-window embedding (sliding_window.py) -> denoise -> temporally
constrained clustering (clustering.py) -> SegmentMetadata records
(schema.py). This is the first stage of the indexing pipeline that feeds
#70's FAISS index.

WEEK 4 SCOPE -- DO NOT RUN ON REAL VIDEOS YET:
This module is implement-and-test-only this week. The Phase 1 checkpoint is
still training on Machine 1 (see training/phase1.py); the first checkpoint
doesn't sync to Machine 2 until Week 5
(training/phase1.py:sync_checkpoints_to_machine2()). Running segment_video()
against an untrained/undertrained checkpoint now would cluster embeddings
that are effectively noise, and every resulting segment boundary would have
to be thrown out and rebuilt once a real checkpoint lands -- wasted compute
for zero signal. tests/ exercises this module with dummy X-Encoder/Predictor
stand-ins (see tests/dummy_components.py) purely to check the pipeline's
shape/schema/contiguity logic, NOT to validate segmentation quality -- that
validation can only happen for real starting Week 5, against #21/#44's real
components (facebook/vjepa2-vitl-fpc64-256 + the trained Qwen2.5-1.5B
predictor).
"""

from indexing.segmentation.sliding_window import (
    DEFAULT_STRIDE,
    DEFAULT_WINDOW_SIZE,
    DEFAULT_DENOISE_KERNEL,
    compute_embedding_stream,
    denoise_embedding_stream,
)
from indexing.segmentation.clustering import (
    DEFAULT_WARD_DISTANCE_THRESHOLD,
    cluster_embedding_stream,
    group_into_segments,
)
from indexing.segmentation.schema import SegmentMetadata


def segment_video(x_encoder, predictor, video_id, video_frames, device="cuda",
                   window_size=DEFAULT_WINDOW_SIZE, stride=DEFAULT_STRIDE,
                   denoise_kernel=DEFAULT_DENOISE_KERNEL,
                   distance_threshold=DEFAULT_WARD_DISTANCE_THRESHOLD,
                   batch_size=8, level="coarse"):
    """
    Run the full segmentation pipeline for one video.

    x_encoder, predictor: see sliding_window.compute_embedding_stream's
        docstring for the exact interface each needs to implement (real:
        model.x_encoder.vjepa2.XEncoder / model.predictor.gwen2_5_1_5b.
        VLJEPAPredictor; test doubles: tests/dummy_components.py).
    video_id: opaque string identifying the video (e.g. a filename stem or
        dataset ID) -- stamped onto every emitted SegmentMetadata unchanged.
    video_frames: [T, C, H, W] uint8 tensor, ALREADY DECODED for one video.
        Caller is responsible for decoding (e.g.
        model.x_encoder.vjepa2.XEncoder.load_video_frames for a real video,
        or a plain torch.randint(0, 256, ...) tensor in tests) -- this
        mirrors model/vl_jepa.py's split between encode_video (decodes) and
        encode_frames (doesn't). This pipeline always uses the
        encode_frames path (query-free, batched), same as
        training/phase1.py's validation loop.
    level: stamped onto every emitted SegmentMetadata.level. Only "coarse"
        is used today (see schema.py).

    Returns: list[SegmentMetadata], one per detected segment, in time order,
    covering [0, num_frames) with no gaps and no overlaps.
    """
    num_frames = video_frames.shape[0]

    raw_embeddings, window_starts, _resolved_window_size = compute_embedding_stream(
        x_encoder, predictor, video_frames,
        window_size=window_size, stride=stride, device=device, batch_size=batch_size,
    )
    denoised_embeddings = denoise_embedding_stream(raw_embeddings, kernel_size=denoise_kernel)

    labels = cluster_embedding_stream(denoised_embeddings, distance_threshold=distance_threshold)
    frame_segments = group_into_segments(labels, window_starts, num_frames)

    segments = []
    for frame_seg in frame_segments:
        start_frame = frame_seg["start_frame"]
        end_frame = frame_seg["end_frame"]
        w_lo, w_hi = frame_seg["window_idx_start"], frame_seg["window_idx_end"]

        # Representative embedding for the segment: mean-pool the (already
        # denoised) S_hat_Y over exactly the windows clustering.py assigned
        # to this segment -- not a fresh re-encode of the segment's frame
        # range, since the X-Encoder/Predictor were only ever run at the
        # fixed sliding-window granularity in step 1.
        segment_embedding = denoised_embeddings[w_lo: w_hi + 1].mean(dim=0).numpy()

        # Frame shown as the keyframe for KIS/Q&A retrieval results -- the
        # temporal midpoint of the segment, clamped so a possible 1-frame
        # segment (end_frame == start_frame + 1) still satisfies
        # SegmentMetadata's start_frame <= midpoint_frame < end_frame.
        midpoint_frame = (start_frame + end_frame) // 2
        if midpoint_frame >= end_frame:
            midpoint_frame = end_frame - 1

        segments.append(SegmentMetadata(
            video_id=video_id,
            start_frame=start_frame,
            end_frame=end_frame,
            midpoint_frame=midpoint_frame,
            embedding=segment_embedding,
            level=level,
        ))

    return segments
