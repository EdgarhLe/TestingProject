import torch

from indexing.segmentation.sliding_window import (
    make_window_starts,
    compute_embedding_stream,
    denoise_embedding_stream,
)
from tests.utils.dummy_components import (
    DummyXEncoder,
    DummyPredictor,
    make_two_scene_video,
)


def test_make_window_starts_covers_full_video_without_dropping_the_tail():
    starts = make_window_starts(num_frames=100, window_size=20, stride=7)
    assert starts[0] == 0
    assert starts[-1] == 80          # last window must end exactly at num_frames (100)
    assert all(0 <= s <= 80 for s in starts)
    assert starts == sorted(starts)  # strictly increasing, time order preserved


def test_make_window_starts_short_video_returns_single_window():
    starts = make_window_starts(num_frames=10, window_size=64, stride=8)
    assert starts == [0]


def test_make_window_starts_rejects_empty_video():
    import pytest
    with pytest.raises(ValueError):
        make_window_starts(num_frames=0, window_size=64, stride=8)


def test_compute_embedding_stream_shapes():
    video = make_two_scene_video(frames_per_scene=40)   # 80 frames total
    x_encoder = DummyXEncoder(vision_dim=8)
    predictor = DummyPredictor(vision_dim=8, shared_dim=16)

    embeddings, window_starts, window_size = compute_embedding_stream(
        x_encoder, predictor, video, window_size=20, stride=5, device="cpu", batch_size=4,
    )

    assert embeddings.shape == (len(window_starts), 16)
    assert window_size == 20
    assert window_starts[-1] + window_size == video.shape[0]  # tail is covered
    assert embeddings.device.type == "cpu"


def test_compute_embedding_stream_pads_short_video_to_window_size():
    video = make_two_scene_video(frames_per_scene=5)   # 10 frames total, shorter than window_size
    x_encoder = DummyXEncoder(vision_dim=8)
    predictor = DummyPredictor(vision_dim=8, shared_dim=16)

    embeddings, window_starts, window_size = compute_embedding_stream(
        x_encoder, predictor, video, window_size=64, stride=8, device="cpu", batch_size=4,
    )
    assert window_starts == [0]
    assert embeddings.shape == (1, 16)


def test_denoise_embedding_stream_preserves_shape_and_smooths():
    torch.manual_seed(0)
    embeddings = torch.randn(50, 16)
    denoised = denoise_embedding_stream(embeddings, kernel_size=5)

    assert denoised.shape == embeddings.shape
    # Denoising should reduce frame-to-frame variance versus the raw stream.
    raw_variation = (embeddings[1:] - embeddings[:-1]).pow(2).mean()
    denoised_variation = (denoised[1:] - denoised[:-1]).pow(2).mean()
    assert denoised_variation < raw_variation


def test_denoise_embedding_stream_even_kernel_is_bumped_to_odd_and_still_shape_safe():
    embeddings = torch.randn(20, 16)
    denoised = denoise_embedding_stream(embeddings, kernel_size=4)
    assert denoised.shape == embeddings.shape


def test_denoise_embedding_stream_kernel_one_is_identity():
    embeddings = torch.randn(20, 16)
    denoised = denoise_embedding_stream(embeddings, kernel_size=1)
    assert torch.equal(denoised, embeddings)
