"""
indexing/segmentation/tests/dummy_components.py

Lightweight stand-ins for model.x_encoder.vjepa2.XEncoder and
model.predictor.gwen2_5_1_5b.VLJEPAPredictor, used ONLY by this package's
unit tests. Per the Week 4 scope note in pipeline.py, we deliberately do NOT
download or run the real facebook/vjepa2-vitl-fpc64-256 / Qwen2.5-1.5B
models here -- these dummies exist purely to exercise the sliding-window /
clustering / schema plumbing against known, controllable input, not to
validate segmentation quality (that needs a real trained checkpoint,
starting Week 5).

DummyXEncoder.encode_frames and DummyPredictor.__call__ implement the exact
call signatures compute_embedding_stream() actually uses:
    x_encoder.encode_frames(batched_frames)      -> [B, N_tokens, vision_dim]
    predictor(visual_embeds=visual_embeds)       -> [B, shared_dim]
"""

import torch
import torch.nn as nn

from indexing.segmentation.schema import EMBEDDING_DIM


class DummyXEncoder:
    """
    encode_frames(batch) -> [B, N_tokens=1, vision_dim] where the single
    token's value is derived deterministically from the batch's per-window
    mean pixel intensity (plus a few fixed per-channel offsets so it isn't
    literally a constant vector). This lets tests construct videos with two
    (or more) visually distinct regions and know, without any randomness,
    that the resulting embedding stream should separate into that many
    clusters.
    """

    def __init__(self, vision_dim=8):
        self.vision_dim = vision_dim
        # Fixed, non-random per-dimension offsets/scales so output is
        # reproducible across runs without seeding a global RNG.
        self._offsets = torch.arange(vision_dim, dtype=torch.float32) * 0.1

    def encode_frames(self, video_frames):
        # video_frames: [B, T, C, H, W] uint8
        mean_intensity = video_frames.float().mean(dim=(1, 2, 3, 4))   # [B]
        features = mean_intensity[:, None] + self._offsets[None, :]     # [B, vision_dim]
        return features.unsqueeze(1)   # [B, 1, vision_dim]


class DummyPredictor(nn.Module):
    """
    A minimal stand-in for VLJEPAPredictor's "visual only" forward mode: a
    single fixed (seeded, untrained-but-deterministic) Linear layer mapping
    pooled visual features -> shared_dim. Fixed seed so distances between
    outputs are reproducible across test runs; NOT meant to resemble a real
    trained predictor in any other way.
    """

    def __init__(self, vision_dim=8, shared_dim=EMBEDDING_DIM, seed=0):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        weight = torch.randn(shared_dim, vision_dim, generator=generator) * 0.5
        bias = torch.zeros(shared_dim)
        self.projection = nn.Linear(vision_dim, shared_dim)
        with torch.no_grad():
            self.projection.weight.copy_(weight)
            self.projection.bias.copy_(bias)

    def forward(self, visual_embeds=None, input_ids=None, attention_mask=None):
        assert visual_embeds is not None, "DummyPredictor only supports the query-free, visual-only mode"
        pooled = visual_embeds.mean(dim=1)   # [B, vision_dim] -- mean over the (here, single) token
        return self.projection(pooled)       # [B, shared_dim]

    def __call__(self, *args, **kwargs):
        return nn.Module.__call__(self, *args, **kwargs)


def make_two_scene_video(frames_per_scene=100, height=8, width=8, channels=3,
                          low_value=10, high_value=200):
    """
    A synthetic [T, C, H, W] uint8 "video" with two visually distinct,
    contiguous halves (a dark scene then a bright scene) -- exactly the kind
    of content a correct temporal segmentation should split into 2 segments
    with a boundary near the midpoint, and NOT merge the two halves together
    just because e.g. a future third scene happened to look similar to the
    first.
    """
    total_frames = frames_per_scene * 2
    video = torch.empty((total_frames, channels, height, width), dtype=torch.uint8)
    video[:frames_per_scene] = low_value
    video[frames_per_scene:] = high_value
    return video


def make_three_scene_video(frames_per_scene=80, height=8, width=8, channels=3,
                            values=(10, 200, 10)):
    """
    Three contiguous scenes where the first and third look identical
    (same pixel value). A correct temporally-CONSTRAINED clustering must
    still emit 3 segments (not merge scene 1 and scene 3 into one cluster
    just because their embeddings are near-identical) -- this is the
    canonical regression case for the temporal connectivity constraint in
    clustering.build_temporal_connectivity.
    """
    total_frames = frames_per_scene * len(values)
    video = torch.empty((total_frames, channels, height, width), dtype=torch.uint8)
    for i, value in enumerate(values):
        video[i * frames_per_scene:(i + 1) * frames_per_scene] = value
    return video
