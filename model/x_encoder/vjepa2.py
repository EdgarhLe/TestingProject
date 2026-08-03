"""
X-Encoder: wrapper around V-JEPA2 (facebook/vjepa2-vitl-fpc64-256) used as the
visual encoder branch of the VL-JEPA pipeline.

Responsibilities of this module:
  - Load the model + processor from the HF hub, move to the right device.
  - Freeze all parameters (the X-Encoder is never trained).
  - Decode a video (local path or URL) -> sample frames -> preprocess -> encode
    into visual embeddings [B, N_visual_tokens, vision_dim].
  - Expose `vision_dim` so external code (VLJEPAPredictor) can use it to
    initialize vision_projection.
"""

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoVideoProcessor, AutoModel
from torchcodec.decoders import VideoDecoder


class XEncoder(nn.Module):
    def __init__(self, hf_repo="facebook/vjepa2-vitl-fpc64-256", device="cuda",
                 num_frames=64, freeze=True):
        super().__init__()
        self.device = device
        self.num_frames = num_frames

        self.model = AutoModel.from_pretrained(hf_repo).to(device)
        self.processor = AutoVideoProcessor.from_pretrained(hf_repo)

        if freeze:
            self.freeze()

        # Infer vision_dim from the config if available; otherwise it gets set
        # after the first encode call
        self.vision_dim = getattr(self.model.config, "hidden_size", None)

    # ------------------------------------------------------------------
    # Freeze / check state
    # ------------------------------------------------------------------
    def freeze(self):
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

    @property
    def is_frozen(self):
        return not any(p.requires_grad for p in self.model.parameters())

    # ------------------------------------------------------------------
    # Frame index sampling -- kept separate so other sampling strategies are
    # easy to add later
    # ------------------------------------------------------------------
    @staticmethod
    def sample_frame_indices(num_frames, strategy="uniform"):
        """
        Default: take the first num_frames consecutively (same as the original
        code: np.arange(0, 64)). For more complex sampling (random clips,
        evenly spaced stride across the whole video, etc.) add a new strategy
        here.
        """
        if strategy == "uniform":
            return np.arange(0, num_frames)
        raise NotImplementedError(f"Sampling strategy '{strategy}' is not supported")

    # ------------------------------------------------------------------
    # Decode video -> frame tensor [T, C, H, W]
    # ------------------------------------------------------------------
    def load_video_frames(self, video_path_or_url, frame_idx=None):
        vr = VideoDecoder(video_path_or_url)
        if frame_idx is None:
            frame_idx = self.sample_frame_indices(self.num_frames)
        frames = vr.get_frames_at(indices=frame_idx).data  # T x C x H x W
        return frames

    # ------------------------------------------------------------------
    # Encode: video (path/url) -> visual embeddings
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_video(self, video_path_or_url, frame_idx=None):
        frames = self.load_video_frames(video_path_or_url, frame_idx=frame_idx)
        inputs = self.processor(frames, return_tensors="pt").to(self.device)
        embeddings = self.model.get_vision_features(**inputs)  # [B, N_visual_tokens, vision_dim]

        if self.vision_dim is None:
            self.vision_dim = embeddings.shape[-1]

        return embeddings

    @torch.no_grad()
    def encode_frames(self, video_frames):
        """
        Encode an ALREADY-DECODED batched pixel tensor (B, T, C, H, W) coming
        directly from the training DataLoader (training/data/unified_dataset.py
        via training/data/prompt_ensemble_wrapper.py) -- unlike encode_video()
        above (which takes a path/URL and decodes one video at a time via
        VideoDecoder, meant for inference/demo), this skips decoding entirely
        and processes a real training batch of B videos at once.

        Pixel format: confirmed uint8, [0, 255] (see
        training/data/transforms/resize.py's docstring and return type --
        value normalization is deliberately NOT done there, left to whatever
        consumes it). This is exactly what HF image/video processors expect
        by default (do_rescale=True, rescale_factor=1/255), so no override is
        passed here -- an earlier version of this method passed
        do_rescale=False based on a since-corrected claim that the data was
        already [0,1] float; that was wrong and has been reverted.

        NOTE: this assumes AutoVideoProcessor accepts a (B, T, C, H, W) tensor
        directly and batches internally -- not verified end-to-end against a
        downloaded facebook/vjepa2-vitl-fpc64-256 model in this environment
        (no network access here). Please sanity-check output shape AND actual
        pixel value scale reaching the model the first time this runs for real.
        """
        video_frames = video_frames.to(self.device)
        inputs = self.processor(video_frames, return_tensors="pt").to(self.device)
        embeddings = self.model.get_vision_features(**inputs)  # expected [B, N_visual_tokens, vision_dim]

        if self.vision_dim is None:
            self.vision_dim = embeddings.shape[-1]

        return embeddings

    @torch.no_grad()
    def encode_videos(self, video_paths_or_urls, frame_idx=None):
        """Encode multiple videos, returning a list of embeddings (each video may have a
        different N_visual_tokens)."""
        return [self.encode_video(v, frame_idx=frame_idx) for v in video_paths_or_urls]

    def forward(self, video_path_or_url, frame_idx=None):
        return self.encode_video(video_path_or_url, frame_idx=frame_idx)


# ---------------------------------------------------------------------
# Standalone usage example (equivalent to the original snippet you provided)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    x_encoder = XEncoder(device=device)  # defaults: num_frames=64, freeze=True
    assert x_encoder.is_frozen

    video_url = ("https://huggingface.co/datasets/nateraw/kinetics-mini/resolve/main/"
                 "val/archery/-Qz25rXdMjE_000014_000024.mp4")

    video_embeddings = x_encoder.encode_video(video_url)
    print("Video embeddings shape:", video_embeddings.shape)
    print("Inferred vision_dim:", x_encoder.vision_dim)

    # ---- Integration with the VLJEPAPredictor built in a previous step ----
    # from model.predictor.predictor import build_model, MAX_QUERY_LEN
    #
    # predictor_model, tokenizer = build_model(device=device, vision_dim=x_encoder.vision_dim)
    #
    # queries = ["What action is happening in this video?"]
    # query_inputs = tokenizer(queries, return_tensors="pt", padding=True,
    #                           truncation=True, max_length=MAX_QUERY_LEN).to(device)
    #
    # with torch.no_grad():
    #     joint_embedding = predictor_model(
    #         input_ids=query_inputs["input_ids"],
    #         attention_mask=query_inputs["attention_mask"],
    #         visual_embeds=video_embeddings,
    #     )
    # print("Joint embedding shape:", joint_embedding.shape)  # [batch, shared_embedding_dim]