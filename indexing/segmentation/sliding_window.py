"""
indexing/segmentation/sliding_window.py

Step 1 of the semantic segmentation pipeline (#69): run the frozen X-Encoder
(#21, model/x_encoder/vjepa2.py) + trainable Predictor (#44,
model/predictor/gwen2_5_1_5b.py) over a video with a small, fixed frame
stride, producing a continuous, time-ordered stream of predicted embeddings
S_hat_Y. clustering.py then cuts this stream into segments. There is
deliberately no Shot Boundary Detection anywhere in this pipeline -- segment
boundaries emerge purely from where the embedding stream's semantic content
changes (see the "So Với Cách Làm Truyền Thống" slide).

IMPORTANT -- query-free inference:
Phase 1 training (configs/training.yaml: phases.phase1.query_conditioned:
false) trains the Predictor to produce a meaningful S_hat_Y from
visual_embeds ALONE. training/phase1.py's run_validation() and
run_validation_retrieval() both already call the Predictor this way --
`model.predictor(visual_embeds=visual_embeds)`, no queries/input_ids -- which
is VLJEPAPredictor.forward()'s "visual only" mode (mode 2 in its docstring).
Segmentation reuses that exact same query-free call: there is no text query
available at index-build time, so passing one in would exercise a code path
the Phase 1 checkpoint was never optimized for.
"""

import torch


DEFAULT_WINDOW_SIZE = 64     # frames per window -- matches XEncoder's default num_frames
                              # (model/x_encoder/vjepa2.py), so each window is a valid,
                              # full-length clip for the frozen X-Encoder.
DEFAULT_STRIDE = 8           # coarse pass, per spec: ~5-10 frame stride between window starts.
DEFAULT_DENOISE_KERNEL = 3   # local average pooling kernel width, in WINDOW-STEPS (not frames).


def make_window_starts(num_frames, window_size, stride):
    """
    Start-frame index of every sliding window over a video with `num_frames`
    frames. Windows fully cover [0, num_frames): the last window is forced to
    end exactly at num_frames (rather than possibly stopping short by up to
    stride-1 frames), so the tail of the video is never silently dropped from
    the embedding stream.

    If the video is shorter than window_size, returns a single window
    starting at 0 -- the caller pads it to window_size (see
    compute_embedding_stream), matching how a short clip would be looped/
    padded for the X-Encoder elsewhere in this codebase.
    """
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    if num_frames <= window_size:
        return [0]

    last_start = num_frames - window_size
    starts = list(range(0, last_start + 1, stride))
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


@torch.no_grad()
def compute_embedding_stream(x_encoder, predictor, video_frames, window_size=DEFAULT_WINDOW_SIZE,
                              stride=DEFAULT_STRIDE, device="cuda", batch_size=8):
    """
    video_frames: [T, C, H, W] uint8 tensor for ONE already-decoded video.
    Pixel format matches training (see model/x_encoder/vjepa2.py's
    encode_frames docstring: uint8, [0, 255]) -- this function calls
    x_encoder.encode_frames(), the same batched, pre-decoded-tensor path
    training/phase1.py's training step uses, rather than
    x_encoder.encode_video() (which decodes a path/URL one video at a time
    and isn't meant for a tensor we've already decoded/sampled ourselves).

    x_encoder: model.x_encoder.vjepa2.XEncoder instance (or a stand-in with
        the same encode_frames(batched_frames) -> [B, N_tokens, vision_dim]
        interface -- see tests/dummy_components.py).
    predictor: model.predictor.gwen2_5_1_5b.VLJEPAPredictor instance (or a
        stand-in with the same __call__(visual_embeds=...) -> [B, shared_dim]
        interface).

    Returns:
      embeddings: FloatTensor [num_windows, shared_dim] -- S_hat_Y per window,
                  in time order, moved to CPU.
      window_starts: list[int] -- start_frame of each window, same order as
                     `embeddings`.
      window_size: int -- passed straight through so downstream code
                   (clustering.group_into_segments) can compute end_frame
                   without needing a second argument threaded everywhere.
    """
    num_frames = video_frames.shape[0]
    window_starts = make_window_starts(num_frames, window_size, stride)

    all_embeds = []
    for batch_start in range(0, len(window_starts), batch_size):
        batch_window_starts = window_starts[batch_start: batch_start + batch_size]
        batch_windows = []
        for start in batch_window_starts:
            end = min(start + window_size, num_frames)
            window = video_frames[start:end]
            if window.shape[0] < window_size:
                # Only reachable when num_frames <= window_size (see
                # make_window_starts, which then only ever returns [0]) --
                # pad by repeating the last frame so the X-Encoder always
                # sees a fixed-length clip, same as a short real video would
                # need to be handled before hitting the frozen encoder.
                pad = window[-1:].repeat(window_size - window.shape[0], 1, 1, 1)
                window = torch.cat([window, pad], dim=0)
            batch_windows.append(window)

        stacked = torch.stack(batch_windows, dim=0).to(device)   # [b, window_size, C, H, W]
        visual_embeds = x_encoder.encode_frames(stacked)          # [b, N_visual_tokens, vision_dim]
        s_hat_y = predictor(visual_embeds=visual_embeds)          # [b, shared_dim] -- query-free
        all_embeds.append(s_hat_y.detach().cpu())

    embeddings = torch.cat(all_embeds, dim=0)   # [num_windows, shared_dim]
    return embeddings, window_starts, window_size


def denoise_embedding_stream(embeddings, kernel_size=DEFAULT_DENOISE_KERNEL):
    """
    Local average pooling along the time axis, to denoise the raw per-window
    S_hat_Y stream before clustering (spec step 1). kernel_size is forced odd
    (bumped up by 1 if given even) so the pooling window is centered on every
    timestep.

    embeddings: FloatTensor [num_windows, shared_dim]
    Returns: FloatTensor [num_windows, shared_dim], same shape. Windows near
    either edge of the video use a smaller, asymmetric (clipped) kernel
    rather than zero-padding, which would otherwise bias the first/last
    windows' embeddings toward the origin instead of just averaging over
    fewer real neighbors.
    """
    if kernel_size <= 1:
        return embeddings
    if kernel_size % 2 == 0:
        kernel_size += 1

    num_windows = embeddings.shape[0]
    half = kernel_size // 2
    pooled = torch.empty_like(embeddings)
    for i in range(num_windows):
        lo = max(0, i - half)
        hi = min(num_windows, i + half + 1)
        pooled[i] = embeddings[lo:hi].mean(dim=0)
    return pooled
