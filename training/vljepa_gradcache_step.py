"""
training/vljepa_gradcache_step.py

Wires VLJEPA (model/main.py) together with GradCache-style accumulation
(training/gradcache.py), consuming training/data/loader.py's
(video_frames, captions) batches.

No prompt ensembling: Phase 1 uses one raw caption per sample directly as
the Y-Encoder target -- no fixed event/content prompts, no averaging
(dropped entirely for Phase 1 -- was #49's design, no longer used).

No query field: matches configs/training.yaml's phase1.query_conditioned:
false -- Predictor runs visual-only (no X_Q), which VLJEPAPredictor.forward()
already supports. Phase 2 turns query_conditioned back on; when that
DataLoader contract exists, only make_forward_fn needs a query branch added.
"""

import torch

from model.vl_jepa import VLJEPA, _PRECISION_TO_DTYPE
from training.losses.info_nce_loss import DEFAULT_UNIFORMITY_LAMBDA, bidirectional_infonce_loss


def prepare_micro_batches(model: VLJEPA, raw_micro_batches, device="cuda", precision="bf16"):
    """
    Batch all micro-batch frames together for ONE X-Encoder forward pass,
    then split back. Massively more efficient than looping.
    """
    amp_dtype = _PRECISION_TO_DTYPE[precision]
    amp_enabled = device.startswith("cuda") and precision != "fp32"

    # Extract all frames and captions
    all_video_frames = []
    all_captions = []
    for video_frames, captions in raw_micro_batches:
        all_video_frames.append(video_frames)
        all_captions.extend(captions)
    
    # Batch encode all frames at once
    batched_frames = torch.cat(all_video_frames, dim=0).to(device)  # (B*accumulation_steps, T, C, H, W)
    with torch.inference_mode():
        batched_visual_embeds = model.x_encoder.encode_frames(batched_frames)  # [B*acc, N_tokens, D]
    
    # Tokenize all captions at once
    input_ids, attention_mask = model.y_encoder.tokenize(all_captions)
    
    # Split back into micro-batch structure for GradCache
    prepared = []
    idx = 0
    for video_frames, captions in raw_micro_batches:
        batch_size = len(captions)
        prepared.append({
            "visual_embeds": batched_visual_embeds[idx:idx+batch_size],
            "input_ids": input_ids[idx:idx+batch_size].to(device),
            "attention_mask": attention_mask[idx:idx+batch_size].to(device),
        })
        idx += batch_size
    return prepared


def make_forward_fn(model: VLJEPA, device="cuda", precision="bf16", query_conditioned=False):
    """
    Returns forward_fn(micro_batch) -> (s_hat_y, s_y), both [batch_size, shared_dim].

    query_conditioned=False (Phase 1, current DataLoader contract): Predictor
    runs visual-only, no X_Q.

    query_conditioned=True (Phase 2, NOT wired yet): the current DataLoader
    contract has no query field, so this branch is a placeholder.
    """
    amp_dtype = _PRECISION_TO_DTYPE[precision]
    amp_enabled = device.startswith("cuda") and precision != "fp32"

    def forward_fn(micro_batch):
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
            if query_conditioned:
                raise NotImplementedError(
                    "query_conditioned=True (Phase 2) is not wired yet -- the current "
                    "DataLoader contract has no query field. Update this branch once "
                    "Phase 2's batch format is confirmed."
                )
            s_hat_y = model.predictor(visual_embeds=micro_batch["visual_embeds"])   # visual-only, [B, D]

            input_ids = micro_batch["input_ids"]
            attention_mask = micro_batch["attention_mask"]
            s_y = model.y_encoder(input_ids=input_ids, attention_mask=attention_mask)   # [B, D]

        return s_hat_y, s_y

    return forward_fn


def vljepa_gradcache_training_step(model: VLJEPA, optimizer, raw_micro_batches,
                                    device="cuda", precision="bf16", query_conditioned=False,
                                    uniformity_lambda=DEFAULT_UNIFORMITY_LAMBDA):
    """
    One full accumulation-window training step: len(raw_micro_batches) micro-batches
    (e.g. 16, matching configs/training.yaml's gradient_accumulation_steps) ->
    one optimizer.step(), with correct in-batch-negative InfoNCE AND uniformity
    regularization across the whole accumulation window (see training/gradcache.py's
    module docstring for why naive accumulation would silently break both terms).
    """
    from training.gradcache import gradcache_infonce_step

    model.train()
    model.x_encoder.eval()   # frozen -> always keep it in eval()

    prepared = prepare_micro_batches(model, raw_micro_batches, device=device, precision=precision)
    forward_fn = make_forward_fn(model, device=device, precision=precision, query_conditioned=query_conditioned)

    def loss_fn(s_hat_y, s_y):
        return bidirectional_infonce_loss(
            s_hat_y, s_y, model.logit_scale, uniformity_lambda=uniformity_lambda,
        )

    stats = gradcache_infonce_step(prepared, forward_fn, loss_fn, optimizer, device=device)
    model.clamp_logit_scale()
    return stats