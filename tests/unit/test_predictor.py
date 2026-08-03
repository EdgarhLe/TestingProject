"""
model/predictor/tests/test_predictor.py

Step 4 — Unit tests:
    1. Forward pass with dummy input [batch, num_visual_tokens, hidden_dim] -> output [batch, shared_dim].
    2. Verify that the bidirectional mask differs from the causal mask.
    3. Verify that gradients flow through the Predictor and do NOT flow through the frozen X-Encoder/backbone.

Uses a small Qwen2Config (hidden_size=32, 4 layers) with random weights instead of
loading the real Qwen2.5-1.5B model -> the test runs quickly, offline, and without
GPU/network access, while still exercising the real logic (apply_upper_half_layers,
patch_bidirectional_mask, VLJEPAPredictor) because it shares code with production
(predictor.py) rather than mocking it out.
"""

import copy

import pytest
import torch
from transformers import Qwen2Config, Qwen2Model, masking_utils

from model.predictor.gwen2_5_1_5b import (
    VLJEPAPredictor,
    apply_upper_half_layers,
    patch_bidirectional_mask,
)

HIDDEN_DIM = 32
NUM_LAYERS = 4          # sau khi cắt upper-half -> còn 2 layer
VOCAB_SIZE = 100
SHARED_DIM = 16
VISION_DIM = 8


def make_tiny_backbone():
    config = Qwen2Config(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_DIM,
        intermediate_size=HIDDEN_DIM * 2,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        attn_implementation="eager",   # must be "eager" so the _update_causal_mask patch takes effect
    )
    return Qwen2Model(config)


@pytest.fixture
def tiny_backbone():
    return make_tiny_backbone()


@pytest.fixture
def predictor(tiny_backbone):
    apply_upper_half_layers(tiny_backbone)
    patch_bidirectional_mask(tiny_backbone)
    return VLJEPAPredictor(
        tiny_backbone, hidden_dim=HIDDEN_DIM, shared_dim=SHARED_DIM,
        vision_dim=VISION_DIM, freeze_backbone=True,
    )


# ---------------------------------------------------------------------
# 1. Forward pass with dummy visual input -> correct output shape
# ---------------------------------------------------------------------
def test_forward_dummy_visual_input_shape(predictor):
    batch, num_visual_tokens = 3, 5
    dummy_visual = torch.randn(batch, num_visual_tokens, VISION_DIM)

    out = predictor(visual_embeds=dummy_visual)

    assert out.shape == (batch, SHARED_DIM)
    assert torch.isfinite(out).all()


def test_forward_visual_and_text_shape(predictor):
    batch, num_visual_tokens, num_text_tokens = 2, 4, 6
    dummy_visual = torch.randn(batch, num_visual_tokens, VISION_DIM)
    input_ids = torch.randint(0, VOCAB_SIZE, (batch, num_text_tokens))
    attention_mask = torch.ones(batch, num_text_tokens, dtype=torch.long)

    out = predictor(input_ids=input_ids, attention_mask=attention_mask, visual_embeds=dummy_visual)

    assert out.shape == (batch, SHARED_DIM)


# ---------------------------------------------------------------------
# 2. Bidirectional mask must differ from the causal mask
# ---------------------------------------------------------------------
def test_bidirectional_mask_differs_from_causal():
    causal_backbone = make_tiny_backbone()   # no patch -> config.is_causal defaults to True/causal
    bidir_backbone = copy.deepcopy(causal_backbone)
    patch_bidirectional_mask(bidir_backbone)  # set config.is_causal = False

    batch, seq_len = 2, 6
    dummy_embeds = torch.randn(batch, seq_len, HIDDEN_DIM)
    attention_mask = torch.ones(batch, seq_len, dtype=torch.long)   # no padding

    causal_mask = masking_utils.create_causal_mask(
        config=causal_backbone.config, inputs_embeds=dummy_embeds,
        attention_mask=attention_mask, past_key_values=None,
    )
    bidir_mask = masking_utils.create_causal_mask(
        config=bidir_backbone.config, inputs_embeds=dummy_embeds,
        attention_mask=attention_mask, past_key_values=None,
    )

    # Causal must always return a real mask (blocking future tokens), not None
    assert causal_mask is not None
    min_dtype = torch.finfo(dummy_embeds.dtype).min

    triu_i, triu_j = torch.triu_indices(seq_len, seq_len, offset=1)  # only positions j > i
    causal_future_values = causal_mask[0, 0][triu_i, triu_j]
    assert (causal_future_values == min_dtype).all(), \
        "Causal mask phải chặn mọi vị trí tương lai (j > i)"

    # Bidirectional: when there is no padding, the library may optimize by returning None
    # (meaning "no mask needed" = attend to everything) instead of a tensor of all zeros.
    # Both cases (None, or a tensor that does not mask future tokens) are valid -> normalize before comparing.
    if bidir_mask is None:
        # None means nothing is masked at all -> definitely different from causal (which blocks the future)
        pass
    else:
        assert not torch.equal(causal_mask, bidir_mask), \
            "Bidirectional mask must differ from the causal mask, otherwise the patch has no effect"
        bidir_future_values = bidir_mask[0, 0][triu_i, triu_j]
        assert (bidir_future_values == 0).all(), \
            "Bidirectional mask must not block future positions when there is no padding"


def test_bidirectional_mask_still_respects_padding():
    """Bidirectional does not mean ignoring padding — PAD tokens must still be masked."""
    backbone = make_tiny_backbone()
    patch_bidirectional_mask(backbone)

    batch, seq_len = 1, 5
    dummy_embeds = torch.randn(batch, seq_len, HIDDEN_DIM)
    # The last 2 tokens are padding (0)
    attention_mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.long)

    mask = masking_utils.create_causal_mask(
        config=backbone.config, inputs_embeds=dummy_embeds,
        attention_mask=attention_mask, past_key_values=None,
    )
    min_dtype = torch.finfo(dummy_embeds.dtype).min

    # Columns corresponding to padding (indices 3, 4) must be masked in EVERY row
    assert (mask[0, 0, :, 3] == min_dtype).all()
    assert (mask[0, 0, :, 4] == min_dtype).all()
    # Columns corresponding to real tokens (indices 0..2) must not be masked
    assert (mask[0, 0, :, 0] == 0).all()


# ---------------------------------------------------------------------
# 3. Gradients flow through the Predictor and NOT through the frozen backbone
# ---------------------------------------------------------------------
def test_gradient_flows_through_predictor_not_backbone_visual_only(predictor):
    """This forward pass uses only visual_embeds -> text_projection is NOT called, so it is not checked here."""
    batch, num_visual_tokens = 2, 4
    dummy_visual = torch.randn(batch, num_visual_tokens, VISION_DIM)

    out = predictor(visual_embeds=dummy_visual)
    loss = out.sum()
    loss.backward()

    # --- Backbone: frozen, requires_grad=False -> .grad must be None ---
    backbone_params = list(predictor.backbone.parameters())
    assert len(backbone_params) > 0
    for p in backbone_params:
        assert not p.requires_grad
        assert p.grad is None

    # --- Only vision_projection + predictor_head participate in this forward pass ---
    used_modules = [predictor.vision_projection, predictor.predictor_head]
    for module in used_modules:
        for p in module.parameters():
            assert p.requires_grad
            assert p.grad is not None
            assert torch.any(p.grad != 0), f"Gradient toàn 0 ở {module}"

    # text_projection is unused in this pass -> .grad must be None (no backward pass reached it)
    for p in predictor.text_projection.parameters():
        assert p.requires_grad          # still trainable, just no gradient in this forward pass
        assert p.grad is None


def test_gradient_flows_through_predictor_with_text_and_visual(predictor):
    """This forward pass uses both visual_embeds + input_ids -> all 3 modules (vision/text projection, predictor_head) receive gradients."""
    batch, num_visual_tokens, num_text_tokens = 2, 4, 6
    dummy_visual = torch.randn(batch, num_visual_tokens, VISION_DIM)
    input_ids = torch.randint(0, VOCAB_SIZE, (batch, num_text_tokens))
    attention_mask = torch.ones(batch, num_text_tokens, dtype=torch.long)

    out = predictor(input_ids=input_ids, attention_mask=attention_mask, visual_embeds=dummy_visual)
    out.sum().backward()

    for p in predictor.backbone.parameters():
        assert not p.requires_grad
        assert p.grad is None

    trainable_modules = [predictor.vision_projection, predictor.text_projection, predictor.predictor_head]
    for module in trainable_modules:
        for p in module.parameters():
            assert p.requires_grad
            assert p.grad is not None
            assert torch.any(p.grad != 0), f"Gradient toàn 0 ở {module}"


def test_backbone_stays_frozen_after_multiple_forward_backward(predictor):
    """Ensure requires_grad does not get accidentally re-enabled (e.g. wrong .train() call, accidental unfreeze, etc.)."""
    for _ in range(3):
        predictor.zero_grad(set_to_none=True)
        dummy_visual = torch.randn(2, 4, VISION_DIM)
        out = predictor(visual_embeds=dummy_visual)
        out.sum().backward()

    assert not any(p.requires_grad for p in predictor.backbone.parameters())


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))