"""
model/predictor/gwen2_5_1_5b.py

Predictor for the VL-JEPA-style pipeline, built on top of Qwen2.5:
  - Step 2: load the backbone, keep the upper-half layers, replace the causal
    mask with a bidirectional mask.
  - Step 3: vision_projection + text_projection + predictor_head (shared
    embedding space).

Each step is split into its own function (apply_upper_half_layers,
patch_bidirectional_mask) so unit tests can call them directly on a small/fake
backbone, without needing to download the real Qwen2.5-1.5B (see
tests/test_predictor.py).
"""

import types
import gc

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM


# Read configs/model.yaml
with open("configs/model.yaml", "r") as f:
    import yaml
    model_config = yaml.safe_load(f)

MODEL_ID = model_config["predictor"]["model_name"]
SHARED_EMBED_DIM = model_config["predictor"]["projection_dim"]
MAX_QUERY_LEN = model_config["predictor"]["max_context_tokens"]


def apply_upper_half_layers(backbone):
    """
    Trim backbone.layers down to the upper half (index num_layers//2 -> end).
    Mutates and returns the same backbone in place, so it can be reused both
    in the real loading path and in tests.
    """
    num_layers = len(backbone.layers)
    half = num_layers // 2

    kept_layers = nn.ModuleList(backbone.layers[half:])
    for _ in range(half):
        del backbone.layers[0]

    backbone.layers = kept_layers
    backbone.config.num_hidden_layers = len(kept_layers)

    # Reindex the retained layers so cache lookups start at 0 again.
    for layer_idx, layer in enumerate(backbone.layers):
        if hasattr(layer, "layer_idx"):
            layer.layer_idx = layer_idx
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "layer_idx"):
            layer.self_attn.layer_idx = layer_idx

    del kept_layers
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return backbone


def patch_bidirectional_mask(backbone):
    """
    Older transformers versions exposed Model._update_causal_mask which could
    be monkeypatched directly. Since transformers >= 4.5x (masking_utils
    refactor), mask creation moved to a module-level function
    create_causal_mask, which already supports this power feature:

        if not getattr(config, "is_causal", True):
            return create_bidirectional_mask(...)

    So the correct and most stable approach now is to set config.is_causal =
    False, no monkeypatching needed. This function checks the installed
    version and raises a clear error if it is too old to have this flag.
    """
    import transformers
    from transformers import masking_utils

    if not hasattr(masking_utils, "create_bidirectional_mask"):
        raise RuntimeError(
            "The installed transformers version does not have "
            f"create_bidirectional_mask (found {transformers.__version__}). "
            "You need a newer transformers with the masking_utils refactor, "
            "or rewrite this patch using the old API (_update_causal_mask) "
            "for that version."
        )

    backbone.config.is_causal = False
    return backbone


def load_bidirectional_backbone(model_id=MODEL_ID, device="cuda", dtype=torch.bfloat16):
    """
    dtype defaults to bfloat16, matching configs/training.yaml's
    precision: "bf16" and how y_encoder.py already loads its backbone --
    training runs under bf16 autocast regardless of storage dtype, so
    keeping these weights in float32 bought nothing numerically and only
    inflated checkpoint size (~1.78GB extra for the predictor alone: full
    embed_tokens table, ~233M params, plus the kept upper-half decoder
    layers, ~650M params, stored at 4 bytes/param instead of 2).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    causal_lm = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device)
    backbone = causal_lm.model

    apply_upper_half_layers(backbone)
    patch_bidirectional_mask(backbone)

    hidden_dim = causal_lm.config.hidden_size
    return backbone, tokenizer, hidden_dim


class VLJEPAPredictor(nn.Module):
    """Predictor = vision_projection + text_projection + backbone + pooling + head."""

    def __init__(self, backbone, hidden_dim, shared_dim, vision_dim=None, freeze_backbone=False):
        """
        freeze_backbone defaults to False: per the paper, the upper 8 layers of
        the backbone (Llama-3.2-1B in the paper, Qwen2.5-1.5B here) are the
        TRAINABLE part of the Predictor ("490M trainable parameters"), unlike
        the X-Encoder which is always frozen. Only set this to True if you have
        a specific reason to freeze the backbone (e.g. debugging, ablation) --
        don't leave it defaulted to True and forget about it, since then the
        predictor learns almost nothing from the data (only 3 small linear
        layers would be trained).
        """
        super().__init__()
        self.backbone = backbone
        self.hidden_dim = hidden_dim

        self.vision_projection = (
            nn.Linear(vision_dim, hidden_dim) if vision_dim is not None else None
        )
        self.text_projection = nn.Linear(hidden_dim, hidden_dim)
        self.predictor_head = nn.Linear(hidden_dim, shared_dim)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
        if self.vision_projection is not None:
            for p in self.vision_projection.parameters():
                p.requires_grad_(True)
        for p in self.text_projection.parameters():
            p.requires_grad_(True)
        for p in self.predictor_head.parameters():
            p.requires_grad_(True)

    def forward(self, input_ids=None, attention_mask=None,
                visual_embeds=None, visual_attention_mask=None):
        """
        Supports 3 modes:
          1) text only: input_ids + attention_mask
          2) visual only (dummy input tests): visual_embeds + visual_attention_mask,
             no input_ids
          3) both: visual_embeds + input_ids + attention_mask (concatenated)
        """
        assert input_ids is not None or visual_embeds is not None, \
            "Need at least one of input_ids or visual_embeds"

        text_embeds = None
        if input_ids is not None:
            raw_text_embeds = self.backbone.embed_tokens(input_ids)
            text_embeds = self.text_projection(raw_text_embeds)

        if visual_embeds is not None:
            assert self.vision_projection is not None, \
                "Model must be initialized with vision_dim to use visual_embeds"
            visual_proj = self.vision_projection(visual_embeds)

            if visual_attention_mask is None:
                visual_attention_mask = torch.ones(
                    visual_proj.shape[0], visual_proj.shape[1],
                    dtype=torch.long, device=visual_proj.device,
                )

            if text_embeds is not None:
                inputs_embeds = torch.cat([visual_proj, text_embeds], dim=1)
                combined_mask = torch.cat([visual_attention_mask, attention_mask], dim=1)
            else:
                inputs_embeds = visual_proj
                combined_mask = visual_attention_mask
        else:
            inputs_embeds = text_embeds
            combined_mask = attention_mask

        out = self.backbone(inputs_embeds=inputs_embeds, attention_mask=combined_mask)
        hidden_states = out.last_hidden_state

        mask = combined_mask.unsqueeze(-1).to(hidden_states.dtype)
        pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)

        return self.predictor_head(pooled)


def build_model(model_id=MODEL_ID, shared_dim=SHARED_EMBED_DIM, vision_dim=None,
                 device="cuda", freeze_backbone=False, dtype=torch.bfloat16):
    backbone, tokenizer, hidden_dim = load_bidirectional_backbone(model_id, device, dtype=dtype)
    model = VLJEPAPredictor(backbone, hidden_dim, shared_dim, vision_dim=vision_dim,
                             freeze_backbone=freeze_backbone).to(device)
    return model, tokenizer