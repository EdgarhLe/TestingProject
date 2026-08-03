"""
model/y_encoder/bge_m3.py

Y-Encoder — text/query encoder that maps text into the same shared embedding
space as the Predictor.

Architecture:
        text -> tokenizer (bge-m3) -> backbone (bge-m3, XLM-RoBERTa-large) -> pooling
                 -> linear projection (hidden_dim -> shared_dim)

LR strategy (important):
        - `backbone` (pretrained bge-m3 weights) trains with LR = base_lr * lr_multiplier
            (default 0.05) so we do NOT destroy the existing multilingual alignment.
        - `projection` (newly initialized linear head, nothing to preserve) trains with
            LR = base_lr, same as the Predictor.
            -> This is why build_optimizer_param_groups splits backbone and projection into
                 two separate param groups instead of applying the multiplier to the entire
                 Y-Encoder. If you really want 0.05x for the projection head too, call
                 build_optimizer_param_groups(..., apply_multiplier_to_projection=True).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

Y_ENCODER_MODEL_ID = "BAAI/bge-m3"
SHARED_EMBED_DIM = 1536
DEFAULT_Y_ENCODER_LR_MULTIPLIER = 0.05
MAX_TEXT_LEN = 512


# ---------------------------------------------------------------------
# Load the real backbone (requires network access to download from HuggingFace Hub)
# ---------------------------------------------------------------------
def load_y_encoder_backbone(model_name=Y_ENCODER_MODEL_ID, device="cuda", torch_dtype=torch.float16):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # add_pooling_layer=False: we do CLS/mean pooling manually in YEncoder,
    # so the default XLM-RoBERTa pooler (Dense+Tanh) is unnecessary and would add
    # parameters that never get used in forward (and never receive gradients), which
    # can be confusing during debugging.
    backbone = AutoModel.from_pretrained(
        model_name,
        add_pooling_layer=False,
        torch_dtype=torch_dtype,
    ).to(device)
    hidden_dim = backbone.config.hidden_size
    return backbone, tokenizer, hidden_dim


# ---------------------------------------------------------------------
# Y-Encoder module
# ---------------------------------------------------------------------
class YEncoder(nn.Module):
    def __init__(self, backbone, tokenizer, hidden_dim, shared_dim=SHARED_EMBED_DIM,
                 pooling="cls", freeze_backbone=False):
        """
        backbone, tokenizer, hidden_dim: taken from load_y_encoder_backbone(...) in real use,
            or passed as a tiny/dummy model in offline unit tests (see tests/test_y_encoder.py).
        pooling: "cls" (default, matches bge-m3 dense embedding behavior) or "mean".
        freeze_backbone: default False — the Y-Encoder still trains (unlike the frozen
            X-Encoder), just with a much smaller LR (see build_optimizer_param_groups).
        """
        super().__init__()
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.hidden_dim = hidden_dim
        self.pooling = pooling

        self.projection = nn.Linear(hidden_dim, shared_dim)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    def _pool(self, hidden_states, attention_mask):
        if self.pooling == "cls":
            return hidden_states[:, 0]
        elif self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        raise ValueError(f"Pooling '{self.pooling}' is not supported")

    def forward(self, input_ids, attention_mask, **kwargs):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._pool(out.last_hidden_state, attention_mask)
        return self.projection(pooled)                     # [B, shared_dim]

    @torch.no_grad()
    def encode_texts(self, texts, max_length=MAX_TEXT_LEN, device=None):
        """Inference helper: text (list[str]) -> embedding [B, shared_dim], no gradients."""
        device = device or next(self.parameters()).device
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length,
        ).to(device)
        return self.forward(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
    
    def tokenize(self, texts, max_length=MAX_TEXT_LEN, device=None):
        """Tokenize helper: text (list[str]) -> input_ids, attention_mask (for forward)."""
        device = device or next(self.parameters()).device
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length,
        ).to(device)
        return inputs["input_ids"], inputs["attention_mask"]


def build_y_encoder(model_name=Y_ENCODER_MODEL_ID, shared_dim=SHARED_EMBED_DIM,
                     device="cuda", pooling="cls", freeze_backbone=False, torch_dtype=torch.float16):
    backbone, tokenizer, hidden_dim = load_y_encoder_backbone(model_name, device, torch_dtype=torch_dtype)
    model = YEncoder(backbone, tokenizer, hidden_dim, shared_dim=shared_dim,
                      pooling=pooling, freeze_backbone=freeze_backbone).to(device)
    return model


# ---------------------------------------------------------------------
# Optimizer param groups: separate LR multiplier for the Y-Encoder backbone
# ---------------------------------------------------------------------
def build_optimizer_param_groups(predictor, y_encoder, base_lr,
                                  y_encoder_lr_multiplier=DEFAULT_Y_ENCODER_LR_MULTIPLIER,
                                  apply_multiplier_to_projection=False):
    """
    Return a list of param_groups that can be passed directly to torch.optim.AdamW(param_groups):

        [Predictor (toàn bộ phần trainable)]      -> lr = base_lr
        [Y-Encoder backbone (bge-m3 pretrained)]  -> lr = base_lr * y_encoder_lr_multiplier
        [Y-Encoder projection head (mới khởi tạo)] -> lr = base_lr
            (trừ khi apply_multiplier_to_projection=True)

    Only include parameters with requires_grad=True (skip frozen parts,
    such as the X-Encoder inside the Predictor).
    """
    predictor_params = [p for p in predictor.parameters() if p.requires_grad]
    y_backbone_params = [p for p in y_encoder.backbone.parameters() if p.requires_grad]
    y_projection_params = [p for p in y_encoder.projection.parameters() if p.requires_grad]

    projection_lr = (
        base_lr * y_encoder_lr_multiplier if apply_multiplier_to_projection else base_lr
    )

    param_groups = []
    if predictor_params:
        param_groups.append({"params": predictor_params, "lr": base_lr, "name": "predictor"})
    if y_backbone_params:
        param_groups.append({
            "params": y_backbone_params,
            "lr": base_lr * y_encoder_lr_multiplier,
            "name": "y_encoder_backbone",
        })
    if y_projection_params:
        param_groups.append({
            "params": y_projection_params,
            "lr": projection_lr,
            "name": "y_encoder_projection",
        })

    return param_groups