"""
model/vl_jepa.py

Assembles the full VL-JEPA pipeline (per Fig. 1 / §3.1 of the VL-JEPA paper,
arXiv:2512.10942):

    X_V --[X-Encoder, frozen]--> S_V  --\
                                          +--[Predictor]--> S_hat_Y  --\
    X_Q (textual query) ----------------/                              +-- L = InfoNCE(S_hat_Y, S_Y)
                                                                       /
    Y (textual target) --[Y-Encoder, LR x0.05]----------------------> S_Y

Training objective (§2 Methodology): bidirectional InfoNCE loss; train the
Predictor and Y-Encoder JOINTLY ("jointly with bi-directional InfoNCE loss,
enabling them to mutually learn from each other"). The X-Encoder is always
frozen and is not included in the optimizer.
"""

import torch
import torch.nn as nn
import math

# Single source of truth for the shared embedding dimension across the whole
# pipeline. Defined here explicitly (not imported from model/predictor or
# model/y_encoder) so this file never silently inherits whichever default one
# of those modules happens to use standalone -- matches configs/model.yaml's
# top-level `embedding_dim: 1536`.
SHARED_EMBED_DIM = 1536

from model.x_encoder.vjepa2 import XEncoder
from model.predictor.gwen2_5_1_5b import (
    MAX_QUERY_LEN,
    VLJEPAPredictor,
    build_model as build_predictor,
)
from model.y_encoder.bge_m3 import (
    DEFAULT_Y_ENCODER_LR_MULTIPLIER,
    MAX_TEXT_LEN,
    YEncoder,
    build_optimizer_param_groups,
    build_y_encoder,
)
from training.losses.info_nce_loss import DEFAULT_UNIFORMITY_LAMBDA, bidirectional_infonce_loss


# =======================================================================
# 1. Model: VLJEPA = X-Encoder (frozen) + Predictor + Y-Encoder
# =======================================================================
class VLJEPA(nn.Module):
    def __init__(self, x_encoder: XEncoder, predictor: VLJEPAPredictor, y_encoder: YEncoder,
                 init_logit_scale=1 / 0.07):
        super().__init__()
        self.x_encoder = x_encoder      # frozen, NOT included in the optimizer
        self.predictor = predictor      # TRAINABLE (8 Qwen2.5 layers + 3 linear projections) -- this is
                                         # the "490M trainable parameters" from the paper, must NOT be frozen
        self.y_encoder = y_encoder      # backbone LR x0.05, projection at the full LR

        # Learnable logit_scale, like CLIP: logits = scale * (P_norm @ T_norm^T)
        # Store the log-scale so it stays positive and trains stably; clamp to prevent scale explosion.
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(float(init_logit_scale))))

        for p in self.x_encoder.parameters():
            p.requires_grad_(False)
        self.x_encoder.eval()

    def clamp_logit_scale(self, max_scale=100.0):
        """Call after every optimizer.step() -- CLIP convention to prevent logit_scale from exploding."""
        with torch.no_grad():
            self.logit_scale.clamp_(max=math.log(torch.tensor(max_scale)))

    def enable_gradient_checkpointing(self):
        """
        Per configs/training.yaml: gradient_checkpointing: true.
        Only enable this on the TRAINABLE backbones (predictor.backbone,
        y_encoder.backbone) -- enabling it on the X-Encoder (always frozen,
        runs inside torch.no_grad()) would be pointless, adding extra compute
        (checkpointing recomputes the forward pass during backward) with no
        VRAM savings since there's no gradient to compute there anyway.

        Note: predictor.backbone receives its input via inputs_embeds (from
        vision_projection/text_projection, both TRAINABLE Linear layers), and
        y_encoder.backbone receives input_ids and embeds them itself using its
        own TRAINABLE embedding layer -> in both cases the input entering the
        checkpointed section already has requires_grad=True, so there's no
        need to additionally call `enable_input_require_grads()` (a hack only
        needed when checkpointing is applied to a section whose input passed
        through a frozen layer right before it).
        """
        self.predictor.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        self.y_encoder.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    # -------------------------------------------------------------
    # Encode each branch separately -- useful for retrieval/classification
    # (no need to always run all 3 encoders at once)
    # -------------------------------------------------------------
    @torch.no_grad()
    def encode_visual(self, video_paths_or_urls):
        """list[str] (path/url) -> visual_embeds [B, Tv, vision_dim] (Tv can differ per video, so
        return a list if shapes are not uniform; here we assume the same num_frames so stacking works)."""
        embeds = [self.x_encoder.encode_video(v) for v in video_paths_or_urls]  # each one [1, Tv, D]
        return torch.cat(embeds, dim=0)   # [B, Tv, D]

    def predict(self, visual_embeds, queries, tokenizer, device):
        """(S_V, X_Q) -> S_hat_Y"""
        query_inputs = tokenizer(
            queries, return_tensors="pt", padding=True, truncation=True, max_length=MAX_QUERY_LEN,
        ).to(device)
        return self.predictor(
            input_ids=query_inputs["input_ids"],
            attention_mask=query_inputs["attention_mask"],
            visual_embeds=visual_embeds,
        )

    def encode_target(self, targets, train_mode=False):
        """Y -> S_Y. train_mode=True keeps gradients (used in the training step),
        train_mode=False uses encode_texts (no_grad, for when you only need the embedding for
        comparison/retrieval)."""
        if train_mode:
            input_ids, attention_mask = self.y_encoder.tokenize(targets)
            return self.y_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.y_encoder.encode_texts(targets)

    def forward(self, visual_embeds, queries, targets, predictor_tokenizer, train_mode=True):
        """A full forward pass: (S_V, X_Q, Y) -> (S_hat_Y, S_Y)."""
        device = visual_embeds.device
        s_hat_y = self.predict(visual_embeds, queries, predictor_tokenizer, device)
        s_y = self.encode_target(targets, train_mode=train_mode)
        return s_hat_y, s_y


def build_vljepa(
    predictor_model_id="Qwen/Qwen2.5-1.5B",
    y_encoder_model_id="BAAI/bge-m3",
    x_encoder_model_id="facebook/vjepa2-vitl-fpc64-256",
    shared_dim=SHARED_EMBED_DIM,
    device="cuda",
    freeze_predictor_backbone=False,   # see VLJEPAPredictor's docstring: defaults to NOT frozen
):
    """Load all 3 real components from HuggingFace. Requires network access and enough VRAM/RAM."""
    x_encoder = XEncoder(hf_repo=x_encoder_model_id, device=device, freeze=True)

    predictor, predictor_tokenizer = build_predictor(
        model_id=predictor_model_id, shared_dim=shared_dim,
        vision_dim=x_encoder.vision_dim, device=device,
        freeze_backbone=freeze_predictor_backbone,
    )

    y_encoder = build_y_encoder(
        model_name=y_encoder_model_id, shared_dim=shared_dim, device=device, freeze_backbone=False,
    )

    model = VLJEPA(x_encoder, predictor, y_encoder).to(device)
    return model, predictor_tokenizer


# =======================================================================
# 2. Loss: bidirectional InfoNCE + uniformity regularization
#    -> imported from training/losses/info_nce_loss.py (issue #46), see the
#    import at the top of this file. Do NOT redefine it here -- an earlier
#    version of this file had a duplicate, simplified bidirectional_infonce_loss
#    (no uniformity_lambda term) defined locally, which silently diverged from
#    the real #46 implementation and configs/training.yaml's
#    loss.uniformity_lambda: 0.01 setting. That duplicate has been removed.
# =======================================================================


# =======================================================================
# 3. Optimizer: param groups with separate LRs for each part
# =======================================================================
def resolve_optimizer_cls(name: str):
    """
    Maps a string name from configs/training.yaml (e.g. 'adamw_8bit') to the
    real optimizer class. bitsandbytes is imported ONLY when actually needed
    (guarded) -- a machine without bitsandbytes can still import this module
    fine, as long as resolve_optimizer_cls("adamw_8bit") is never called.
    """
    name = name.lower()
    if name in ("adamw", "adamw_torch", "adamw_fp32"):
        return torch.optim.AdamW
    if name in ("adamw_8bit", "adamw8bit", "bnb_adamw8bit"):
        try:
            import bitsandbytes as bnb
        except ImportError as e:
            raise ImportError(
                "configs/training.yaml requests optimizer='adamw_8bit' (bitsandbytes) "
                "but bitsandbytes is not installed. Install it with `pip install bitsandbytes`, "
                "or switch the config to 'adamw' to use plain torch.optim.AdamW instead."
            ) from e
        return bnb.optim.AdamW8bit
    raise ValueError(f"Unrecognized optimizer '{name}' in config")


def build_vljepa_optimizer(model: VLJEPA, base_lr,
                            y_encoder_lr_multiplier=DEFAULT_Y_ENCODER_LR_MULTIPLIER,
                            weight_decay=0.01,
                            optimizer_cls=None,
                            optimizer_name=None):
    """
    X-Encoder: frozen -> NO param group at all (no optimizer state cost).
    Predictor (TRAINABLE, see VLJEPA.__init__): base_lr.
    Y-Encoder backbone (pretrained bge-m3): base_lr * y_encoder_lr_multiplier.
    Y-Encoder projection (freshly initialized): base_lr.
    logit_scale: base_lr (a small parameter, grouped with base_lr for simplicity).

    optimizer_cls: pass the class directly (e.g. bnb.optim.AdamW8bit) if already imported.
    optimizer_name: or pass a string from config (e.g. "adamw_8bit") to auto-resolve it.
    Default (nothing passed): plain torch.optim.AdamW.
    """
    if optimizer_cls is None:
        optimizer_cls = resolve_optimizer_cls(optimizer_name) if optimizer_name else torch.optim.AdamW

    param_groups = build_optimizer_param_groups(
        model.predictor, model.y_encoder, base_lr=base_lr,
        y_encoder_lr_multiplier=y_encoder_lr_multiplier,
    )
    param_groups.append({"params": [model.logit_scale], "lr": base_lr, "name": "logit_scale"})

    optimizer = optimizer_cls(param_groups, weight_decay=weight_decay)
    return optimizer


# =======================================================================
# 4. Training step
# =======================================================================
_PRECISION_TO_DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def training_step(model: VLJEPA, optimizer, video_paths, queries, targets,
                   predictor_tokenizer, device="cuda", precision="bf16",
                   uniformity_lambda=DEFAULT_UNIFORMITY_LAMBDA):
    """
    precision: "bf16" (default, matches configs/training.yaml), "fp16", or
    "fp32" (disables autocast entirely). bf16 doesn't need a GradScaler
    (unlike fp16), so there's no scaler logic here -- if you switch to "fp16"
    and see NaN/inf losses, you'll need to add a torch.cuda.amp.GradScaler,
    which isn't wired up yet since the config default is bf16.

    NOTE: this is a simple, non-GradCache demo step (single forward+backward,
    real batch_size = len(video_paths)). For actual Phase 1 training with
    batch_size=1 + gradient_accumulation_steps=16 (configs/training.yaml),
    use training/vljepa_gradcache_step.py's vljepa_gradcache_training_step()
    instead -- see that module's docstring for why naive accumulation would
    break both the InfoNCE alignment term AND the uniformity term here (both
    need a real multi-sample batch, not batch_size=1).
    """
    if precision not in _PRECISION_TO_DTYPE:
        raise ValueError(f"Invalid precision '{precision}', choose one of {list(_PRECISION_TO_DTYPE)}")
    amp_dtype = _PRECISION_TO_DTYPE[precision]
    amp_enabled = device.startswith("cuda") and precision != "fp32"

    model.train()
    model.x_encoder.eval()   # frozen -> always keep it in eval() (disables dropout if any, though it
                              # usually doesn't matter since it's frozen)

    with torch.no_grad():
        visual_embeds = model.encode_visual(video_paths).to(device)

    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
        s_hat_y, s_y = model(visual_embeds, queries, targets, predictor_tokenizer, train_mode=True)
        loss, stats = bidirectional_infonce_loss(
            s_hat_y, s_y, model.logit_scale, uniformity_lambda=uniformity_lambda,
        )

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    model.clamp_logit_scale()

    stats["loss"] = loss.item()
    return stats


# =======================================================================
# Demo (REQUIRES network access and enough GPU/RAM to load the real
# Qwen2.5-1.5B + bge-m3 + V-JEPA2 ViT-L models -> will NOT run in an offline
# environment. See model/tests/test_main.py for an equivalent pipeline that
# runs offline using small mock components.)
# =======================================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, predictor_tokenizer = build_vljepa(device=device)
    model.enable_gradient_checkpointing()   # configs/training.yaml: gradient_checkpointing: true

    # optimizer_name="adamw_8bit" matches configs/training.yaml (bitsandbytes).
    # Switch to optimizer_name="adamw" if the machine doesn't have bitsandbytes or is running on CPU.
    optimizer = build_vljepa_optimizer(model, base_lr=5e-5, optimizer_name="adamw_8bit")

    video_url = ("https://huggingface.co/datasets/nateraw/kinetics-mini/resolve/main/"
                 "val/archery/-Qz25rXdMjE_000014_000024.mp4")

    video_paths = [video_url, video_url]   # simulated batch of 2, reusing the same video for the demo
    queries = ["What action is happening in this video?", "Describe the main activity."]
    targets = ["A person is shooting an arrow with a bow.", "Someone practicing archery."]

    stats = training_step(model, optimizer, video_paths, queries, targets,
                           predictor_tokenizer, device=device, precision="bf16")
    print("Training step stats:", stats)