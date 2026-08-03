"""Bidirectional InfoNCE loss used by VL-JEPA training."""

import torch
import torch.nn.functional as F


DEFAULT_UNIFORMITY_LAMBDA = 0.01


def _uniformity_loss(embeds):
    """Uniformity regularization on normalized embeddings (Wang & Isola, 2020)."""
    embeds = F.normalize(embeds, dim=-1)
    if embeds.shape[0] < 2:
        return embeds.new_zeros(())

    pairwise_sq_dist = torch.pdist(embeds, p=2).pow(2)
    return torch.log(torch.exp(-2 * pairwise_sq_dist).mean())


def bidirectional_infonce_loss(
    pred_embeds,
    target_embeds,
    logit_scale,
    uniformity_lambda=DEFAULT_UNIFORMITY_LAMBDA,
):
    """
    pred_embeds, target_embeds: [B, shared_dim] (S_hat_Y, S_Y in the same batch).
    The diagonal (i, i) is the true positive pair; pairs (i, j != i) in the batch are negatives
    (in-batch negatives, CLIP/InfoNCE style).

    Returns (loss, stats) — stats are for logging/monitoring only and are not used for backward.

    Cast to float32 up front, regardless of the caller's autocast context:
    pred_embeds and target_embeds can arrive with DIFFERENT dtypes even when
    both were produced under the same bf16 autocast region -- ops like
    F.normalize/torch.norm are kept in fp32 by autocast's numerical-stability
    policy while plain Linear layers stay in bf16, so two embeddings that
    took different code paths upstream (e.g. one straight out of a Linear,
    the other through an extra normalize+average step) can legitimately end
    up as different dtypes. logit_scale is also always fp32 (a plain
    nn.Parameter, never autocast). Normalizing dtype here once, rather than
    relying on every caller to keep dtypes consistent, avoids
    "mat1 and mat2 must have the same dtype" RuntimeErrors and matches
    standard practice for contrastive losses under mixed precision (compute
    the loss itself in fp32 even when the forward pass runs in bf16/fp16).
    """
    pred_embeds = pred_embeds.float()
    target_embeds = target_embeds.float()

    pred_norm = F.normalize(pred_embeds, dim=-1)
    target_norm = F.normalize(target_embeds, dim=-1)

    scale = logit_scale.float().exp()
    logits_per_pred = scale * pred_norm @ target_norm.t()      # [B, B] : predictor -> target
    logits_per_target = logits_per_pred.t()                    # [B, B] : target -> predictor

    batch_size = pred_embeds.shape[0]
    labels = torch.arange(batch_size, device=pred_embeds.device)

    loss_pred_to_target = F.cross_entropy(logits_per_pred, labels)
    loss_target_to_pred = F.cross_entropy(logits_per_target, labels)

    pred_uniformity = _uniformity_loss(pred_embeds)
    target_uniformity = _uniformity_loss(target_embeds)
    uniformity_loss = (pred_uniformity + target_uniformity) / 2

    bidirectional_loss = loss_pred_to_target + loss_target_to_pred
    loss = bidirectional_loss + (uniformity_lambda * uniformity_loss)

    with torch.no_grad():
        pred_to_target_acc = (logits_per_pred.argmax(dim=-1) == labels).float().mean()
        target_to_pred_acc = (logits_per_target.argmax(dim=-1) == labels).float().mean()

    stats = {
        "loss_pred_to_target": loss_pred_to_target.item(),
        "loss_target_to_pred": loss_target_to_pred.item(),
        "bidirectional_loss": bidirectional_loss.item(),
        "uniformity_loss": uniformity_loss.item(),
        "uniformity_lambda": float(uniformity_lambda),
        "pred_to_target_acc": pred_to_target_acc.item(),
        "target_to_pred_acc": target_to_pred_acc.item(),
    }
    return loss, stats