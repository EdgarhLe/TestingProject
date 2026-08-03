"""
tests/unit/test_main.py

Standalone unit tests for the bidirectional InfoNCE loss in training/losses/info_nce_loss.py
(as required by Issue #46):
    1. Loss is finite and stats are present.
    2. Loss is symmetric when swapping predictor/target embeddings.
    3. Loss decreases on toy data when positive pairs are aligned.
"""

import torch
import pytest

from training.losses.info_nce_loss import bidirectional_infonce_loss


def test_bidirectional_infonce_loss_is_finite_and_reports_stats():
    pred_embeds = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    target_embeds = torch.tensor(
        [[0.9, 0.1, 0.0], [0.1, 0.9, 0.0]],
        dtype=torch.float32,
    )

    loss, stats = bidirectional_infonce_loss(
        pred_embeds,
        target_embeds,
        logit_scale=torch.tensor(0.0),
        uniformity_lambda=0.01,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert isinstance(stats, dict)
    for key in [
        "loss_pred_to_target",
        "loss_target_to_pred",
        "bidirectional_loss",
        "uniformity_loss",
        "pred_to_target_acc",
        "target_to_pred_acc",
    ]:
        assert key in stats
        assert stats[key] == pytest.approx(stats[key])


def test_bidirectional_infonce_loss_is_symmetric():
    pred_embeds = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    target_embeds = torch.tensor(
        [[0.8, 0.2, 0.0], [0.1, 0.9, 0.0], [0.0, 0.2, 0.8]],
        dtype=torch.float32,
    )

    forward_loss, _ = bidirectional_infonce_loss(
        pred_embeds,
        target_embeds,
        logit_scale=torch.tensor(0.2),
        uniformity_lambda=0.03,
    )
    reverse_loss, _ = bidirectional_infonce_loss(
        target_embeds,
        pred_embeds,
        logit_scale=torch.tensor(0.2),
        uniformity_lambda=0.03,
    )

    assert forward_loss.item() == pytest.approx(reverse_loss.item(), rel=1e-6, abs=1e-6)


def test_bidirectional_infonce_loss_decreases_when_pairs_are_aligned():
    aligned_pred = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0]],
        dtype=torch.float32,
    )
    aligned_target = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0]],
        dtype=torch.float32,
    )
    misaligned_target = torch.tensor(
        [[0.0, 1.0], [1.0, 0.0]],
        dtype=torch.float32,
    )

    aligned_loss, _ = bidirectional_infonce_loss(
        aligned_pred,
        aligned_target,
        logit_scale=torch.tensor(2.0),
        uniformity_lambda=0.0,
    )
    misaligned_loss, _ = bidirectional_infonce_loss(
        aligned_pred,
        misaligned_target,
        logit_scale=torch.tensor(2.0),
        uniformity_lambda=0.0,
    )

    assert aligned_loss.item() < misaligned_loss.item()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
