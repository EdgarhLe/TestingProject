"""
training/gradcache.py

GradCache-style gradient accumulation for contrastive (InfoNCE) losses.

Why this file exists
---------------------
Standard gradient accumulation (forward + backward each micro-batch, let
.grad accumulate, call optimizer.step() every N micro-batches) works fine for
losses that are computed independently per-sample, like language modeling
cross-entropy. It does NOT work correctly for in-batch-negative contrastive
losses (InfoNCE) when the micro-batch size is smaller than what you need for
useful negatives.

configs/training.yaml uses batch_size=1 with gradient_accumulation_steps=16.
With batch_size=1, a naive per-micro-batch InfoNCE loss has a 1x1 logits
matrix -> zero negatives -> the loss is trivially ~0 every step and no real
contrastive learning signal ever reaches the model, even though training logs
would look "normal" (loss decreasing, no errors). This would silently waste
the whole Phase 1 run.

GradCache (Gao, Yao, Callan -- "Scaling Deep Contrastive Learning Batch Size
under Memory Limited Setup", 2021) fixes this while keeping peak memory
bounded to a single micro-batch's activations, via 2 passes:

  Pass 1 (representation pass, NO autograd graph):
      Forward every micro-batch under torch.no_grad() to get its
      representations (s_hat_y, s_y). Clone each into a fresh leaf tensor
      with requires_grad=True (detached from the model entirely). Concatenate
      all micro-batch representations into one full "effective batch" and
      compute the REAL contrastive loss on it (so all 16 micro-batches worth
      of negatives are present at once). Backward ONLY as far as these leaf
      tensors, giving d(loss)/d(representation) per micro-batch. This never
      touches the model's parameters and never builds the model's
      computation graph, so it's cheap.

  Pass 2 (gradient pass, ONE micro-batch of real graph at a time):
      Re-forward each micro-batch WITH the model's real computation graph
      (one micro-batch at a time, so peak memory = 1 micro-batch of
      activations, not 16), then call
      `torch.autograd.backward([s_hat_y_i, s_y_i], grad_tensors=[...])` using
      the cached d(loss)/d(representation) from pass 1. This propagates the
      correct gradient into the model's parameters via the chain rule.
      .grad accumulates across the 16 micro-batches exactly like normal
      gradient accumulation, but now it's mathematically IDENTICAL to having
      computed the loss on the full concatenated batch of 16 and calling
      .backward() once -- just computed in a memory-bounded way.

Correctness requirement: dropout (or any other source of randomness inside
the model's forward) must produce IDENTICAL masks in pass 1 and pass 2 for
the same micro-batch, otherwise pass 2's gradient doesn't correspond to the
representation pass 1 actually used. We handle this by snapshotting the RNG
state right before each micro-batch's pass-1 forward, and restoring that
exact state before that micro-batch's pass-2 forward.
"""

import torch


# ---------------------------------------------------------------------
# RNG state snapshot/restore, so dropout masks match between pass 1 and pass 2
# ---------------------------------------------------------------------
def _get_rng_state(device):
    state = {"cpu": torch.random.get_rng_state()}
    if isinstance(device, torch.device) and device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    elif isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state


def _set_rng_state(state, device):
    torch.random.set_rng_state(state["cpu"])
    if "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"], device)


# ---------------------------------------------------------------------
# Core GradCache step
# ---------------------------------------------------------------------
def gradcache_infonce_step(micro_batches, forward_fn, loss_fn, optimizer, device="cuda"):
    """
    micro_batches: list of length `accumulation_steps`. Each element is a single
        micro-batch in whatever format forward_fn expects (e.g. a dict with
        video_paths/queries/targets). This function is agnostic to that format
        -- it never inspects micro_batches itself, only passes each one to
        forward_fn. That's what makes it independent of the DataLoader's
        exact schema (#49): once the DataLoader lands, `forward_fn` is the
        only piece that needs to know its interface.

    forward_fn(micro_batch) -> (s_hat_y, s_y): runs the model's forward on
        ONE micro-batch and returns its (predictor_output, target_output)
        pair. Must be callable both under torch.no_grad() (pass 1) and with
        gradients enabled (pass 2) -- i.e. it should not itself force
        no_grad internally for the parts that need to be trained. It's fine
        (and expected) for forward_fn to internally use torch.no_grad() for
        any genuinely frozen sub-component, e.g. the X-Encoder.

    loss_fn(s_hat_y_full, s_y_full) -> (loss, stats): the real contrastive
        loss (e.g. bidirectional_infonce_loss), computed on the FULL
        concatenated batch across all micro-batches, so it sees
        len(micro_batches) worth of negatives.

    optimizer: the optimizer whose .zero_grad()/.step() bracket this
        accumulation window. Gradients accumulate into model.parameters().grad
        across all micro-batches in pass 2; optimizer.step() is called once
        at the end, exactly like normal gradient accumulation.

    Returns: stats dict from loss_fn, plus "loss" (float).
    """
    optimizer.zero_grad(set_to_none=True)

    # ---- Pass 1: representation pass (no model graph) ----
    rng_states = []
    detached_s_hat_y, detached_s_y = [], []

    for micro_batch in micro_batches:
        rng_states.append(_get_rng_state(device))
        with torch.no_grad():
            s_hat_y, s_y = forward_fn(micro_batch)
        detached_s_hat_y.append(s_hat_y.clone().requires_grad_(True))
        detached_s_y.append(s_y.clone().requires_grad_(True))

    full_s_hat_y = torch.cat(detached_s_hat_y, dim=0)
    full_s_y = torch.cat(detached_s_y, dim=0)

    loss, stats = loss_fn(full_s_hat_y, full_s_y)
    loss.backward()   # only reaches the detached leaves above -- cheap, no model graph involved

    # NOTE: full_s_hat_y / full_s_y are the OUTPUT of torch.cat, so they are
    # non-leaf tensors -- PyTorch does not populate .grad on non-leaf tensors
    # by default (it would silently be None; accessing it also raises a
    # UserWarning). The actual per-micro-batch gradients live on the original
    # leaves in detached_s_hat_y / detached_s_y instead (torch.cat's backward
    # correctly scatters the gradient back to each leaf that fed into it), so
    # read .grad from there directly -- no need to re-split anything.
    grad_s_hat_y_chunks = [t.grad for t in detached_s_hat_y]
    grad_s_y_chunks = [t.grad for t in detached_s_y]

    # ---- Pass 2: gradient pass, one micro-batch of real graph at a time ----
    for micro_batch, rng_state, grad_s_hat_y_i, grad_s_y_i in zip(
        micro_batches, rng_states, grad_s_hat_y_chunks, grad_s_y_chunks
    ):
        _set_rng_state(rng_state, device)   # reproduce the exact dropout masks pass 1 used
        s_hat_y_i, s_y_i = forward_fn(micro_batch)   # real graph, ONE micro-batch only
        torch.autograd.backward(
            [s_hat_y_i, s_y_i], grad_tensors=[grad_s_hat_y_i, grad_s_y_i]
        )
        # s_hat_y_i / s_y_i and their graph go out of scope here and get freed
        # before the next micro-batch's forward -- this is what bounds peak
        # memory to a single micro-batch.

    optimizer.step()
    stats["loss"] = loss.item()
    return stats