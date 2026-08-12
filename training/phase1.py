"""
training/phase1.py

Phase 1 entrypoint: general multilingual pretraining, query-free
(configs/training.yaml: phases.phase1.query_conditioned: false).

Curriculum stage transitions (A -> B -> C) are MANUAL, per-week team
decisions, not auto-detected from wall-clock time. This script runs ONE
stage per invocation: rerun it with --stage b once the team decides Stage A
is done, etc.

Data pipeline (training/data/, flat files, no wrapper classes):
  - training/data/download/{cc3m,panda70m,main}.py -- raw metadata/media download.
  - training/data/preprocess.py -- builds phase1_combined.jsonl (run once, before
    training; NOT called from this script).
  - training/data/loader.py -- build_phase1_loader() reads phase1_combined.jsonl,
    yields (video_frames, captions) batches. No prompt ensembling: one raw
    caption per sample.

Checkpointing: two subdirectories under checkpoints.save_dir (configs/training.yaml),
not one flat directory -- resume/ (weights + optimizer, local only) and
deploy/ (weights only). After each save, scripts/sync_checkpoint.sh gets
called to rsync deploy/ to Machine 2 (reads MACHINE2_HOST/REMOTE_DIR from
.env itself) -- pass --skip-rsync to disable this. See save_checkpoint()
and sync_checkpoints_to_machine2() for why the split matters.
"""

import argparse
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml

from model.vl_jepa import build_vljepa, build_vljepa_optimizer, _PRECISION_TO_DTYPE
from training.losses.info_nce_loss import DEFAULT_UNIFORMITY_LAMBDA, bidirectional_infonce_loss
from training.data.loader import build_phase1_loader, build_phase1_val_loader
from training.vljepa_gradcache_step import vljepa_gradcache_training_step

# Global reference to the currently running sync process (none if idle)
_last_sync_process = None


class _TeeStream:
    """Write stream output to multiple destinations (e.g., terminal + log file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def setup_file_logging(log_file_path):
    """Mirror stdout/stderr to a log file while preserving terminal output."""
    log_path = Path(log_file_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = open(log_path, "a", encoding="utf-8", buffering=1)

    sys.stdout = _TeeStream(sys.__stdout__, log_handle)
    sys.stderr = _TeeStream(sys.__stderr__, log_handle)

    print(f"\n===== phase1.py started at {datetime.now().isoformat()} =====")
    print(f"Logging to: {log_path.resolve()}")
    return log_handle


def load_env_file(env_path=".env"):
    """Best-effort .env loader that does not override existing environment variables."""
    path = Path(env_path)
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]

        os.environ.setdefault(key, value)


# =======================================================================
# Config loading -- plain dicts read straight from yaml, no wrapper classes
# =======================================================================
def load_model_config(model_config_path="configs/model.yaml"):
    with open(model_config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_training_config(training_config_path="configs/training.yaml"):
    with open(training_config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_train_jsonl_path(dataset_config_path="configs/dataset.yaml"):
    data_root_value = os.environ.get("DATA_ROOT")
    if not data_root_value:
        raise KeyError(
            "DATA_ROOT is not set. Export it in the shell or add it to .env, "
            "then rerun training/phase1.py."
        )
    data_root = Path(data_root_value)
    with open(dataset_config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return data_root / raw["output"]["train_jsonl"]


# =======================================================================
# Checkpointing
#
# Two separate files per save point, not one:
#   - step_{N}.resume.pt : weights + optimizer state -- needed to resume
#     training on Machine 1, NEVER rsynced (optimizer state is dead weight
#     for Machine 2, which only ever reads weights for indexing/inference).
#   - step_{N}.deploy.pt : weights only -- what actually gets rsynced.
# Splitting these (instead of one file with a keep_optimizer flag controlling
# what gets rsynced) means a resume is always possible from the last local
# save regardless of what's been synced, and the sync payload is smaller
# without extra bookkeeping about which single file serves both purposes.
# =======================================================================
def _model_state_dict(model, keep_x_encoder=False):
    state = {
        "predictor": model.predictor.state_dict(),
        "y_encoder": model.y_encoder.state_dict(),
        "logit_scale": model.logit_scale.data,
    }
    if keep_x_encoder:
        state["x_encoder"] = model.x_encoder.state_dict()
    return state


def save_checkpoint(model, optimizer, step, checkpoint_root, keep_x_encoder=False):
    """
    Writes into TWO subdirectories, not a flat directory:
        {checkpoint_root}/resume/step_{N}.resume.pt   (weights + optimizer, local only)
        {checkpoint_root}/deploy/step_{N}.deploy.pt    (weights only)

    This layout is not arbitrary -- scripts/sync_checkpoint.sh syncs
    training/checkpoints/deploy/ as a WHOLE DIRECTORY (its own default
    CHECKPOINT_DIR), so .resume.pt files must live somewhere that script
    never touches, or optimizer state would get shipped to Machine 2 every
    time regardless of any per-file filtering here.

    Returns (resume_path, deploy_path).
    """
    checkpoint_root = Path(checkpoint_root)
    resume_dir = checkpoint_root / "resume"
    deploy_dir = checkpoint_root / "deploy"
    resume_dir.mkdir(parents=True, exist_ok=True)
    deploy_dir.mkdir(parents=True, exist_ok=True)

    model_state = _model_state_dict(model, keep_x_encoder=keep_x_encoder)

    resume_path = resume_dir / f"step_{step}.resume.pt"
    torch.save({"step": step, **model_state, "optimizer": optimizer.state_dict()}, resume_path)

    deploy_path = deploy_dir / f"step_{step}.deploy.pt"
    torch.save({"step": step, **model_state}, deploy_path)

    return resume_path, deploy_path


def load_checkpoint(model, optimizer, checkpoint_path, device="cuda"):
    """Loads a .resume.pt (has optimizer state) to continue training. A .deploy.pt
    has no optimizer state and isn't meant for this -- use it on Machine 2 instead."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.predictor.load_state_dict(checkpoint["predictor"])
    model.y_encoder.load_state_dict(checkpoint["y_encoder"])
    model.logit_scale.data = checkpoint["logit_scale"].to(device)
    if "optimizer" not in checkpoint:
        raise ValueError(
            f"{checkpoint_path} has no optimizer state -- looks like a .deploy.pt, "
            "not a .resume.pt. Resume from the .resume.pt file for this step instead."
        )
    optimizer.load_state_dict(checkpoint["optimizer"])
    if "x_encoder" in checkpoint:
        model.x_encoder.load_state_dict(checkpoint["x_encoder"])
    return checkpoint["step"]


def sync_checkpoints_to_machine2(checkpoint_root):
    """
    Non‑blocking rsync of the deploy/ directory to Machine 2.

    - Starts scripts/sync_checkpoint.sh as a background subprocess.
    - If a previous sync is still running, this call is silently skipped
      (the next sync will pick up all pending checkpoints).
    - Failures are printed but never raised; training continues uninterrupted.
    """
    global _last_sync_process

    # Check if a previous sync is still running
    if _last_sync_process is not None:
        if _last_sync_process.poll() is None:   # process still alive
            # Already syncing – skip this one; next call will cover it
            return False
        # Process has finished, we can clean up
        _last_sync_process = None

    deploy_dir = str(Path(checkpoint_root) / "deploy")
    env = dict(os.environ)
    env["CHECKPOINT_DIR"] = deploy_dir

    # Start rsync in the background, discarding its output
    try:
        proc = subprocess.Popen(
            ["bash", "scripts/sync_checkpoint.sh"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        _last_sync_process = proc
        return True
    except Exception as e:
        print(f"[rsync] Failed to start sync: {e} -- checkpoint saved locally but NOT synced.")
        return False


# =======================================================================
# Training loop for one curriculum stage
# =======================================================================
def _restartable_batches(loader):
    """Iterates a DataLoader, restarting it (new epoch) whenever it's exhausted, so
    num_steps can exceed one pass over the dataset."""
    while True:
        for batch in loader:
            yield batch

@torch.no_grad()
def run_validation(model, val_loader, device, precision, uniformity_lambda, accumulation_steps=64):
    model.eval()

    all_s_hat_y = []
    all_s_y = []
    total_samples = 0
    amp_dtype = _PRECISION_TO_DTYPE[precision]
    amp_enabled = device.startswith("cuda") and precision != "fp32"

    for i, (video_frames, captions) in enumerate(val_loader):
        if i >= accumulation_steps:
            break

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
            visual_embeds = model.x_encoder.encode_frames(video_frames.to(device))
            s_hat_y = model.predictor(visual_embeds=visual_embeds)

            input_ids, attention_mask = model.y_encoder.tokenize(captions)
            s_y = model.y_encoder(input_ids=input_ids, attention_mask=attention_mask)

        all_s_hat_y.append(s_hat_y.detach().cpu())
        all_s_y.append(s_y.detach().cpu())
        total_samples += video_frames.shape[0]

    if total_samples < 2:
        return {"val_loss": float("nan"), "val_align": float("nan"),
                "val_uniform": float("nan"), "val_pred2tgt_acc": float("nan"),
                "val_tgt2pred_acc": float("nan")}

    # Concatenate all accumulated embeddings
    full_s_hat_y = torch.cat(all_s_hat_y, dim=0)
    full_s_y = torch.cat(all_s_y, dim=0)

    loss, stats = bidirectional_infonce_loss(
        full_s_hat_y, full_s_y, model.logit_scale.detach().cpu(), uniformity_lambda=uniformity_lambda,
    )

    return {
        "val_loss": loss.item(),
        "val_align": stats["bidirectional_loss"],
        "val_uniform": stats["uniformity_loss"],
        "val_pred2tgt_acc": stats["pred_to_target_acc"],
        "val_tgt2pred_acc": stats["target_to_pred_acc"],
    }


@torch.no_grad()
def run_validation_retrieval(model, val_loader, device, precision, num_batches=64):
    """Compute video→text recall@1, @5, @10 on val set."""
    model.eval()
    
    all_video_embeds = []
    all_text_embeds = []
    amp_dtype = _PRECISION_TO_DTYPE[precision]
    amp_enabled = device.startswith("cuda") and precision != "fp32"
    
    for i, (video_frames, captions) in enumerate(val_loader):
        if i >= num_batches:
            break
        
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                # Get vision embeddings: pass directly to predictor
                v_embeds = model.x_encoder.encode_frames(video_frames.to(device))  # [B, N_tokens, vision_dim]
                v_projected = model.predictor(visual_embeds=v_embeds)  # [B, shared_dim]
                
                # Get text embeddings from Y-Encoder
                input_ids, attention_mask = model.y_encoder.tokenize(captions)
                t_embeds = model.y_encoder(input_ids=input_ids.to(device), 
                                          attention_mask=attention_mask.to(device))  # [B, shared_dim]
        
            all_video_embeds.append(v_projected.detach().cpu())
            all_text_embeds.append(t_embeds.detach().cpu())
    
    v_embeds = torch.cat(all_video_embeds, dim=0)  # [N, shared_dim]
    t_embeds = torch.cat(all_text_embeds, dim=0)   # [N, shared_dim]
    
    # Normalize
    v_embeds = v_embeds / v_embeds.norm(dim=1, keepdim=True)
    t_embeds = t_embeds / t_embeds.norm(dim=1, keepdim=True)
    
    # Compute similarity
    sim = v_embeds @ t_embeds.T  # [N, N]
    
    # Video→Text retrieval
    rankings = sim.argsort(dim=1, descending=True)
    labels = torch.arange(len(v_embeds), device=v_embeds.device)
    
    r1 = (rankings[:, 0] == labels).float().mean()
    r5 = ((rankings[:, :5] == labels.unsqueeze(1)).any(dim=1)).float().mean()
    r10 = ((rankings[:, :10] == labels.unsqueeze(1)).any(dim=1)).float().mean()
    
    return {"r@1": r1.item(), "r@5": r5.item(), "r@10": r10.item()}


def run_stage(model, optimizer, loader, stage_name, num_steps, gradient_accumulation_steps,
              precision, uniformity_lambda, checkpoint_root, save_every_n_steps,
              skip_rsync=False, log_every_n_steps=100, start_step=0,
              val_loader=None, val_every_n_steps=500, val_accumulation_steps=64):
    device = next(model.parameters()).device.type
    batch_stream = _restartable_batches(loader)

    # Helper to fetch all micro-batches for one training step
    def fetch_micro_batches():
        return [next(batch_stream) for _ in range(gradient_accumulation_steps)]

    # Fetch the very first batch synchronously (no overlap possible yet)
    current_micro_batches = fetch_micro_batches()

    for step in range(start_step + 1, num_steps + 1):
        # Start loading the next batch in a background thread
        next_batches_holder = []
        prefetch_error = []
        def load_next():
            try:
                next_batches_holder.append(fetch_micro_batches())
            except Exception as e:
                prefetch_error.append(e)
        prefetch_thread = threading.Thread(target=load_next, daemon=True)
        prefetch_thread.start()

        # ---- GPU work with the CURRENT batch ----
        t0 = time.time()
        stats = vljepa_gradcache_training_step(
            model, optimizer, current_micro_batches,
            device=device, precision=precision, uniformity_lambda=uniformity_lambda,
        )
        step_time = time.time() - t0

        # Wait for the background loading to finish (it may already be done)
        prefetch_thread.join()
        if prefetch_error:
            raise RuntimeError(
                f"[{stage_name}] data prefetch failed at step {step}. "
                "See nested exception for root cause."
            ) from prefetch_error[0]
        if not next_batches_holder:
            raise RuntimeError(
                f"[{stage_name}] data prefetch produced no batches at step {step} "
                "without raising an exception."
            )
        # The next batch is now ready in next_batches_holder[0]
        current_micro_batches = next_batches_holder[0]

        # Logging (unchanged)
        if step % log_every_n_steps == 0 or step == num_steps:
            print(
                f"[{stage_name}] step {step}/{num_steps} | loss={stats['loss']:.4f} "
                f"| align={stats['bidirectional_loss']:.4f} | uniform={stats['uniformity_loss']:.4f} "
                f"| pred->tgt acc={stats['pred_to_target_acc']:.3f} "
                f"| tgt->pred acc={stats['target_to_pred_acc']:.3f} | {step_time:.2f}s/step"
            )

        # Checkpointing (unchanged)
        if step % save_every_n_steps == 0:
            resume_path, deploy_path = save_checkpoint(model, optimizer, step, checkpoint_root)
            print(f"[{stage_name}] checkpoint saved: {resume_path}, {deploy_path}")
            if not skip_rsync:
                sync_checkpoints_to_machine2(checkpoint_root)

        # Validation (now uses the passed-in val_loader and val_every_n_steps)
        if val_loader and (step % val_every_n_steps == 0 or step == num_steps):
            val_stats = run_validation(model, val_loader, device, precision, uniformity_lambda, val_accumulation_steps)
            val_retrieval = run_validation_retrieval(model, val_loader, device, precision)
            print(
                f"[{stage_name} VAL] step {step} "
                f"loss={val_stats['val_loss']:.4f} "
                f"R@1={val_retrieval['r@1']:.3f} R@5={val_retrieval['r@5']:.3f} R@10={val_retrieval['r@10']:.3f}"
            )
            del val_stats, val_retrieval
            if device == "cuda":
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            model.train()   # return to training mode

    # Final checkpoint
    save_checkpoint(model, optimizer, num_steps, checkpoint_root)
    print(f"[{stage_name}] final checkpoint saved at step {num_steps}")
    if not skip_rsync:
        sync_checkpoints_to_machine2(checkpoint_root)


# =======================================================================
# --dry-run: validate everything that can fail before a multi-GB model
# download starts. Explicitly does NOT touch build_vljepa(), any HF
# download, optimizer.step(), or the training loop.
# =======================================================================
def run_dry_run(args, model_cfg, training_cfg, train_jsonl_path, frames_per_clip, loader):
    import shutil

    checks = []

    def check(name, passed, detail=""):
        checks.append((name, passed, detail))

    check("DATA_ROOT set", "DATA_ROOT" in os.environ, os.environ.get("DATA_ROOT", "(not set)"))

    x_cfg = model_cfg.get("x_encoder", {})
    y_cfg = model_cfg.get("y_encoder", {})
    for key_name, value in [
        ("predictor.model_name", model_cfg.get("predictor", {}).get("model_name")),
        ("y_encoder.model_name", y_cfg.get("model_name")),
        ("x_encoder.model_name", x_cfg.get("model_name")),
        ("x_encoder.image_size", x_cfg.get("image_size")),
        ("embedding_dim", model_cfg.get("embedding_dim")),
        ("y_encoder.lr_multiplier", y_cfg.get("lr_multiplier")),
    ]:
        check(f"configs/model.yaml has {key_name}", value is not None, str(value))

    base_lr_present = args.base_lr is not None or model_cfg.get("base_learning_rate") is not None
    check("base_learning_rate available (config or --base-lr)", base_lr_present)

    x_encoder_model_id = x_cfg.get("model_name")
    check(
        "x_encoder.model_name is not the known-bad placeholder",
        x_encoder_model_id != "facebook/vjepa2-vitl",
        x_encoder_model_id,
    )

    hw = training_cfg.get("hardware", {})
    for key in ("precision", "gradient_checkpointing", "optimizer", "batch_size", "gradient_accumulation_steps"):
        check(f"configs/training.yaml hardware.{key}", key in hw, hw.get(key))

    check(
        "phases.phase1.query_conditioned == False",
        training_cfg.get("phases", {}).get("phase1", {}).get("query_conditioned") is False,
    )
    ckpt_cfg = training_cfg.get("checkpoints", {})
    for key in ("save_dir", "save_every_n_steps"):
        check(f"configs/training.yaml checkpoints.{key}", key in ckpt_cfg, ckpt_cfg.get(key))
    check(
        "configs/training.yaml loss.uniformity_lambda",
        "uniformity_lambda" in training_cfg.get("loss", {}),
        training_cfg.get("loss", {}).get("uniformity_lambda"),
    )

    check(f"curriculum stage_{args.stage} exists", frames_per_clip is not None,
          f"frames_per_clip={frames_per_clip}")

    jsonl_ok = train_jsonl_path.exists() and train_jsonl_path.stat().st_size > 0
    check("phase1_combined.jsonl exists and is non-empty", jsonl_ok, str(train_jsonl_path))

    try:
        video_frames, captions = next(iter(loader))
        detail = f"video_frames {tuple(video_frames.shape)} {video_frames.dtype}, {len(captions)} captions"
        check("DataLoader builds and yields a valid batch", True, detail)
    except Exception as e:
        check("DataLoader builds and yields a valid batch", False, str(e))

    optimizer_name = hw.get("optimizer", "")
    if optimizer_name.lower() in ("adamw_8bit", "adamw8bit", "bnb_adamw8bit"):
        try:
            import bitsandbytes  # noqa: F401
            check("bitsandbytes importable (optimizer=adamw_8bit)", True)
        except ImportError as e:
            check("bitsandbytes importable (optimizer=adamw_8bit)", False, str(e))
    else:
        check(f"optimizer={optimizer_name!r} (not 8-bit, bitsandbytes not required)", True)

    checkpoint_dir = Path(ckpt_cfg.get("save_dir", "training/checkpoints"))
    try:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        test_file = checkpoint_dir / ".dry_run_write_test"
        test_file.write_text("ok")
        test_file.unlink()
        check("checkpoint save_dir is writable", True, str(checkpoint_dir))
    except OSError as e:
        check("checkpoint save_dir is writable", False, str(e))

    if not args.skip_rsync:
        sync_script = Path("scripts/sync_checkpoint.sh")
        check("scripts/sync_checkpoint.sh exists", sync_script.exists(), str(sync_script))
        check("rsync binary available", shutil.which("rsync") is not None)
        # Best-effort, non-fatal: MACHINE2_HOST may be set in .env (which sync_checkpoint.sh
        # sources itself) rather than the current shell's environment -- a warning here can
        # be a false negative, so this doesn't fail the dry run outright.
        env_path = Path(".env")
        has_machine2_host = "MACHINE2_HOST" in os.environ or (
            env_path.exists() and "MACHINE2_HOST" in env_path.read_text()
        )
        if not has_machine2_host:
            print("[warn] MACHINE2_HOST not found in environment or .env -- "
                  "scripts/sync_checkpoint.sh will fail at actual sync time unless it's set "
                  "before then. Not treated as a dry-run failure since .env may be added later.")

    print(f"[info] torch.cuda.is_available() = {torch.cuda.is_available()} (informational, not pass/fail)")

    print("\n=== Dry-run results ===")
    all_passed = True
    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        all_passed = all_passed and passed
        line = f"[{status}] {name}"
        if detail:
            line += f"  ({detail})"
        print(line)

    print(f"\n{'ALL CHECKS PASSED' if all_passed else 'SOME CHECKS FAILED'} -- "
          "no model was downloaded, no training ran.")
    return all_passed


# =======================================================================
# Entrypoint
# =======================================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["a", "b", "c"], required=True)
    parser.add_argument("--num-steps", type=int, required=True)
    parser.add_argument("--base-lr", type=float, default=None,
                         help="Overrides configs/model.yaml's base_learning_rate if given.")
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--skip-rsync", action="store_true",
                         help="Don't call scripts/sync_checkpoint.sh after each checkpoint.")
    parser.add_argument("--dry-run", action="store_true",
                         help="Validate config/env/data without downloading models or training.")
    parser.add_argument("--model-config", type=str, default="configs/model.yaml")
    parser.add_argument("--training-config", type=str, default="configs/training.yaml")
    parser.add_argument("--dataset-config", type=str, default="configs/dataset.yaml")
    parser.add_argument("--log-file", type=str, default="training/logs/training_log.txt",
                        help="Append all stdout/stderr logs to this file while still printing to terminal.")
    parser.add_argument("--val-jsonl", type=str, default=None,
                    help="Path to a validation JSONL (e.g., phase1_val.jsonl). If not given, no validation.")
    parser.add_argument("--val-every-n-steps", type=int, default=500,
                        help="Run validation every this many steps (if --val-jsonl provided).")
    parser.add_argument("--val-accumulation-steps", type=int, default=64,
                    help="Accumulate this many validation batches before computing the loss "
                         "(so InfoNCE has enough negatives).")
    args = parser.parse_args()

    # Populate env vars from .env when running outside shells that don't export them.
    load_env_file()

    setup_file_logging(args.log_file)

    model_cfg = load_model_config(args.model_config)
    training_cfg = load_training_config(args.training_config)

    if training_cfg["phases"]["phase1"]["query_conditioned"]:
        raise ValueError(
            "configs/training.yaml phases.phase1.query_conditioned is True, but "
            "training/vljepa_gradcache_step.py's query_conditioned=True path is not "
            "implemented yet. Either fix the config back to False for Phase 1, or "
            "implement that path first."
        )

    base_lr = args.base_lr if args.base_lr is not None else model_cfg.get("base_learning_rate")
    if base_lr is None:
        raise ValueError(
            "No base_lr available: pass --base-lr, or add base_learning_rate to configs/model.yaml."
        )
    # PyYAML does NOT parse bare exponential notation like '5e-5' (no decimal point) as a
    # float -- it silently loads as the STRING "5e-5" instead. Cast explicitly.
    base_lr = float(base_lr)

    stage_key = f"stage_{args.stage}"
    if stage_key not in training_cfg["curriculum"]:
        raise ValueError(f"No curriculum stage named {stage_key!r} in configs/training.yaml")
    stage_cfg = training_cfg["curriculum"][stage_key]
    frames_per_clip = stage_cfg["frames_per_clip"]
    print(f"Stage {args.stage.upper()}: frames_per_clip={frames_per_clip} ({stage_cfg.get('description', '')})")

    # Build the DataLoader FIRST -- fail fast instead of only finding out after
    # several minutes spent downloading Qwen2.5-1.5B + bge-m3 + V-JEPA2.
    train_jsonl_path = resolve_train_jsonl_path(args.dataset_config)

    # Validation JSONL from config (default) or CLI override
    if args.val_jsonl:
        val_jsonl_path = Path(args.val_jsonl)
    else:
        # You can either read it directly from dataset.yaml or add a resolve_val_jsonl_path function
        with open(args.dataset_config, encoding="utf-8") as f:
            dataset_cfg = yaml.safe_load(f)
        val_jsonl_path = Path(os.environ["DATA_ROOT"]) / dataset_cfg["output"]["val_jsonl"]

    batch_size = training_cfg["hardware"]["batch_size"]
    num_workers = args.num_workers

    train_loader = build_phase1_loader(train_jsonl_path, frames_per_clip, batch_size, num_workers)

    val_loader = None
    if args.val_jsonl or (dataset_cfg["output"].get("val_jsonl") is not None):
        val_loader = build_phase1_val_loader(val_jsonl_path, frames_per_clip, batch_size, num_workers)

    if args.dry_run:
        import sys
        passed = run_dry_run(args, model_cfg, training_cfg, train_jsonl_path, frames_per_clip, train_loader)
        sys.exit(0 if passed else 1)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    x_encoder_model_id = model_cfg["x_encoder"]["model_name"]
    if x_encoder_model_id == "facebook/vjepa2-vitl":
        print(
            "WARNING: configs/model.yaml x_encoder.model_name is 'facebook/vjepa2-vitl' -- "
            "the confirmed working repo is 'facebook/vjepa2-vitl-fpc64-256'. This WILL "
            "fail to load -- update configs/model.yaml."
        )

    model, _ = build_vljepa(
        predictor_model_id=model_cfg["predictor"]["model_name"],
        y_encoder_model_id=model_cfg["y_encoder"]["model_name"],
        x_encoder_model_id=x_encoder_model_id,
        shared_dim=model_cfg["embedding_dim"],
        device=device,
    )

    if training_cfg["hardware"]["gradient_checkpointing"]:
        model.enable_gradient_checkpointing()

    optimizer = build_vljepa_optimizer(
        model,
        base_lr=base_lr,
        y_encoder_lr_multiplier=model_cfg["y_encoder"]["lr_multiplier"],
        optimizer_name=training_cfg["hardware"]["optimizer"],
    )

    val_accumulation_steps = args.val_accumulation_steps

    start_step = 0
    if args.resume_from:
        start_step = load_checkpoint(model, optimizer, args.resume_from, device=device)
        print(f"Resumed from {args.resume_from} at step {start_step}")

    run_stage(
        model, optimizer, train_loader,
        stage_name=f"phase1_{stage_key}",
        num_steps=args.num_steps,
        gradient_accumulation_steps=training_cfg["hardware"]["gradient_accumulation_steps"],
        precision=training_cfg["hardware"]["precision"],
        uniformity_lambda=training_cfg["loss"]["uniformity_lambda"],
        checkpoint_root=training_cfg["checkpoints"]["save_dir"],
        save_every_n_steps=training_cfg["checkpoints"]["save_every_n_steps"],
        skip_rsync=args.skip_rsync,
        start_step=start_step,
        val_loader=val_loader,                # the loader built earlier
        val_every_n_steps=args.val_every_n_steps,   # from CLI
        val_accumulation_steps=val_accumulation_steps,
    )


if __name__ == "__main__":
    main()