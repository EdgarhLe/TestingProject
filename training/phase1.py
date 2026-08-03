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
import time
from pathlib import Path

import torch
import yaml

from model.vl_jepa import build_vljepa, build_vljepa_optimizer
from training.data.loader import build_phase1_loader
from training.vljepa_gradcache_step import vljepa_gradcache_training_step


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
    data_root = Path(os.environ["DATA_ROOT"])
    with open(dataset_config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return data_root / raw["output"]["train_jsonl"]


# =======================================================================
# Data readiness (training/data/check_readiness.py, if it exists -- never
# seen its real content in this conversation, best-effort call)
# =======================================================================
def check_readiness(stage):
    result = subprocess.run(
        ["python", "training/data/check_readiness.py", "--stage", f"stage_{stage}"],
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"training/data/check_readiness.py --stage stage_{stage} failed "
            f"(exit code {result.returncode}). Fix data readiness before training, "
            "or pass --skip-readiness-check to bypass (not recommended)."
        )


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
    Shells out to scripts/sync_checkpoint.sh -- does NOT reimplement rsync
    directly here. That script already handles: reading MACHINE2_HOST/
    REMOTE_DIR from .env, validating both are set, checking the local
    checkpoint dir exists, and the actual rsync invocation with the flags
    Machine 2 (a Windows box, per REMOTE_DIR's /c/Users/... default) needs
    (--no-perms --no-owner --no-group). Duplicating any of that here would
    just be a second, divergent copy of logic that already works.

    Syncs the ENTIRE deploy/ directory each call, not a single file -- rsync
    only transfers deltas, so calling this after every checkpoint save is
    cheap, not wasteful.

    Best-effort: a sync failure (e.g. .env missing, Machine 2 unreachable)
    prints a warning and returns False rather than raising -- the checkpoint
    is already safe locally regardless, and training shouldn't die because
    Machine 2 is briefly unreachable. Re-run scripts/sync_checkpoint.sh
    manually later to catch up.
    """
    env = dict(os.environ)
    env["CHECKPOINT_DIR"] = str(Path(checkpoint_root) / "deploy")

    result = subprocess.run(["bash", "scripts/sync_checkpoint.sh"], env=env)
    if result.returncode != 0:
        print(f"[rsync] scripts/sync_checkpoint.sh failed (exit {result.returncode}) -- "
              "checkpoint saved locally but NOT synced to Machine 2 this time.")
        return False
    return True


# =======================================================================
# Training loop for one curriculum stage
# =======================================================================
def _restartable_batches(loader):
    """Iterates a DataLoader, restarting it (new epoch) whenever it's exhausted, so
    num_steps can exceed one pass over the dataset."""
    while True:
        for batch in loader:
            yield batch


def run_stage(model, optimizer, loader, stage_name, num_steps, gradient_accumulation_steps,
              precision, uniformity_lambda, checkpoint_root, save_every_n_steps,
              skip_rsync=False, log_every_n_steps=10, start_step=0):
    device = next(model.parameters()).device.type
    batch_stream = _restartable_batches(loader)

    for step in range(start_step + 1, num_steps + 1):
        raw_micro_batches = [next(batch_stream) for _ in range(gradient_accumulation_steps)]

        t0 = time.time()
        stats = vljepa_gradcache_training_step(
            model, optimizer, raw_micro_batches,
            device=device, precision=precision, uniformity_lambda=uniformity_lambda,
        )
        step_time = time.time() - t0

        if step % log_every_n_steps == 0 or step == num_steps:
            print(
                f"[{stage_name}] step {step}/{num_steps} | loss={stats['loss']:.4f} "
                f"| align={stats['bidirectional_loss']:.4f} | uniform={stats['uniformity_loss']:.4f} "
                f"| pred->tgt acc={stats['pred_to_target_acc']:.3f} "
                f"| tgt->pred acc={stats['target_to_pred_acc']:.3f} | {step_time:.2f}s/step"
            )

        if step % save_every_n_steps == 0:
            resume_path, deploy_path = save_checkpoint(model, optimizer, step, checkpoint_root)
            print(f"[{stage_name}] checkpoint saved: {resume_path}, {deploy_path}")
            if not skip_rsync:
                sync_checkpoints_to_machine2(checkpoint_root)

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
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--skip-rsync", action="store_true",
                         help="Don't call scripts/sync_checkpoint.sh after each checkpoint.")
    parser.add_argument("--skip-readiness-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                         help="Validate config/env/data without downloading models or training.")
    parser.add_argument("--model-config", type=str, default="configs/model.yaml")
    parser.add_argument("--training-config", type=str, default="configs/training.yaml")
    parser.add_argument("--dataset-config", type=str, default="configs/dataset.yaml")
    args = parser.parse_args()

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

    if not args.skip_readiness_check:
        check_readiness(args.stage)

    stage_key = f"stage_{args.stage}"
    if stage_key not in training_cfg["curriculum"]:
        raise ValueError(f"No curriculum stage named {stage_key!r} in configs/training.yaml")
    stage_cfg = training_cfg["curriculum"][stage_key]
    frames_per_clip = stage_cfg["frames_per_clip"]
    print(f"Stage {args.stage.upper()}: frames_per_clip={frames_per_clip} ({stage_cfg.get('description', '')})")

    # Build the DataLoader FIRST -- fail fast instead of only finding out after
    # several minutes spent downloading Qwen2.5-1.5B + bge-m3 + V-JEPA2.
    train_jsonl_path = resolve_train_jsonl_path(args.dataset_config)
    loader = build_phase1_loader(
        jsonl_path=train_jsonl_path,
        curriculum_frames=frames_per_clip,
        batch_size=training_cfg["hardware"]["batch_size"],
        num_workers=args.num_workers,
    )

    if args.dry_run:
        import sys
        passed = run_dry_run(args, model_cfg, training_cfg, train_jsonl_path, frames_per_clip, loader)
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

    start_step = 0
    if args.resume_from:
        start_step = load_checkpoint(model, optimizer, args.resume_from, device=device)
        print(f"Resumed from {args.resume_from} at step {start_step}")

    run_stage(
        model, optimizer, loader,
        stage_name=f"phase1_{stage_key}",
        num_steps=args.num_steps,
        gradient_accumulation_steps=training_cfg["hardware"]["gradient_accumulation_steps"],
        precision=training_cfg["hardware"]["precision"],
        uniformity_lambda=training_cfg["loss"]["uniformity_lambda"],
        checkpoint_root=training_cfg["checkpoints"]["save_dir"],
        save_every_n_steps=training_cfg["checkpoints"]["save_every_n_steps"],
        skip_rsync=args.skip_rsync,
        start_step=start_step,
    )


if __name__ == "__main__":
    main()