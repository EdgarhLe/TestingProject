"""Build phase1_combined.jsonl — training/data/preprocess.py.

Includes InternVid-10M, Panda-70M, and Recap-DataComp-1B (images).

Deduplication & splitting strategy:
  - Exact file paths: deduplicated per source and across all sources.
  - Video‑level splits: for InternVid and Panda‑70M, all clips from the same
    YouTube video are treated as a single group. Groups are randomly split
    into train/val with a fixed seed, ensuring NO clip from a video appears
    in both train and val. This is data‑efficient (all clips kept) and
    prevents near‑duplicate leakage.
  - Images (Recap-DataComp-1B) are split individually (no grouping needed).
  - Shuffling happens within each split before writing, for consistent
    DataLoader behavior.
"""

import argparse
import json
import os
import random
from collections import defaultdict
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import cv2
import yaml
from PIL import Image, UnidentifiedImageError


def _is_likely_valid_jpeg(path: Path) -> bool:
    """Fast JPEG sanity check: SOI/EOI markers only (no full decode)."""
    try:
        with open(path, "rb") as f:
            head = f.read(2)
            if head != b"\xff\xd8":
                return False
            f.seek(-2, os.SEEK_END)
            tail = f.read(2)
            return tail == b"\xff\xd9"
    except OSError:
        return False


def _validate_image(path: Path, mode: str) -> tuple[bool, str]:
    """Validate image with configurable strictness.

    mode=none: no validation beyond existence.
    mode=fast: cheap JPEG marker check.
    mode=strict: PIL parse check via Image.verify().
    """
    if mode == "none":
        return True, ""

    if mode == "fast":
        ok = _is_likely_valid_jpeg(path)
        return ok, "jpeg marker check failed" if not ok else ""

    # mode == "strict"
    try:
        with Image.open(path) as img:
            img.verify()
        return True, ""
    except (UnidentifiedImageError, OSError) as e:
        return False, str(e)


def _validate_video(path: Path) -> tuple[bool, str]:
    """Cheap video sanity check: ensure OpenCV can see at least one frame."""
    capture = cv2.VideoCapture(str(path))
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            return False, "no readable frames"

        ok, _ = capture.read()
        if not ok:
            return False, "first frame unreadable"

        return True, ""
    finally:
        capture.release()


def _delete_corrupt_file(path: Path, label: str, reason: str) -> None:
    """Delete a corrupt media file from disk and report what happened."""
    try:
        path.unlink(missing_ok=True)
        print(f"   [deleted corrupt {label}] {path} ({reason})")
    except OSError as e:
        print(f"   [failed to delete corrupt {label}] {path} ({reason}; {e})")


def _resolve_data_root() -> Path:
    value = os.environ.get("DATA_ROOT")
    if value:
        return Path(value)

    with suppress(ImportError):
        from dotenv import load_dotenv
        load_dotenv()
        value = os.environ.get("DATA_ROOT")
        if value:
            return Path(value)

    env_candidates = [Path(".env"), Path(__file__).resolve().parents[2] / ".env"]
    seen = set()
    for env_path in env_candidates:
        resolved = env_path.resolve()
        if resolved in seen or not env_path.exists():
            continue
        seen.add(resolved)
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            if "=" not in line:
                continue
            key, raw_value = line.split("=", 1)
            if key.strip() != "DATA_ROOT":
                continue
            parsed = raw_value.strip()
            if len(parsed) >= 2 and parsed[0] == parsed[-1] and parsed[0] in {'"', "'"}:
                parsed = parsed[1:-1]
            if parsed:
                return Path(parsed)

    raise SystemExit("DATA_ROOT not set (set it in environment or .env)")


@dataclass
class PreprocessSummary:
    recap_datacomp1b_total: int = 0
    recap_datacomp1b_ok: int = 0
    recap_datacomp1b_missing_file: int = 0
    recap_datacomp1b_duplicate_path: int = 0
    recap_datacomp1b_corrupt_file: int = 0
    panda70m_total: int = 0
    panda70m_ok: int = 0
    panda70m_missing_file: int = 0
    panda70m_duplicate_path: int = 0
    panda70m_corrupt_file: int = 0
    internvid_total: int = 0
    internvid_ok: int = 0
    internvid_missing_file: int = 0
    internvid_duplicate_path: int = 0
    internvid_corrupt_file: int = 0
    total_duplicates_across_sources: int = 0
    video_groups: int = 0               # number of unique YouTube videos in video datasets
    val_video_groups: int = 0           # number assigned to val


def _load_paths(dataset_config_path: str = "configs/dataset.yaml") -> dict:
    data_root = _resolve_data_root()
    with open(dataset_config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cache_subdir = raw["cache"]["subdir"]
    return {
        "data_root": data_root,
        "cache_subdir": cache_subdir,
        "train_jsonl": data_root / raw["output"]["train_jsonl"],
        "val_jsonl": data_root / raw["output"]["val_jsonl"],
        "val_ratio": float(raw["output"].get("val_ratio", 0.05)),
        "recap_datacomp1b_metadata": data_root / raw["sources"]["recap_datacomp1b"]["metadata_path"],
        "panda70m_metadata": data_root / raw["sources"]["panda70m"]["metadata_path"],
        "internvid_metadata": data_root / raw["sources"]["internvid"]["metadata_path"],
    }


def _extract_youtube_id(filename: str) -> str | None:
    stem = Path(filename).stem
    if "_" in stem:
        return stem.split("_", 1)[0]
    return None


def _process_recap_datacomp1b(paths: dict, summary: PreprocessSummary, image_validation: str) -> list[dict]:
    metadata_path = paths["recap_datacomp1b_metadata"]
    if not metadata_path.exists():
        print(f"recap_datacomp1b: no metadata at {metadata_path}, skipping")
        return []

    rows = []
    seen_paths = set()
    with open(metadata_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            caption = (record.get("caption") or "").strip()
            if not record.get("id") or not caption:
                continue

            summary.recap_datacomp1b_total += 1
            rel_path = f"{paths['cache_subdir']}/recap_datacomp1b/{record['id']}.jpg"
            if rel_path in seen_paths:
                summary.recap_datacomp1b_duplicate_path += 1
                continue
            seen_paths.add(rel_path)

            dest = paths["data_root"] / rel_path
            if not dest.exists():
                summary.recap_datacomp1b_missing_file += 1
                print(f"   [missing file] {rel_path}")
                continue

            is_valid, reason = _validate_image(dest, image_validation)
            if not is_valid:
                summary.recap_datacomp1b_corrupt_file += 1
                _delete_corrupt_file(dest, "image", reason)
                continue

            summary.recap_datacomp1b_ok += 1
            rows.append({
                "data_path": rel_path,
                "caption": caption,
                "media_type": "image",
                "num_frames": 1,
                "duration": None,
                "source": "recap_datacomp1b",
            })

    return rows


def _process_panda70m(paths: dict, summary: PreprocessSummary) -> list[dict]:
    metadata_path = paths["panda70m_metadata"]
    if not metadata_path.exists():
        print(f"panda70m: no metadata at {metadata_path}, skipping")
        return []

    rows = []
    seen_paths = set()
    with open(metadata_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            summary.panda70m_total += 1

            rel_path = f"{paths['cache_subdir']}/panda70m/{record['id']}.mp4"
            if rel_path in seen_paths:
                summary.panda70m_duplicate_path += 1
                continue
            seen_paths.add(rel_path)

            dest = paths["data_root"] / rel_path
            if not dest.exists():
                summary.panda70m_missing_file += 1
                print(f"   [missing file] {rel_path}")
                continue

            is_valid, reason = _validate_video(dest)
            if not is_valid:
                summary.panda70m_corrupt_file += 1
                _delete_corrupt_file(dest, "video", reason)
                continue

            summary.panda70m_ok += 1
            rows.append({
                "data_path": rel_path,
                "caption": record["caption"],
                "media_type": "video",
                "num_frames": int(record["num_frames"]),
                "duration": float(record["duration"]),
                "source": "panda70m",
            })

    return rows


def _process_internvid(paths: dict, summary: PreprocessSummary) -> list[dict]:
    metadata_path = paths["internvid_metadata"]
    if not metadata_path.exists():
        print(f"internvid: no metadata at {metadata_path}, skipping")
        return []

    rows = []
    seen_paths = set()
    with open(metadata_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print("   [malformed metadata line] internvid: skipping")
                continue
            summary.internvid_total += 1

            rel_path = f"{paths['cache_subdir']}/internvid/{record['id']}.mp4"
            if rel_path in seen_paths:
                summary.internvid_duplicate_path += 1
                continue
            seen_paths.add(rel_path)

            dest = paths["data_root"] / rel_path
            if not dest.exists():
                summary.internvid_missing_file += 1
                print(f"   [missing file] {rel_path}")
                continue

            is_valid, reason = _validate_video(dest)
            if not is_valid:
                summary.internvid_corrupt_file += 1
                _delete_corrupt_file(dest, "video", reason)
                continue

            summary.internvid_ok += 1
            rows.append({
                "data_path": rel_path,
                "caption": record["caption"],
                "media_type": "video",
                "num_frames": int(record["num_frames"]),
                "duration": float(record["duration"]),
                "source": "internvid",
            })

    return rows


def build_manifest(dataset_config_path: str = "configs/dataset.yaml", image_validation: str = "fast") -> PreprocessSummary:
    paths = _load_paths(dataset_config_path)
    summary = PreprocessSummary()

    # ---------------- Load and deduplicate per source ----------------
    print("=== Recap-DataComp-1B: checking images ===")
    print(f"recap_datacomp1b image validation mode: {image_validation}")
    image_rows = _process_recap_datacomp1b(paths, summary, image_validation=image_validation)
    print(f"recap_datacomp1b: {summary.recap_datacomp1b_ok}/{summary.recap_datacomp1b_total} ok "
            f"({summary.recap_datacomp1b_missing_file} missing, "
            f"{summary.recap_datacomp1b_corrupt_file} corrupt, "
            f"{summary.recap_datacomp1b_duplicate_path} dup paths)")

    print("=== Panda-70M: checking clips ===")
    panda_rows = _process_panda70m(paths, summary)
    print(f"panda70m: {summary.panda70m_ok}/{summary.panda70m_total} ok "
            f"({summary.panda70m_missing_file} missing, {summary.panda70m_corrupt_file} corrupt, "
            f"{summary.panda70m_duplicate_path} dup paths)")

    print("=== InternVid: checking clips ===")
    internvid_rows = _process_internvid(paths, summary)
    print(f"internvid: {summary.internvid_ok}/{summary.internvid_total} ok "
            f"({summary.internvid_missing_file} missing, {summary.internvid_corrupt_file} corrupt, "
            f"{summary.internvid_duplicate_path} dup paths)")

    # ---------------- Combine and exact-path cross-dedup ----------------
    all_rows = image_rows + panda_rows + internvid_rows
    print(f"Total rows before cross-source dedup: {len(all_rows)}")

    seen = set()
    deduped = []
    for row in all_rows:
        p = row["data_path"]
        if p in seen:
            summary.total_duplicates_across_sources += 1
            continue
        seen.add(p)
        deduped.append(row)
    print(f"After exact-path dedup: {len(deduped)} samples "
          f"({summary.total_duplicates_across_sources} duplicates removed)")

    # ---------------- Video‑level train/val split ----------------
    # Separate images (no grouping) from videos (group by YouTube ID)
    images = [r for r in deduped if r["media_type"] == "image"]
    videos = [r for r in deduped if r["media_type"] == "video"]

    # Group videos by YouTube ID
    video_groups = defaultdict(list)
    for v in videos:
        yt_id = _extract_youtube_id(v["data_path"])
        if yt_id is None:
            # fallback: treat each as its own group
            yt_id = v["data_path"]
        video_groups[yt_id].append(v)

    group_ids = list(video_groups.keys())
    summary.video_groups = len(group_ids)

    # Shuffle groups with fixed seed, then split by val_ratio
    random.seed(42)
    random.shuffle(group_ids)
    n_val_groups = int(len(group_ids) * paths["val_ratio"])
    val_group_ids = set(group_ids[:n_val_groups])
    summary.val_video_groups = len(val_group_ids)

    train_videos = []
    val_videos = []
    for gid in group_ids:
        clips = video_groups[gid]
        if gid in val_group_ids:
            val_videos.extend(clips)
        else:
            train_videos.extend(clips)

    print(f"Video groups: {len(group_ids)} total, {len(val_group_ids)} val "
          f"(val ratio = {paths['val_ratio']:.2%})")

    # ---------------- Split images (individual) ----------------
    random.shuffle(images)
    n_val_images = int(len(images) * paths["val_ratio"])
    val_images = images[:n_val_images]
    train_images = images[n_val_images:]

    # Combine
    train_rows = train_images + train_videos
    val_rows = val_images + val_videos

    # Shuffle within each split for a clean DataLoader (deterministic seed)
    random.shuffle(train_rows)
    random.shuffle(val_rows)

    # ---------------- Write JSONL ----------------
    paths["train_jsonl"].parent.mkdir(parents=True, exist_ok=True)
    with open(paths["train_jsonl"], "w", encoding="utf-8") as f:
        for row in train_rows:
            f.write(json.dumps(row) + "\n")

    with open(paths["val_jsonl"], "w", encoding="utf-8") as f:
        for row in val_rows:
            f.write(json.dumps(row) + "\n")

    print(f"\nTraining samples: {len(train_rows)} -> {paths['train_jsonl']}")
    print(f"Validation samples: {len(val_rows)} -> {paths['val_jsonl']}")
    print(f"(Image val: {len(val_images)}, Video val: {len(val_videos)})")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", type=str, default="configs/dataset.yaml")
    parser.add_argument(
        "--image-validation",
        choices=["none", "fast", "strict"],
        default="fast",
        help=(
            "Image validation strictness for recap_datacomp1b: "
            "none=existence only, fast=JPEG marker check, strict=PIL verify"
        ),
    )
    args = parser.parse_args()
    build_manifest(args.dataset_config, image_validation=args.image_validation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())