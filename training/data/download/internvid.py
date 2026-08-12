"""InternVid-10M download — training/data/download/internvid.py

Downloads a filtered subset of InternVid-10M based on aesthetic score.
Clips are cut from source YouTube videos using the provided timestamps.
Disk‑friendly: only a small LRU cache of source videos is kept.
"""

import json
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yt_dlp
from datasets import load_dataset

from training.data.download.panda70m import cut_clip, probe_clip


def _ffmpeg_exe() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


@dataclass
class DownloadSummary:
    total: int = 0
    downloaded: int = 0
    skipped: int = 0
    failed: int = 0
    broken_cut: int = 0


# How many source videos to keep on disk at once (LRU cache)
_MAX_CACHED_SOURCES = 50


def _download_source(url: str, dest_dir: Path, video_id: str) -> Optional[Path]:
    """Download a source video into dest_dir. Returns path or None on failure."""
    ydl_opts = {
        "format": "best[height<=480][ext=mp4]/best",
        "outtmpl": str(dest_dir / f"{video_id}.%(ext)s"),
        "ffmpeg_location": str(Path(_ffmpeg_exe()).parent),
        "quiet": True,
        "no_warnings": True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            downloaded = dest_dir / f"{info['id']}.{info.get('ext', 'mp4')}"
            return downloaded
    except Exception:
        return None


def download_internvid(
    data_root: Path,
    max_videos: int = 50_000,
    min_aesthetic_score: float = 0.55,
    resume: bool = True,
    cookies_from_browser: Optional[str] = None,
) -> DownloadSummary:
    clip_dir = data_root / "cache" / "internvid"
    clip_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = data_root / "internvid_metadata.jsonl"
    summary = DownloadSummary()

    # Resume: load already completed clip IDs
    completed_ids = set()
    if resume and metadata_path.is_file():
        with open(metadata_path, encoding="utf-8") as f:
            for line in f:
                try:
                    completed_ids.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    pass

    print("Loading InternVid-10M metadata (streaming)...")
    dataset = load_dataset(
        "OpenGVLab/InternVid-10M-FLT",
        "InternVid-18M-AES",
        split="AES_vc2vicuna",
        streaming=True,
        token=True,
    )

    # Temporary directory for downloaded source videos
    tmp_source_dir = data_root / ".tmp_internvid_sources"
    tmp_source_dir.mkdir(parents=True, exist_ok=True)

    # LRU cache: video_id -> source file path
    source_cache = OrderedDict()

    file_mode = "a" if resume else "w"
    with open(metadata_path, file_mode, encoding="utf-8") as meta_f:
        for row in dataset:
            if summary.total >= max_videos:
                break

            aesthetic = row.get("Aesthetic_Score", 0)
            if aesthetic < min_aesthetic_score:
                continue

            youtube_id = row["YoutubeID"]
            start = row["Start_timestamp"]
            end = row["End_timestamp"]
            caption = row["Caption"].strip()
            if not caption:
                continue

            # Unique clip ID (sanitise timestamps for filenames)
            safe_start = start.replace(":", "-").replace(".", "-")
            safe_end = end.replace(":", "-").replace(".", "-")
            clip_id = f"{youtube_id}_{safe_start}_{safe_end}"

            summary.total += 1

            if clip_id in completed_ids:
                summary.skipped += 1
                continue

            # --- get source video (cached) ---
            if youtube_id not in source_cache:
                # evict oldest if cache is full
                while len(source_cache) >= _MAX_CACHED_SOURCES:
                    old_id, old_path = source_cache.popitem(last=False)
                    if old_path and old_path.exists():
                        old_path.unlink()
                url = f"https://www.youtube.com/watch?v={youtube_id}"
                source_path = _download_source(url, tmp_source_dir, youtube_id)
                if source_path is None:
                    summary.failed += 1
                    continue
                source_cache[youtube_id] = source_path
            else:
                # move to end (most recently used)
                source_cache.move_to_end(youtube_id)
                source_path = source_cache[youtube_id]

            # --- cut the segment ---
            clip_path = clip_dir / f"{clip_id}.mp4"
            if not clip_path.exists():
                success = cut_clip(source_path, start, end, clip_path)
                if not success:
                    summary.broken_cut += 1
                    continue

            # --- probe final clip ---
            probed = probe_clip(clip_path)
            if probed is None:
                summary.broken_cut += 1
                clip_path.unlink(missing_ok=True)
                continue
            num_frames, duration = probed

            meta_f.write(json.dumps({
                "id": clip_id,
                "caption": caption,
                "num_frames": num_frames,
                "duration": duration,
                "source": "internvid",
            }) + "\n")
            meta_f.flush()
            summary.downloaded += 1

    # Clean up remaining cached sources
    for p in source_cache.values():
        if p and p.exists():
            p.unlink()
    return summary