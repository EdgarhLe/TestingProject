"""Panda-70M download and clip preparation — training/data/download/panda70m.py.

CLI lives in training/data/download/main.py (shared with cc3m), not here.
"""

import ast
import json
import subprocess
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

# multimodalart/panda-70m on HuggingFace mirrors the official Panda-70M CSVs;
# train_2m is the largest split actually populated on this mirror (800K real
# source videos) — the raw `train`/`train_10m` splits are near-empty there.
_PANDA70M_ROWS_API = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=multimodalart%2Fpanda-70m&config=default&split=train_2m"
    "&offset={offset}&length={length}"
)
_HF_API_PAGE_SIZE = 100  # datasets-server's hard per-request cap
_MAX_RETRIES = 5
_BACKOFF_SECONDS = 2.0  # doubles each retry: 2s, 4s, 8s, 16s, 32s


def _ffmpeg_exe() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


@dataclass(frozen=True)
class ClipTask:
    clip_id: str
    video_id: str
    youtube_url: str
    start: str
    end: str
    caption: str


@dataclass
class PrepSummary:
    total_tasks: int = 0
    broken_download: int = 0
    broken_clip: int = 0
    empty_caption: int = 0
    ok: int = 0


def _fetch_page(url: str, timeout: float = 30.0, log_prefix: str = "panda70m") -> dict | None:
    """Fetch one page, retrying transient failures with exponential backoff.
    Returns None (not raises) if every attempt fails -- the caller decides
    whether that's fatal.

    429 (Too Many Requests) is handled separately: honors a Retry-After
    header if the server sends one, and backs off 3x longer by default
    otherwise -- retrying at the same short interval as a generic 502 just
    triggers another 429 immediately. Same helper as cc3m.py's, since both
    hit the same datasets-server API and can hit the same rate limit.
    """
    last_error: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            last_error = e
            if e.code == 429:
                retry_after = e.headers.get("Retry-After") if e.headers else None
                try:
                    wait = float(retry_after) if retry_after else _BACKOFF_SECONDS * (2 ** (attempt - 1)) * 3
                except ValueError:
                    wait = _BACKOFF_SECONDS * (2 ** (attempt - 1)) * 3
                print(f"{log_prefix}: rate limited (429), waiting {wait:.0f}s (attempt {attempt}/{_MAX_RETRIES})")
            else:
                wait = _BACKOFF_SECONDS * (2 ** (attempt - 1))
                print(f"{log_prefix}: request failed (attempt {attempt}/{_MAX_RETRIES}): {e}")
            if attempt < _MAX_RETRIES:
                time.sleep(wait)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_error = e
            print(f"{log_prefix}: request failed (attempt {attempt}/{_MAX_RETRIES}): {e}")
            if attempt < _MAX_RETRIES:
                time.sleep(_BACKOFF_SECONDS * (2 ** (attempt - 1)))
    print(f"{log_prefix}: giving up on this page after {_MAX_RETRIES} attempts ({last_error}) -- "
          "keeping whatever rows were already fetched.")
    return None


def fetch_real_panda70m_rows(n: int) -> list[dict[str, Any]]:
    """Page through the real Panda-70M HF mirror via the datasets-server
    REST API — public, no auth needed. A page that fails after retries stops
    paging rather than crashing the whole run -- returns whatever rows were
    successfully fetched before the failure."""
    rows: list[dict[str, Any]] = []
    while len(rows) < n:
        page_len = min(_HF_API_PAGE_SIZE, n - len(rows))
        url = _PANDA70M_ROWS_API.format(offset=len(rows), length=page_len)

        payload = _fetch_page(url)
        if payload is None:
            break  # give up on further paging, keep what we have

        page_rows = [r["row"] for r in payload.get("rows", [])]
        if not page_rows:
            break
        rows.extend(page_rows)

    if len(rows) < n:
        print(f"panda70m: got {len(rows)}/{n} rows requested (a page request may have failed)")
    return rows[:n]


def expand_to_clip_tasks(rows: list[dict[str, Any]]) -> list[ClipTask]:
    """Expand each source-video row into one ClipTask per sub-clip.

    timestamp/caption are Python-list-literal strings in the raw CSV/HF
    mirror (e.g. "[['0:00:16.300', '0:00:32.566'], ...]"), NOT JSON —
    ast.literal_eval, not json.loads (verified against the real mirror).
    """
    tasks: list[ClipTask] = []
    for row in rows:
        video_id = row["videoID"]
        try:
            timestamps = ast.literal_eval(row["timestamp"])
            captions = ast.literal_eval(row["caption"])
        except (ValueError, SyntaxError):
            continue

        for index, ((start, end), caption) in enumerate(zip(timestamps, captions)):
            tasks.append(
                ClipTask(
                    clip_id=f"{video_id}_{index:03d}",
                    video_id=video_id,
                    youtube_url=row["url"],
                    start=start,
                    end=end,
                    caption=str(caption).strip(),
                )
            )
    return tasks


def _existing_video_file(dest_dir: Path, video_id: str) -> Path | None:
    """A source video already downloaded by a previous (interrupted) run —
    reused as-is so a resumed run never re-downloads it."""
    for candidate in dest_dir.glob(f"{video_id}.*"):
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def download_source_video(youtube_url: str, dest_dir: Path, cookies_from_browser: str | None = None) -> Path | None:
    import yt_dlp

    dest_dir.mkdir(parents=True, exist_ok=True)

    video_id = youtube_url.rstrip("/").rsplit("=", 1)[-1]
    existing = _existing_video_file(dest_dir, video_id)
    if existing is not None:
        return existing

    ydl_opts = {
        "format": "18/best[height<=480][ext=mp4]/best",
        "outtmpl": str(dest_dir / "%(id)s.%(ext)s"),
        "ffmpeg_location": str(Path(_ffmpeg_exe()).parent),
        "quiet": True,
        "no_warnings": True,
    }
    if cookies_from_browser:
        # Real logged-in browser cookies make requests look like an
        # authenticated human, not anonymous automated scraping — YouTube's
        # "Sign in to confirm you're not a bot" block is IP+request-pattern
        # based, seen for real in this project's own testing.
        ydl_opts["cookiesfrombrowser"] = (cookies_from_browser,)
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
        return dest_dir / f"{info['id']}.{info.get('ext', 'mp4')}"
    except Exception:
        return None


def cut_clip(source_video: Path, start: str, end: str, output_path: Path) -> bool:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        _ffmpeg_exe(), "-y", "-nostdin",
        "-i", str(source_video),
        "-ss", start, "-to", end,
        "-c:v", "libx264", "-c:a", "aac",
        "-loglevel", "quiet",
        str(output_path),
    ]
    result = subprocess.run(command, capture_output=True)
    return result.returncode == 0 and output_path.is_file() and output_path.stat().st_size > 0


def probe_clip(clip_path: Path) -> tuple[int, float] | None:
    """Return (num_frames, duration_seconds) if the clip actually opens and
    has real content — None means the clip is unusable."""
    capture = cv2.VideoCapture(str(clip_path))
    if not capture.isOpened():
        return None
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0 or frame_count <= 0:
            return None
        return frame_count, frame_count / fps
    finally:
        capture.release()


def _load_completed_ids(output_metadata_path: Path) -> set[str]:
    """clip_ids already written to a previous (possibly interrupted) run's
    metadata file — resume skips these entirely, no re-download/re-cut.

    A malformed line (most likely the last line, truncated because the
    process was killed mid-write before flushing a full JSON line + newline
    -- exactly the scenario --resume exists to recover from) is skipped
    with a warning rather than crashing the whole resume attempt. The file
    itself is also rewritten to drop that line (see prepare_panda70m()'s
    resume handling below) so it doesn't linger and get silently skipped
    again on every future resume.
    """
    if not output_metadata_path.is_file():
        return set()
    ids: set[str] = set()
    with open(output_metadata_path, encoding="utf-8") as f:
        for line_number, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError) as e:
                print(f"panda70m: skipping malformed line {line_number} in {output_metadata_path} "
                      f"(likely truncated by a previous interrupted run): {e}")
    return ids


def _repair_metadata_file(output_metadata_path: Path) -> None:
    """Rewrites output_metadata_path keeping only lines that parse as valid
    JSON with an "id" field -- drops any truncated/corrupt line left behind
    by a previous interrupted run, so resume's append mode doesn't build on
    top of garbage. No-op if the file doesn't exist or has nothing to drop."""
    if not output_metadata_path.is_file():
        return
    good_lines = []
    dropped = 0
    with open(output_metadata_path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                json.loads(stripped)["id"]
            except (json.JSONDecodeError, KeyError):
                dropped += 1
                continue
            good_lines.append(stripped)
    if dropped:
        with open(output_metadata_path, "w", encoding="utf-8") as f:
            f.write("\n".join(good_lines) + ("\n" if good_lines else ""))
        print(f"panda70m: repaired {output_metadata_path} -- dropped {dropped} malformed line(s)")


def _cleanup_source_if_done(video_id, tasks_per_video, processed_per_video, downloaded_sources):
    """Delete a raw source video once every clip task for it has been
    processed (success or failure). Raw source videos are full-length
    (capped at 480p, but still much larger than any single cut clip) and
    were previously never cleaned up -- they'd silently accumulate in
    tmp_video_dir for the life of the run, consuming meaningfully more disk
    than the final clips alone at any real scale.
    """
    if processed_per_video[video_id] < tasks_per_video[video_id]:
        return  # more tasks for this video still pending -- not done yet
    source_path = downloaded_sources.get(video_id)
    if source_path is not None and source_path.exists():
        source_path.unlink()
    downloaded_sources[video_id] = None  # mark handled so this doesn't retry the unlink


def prepare_panda70m(
    n_source_videos: int,
    output_metadata_path: Path,
    clip_output_dir: Path,
    tmp_video_dir: Path,
    cookies_from_browser: str | None = None,
    resume: bool = False,
) -> PrepSummary:
    """Download n_source_videos real Panda-70M source videos, cut every
    sub-clip, verify each, and write the resolved metadata JSONL.

    Every task is accounted for in the returned summary — nothing is
    silently dropped.

    cookies_from_browser: a browser name yt-dlp recognizes (chrome, edge,
        firefox, brave, opera, vivaldi, safari) to authenticate downloads
        with real logged-in cookies — works around YouTube's "Sign in to
        confirm you're not a bot" block.
    resume: if True, append to an existing output_metadata_path instead of
        overwriting it, and skip any clip already recorded there.
    """
    rows = fetch_real_panda70m_rows(n_source_videos)
    tasks = expand_to_clip_tasks(rows)
    summary = PrepSummary(total_tasks=len(tasks))

    tasks_per_video = Counter(task.video_id for task in tasks)
    processed_per_video: Counter = Counter()

    clip_output_dir.mkdir(parents=True, exist_ok=True)
    output_metadata_path.parent.mkdir(parents=True, exist_ok=True)

    completed_ids = set()
    if resume:
        _repair_metadata_file(output_metadata_path)
        completed_ids = _load_completed_ids(output_metadata_path)
    file_mode = "a" if resume else "w"

    downloaded_sources: dict[str, Path | None] = {}
    with open(output_metadata_path, file_mode, encoding="utf-8") as out:
        for task in tasks:
            try:
                if task.clip_id in completed_ids:
                    summary.ok += 1
                    continue

                if not task.caption:
                    summary.empty_caption += 1
                    continue

                if task.video_id not in downloaded_sources:
                    downloaded_sources[task.video_id] = download_source_video(
                        task.youtube_url, tmp_video_dir, cookies_from_browser
                    )
                source_video = downloaded_sources[task.video_id]
                if source_video is None:
                    summary.broken_download += 1
                    continue

                clip_path = clip_output_dir / f"{task.clip_id}.mp4"
                if not cut_clip(source_video, task.start, task.end, clip_path):
                    summary.broken_clip += 1
                    continue

                probed = probe_clip(clip_path)
                if probed is None:
                    summary.broken_clip += 1
                    clip_path.unlink(missing_ok=True)
                    continue

                num_frames, duration = probed
                out.write(
                    json.dumps({
                        "id": task.clip_id,
                        "url": task.youtube_url,
                        "caption": task.caption,
                        "num_frames": num_frames,
                        "duration": duration,
                    }) + "\n"
                )
                summary.ok += 1
            finally:
                processed_per_video[task.video_id] += 1
                _cleanup_source_if_done(task.video_id, tasks_per_video, processed_per_video, downloaded_sources)

    return summary