"""
transcribe.py — Batch Vietnamese ASR with segment-level timestamps.

Model choice (investigated after PhoWhisper and Whisper large-v3 both proved
unsatisfactory on news-format video — high WER, dropped speech under BGM/noise):

  Qwen3-ASR-1.7B (Apache-2.0, free, ~1.7B params)
    - On the "News-Multilingual" benchmark (incl. Vietnamese): WER 12.80 vs.
      14.80 for Whisper-large-v3.
    - On heavy background noise: WER 16.17 vs. 63.17 for Whisper-large-v3.
    - On audio with background music: Whisper is reported as failing outright
      (N/A) on some benchmarks; Qwen3-ASR still produces usable output.
    - This directly targets the "missing speech in news video" symptom, which
      is typically Whisper's internal VAD dropping segments under music/SFX.

Timestamps: Qwen3-ASR ships a companion forced aligner (Qwen3-ForcedAligner-0.6B)
for word-level timestamps, but it does NOT officially support Vietnamese yet
(supported: zh, en, yue, fr, de, it, ja, ko, pt, ru). So instead of word-level
alignment, this script uses Silero VAD to find speech regions up front and
treats each VAD region as one timestamped segment.

Segment length: no cap by default (asr.max_segment_s in configs/model.yaml,
default None). Segment boundaries follow natural VAD speech regions as-is —
including cases like a fast-cut news headline montage where multiple stories
run together with no pause between them. Splitting on a fixed time cap
doesn't know where topic boundaries actually are, so at small dataset scale
it either does nothing (segment already shorter than the cap) or cuts a
single well-aligned segment into arbitrary pieces mid-sentence for no
benefit. Multi-topic segments are accepted as ordinary label noise for the
contrastive (InfoNCE) objective rather than special-cased in preprocessing;
revisit with real capping/splitting only if this is shown empirically to
hurt downstream retrieval quality.

One-time setup on Machine 2 (not per run):

    conda create -n qwen3-asr python=3.12 -y && conda activate qwen3-asr
    pip install -U qwen-asr torch torchaudio soundfile python-dotenv PyYAML
    # optional, faster + lower VRAM:
    pip install -U flash-attn --no-build-isolation

    # Pre-download weights (~1.7B ASR model, few GB) so Machine 2 doesn't need
    # to hit the network on every run:
    huggingface-cli download Qwen/Qwen3-ASR-1.7B --local-dir ./models/Qwen3-ASR-1.7B

Data root: DATASET_ROOT is read from a .env file (default: ./.env, override
with --env-file), not hardcoded. By default the script globs all videos
recursively under DATASET_ROOT; pass --video-dir to point at a subfolder
instead. --out-dir also resolves relative paths against DATASET_ROOT.

.env example:
    DATASET_ROOT=/mnt/data/vn_news_dataset

Resume: if an output transcript already exists for a video, that video is
skipped (use --overwrite to force re-processing). Safe to re-run the same
command after an interrupted batch — already-done videos won't be redone,
and the manifest is merged rather than overwritten.

Usage:
    # Quick manual test — first N videos found under DATASET_ROOT
    python transcribe.py --limit 3 --out-dir /tmp/asr_test

    # Full batch run over everything under DATASET_ROOT (safe to re-run/resume)
    python transcribe.py --out-dir transcripts/

    # Restrict to a subfolder instead of all of DATASET_ROOT
    python transcribe.py --video-dir videos/test --out-dir transcripts/test
"""

import argparse
import json
import logging
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import torch
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("transcribe")

VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".avi", ".webm")

# Read configs/model.yaml
with open("configs/model.yaml", "r", encoding="utf-8") as f:
    import yaml
    model_config = yaml.safe_load(f)

DEFAULT_MODEL_PATH = model_config["asr"]["model_name"]  # or a local dir from the setup step above

# Merge VAD segments closer than this many seconds, to avoid too many tiny
# calls into the ASR model (each call has fixed overhead).
VAD_MERGE_GAP_S = model_config["asr"]["vad_merge_gap_s"]  # e.g. 0.1
# Pad each speech segment by this much on each side so words at the edges
# aren't clipped.
VAD_PAD_S = model_config["asr"]["vad_pad_s"]  # e.g. 0.25
# Hard cap on segment duration used as a training label. Default: no cap —
# natural VAD segments are used as-is. A segment spanning multiple topics
# (e.g. a fast-cut news headline montage with no pause between stories) is
# accepted as ordinary label noise rather than special-cased away; at the
# current dataset scale this is a small fraction of each video's duration,
# within what InfoNCE with in-batch negatives normally tolerates. Set
# asr.max_segment_s in configs/model.yaml to a positive number only if this
# is later shown empirically to hurt retrieval quality.
MAX_SEGMENT_S = model_config["asr"].get("max_segment_s")  # None -> disabled


@dataclass
class Segment:
    start: float
    end: float
    text: str


def load_dataset_root(env_file: str) -> Path:
    """Load DATASET_ROOT from a .env file (falls back to an already-exported
    env var if the file doesn't exist or doesn't set it)."""
    env_path = Path(env_file)
    if env_path.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path, override=False)
        except ImportError:
            # Minimal fallback parser if python-dotenv isn't installed.
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

    root = os.environ.get("DATASET_ROOT")
    if not root:
        raise EnvironmentError(
            f"DATASET_ROOT is not set. Add it to {env_file} (e.g. "
            f"DATASET_ROOT=/mnt/data/vn_news_dataset) or export it in the shell."
        )
    return Path(root)


def resolve_path(path: Path, dataset_root: Path) -> Path:
    """Relative paths are resolved against DATASET_ROOT; absolute paths pass through."""
    return path if path.is_absolute() else dataset_root / path


def find_videos(video_dir: Path) -> List[Path]:
    """Glob all video files under a directory (recursive)."""
    return sorted(p for p in video_dir.rglob("*") if p.suffix.lower() in VIDEO_EXTS)


def extract_audio(video_path: Path, out_wav: Path) -> str:
    """Extract 16kHz mono PCM WAV from a video file via ffmpeg.

    Returns ffmpeg's stderr so callers can parse metadata (e.g. fps) from it
    instead of running a second full decode pass just to read the fps.
    """
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-ac", "1", "-ar", "16000", "-vn",
        str(out_wav),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {video_path}:\n{result.stderr}")
    return result.stderr


def parse_fps_from_ffmpeg_stderr(stderr: str, video_path: Path) -> float:
    """Parse the 'NN fps' / 'NUM/DEN fps' token out of ffmpeg's stream info,
    which is already present in the stderr from the extract_audio() call —
    avoids a second `ffmpeg -f null -` decode of the whole video just for fps.
    """
    fps_match = re.search(r"Video:.*?,\s*(\d+(?:/\d+)?)\s*fps", stderr, re.IGNORECASE)
    if not fps_match:
        fps_match = re.search(r"(\d+(?:\.\d+)?)\s*fps", stderr)
    if not fps_match:
        raise RuntimeError(f"Could not determine FPS for {video_path}")

    fps_str = fps_match.group(1)
    if "/" in fps_str:
        num, den = map(int, fps_str.split("/"))
        if den == 0:
            raise RuntimeError(f"Invalid FPS ratio {fps_str} for {video_path}")
        return num / den
    return float(fps_str)


def cap_segment_lengths(regions: List[tuple], max_len_s: float) -> List[tuple]:
    """Split any (start, end) region longer than max_len_s into near-equal
    sub-regions, each capped at max_len_s. Splitting happens BEFORE the ASR
    call, so each sub-region gets its own real transcription rather than a
    word-count-proportional guess sliced out of one longer transcript."""
    if max_len_s is None or max_len_s <= 0:
        return regions
    capped = []
    for start, end in regions:
        dur = end - start
        if dur <= max_len_s:
            capped.append((start, end))
            continue
        n_pieces = math.ceil(dur / max_len_s)
        piece_len = dur / n_pieces
        for i in range(n_pieces):
            piece_start = start + i * piece_len
            piece_end = end if i == n_pieces - 1 else start + (i + 1) * piece_len
            capped.append((piece_start, piece_end))
    return capped


def run_vad(model, utils, wav_path: Path) -> List[tuple]:
    """Return list of (start_s, end_s) speech regions using Silero VAD."""
    torch.set_num_threads(1)

    (get_speech_timestamps, _, read_audio, *_rest) = utils

    wav = read_audio(str(wav_path), sampling_rate=16000)
    raw_segments = get_speech_timestamps(
        wav, model, sampling_rate=16000, return_seconds=True
    )
    regions = [(s["start"], s["end"]) for s in raw_segments]

    # Merge close/adjacent regions and pad edges.
    merged = []
    for start, end in regions:
        start = max(0.0, start - VAD_PAD_S)
        end = end + VAD_PAD_S
        if merged and start - merged[-1][1] <= VAD_MERGE_GAP_S:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    return cap_segment_lengths(merged, MAX_SEGMENT_S)


def load_asr_model(model_path: str = DEFAULT_MODEL_PATH):
    from qwen_asr import Qwen3ASRModel

    log.info(f"Loading {model_path} (device=cuda:0, dtype=bfloat16)")
    return Qwen3ASRModel.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        max_inference_batch_size=8,
        max_new_tokens=512,
    )


def transcribe_video(
    video_path: Path,
    model_asr,
    model_vad,
    utils,
    tmp_dir: Path,
    language: str = "Vietnamese",
) -> tuple:
    """Returns (segments, fps)."""
    wav_path = tmp_dir / f"{video_path.stem}.wav"
    log.info(f"[{video_path.name}] extracting audio")
    ffmpeg_stderr = extract_audio(video_path, wav_path)
    fps = parse_fps_from_ffmpeg_stderr(ffmpeg_stderr, video_path)

    log.info(f"[{video_path.name}] running VAD")
    speech_regions = run_vad(model_vad, utils, wav_path)
    if not speech_regions:
        log.warning(f"[{video_path.name}] VAD found no speech regions — check the source audio")
        return [], fps
    log.info(f"[{video_path.name}] {len(speech_regions)} speech region(s) after merge/cap")

    import soundfile as sf
    audio, sr = sf.read(str(wav_path))
    assert sr == 16000

    # Batch all regions for this video into one call for throughput.
    chunks = []
    for region_start, region_end in speech_regions:
        i0 = int(region_start * sr)
        i1 = min(int(region_end * sr), len(audio))
        chunks.append((audio[i0:i1], sr))

    results = model_asr.transcribe(audio=chunks, language=[language] * len(chunks))

    segments: List[Segment] = []
    for (region_start, region_end), result in zip(speech_regions, results):
        text = result.text.strip()
        if not text:
            continue
        segments.append(Segment(start=round(region_start, 3), end=round(region_end, 3), text=text))

    return segments, fps


def write_json(video_id: str, fps: float, segments: List[Segment], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "video_id": video_id,
        "video_fps": fps,
        "segments": [
            {
                "text": seg.text,
                "start_frame": int(round(seg.start * fps)),
                "end_frame": int(round(seg.end * fps)),
            }
            for seg in segments
        ],
    }
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_srt(segments: List[Segment], out_path: Path) -> None:
    def fmt(t: float) -> str:
        h, rem = divmod(t, 3600)
        m, s = divmod(rem, 60)
        ms = int((s - int(s)) * 1000)
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{ms:03d}"

    lines = []
    for i, seg in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(f"{fmt(seg.start)} --> {fmt(seg.end)}")
        lines.append(seg.text)
        lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video-dir", type=str, default=None,
                         help="Subfolder to glob videos from (relative to DATASET_ROOT unless absolute). "
                              "Defaults to DATASET_ROOT itself.")
    parser.add_argument("--out-dir", type=str, required=True, help="Directory to write transcripts to.")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH,
                         help="HF repo id or local dir for Qwen3-ASR.")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only process the first N videos found (useful for quick manual tests).")
    parser.add_argument("--language", type=str, default="Vietnamese")
    parser.add_argument("--tmp-dir", type=str, default="/tmp/asr_audio")
    parser.add_argument("--env-file", type=str, default=".env",
                         help="Path to .env file containing DATASET_ROOT.")
    parser.add_argument("--overwrite", action="store_true",
                         help="Re-process videos even if an output transcript already exists.")
    args = parser.parse_args()

    dataset_root = load_dataset_root(args.env_file)
    log.info(f"DATASET_ROOT={dataset_root}")

    video_dir = resolve_path(Path(args.video_dir), dataset_root) if args.video_dir else dataset_root
    videos = find_videos(video_dir)
    if not videos:
        log.error(f"No videos found under {video_dir}.")
        sys.exit(1)

    if args.limit is not None:
        log.info(f"Limiting to first {args.limit} of {len(videos)} video(s) found.")
        videos = videos[:args.limit]

    out_dir = resolve_path(Path(args.out_dir), dataset_root)
    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # --- resume: load existing manifest (if any) and skip already-done videos ---
    manifest_path = out_dir / "transcript_manifest.json"
    manifest = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            log.info(f"Loaded existing manifest with {len(manifest)} entrie(s) for resume.")
        except json.JSONDecodeError:
            log.warning(f"Existing manifest at {manifest_path} is corrupt, starting fresh.")

    todo = []
    skipped = 0
    for video_path in videos:
        rel_path = video_path.relative_to(video_dir)
        transcript_path = out_dir / rel_path.with_suffix(".json")
        if transcript_path.exists() and not args.overwrite:
            skipped += 1
            continue
        todo.append((video_path, transcript_path))

    if skipped:
        log.info(f"Resume: skipping {skipped} video(s) with existing output (use --overwrite to redo).")
    if not todo:
        log.info("Nothing to do — all videos already have output.")
        sys.exit(0)

    model_asr = load_asr_model(args.model_path)
    model_vad, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
    )

    ok, failed = 0, []
    for video_path, transcript_path in todo:
        try:
            segments, fps = transcribe_video(video_path, model_asr, model_vad, utils, tmp_dir, language=args.language)
            video_id = video_path.stem
            srt_path = transcript_path.with_suffix(".srt")

            write_json(video_id, fps, segments, transcript_path)
            write_srt(segments, srt_path)
            manifest[str(transcript_path)] = str(video_path)

            log.info(f"[{video_path.name}] done — {len(segments)} segment(s)")
            ok += 1
        except Exception as e:
            log.exception(f"[{video_path.name}] FAILED: {e}")
            failed.append(str(video_path))

    if manifest:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info(f"Wrote manifest ({len(manifest)} total entrie(s)): {manifest_path}")

    log.info(f"Finished: {ok} succeeded, {len(failed)} failed, {skipped} skipped (already done).")
    if failed:
        log.error("Failed videos:\n" + "\n".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()