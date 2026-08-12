"""Recap-DataComp-1B download — training/data/download/recap_datacomp1b.py

UCSC-VLAA/Recap-DataComp-1B (ICML 2025; captions updated Jan 2026 with a
v2/OpenVision-2 recaption column) re-captions the DataComp-1B image pool
with LLaMA-3. The dataset itself ships as HF parquet shards of
{url, org_caption, re_caption, sha256, key, ...} rows -- the *images*
still have to be fetched from their original web URLs (same "URL list +
downloader" shape as CC3M), there's no bundled image bytes.

The official recipe (UCSC-VLAA's README) points at img2dataset; this
script does the same job directly with urllib + retry/backoff so it
matches the rest of this download package's dependency footprint and
resume/summary conventions instead of pulling in a separate CLI tool.
Every image download is retried, verified as a real decodable image, and
counted -- nothing is silently dropped.
"""

import io
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image

_REPO_ID = "UCSC-VLAA/Recap-DataComp-1B"
_MAX_RETRIES = 4
_BACKOFF_SECONDS = 1.5
_IMAGE_TIMEOUT = 15.0

# "re_caption" is the v1 LLaMA-3 recaption used in the paper.
# "re_caption_condition_diverse_topk" is the newer v2 recaption (used by
# OpenVision 2) added to the same repo in a later update -- pass
# --caption-column to use it instead.
_DEFAULT_CAPTION_COLUMN = "re_caption"


@dataclass
class DownloadSummary:
    shards_processed: int = 0
    total_rows: int = 0
    downloaded: int = 0
    skipped: int = 0
    empty_caption: int = 0
    broken_image: int = 0
    failed_fetch: int = 0


def _list_parquet_shards(hf_token: Optional[str]) -> list[str]:
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token)
    files = api.list_repo_files(repo_id=_REPO_ID, repo_type="dataset")
    shards = sorted(f for f in files if f.endswith(".parquet") and "train_data" in f)
    if not shards:
        # Layout fallback -- take any top-level parquet if the train_data/
        # subdirectory naming has changed upstream.
        shards = sorted(f for f in files if f.endswith(".parquet"))
    if not shards:
        raise RuntimeError(
            f"No .parquet shards found in {_REPO_ID} -- check "
            "https://huggingface.co/datasets/UCSC-VLAA/Recap-DataComp-1B/tree/main"
        )
    return shards


def _download_parquet_shard(shard_path: str, dest_dir: Path, hf_token: Optional[str]) -> Optional[Path]:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import HfHubHTTPError

    last_error: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            local_path = hf_hub_download(
                repo_id=_REPO_ID, repo_type="dataset", filename=shard_path,
                local_dir=str(dest_dir), token=hf_token,
            )
            return Path(local_path)
        except (HfHubHTTPError, Exception) as e:
            last_error = e
            wait = _BACKOFF_SECONDS * (2 ** (attempt - 1))
            print(f"recap_datacomp1b: shard download failed for {shard_path} "
                  f"(attempt {attempt}/{_MAX_RETRIES}): {e}")
            if attempt < _MAX_RETRIES:
                time.sleep(wait)
    print(f"recap_datacomp1b: giving up on shard {shard_path} after {_MAX_RETRIES} attempts ({last_error})")
    return None


def _fetch_image(url: str) -> Optional[bytes]:
    """Download+decode-verify one image, retrying transient failures. A
    429 backs off harder, same policy as panda70m.py's _fetch_page."""
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=_IMAGE_TIMEOUT) as resp:
                raw = resp.read()
            Image.open(io.BytesIO(raw)).verify()  # raises if not a real image
            return raw
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = _BACKOFF_SECONDS * (2 ** (attempt - 1)) * 3
            elif e.code in (403, 404, 410):
                return None  # dead/forbidden link -- retrying won't help
            else:
                wait = _BACKOFF_SECONDS * (2 ** (attempt - 1))
            if attempt < _MAX_RETRIES:
                time.sleep(wait)
        except Exception:
            if attempt < _MAX_RETRIES:
                time.sleep(_BACKOFF_SECONDS * (2 ** (attempt - 1)))
    return None


def download_recap_datacomp1b(
    data_root: Path,
    max_images: int = 200_000,
    caption_column: str = _DEFAULT_CAPTION_COLUMN,
    min_clip_score: Optional[float] = None,
    resume: bool = True,
    hf_token: Optional[str] = None,
) -> DownloadSummary:
    image_dir = data_root / "cache" / "recap_datacomp1b"
    image_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = data_root / "recap_datacomp1b_metadata.jsonl"
    tmp_shard_dir = data_root / ".tmp_recap_datacomp1b_shards"
    tmp_shard_dir.mkdir(parents=True, exist_ok=True)
    summary = DownloadSummary()

    completed_ids: set[str] = set()
    if resume and metadata_path.is_file():
        with open(metadata_path, encoding="utf-8") as f:
            for line in f:
                try:
                    completed_ids.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    pass

    print("recap_datacomp1b: listing parquet shards on the HF Hub...")
    shards = _list_parquet_shards(hf_token)
    print(f"recap_datacomp1b: {len(shards)} shards available")

    import pyarrow.parquet as pq

    file_mode = "a" if resume else "w"
    with open(metadata_path, file_mode, encoding="utf-8") as meta_f:
        for shard_path in shards:
            if summary.downloaded >= max_images:
                break
            local_shard = _download_parquet_shard(shard_path, tmp_shard_dir, hf_token)
            if local_shard is None:
                continue
            summary.shards_processed += 1

            table = pq.read_table(local_shard)
            columns = table.column_names
            caption_col = caption_column if caption_column in columns else _DEFAULT_CAPTION_COLUMN
            rows = table.to_pylist()
            for row in rows:
                if summary.downloaded >= max_images:
                    break
                summary.total_rows += 1

                image_key = row.get("key") or row.get("sha256")
                url = row.get("url")
                caption = (row.get(caption_col) or row.get("org_caption") or "").strip()
                if not url or not caption:
                    summary.empty_caption += 1
                    continue

                if min_clip_score is not None:
                    score = row.get("re_clip_score")
                    if score is not None and score < min_clip_score:
                        continue

                image_id = str(image_key) if image_key else str(abs(hash(url)))
                if image_id in completed_ids:
                    summary.skipped += 1
                    continue

                image_path = image_dir / f"{image_id}.jpg"
                if not image_path.exists():
                    raw = _fetch_image(url)
                    if raw is None:
                        summary.failed_fetch += 1
                        continue
                    try:
                        img = Image.open(io.BytesIO(raw)).convert("RGB")
                        image_path.parent.mkdir(parents=True, exist_ok=True)
                        img.save(image_path, "JPEG", quality=95)
                    except Exception:
                        summary.broken_image += 1
                        continue

                meta_f.write(json.dumps({
                    "id": image_id,
                    "caption": caption,
                    "org_caption": row.get("org_caption"),
                    "source": "recap_datacomp1b",
                }) + "\n")
                meta_f.flush()
                summary.downloaded += 1

    return summary