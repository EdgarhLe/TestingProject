"""Build phase1_combined.jsonl — training/data/preprocess.py.

Reads the raw metadata training/data/download/{cc3m,panda70m}.py produced
and writes the single combined manifest training/data/loader.py reads.

CC3M: images aren't downloaded yet at this point (download/cc3m.py only
wrote the caption<TAB>url TSV) — this is where each image actually gets
downloaded and verified to decode. Broken URLs and format issues (downloads
that return HTTP 200 but aren't a valid image) are both caught and reported
here, then excluded from the final manifest.

Panda-70M: download/panda70m.py's prepare_panda70m() already downloaded,
cut, and verified every clip before writing its metadata — nothing left to
re-validate, rows are just converted directly.

Every row written to phase1_combined.jsonl is GUARANTEED to already have its
file on disk. training/data/loader.py relies on this and never downloads
anything itself.
"""

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import yaml
from PIL import Image

_DOWNLOAD_MAX_RETRIES = 3
_DOWNLOAD_BACKOFF_SECONDS = 1.0  # doubles each retry: 1s, 2s


@dataclass
class PreprocessSummary:
    cc3m_total: int = 0
    cc3m_ok: int = 0
    cc3m_broken_url: int = 0
    cc3m_format_issue: int = 0
    panda70m_total: int = 0
    panda70m_ok: int = 0
    panda70m_missing_file: int = 0


def _load_paths(dataset_config_path: str = "configs/dataset.yaml") -> dict:
    """Plain dict of the paths preprocessing needs, resolved against
    DATA_ROOT — no wrapper class, just the values read straight out of the
    yaml file."""
    data_root = Path(os.environ["DATA_ROOT"])
    with open(dataset_config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cache_subdir = raw["cache"]["subdir"]   # e.g. "cache" -- kept relative, see note below
    return {
        "data_root": data_root,
        "cache_subdir": cache_subdir,
        "train_jsonl": data_root / raw["output"]["train_jsonl"],
        "cc3m_metadata": data_root / raw["sources"]["cc3m"]["metadata_path"],
        "panda70m_metadata": data_root / raw["sources"]["panda70m"]["metadata_path"],
    }


def _download_image(url: str, dest: Path, timeout: float = 15.0) -> None:
    """Retries transient failures (502/503/504, timeouts, connection resets)
    with short backoff before giving up -- without this, a brief network
    hiccup gets permanently counted as a "broken URL" in the preprocessing
    report even though the URL itself is fine, undermining the accuracy of
    that report (#48's original requirement)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

    last_error: Exception | None = None
    for attempt in range(1, _DOWNLOAD_MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as f:
                f.write(resp.read())
            tmp.replace(dest)
            return
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as e:
            last_error = e
            tmp.unlink(missing_ok=True)
            if attempt < _DOWNLOAD_MAX_RETRIES:
                time.sleep(_DOWNLOAD_BACKOFF_SECONDS * (2 ** (attempt - 1)))
    raise last_error


def _process_cc3m(paths: dict, summary: PreprocessSummary) -> list[dict]:
    metadata_path = paths["cc3m_metadata"]
    if not metadata_path.exists():
        print(f"cc3m: no metadata at {metadata_path}, skipping")
        return []

    rows = []
    with open(metadata_path, encoding="utf-8") as f:
        for line_number, line in enumerate(f):
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                continue
            caption, url = (p.strip() for p in parts)
            if not caption or not url:
                continue

            summary.cc3m_total += 1
            # data_path stored RELATIVE TO DATA_ROOT (portable across machines) --
            # cache_subdir here is just the short name (e.g. "cache"), not the
            # resolved absolute path.
            rel_path = f"{paths['cache_subdir']}/cc3m/{line_number:08d}.jpg"
            dest = paths["data_root"] / rel_path

            if not dest.exists():
                try:
                    _download_image(url, dest)
                except Exception as e:
                    summary.cc3m_broken_url += 1
                    print(f"   [broken url] line {line_number}: {e}")
                    continue

            try:
                with Image.open(dest) as img:
                    img.convert("RGB")
            except Exception as e:
                summary.cc3m_format_issue += 1
                print(f"   [format issue] line {line_number}: {e}")
                dest.unlink(missing_ok=True)
                continue

            summary.cc3m_ok += 1
            rows.append({
                "data_path": rel_path,
                "caption": caption,
                "media_type": "image",
                "num_frames": 1,
                "duration": None,
                "source": "cc3m",
            })

    return rows


def _process_panda70m(paths: dict, summary: PreprocessSummary) -> list[dict]:
    metadata_path = paths["panda70m_metadata"]
    if not metadata_path.exists():
        print(f"panda70m: no metadata at {metadata_path}, skipping")
        return []

    rows = []
    with open(metadata_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            summary.panda70m_total += 1

            rel_path = f"{paths['cache_subdir']}/panda70m/{record['id']}.mp4"
            dest = paths["data_root"] / rel_path
            if not dest.exists():
                # prepare_panda70m() should never list a clip without the file
                # existing -- if this fires, something deleted it after the fact.
                summary.panda70m_missing_file += 1
                print(f"   [missing file] {rel_path} listed in metadata but not on disk")
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


def build_manifest(dataset_config_path: str = "configs/dataset.yaml") -> PreprocessSummary:
    paths = _load_paths(dataset_config_path)
    summary = PreprocessSummary()

    print("=== CC3M: downloading + validating each image ===")
    cc3m_rows = _process_cc3m(paths, summary)
    print(f"cc3m: {summary.cc3m_ok}/{summary.cc3m_total} ok "
          f"({summary.cc3m_broken_url} broken url, {summary.cc3m_format_issue} format issue)")

    print("=== Panda-70M: converting already-validated clips ===")
    panda70m_rows = _process_panda70m(paths, summary)
    print(f"panda70m: {summary.panda70m_ok}/{summary.panda70m_total} ok "
          f"({summary.panda70m_missing_file} missing file)")

    all_rows = cc3m_rows + panda70m_rows
    paths["train_jsonl"].parent.mkdir(parents=True, exist_ok=True)
    with open(paths["train_jsonl"], "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row) + "\n")

    print(f"\nwrote {len(all_rows)} rows to {paths['train_jsonl']}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", type=str, default="configs/dataset.yaml")
    args = parser.parse_args()
    build_manifest(args.dataset_config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())