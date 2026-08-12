"""CC3M metadata + image download — training/data/download/cc3m.py.

Downloads BOTH the raw metadata (caption/url pairs) AND the actual images
here, at download time -- not in training/data/preprocess.py. Preprocessing
only builds the combined manifest from files that already exist on disk,
matching how Panda-70M already works (prepare_panda70m() downloads + cuts +
validates before ever writing a row to its metadata.jsonl). Doing the actual
image downloads inside preprocessing put CC3M on an inconsistent pattern
from Panda-70M and made preprocessing (which should just be a fast manifest
build, no network I/O) slow.
"""

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

_CC3M_ROWS_API = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=google-research-datasets%2Fconceptual_captions"
    "&config=unlabeled&split=validation&offset={offset}&length={length}"
)
_CC3M_API_PAGE_SIZE = 100  # datasets-server's hard per-request cap
_MAX_RETRIES = 5
_BACKOFF_SECONDS = 2.0  # doubles each retry: 2s, 4s, 8s, 16s, 32s
_IMAGE_MAX_RETRIES = 3
_IMAGE_BACKOFF_SECONDS = 1.0


@dataclass
class Cc3mSummary:
    total: int = 0
    ok: int = 0
    broken_url: int = 0
    format_issue: int = 0


def _fetch_page(url: str, timeout: float = 30.0, log_prefix: str = "cc3m") -> dict | None:
    """Fetch one page, retrying transient failures with exponential backoff.
    Returns None (not raises) if every attempt fails -- the caller decides
    whether that's fatal.

    429 (Too Many Requests) is handled separately from other transient
    errors: honors a Retry-After header if the server sends one, and backs
    off 3x longer by default otherwise -- a 429 means "you're going too
    fast," so retrying at the same short interval as a generic 502 just
    triggers another 429 immediately.
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


def fetch_real_cc3m_rows(n: int) -> list[tuple[str, str]]:
    """Page through the real CC3M validation split via the datasets-server
    REST API — public, no auth needed. Returns (caption, image_url) pairs.

    A page that fails after retries (see _fetch_page) stops paging rather
    than crashing the whole run -- returns whatever rows were successfully
    fetched before the failure, same "skip and report, don't crash"
    philosophy as the broken-URL/format-issue handling elsewhere in this
    pipeline.
    """
    rows: list[tuple[str, str]] = []
    total = None
    while len(rows) < n:
        page_len = min(_CC3M_API_PAGE_SIZE, n - len(rows))
        url = _CC3M_ROWS_API.format(offset=len(rows), length=page_len)

        payload = _fetch_page(url)
        if payload is None:
            break  # give up on further paging, keep what we have

        page_rows = [(row["row"]["caption"], row["row"]["image_url"]) for row in payload["rows"]]
        total = payload["num_rows_total"]
        if not page_rows:
            break
        rows.extend(page_rows)
        if len(rows) >= total:
            break

    if len(rows) < n:
        reason = f"validation split only has {total} rows" if total is not None else "a page request failed"
        print(f"cc3m: {reason}, got {len(rows)}/{n} requested")
    return rows[:n]


def write_cc3m_tsv(rows: list[tuple[str, str]], dest_path: Path) -> int:
    """Write (caption, url) rows to a `caption<TAB>url` TSV file -- kept as the
    raw metadata record/audit trail. Returns the number of rows written."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(dest_path, "w", encoding="utf-8") as f:
        for caption, url in rows:
            clean_caption = caption.replace("\t", " ").replace("\n", " ").strip()
            if not clean_caption or not url.strip():
                continue
            f.write(f"{clean_caption}\t{url.strip()}\n")
            written += 1
    return written


def _download_image(url: str, dest: Path, timeout: float = 15.0) -> None:
    """Retries transient failures (502/503/504, timeouts, connection resets)
    with short backoff before giving up -- without this, a brief network
    hiccup gets permanently counted as a "broken URL" even though the URL
    itself is fine."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

    last_error: Exception | None = None
    for attempt in range(1, _IMAGE_MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as f:
                f.write(resp.read())
            tmp.replace(dest)
            return
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as e:
            last_error = e
            tmp.unlink(missing_ok=True)
            if attempt < _IMAGE_MAX_RETRIES:
                time.sleep(_IMAGE_BACKOFF_SECONDS * (2 ** (attempt - 1)))
    raise last_error


def download_and_validate_images(rows: list[tuple[str, str]], cache_dir: Path) -> Cc3mSummary:
    """Download + decode-validate each row's image into cache_dir/cc3m/{line_number:08d}.jpg.

    Broken URLs (download fails after retries) and format issues (downloads
    fine but doesn't decode as a valid image -- e.g. an HTML error page
    served with a 200 status) are both counted separately and reported, per
    #48's original requirement to report real post-filter counts.
    """
    summary = Cc3mSummary()
    cc3m_dir = cache_dir / "cc3m"
    cc3m_dir.mkdir(parents=True, exist_ok=True)

    for line_number, (_caption, url) in enumerate(rows):
        summary.total += 1
        dest = cc3m_dir / f"{line_number:08d}.jpg"

        if not dest.exists():
            try:
                _download_image(url, dest)
            except Exception as e:
                summary.broken_url += 1
                print(f"cc3m: [broken url] line {line_number}: {e}")
                continue

        try:
            with Image.open(dest) as img:
                img.convert("RGB")
        except Exception as e:
            summary.format_issue += 1
            print(f"cc3m: [format issue] line {line_number}: {e}")
            dest.unlink(missing_ok=True)
            continue

        summary.ok += 1

    return summary


def prepare_cc3m(n: int, dest_path: Path, cache_dir: Path) -> Cc3mSummary:
    """Fetch n real CC3M rows, write the raw metadata TSV, AND download +
    validate every image into cache_dir/cc3m/. Returns a Cc3mSummary."""
    rows = fetch_real_cc3m_rows(n)
    written = write_cc3m_tsv(rows, dest_path)
    print(f"cc3m: wrote {written}/{n} metadata rows to {dest_path}")

    summary = download_and_validate_images(rows, cache_dir)
    print(f"cc3m: {summary.ok}/{summary.total} images ok "
          f"({summary.broken_url} broken url, {summary.format_issue} format issue)")
    return summary
