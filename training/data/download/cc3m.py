"""CC3M metadata download and TSV preparation — training/data/download/cc3m.py."""

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

_CC3M_ROWS_API = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=google-research-datasets%2Fconceptual_captions"
    "&config=unlabeled&split=validation&offset={offset}&length={length}"
)
_CC3M_API_PAGE_SIZE = 100  # datasets-server's hard per-request cap
_MAX_RETRIES = 5
_BACKOFF_SECONDS = 2.0  # doubles each retry: 2s, 4s, 8s, 16s, 32s


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
    """Write (caption, url) rows to a `caption<TAB>url` TSV file. Returns the
    number of rows written."""
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


def prepare_cc3m(n: int, dest_path: Path) -> int:
    """Fetch n real CC3M rows and write them to dest_path. Returns the
    number of rows actually written (may be less than n near the end of
    the validation split, or if some rows had empty caption/url)."""
    rows = fetch_real_cc3m_rows(n)
    written = write_cc3m_tsv(rows, dest_path)
    print(f"cc3m: wrote {written}/{n} rows to {dest_path}")
    return written