"""Unified download CLI — training/data/download/main.py.

Usage:
    python -m training.data.download.main --source cc3m --count 5000
    python -m training.data.download.main --source panda70m --count 500 --cookies-from-browser chrome
    python -m training.data.download.main --source both --cc3m-count 5000 --panda70m-count 500

DATA_ROOT is read from the environment (falls back to a .env file via
python-dotenv if present) — not asked for as a required flag, since every
other script in this repo assumes it's already set.
"""

import argparse
import os
from pathlib import Path

from training.data.download.cc3m import prepare_cc3m
from training.data.download.panda70m import prepare_panda70m


def _resolve_data_root(cli_value: str | None) -> Path:
    if cli_value:
        return Path(cli_value)
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    value = os.environ.get("DATA_ROOT")
    if not value:
        raise SystemExit("DATA_ROOT not set (pass --data-root, or set it in the environment/.env)")
    return Path(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["cc3m", "panda70m", "both"], required=True)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--count", type=int, default=None,
                         help="Row/video count, applies to --source cc3m or panda70m.")
    parser.add_argument("--cc3m-count", type=int, default=1000000, help="Used with --source both.")
    parser.add_argument("--panda70m-count", type=int, default=10000,
                         help="Source videos (not clips) to fetch. Used with --source both.")
    parser.add_argument("--cookies-from-browser",
                         choices=["chrome", "edge", "firefox", "brave", "opera", "vivaldi", "safari"],
                         default=None, help="panda70m only -- see download/panda70m.py.")
    parser.add_argument("--resume", action="store_true", help="panda70m only.")
    args = parser.parse_args()

    data_root = _resolve_data_root(args.data_root)

    if args.source in ("cc3m", "both"):
        count = args.count if args.source == "cc3m" and args.count else args.cc3m_count
        prepare_cc3m(count, data_root / "cc3m" / "metadata.tsv")

    if args.source in ("panda70m", "both"):
        count = args.count if args.source == "panda70m" and args.count else args.panda70m_count
        summary = prepare_panda70m(
            n_source_videos=count,
            output_metadata_path=data_root / "panda70m" / "metadata.jsonl",
            clip_output_dir=data_root / "cache" / "panda70m",
            tmp_video_dir=data_root / ".tmp_panda70m_sources",
            cookies_from_browser=args.cookies_from_browser,
            resume=args.resume,
        )
        print(f"panda70m: {summary.ok}/{summary.total_tasks} clips ok "
              f"(broken_download={summary.broken_download} broken_clip={summary.broken_clip} "
              f"empty_caption={summary.empty_caption})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())