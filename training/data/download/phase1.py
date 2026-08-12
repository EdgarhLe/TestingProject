"""Unified download CLI — training/data/download/main.py.

Usage:
    python -m training.data.download.main --source internvid --count 50000 --resume
    python -m training.data.download.main --source panda70m --source vimix14m --resume
    python -m training.data.download.main --source all --resume
"""

import argparse
import os
from pathlib import Path

from training.data.download.cc3m import prepare_cc3m
from training.data.download.panda70m import prepare_panda70m
from training.data.download.internvid import download_internvid
from training.data.download.recap_datacomp1b import download_recap_datacomp1b

# "both" is kept for backwards compatibility with existing scripts/docs
# that call --source both (cc3m + panda70m only). Use --source all, or
# repeat --source, to pull in the newer sources too.
_LEGACY_BOTH = ("cc3m", "panda70m")
_ALL_SOURCES = ("cc3m", "panda70m", "internvid", "vimix14m", "plm_video_human", "recap_datacomp1b")


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


def _run_source(source: str, data_root: Path, args: argparse.Namespace) -> None:
    if source == "cc3m":
        count = args.count or args.cc3m_count
        prepare_cc3m(count, data_root / "cc3m" / "metadata.tsv", data_root / "cache")

    elif source == "panda70m":
        count = args.count or args.panda70m_count
        summary = prepare_panda70m(
            n_source_videos=count,
            output_metadata_path=data_root / "panda70m" / "metadata.jsonl",
            clip_output_dir=data_root / "cache" / "panda70m",
            tmp_video_dir=data_root / ".tmp_panda70m_sources",
            cookies_from_browser=args.cookies_from_browser,
            resume=args.resume,
        )
        print(f"panda70m: {summary.ok}/{summary.total_tasks} clips ok")

    elif source == "internvid":
        count = args.count or args.internvid_count
        summary = download_internvid(
            data_root=data_root,
            max_videos=count,
            resume=args.resume,
            cookies_from_browser=args.cookies_from_browser,
        )
        print(f"internvid: {summary.downloaded}/{summary.total} downloaded "
              f"(skipped={summary.skipped}, failed={summary.failed})")

    elif source == "recap_datacomp1b":
        count = args.count or args.recap_count
        summary = download_recap_datacomp1b(
            data_root=data_root,
            max_images=count,
            caption_column=args.recap_caption_column,
            min_clip_score=args.recap_min_clip_score,
            resume=args.resume,
            hf_token=args.hf_token,
        )
        print(f"recap_datacomp1b: {summary.downloaded}/{summary.total_rows} downloaded from "
              f"{summary.shards_processed} shard(s) (skipped={summary.skipped}, "
              f"empty_caption={summary.empty_caption}, broken_image={summary.broken_image}, "
              f"failed_fetch={summary.failed_fetch})")

    else:
        raise SystemExit(f"Unknown source: {source}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--source", action="append",
        choices=[*_ALL_SOURCES, "both", "all"],
        default=None,
        help="Repeatable, e.g. --source panda70m"
             "'both' = cc3m+panda70m (legacy shorthand). 'all' = every source below.",
    )
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--count", type=int, default=None,
                         help="Overrides every per-source --*-count flag below for this run.")
    parser.add_argument("--cc3m-count", type=int, default=1_000_000)
    parser.add_argument("--panda70m-count", type=int, default=20_000, help="Number of source videos.")
    parser.add_argument("--internvid-count", type=int, default=50_000)
    parser.add_argument("--recap-count", type=int, default=200_000)

    parser.add_argument("--cookies-from-browser",
                         choices=["chrome", "edge", "firefox", "brave", "opera", "vivaldi", "safari"],
                         default=None, help="panda70m/internvid/plm_video_human: browser cookies for yt-dlp")
    parser.add_argument("--resume", action="store_true", help="Resume any source that supports it.")

    parser.add_argument("--hf-token", type=str, default=None,
                         help="vimix14m/plm_video_human/recap_datacomp1b: HF token, if the dataset needs one. "
                              "Falls back to the HF_TOKEN env var / `huggingface-cli login` if omitted.")
    parser.add_argument("--vimix14m-caption-key", type=str, default="caption",
                         help="JSON key to read the caption from inside each ViMix-14M shard example.")
    parser.add_argument("--recap-caption-column", type=str, default="re_caption",
                         help="Recap-DataComp-1B caption column. Use "
                              "re_caption_condition_diverse_topk for the newer v2/OpenVision-2 recaptions.")
    parser.add_argument("--recap-min-clip-score", type=float, default=None,
                         help="Drop Recap-DataComp-1B rows below this re_clip_score.")

    args = parser.parse_args()
    args.hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    data_root = _resolve_data_root(args.data_root)

    if not args.source:
        raise SystemExit("--source is required (repeatable), e.g. --source panda70m --source vimix14m, "
                          "or --source all")

    sources: list[str] = []
    for s in args.source:
        if s == "all":
            sources.extend(_ALL_SOURCES)
        elif s == "both":
            sources.extend(_LEGACY_BOTH)
        else:
            sources.append(s)
    sources = list(dict.fromkeys(sources))  # de-dupe, preserve order

    for source in sources:
        _run_source(source, data_root, args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())