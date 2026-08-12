#!/usr/bin/env python3
"""
scripts/build_index.py

Builds ONE FAISS index (coarse or fine) end-to-end: for every video in
--videos, run the segmentation pipeline (#68) using a trained checkpoint,
write each segment's metadata into the SegmentStore (#69), and add its
embedding to the FAISS index (indexing/indexes/faiss_index.py). This is the
script DE triggers in Week 5 once the first real checkpoint syncs from
Machine 1 to Machine 2.

    python scripts/build_index.py \\
        --level coarse \\
        --checkpoint path/to/checkpoint.deploy.pt \\
        --videos path/to/video_list.txt \\
        --output INDEX_ROOT/coarse.faiss

Run it twice (once per --level) to build both coarse.faiss and fine.faiss
-- they are independent index files with independent SegmentStore databases
by default (each level's segments are a different sliding-window pass over
the same videos, so they get different segment_id spaces).

DO NOT RUN THIS AGAINST REAL VIDEOS YET. See indexing/segmentation/pipeline.py's
"WEEK 4 SCOPE" docstring -- the checkpoint this script would load doesn't
exist yet (Phase 1 is still training on Machine 1). This week, this script
is implemented and covered by indexing/indexes/tests/, not run for real.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

try:
    from tqdm import tqdm
except ImportError:   # pragma: no cover -- tqdm is a soft dependency, see --no-progress-bar
    tqdm = None

from indexing.segmentation.pipeline import segment_video
from indexing.indexes.segment_store import SegmentStore
from indexing.indexes.faiss_index import SegmentFaissIndex, EMBEDDING_DIM


# ----------------------------------------------------------------------
# Level-specific sliding-window defaults (in FRAMES -- #68's
# sliding_window.py works in frame units, not seconds; the system design's
# "coarse: 5-10s / fine: 0.5-1s" windows are converted here assuming
# ASSUMED_FPS as a PLACEHOLDER. #68's own defaults (window_size=64,
# stride=8) are reused as-is for level=coarse below (they were already
# tuned as the "coarse pass" numbers in #68's spec). level=fine's defaults
# are new, approximating the 0.5-1s target at the same placeholder fps.
#
# MUST be revisited in Week 5 once real video fps/content is known --
# override with --window-size/--stride if a video's real fps differs
# meaningfully from ASSUMED_FPS.
# ----------------------------------------------------------------------
ASSUMED_FPS = 25
LEVEL_DEFAULTS = {
    "coarse": {"window_size": 64, "stride": 8},   # ~2.56s window / ~0.32s stride @25fps
    "fine": {"window_size": 20, "stride": 4},     # ~0.8s window / ~0.16s stride @25fps
}


def load_video_list(path):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def load_checkpoint_model(checkpoint_path, device):
    """
    Loads the frozen X-Encoder (#21) and the trained Predictor (#44) needed
    for query-free segmentation. Predictor weights come from a
    training/phase1.py checkpoint (either a .deploy.pt or .resume.pt --
    both have a "predictor" key, see training/phase1.py's
    _model_state_dict()). The X-Encoder is never trained, so it's loaded
    fresh from HuggingFace rather than from the checkpoint (unless the
    checkpoint happens to include an "x_encoder" key, in which case it's
    used too -- see training/phase1.py's save_checkpoint(keep_x_encoder=...)).
    """
    from model.x_encoder.vjepa2 import XEncoder
    from model.predictor.gwen2_5_1_5b import build_model as build_predictor

    x_encoder = XEncoder(device=device)   # frozen by default, see XEncoder.__init__

    predictor, _tokenizer = build_predictor(device=device, vision_dim=x_encoder.vision_dim)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "predictor" not in checkpoint:
        raise ValueError(
            f"{checkpoint_path} has no 'predictor' key -- not a valid "
            "training/phase1.py checkpoint (expected a .deploy.pt or .resume.pt "
            "written by save_checkpoint())."
        )
    predictor.load_state_dict(checkpoint["predictor"])
    predictor.eval()

    if "x_encoder" in checkpoint:
        x_encoder.load_state_dict(checkpoint["x_encoder"])

    return x_encoder, predictor


def setup_logging(log_file):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )


def build_index(args):
    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    if device != args.device:
        logging.warning(f"--device={args.device} requested but CUDA is not available; using cpu instead.")

    logging.info(f"Loading checkpoint from {args.checkpoint} onto {device} ...")
    x_encoder, predictor = load_checkpoint_model(args.checkpoint, device)

    defaults = LEVEL_DEFAULTS[args.level]
    window_size = args.window_size or defaults["window_size"]
    stride = args.stride or defaults["stride"]
    logging.info(f"level={args.level}: window_size={window_size} frames, stride={stride} frames "
                 f"(assumed_fps={ASSUMED_FPS} -- see LEVEL_DEFAULTS docstring)")

    video_paths = load_video_list(args.videos)
    logging.info(f"Loaded {len(video_paths)} video paths from {args.videos}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    store_path = Path(args.segment_store) if args.segment_store else output_path.with_suffix(".segments.db")
    logging.info(f"FAISS index -> {output_path}")
    logging.info(f"Segment metadata store -> {store_path}")

    use_gpu = {"auto": "auto", "true": True, "false": False}[args.use_gpu_index]
    faiss_index = SegmentFaissIndex(dim=EMBEDDING_DIM, use_gpu=use_gpu)
    logging.info(f"FAISS index backend: {faiss_index.backend}")

    extra_kwargs = {}
    if args.distance_threshold is not None:
        extra_kwargs["distance_threshold"] = args.distance_threshold

    pending_ids, pending_vectors = [], []

    def flush(store):
        nonlocal pending_ids, pending_vectors
        if not pending_ids:
            return
        faiss_index.add(np.array(pending_ids, dtype=np.int64), np.stack(pending_vectors))
        pending_ids, pending_vectors = [], []

    num_segments_total = 0
    num_videos_failed = 0
    t_start = time.time()

    iterator = tqdm(video_paths, desc=f"building {args.level} index", unit="video") if (tqdm and not args.no_progress_bar) else video_paths

    with SegmentStore(store_path) as store:
        for video_path in iterator:
            video_id = Path(video_path).stem
            try:
                video_frames = x_encoder.load_video_frames(video_path)
                segments = segment_video(
                    x_encoder, predictor, video_id, video_frames, device=device,
                    window_size=window_size, stride=stride, level=args.level,
                    **extra_kwargs,
                )
            except Exception:
                logging.exception(f"[{video_id}] segmentation failed, skipping this video")
                num_videos_failed += 1
                continue

            for seg in segments:
                segment_id = store.add(seg)
                pending_ids.append(segment_id)
                pending_vectors.append(seg.embedding)
                num_segments_total += 1
                if len(pending_ids) >= args.flush_every:
                    flush(store)

            if num_segments_total and num_segments_total % args.log_every_n_segments < len(segments):
                logging.info(f"... {num_segments_total} segments so far "
                             f"({time.time() - t_start:.0f}s elapsed)")

        flush(store)
        faiss_index.build()
        faiss_index.save(output_path)

    elapsed = time.time() - t_start
    logging.info(
        f"Done in {elapsed:.0f}s. {len(faiss_index)} segments indexed "
        f"({num_videos_failed}/{len(video_paths)} videos failed and were skipped). "
        f"Index: {output_path}  Metadata store: {store_path}"
    )
    if num_videos_failed:
        logging.warning(
            f"{num_videos_failed} video(s) failed segmentation and were skipped -- "
            "check the log above (or --log-file) for per-video tracebacks."
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--level", choices=["coarse", "fine"], required=True)
    parser.add_argument("--checkpoint", required=True, help="Path to a training/phase1.py checkpoint (.deploy.pt or .resume.pt).")
    parser.add_argument("--videos", required=True, help="Path to a text file, one video path/URL per line.")
    parser.add_argument("--output", required=True, help="Path to write the .faiss index file (e.g. INDEX_ROOT/coarse.faiss).")
    parser.add_argument("--segment-store", default=None,
                         help="Path to the SQLite segment store (#69). Defaults to <output> with the "
                              "extension replaced by '.segments.db'.")
    parser.add_argument("--window-size", type=int, default=None,
                         help="Override the level's default sliding-window size, in frames.")
    parser.add_argument("--stride", type=int, default=None,
                         help="Override the level's default sliding-window stride, in frames.")
    parser.add_argument("--distance-threshold", type=float, default=None,
                         help="Override clustering.DEFAULT_WARD_DISTANCE_THRESHOLD.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-gpu-index", choices=["auto", "true", "false"], default="auto",
                         help="Whether to build the FAISS index itself with GPU/cuVS (CAGRA) acceleration. "
                              "See indexing/indexes/faiss_index.py's module docstring for the caveat on this path.")
    parser.add_argument("--flush-every", type=int, default=2000,
                         help="Add embeddings to the FAISS index in batches of this many segments "
                              "(bounds how much is buffered in memory before being handed to faiss.add()).")
    parser.add_argument("--log-every-n-segments", type=int, default=5000,
                         help="Emit a progress log line roughly every this many segments.")
    parser.add_argument("--log-file", default=None, help="Also write logs to this file.")
    parser.add_argument("--no-progress-bar", action="store_true", help="Disable the tqdm progress bar.")
    args = parser.parse_args()

    setup_logging(args.log_file)
    build_index(args)


if __name__ == "__main__":
    main()
