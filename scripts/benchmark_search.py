#!/usr/bin/env python3
"""
scripts/benchmark_search.py

Standalone benchmark for indexing/indexes/faiss_index.py: builds an index
from N random float32[1536] vectors and measures search latency. Meant to
be rerun at increasing --num-vectors as the real dataset grows (spec: "sẽ
scale lên vài triệu khi full dataset, cần biết baseline") -- this week's
baseline is N=1000; rerun with e.g. --num-vectors 100000 / 1000000 once
larger dummy runs (or real data) are available, to see how latency actually
scales, rather than guessing.

    python scripts/benchmark_search.py --num-vectors 1000
    python scripts/benchmark_search.py --num-vectors 1000 --output bench_1000.json
"""

import argparse
import json
import time

import numpy as np

from indexing.indexes.faiss_index import SegmentFaissIndex, EMBEDDING_DIM, recall_at_k


def run_benchmark(num_vectors, k, num_queries, use_gpu, seed=0, measure_recall=True):
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((num_vectors, EMBEDDING_DIM)).astype(np.float32)
    ids = np.arange(num_vectors, dtype=np.int64)

    index = SegmentFaissIndex(dim=EMBEDDING_DIM, use_gpu=use_gpu)

    t0 = time.perf_counter()
    index.add(ids, vectors)
    index.build()
    build_time_sec = time.perf_counter() - t0

    num_queries = min(num_queries, num_vectors)
    query_idx = rng.choice(num_vectors, size=num_queries, replace=False)
    queries = vectors[query_idx]

    index.search(queries[:1], k=k)   # warm-up, exclude first-call overhead

    t0 = time.perf_counter()
    for i in range(num_queries):
        index.search(queries[i:i + 1], k=k)
    single_query_total_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    index.search(queries, k=k)
    batch_query_total_sec = time.perf_counter() - t0

    results = {
        "num_vectors": num_vectors,
        "k": k,
        "num_queries": num_queries,
        "backend": index.backend,
        "build_time_sec": round(build_time_sec, 4),
        "single_query_avg_ms": round((single_query_total_sec / num_queries) * 1000, 4),
        "batch_query_total_ms": round(batch_query_total_sec * 1000, 4),
        "batch_query_avg_ms": round((batch_query_total_sec / num_queries) * 1000, 4),
    }

    if measure_recall:
        # NOTE: on random Gaussian dummy vectors (as here), recall tends to
        # look very good regardless of ef parameters -- this number is a
        # methodology check, NOT a claim about recall on real VL-JEPA
        # embeddings. See indexing/indexes/faiss_index.py's recall_at_k()
        # docstring. Re-run this benchmark against real embeddings once
        # available (Week 5+) for a number that actually matters.
        results["recall_at_k"] = round(recall_at_k(index, vectors, ids, k=k, num_queries=num_queries, seed=seed + 1), 4)
        results["recall_caveat"] = (
            "measured on random dummy vectors, not representative of real VL-JEPA "
            "embedding recall -- re-measure on real embeddings once available"
        )

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-vectors", type=int, default=1000)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--use-gpu-index", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-recall", action="store_true", help="Skip the recall@k computation (faster, latency-only).")
    parser.add_argument("--output", default=None, help="Optional path to also write results as JSON.")
    args = parser.parse_args()

    use_gpu = {"auto": "auto", "true": True, "false": False}[args.use_gpu_index]
    results = run_benchmark(args.num_vectors, args.k, args.num_queries, use_gpu,
                             seed=args.seed, measure_recall=not args.no_recall)

    print(json.dumps(results, indent=2))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Wrote results to {args.output}")


if __name__ == "__main__":
    main()
