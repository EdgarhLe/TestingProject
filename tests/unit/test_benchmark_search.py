from scripts.benchmark_search import run_benchmark


def test_run_benchmark_returns_expected_keys():
    results = run_benchmark(num_vectors=200, k=5, num_queries=20, use_gpu=False, seed=0)
    expected_keys = {
        "num_vectors", "k", "num_queries", "backend", "build_time_sec",
        "single_query_avg_ms", "batch_query_total_ms", "batch_query_avg_ms",
        "recall_at_k", "recall_caveat",
    }
    assert set(results) == expected_keys
    assert results["num_vectors"] == 200
    assert results["backend"] == "cpu_hnsw"
    assert results["single_query_avg_ms"] >= 0
    assert results["build_time_sec"] >= 0
    assert 0.0 <= results["recall_at_k"] <= 1.0


def test_run_benchmark_can_skip_recall():
    results = run_benchmark(num_vectors=100, k=5, num_queries=10, use_gpu=False, seed=0, measure_recall=False)
    assert "recall_at_k" not in results


def test_run_benchmark_clamps_num_queries_to_num_vectors():
    results = run_benchmark(num_vectors=10, k=3, num_queries=1000, use_gpu=False, seed=1)
    assert results["num_queries"] == 10
