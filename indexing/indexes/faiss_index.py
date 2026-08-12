"""
indexing/indexes/faiss_index.py

FAISS HNSW index wrapper for segment embeddings (#68's output). Two separate
index files are built with this class per the system design:
    coarse.faiss  -- KIS, Q&A stage 1
    fine.faiss    -- Trake, Q&A stage 2
(the split into "coarse"/"fine" is purely which sliding-window granularity
#68 was run with -- this class itself doesn't know or care which one it is;
scripts/build_index.py picks the file name and the segmentation window
size/stride per --level).

Custom integer IDs (segment_id -- see indexing/indexes/segment_store.py):
faiss.IndexHNSWFlat does not natively support custom vector IDs (search
always returns 0..ntotal-1 row indices). The standard fix is to wrap the
index in faiss.IndexIDMap2 -- but that wrapping approach gets genuinely
ambiguous once a GPU/cuVS-built index (see below) is converted from GPU to
CPU as an ALREADY-POPULATED index (does re-wrapping it in IndexIDMap2
double-add the vectors? behavior here isn't something we could verify
without real GPU/cuVS hardware, which this environment doesn't have). To
avoid depending on that specific, hard-to-verify-offline behavior, this
class does its OWN id translation instead of using faiss.IndexIDMap*: it
keeps a plain Python list, `self._segment_ids`, where position i is the
segment_id that was added at physical faiss row i (rows are always assigned
in insertion order, for every faiss index type, GPU or CPU) -- search()
looks up `self._segment_ids[row]` for every raw row index faiss returns.
This is simple enough to reason about and test with certainty on the CPU
backend, and is IDENTICAL code for both backends -- only how
`self._raw_index` gets built differs between them (see `build()`).

Similarity metric: inner product on L2-normalized vectors (cosine
similarity), not raw L2 distance. This is NOT because training used CLIP's
loss -- it didn't (see the system design's "Training objective: Bidirectional
InfoNCE" -- alignment + uniformity regularization, "prevents representation
collapse", not CLIP's symmetric two-encoder loss). The reason is more
specific: the uniformity regularization term (Wang & Isola-style
alignment/uniformity) is defined ON THE UNIT HYPERSPHERE -- it directly
optimizes normalized embeddings for good angular separation. Separately,
model/vl_jepa.py's own logit computation is explicitly `scale *
(P_norm @ T_norm^T)` -- L2-normalized dot product with a learnable scale
(that comment calls this "like CLIP", but it's describing just the logit
computation, not claiming the whole objective is CLIP's). Both training
components -- the hypersphere-defined uniformity term AND the normalized-
dot-product logits -- point at the same conclusion: retrieval should use
cosine similarity on normalized embeddings, which is what `normalize`
(default True) does automatically for both indexed and query vectors.

GPU / cuVS acceleration ("advanced faiss"):
Plain faiss-cpu has no GPU path for HNSW at all -- HNSW's graph traversal
has no classic-GPU-faiss equivalent (GpuIndexFlat/IVFFlat/IVFPQ exist, but
not GpuIndexHNSW). NVIDIA's cuVS integration into faiss adds
`GpuIndexCagra`, a GPU-native graph index that can be BUILT much faster than
CPU HNSW (this matters here: the spec calls out that a full-dataset build
"will take many hours"), and then converted into a CPU-searchable
`IndexHNSWCagra` -- i.e. GPU-accelerated build, then a normal portable CPU
index for serving, with no GPU required at search time.

This class's GPU path (`_build_gpu_cagra_index`) follows the official
recipe from https://github.com/facebookresearch/faiss/wiki/GPU-Faiss-with-cuVS-usage
("CAGRA + HNSW" section):

    faiss::gpu::GpuIndexCagra gpu_cagra_index(res, d);
    gpu_cagra_index.train(n, xb);          // NOT .add() -- see below
    faiss::IndexHNSWCagra* cpu_hnsw_index;
    cpu_hnsw_index->base_level_only = true; // we never add() more vectors after this
    gpu_cagra_index.copyTo(cpu_hnsw_index);

plus explicitly setting `GpuIndexCagraConfig.use_cuvs = True`, which the
same page documents as required to actually dispatch to the cuVS backend
(it is NOT implied just by GpuIndexCagra existing). A second, newer-looking
pattern also appears in the wild (e.g. a 2026 NVIDIA/cuVS deployment
writeup) using `.add(xb)` directly and a plain `faiss.index_gpu_to_cpu()`
call instead of the explicit IndexHNSWCagra/copyTo dance -- this may be a
convenience path added in a later faiss version. Because this environment
has no GPU/cuVS build to verify either against, `_build_gpu_cagra_index`
tries the officially-documented recipe first and falls back to the
`.add()`/`index_gpu_to_cpu()` variant if that raises, before finally
falling back to CPU IndexHNSWFlat (see `build()`) if both GPU attempts
fail. Please check the logs on the real machine to see which path actually
ran, and file a note back here if one of them turns out to be wrong for
your installed faiss/cuVS version.

IMPORTANT CAVEAT: this environment has no GPU and no cuVS-enabled faiss
build available to develop/test against (only plain `faiss-cpu` is
installed here -- `faiss.get_num_gpus()` / `faiss.GpuIndexCagra` etc. simply
don't exist in this build). Both GPU/cuVS code paths below are written from
the documented cuVS/faiss integration (fetched from the official faiss wiki
and cross-checked against a second source) but have NOT been exercised
end-to-end on real GPU hardware. They are feature-detected and wrapped in a
broad try/except that falls back to the CPU path on any failure, so this
class can never leave an index in a broken state -- but please smoke-test
the GPU path on the real machine (the one showing `faiss-gpu-cuvs` in
`conda list`) before relying on it for the Week 5 real build. The CPU path
is the one that is fully tested in this repo's test suite and is what
actually runs whenever GPU/cuVS isn't available or both build attempts
raise.
"""

import logging
from pathlib import Path

import numpy as np
import faiss

logger = logging.getLogger(__name__)

# Must match indexing.segmentation.schema.EMBEDDING_DIM (1536). Duplicated
# here rather than imported for the same reason schema.py duplicates
# model.vl_jepa.SHARED_EMBED_DIM: this module should be importable (e.g. by
# a lightweight search-serving process) without pulling in torch/transformers.
EMBEDDING_DIM = 1536

DEFAULT_M = 32                  # HNSW: number of bi-directional links per node
DEFAULT_EF_CONSTRUCTION = 200   # HNSW: build-time search breadth (bigger = better recall, slower build)
DEFAULT_EF_SEARCH = 64          # HNSW: query-time search breadth (bigger = better recall, slower search)


def _gpu_cuvs_available():
    """
    True only if this faiss build actually has GPU + cuVS's CAGRA index
    available. Checked defensively (missing attributes, or a GPU-less
    machine reporting get_num_gpus() == 0) rather than assumed -- this is
    what makes use_gpu="auto" safe to leave on by default in code that will
    run on machines with and without a GPU faiss build.
    """
    try:
        return (
            hasattr(faiss, "StandardGpuResources")
            and hasattr(faiss, "GpuIndexCagra")
            and hasattr(faiss, "GpuIndexCagraConfig")
            and faiss.get_num_gpus() > 0
        )
    except Exception:
        return False


class SegmentFaissIndex:
    """
    dim: embedding dimensionality (1536, matching the shared VL-JEPA space).
    metric: "ip" (inner product -- the default, used with L2-normalized
        vectors to get cosine similarity, matching training) or "l2".
    normalize: L2-normalize vectors before adding/searching. Only meaningful
        (and only applied) when metric="ip" -- leave True unless you have a
        specific reason to search in raw (unnormalized) embedding space.
    use_gpu: "auto" (default, use GPU/cuVS if available else CPU HNSW),
        True (force GPU/cuVS -- raises if unavailable), False (force CPU
        HNSW).
    """

    def __init__(self, dim=EMBEDDING_DIM, metric="ip", normalize=True, use_gpu="auto",
                 M=DEFAULT_M, ef_construction=DEFAULT_EF_CONSTRUCTION, ef_search=DEFAULT_EF_SEARCH,
                 graph_degree=None):
        if metric not in ("ip", "l2"):
            raise ValueError(f"metric must be 'ip' or 'l2', got {metric!r}")

        self.dim = dim
        self.metric = metric
        self.normalize = normalize and metric == "ip"
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.graph_degree = graph_degree or M   # CAGRA's graph_degree plays roughly the same
                                                  # structural role as HNSW's M

        self.backend = self._resolve_backend(use_gpu)

        self._segment_ids = []          # python list[int], position i == faiss row i's segment_id
        self._raw_index = None          # built immediately for cpu_hnsw, lazily in build() for gpu_cagra
        self._pending_vectors = []      # only used by gpu_cagra, buffered until build()
        self._built = False

        if self.backend == "cpu_hnsw":
            self._raw_index = self._make_cpu_hnsw_index()
            self._built = True

    def _resolve_backend(self, use_gpu):
        if use_gpu is True:
            if not _gpu_cuvs_available():
                raise RuntimeError(
                    "use_gpu=True was requested but no GPU/cuVS-enabled faiss build is available "
                    "in this environment (faiss.get_num_gpus() == 0, or GpuIndexCagra isn't present "
                    "in this faiss build -- e.g. plain faiss-cpu). Install a cuVS-enabled faiss build "
                    "(e.g. `conda install -c pytorch -c nvidia faiss-gpu-cuvs`) on a GPU machine, or "
                    "pass use_gpu='auto'/False."
                )
            return "gpu_cagra"
        if use_gpu is False:
            return "cpu_hnsw"
        if use_gpu == "auto":
            return "gpu_cagra" if _gpu_cuvs_available() else "cpu_hnsw"
        raise ValueError(f"use_gpu must be True, False, or 'auto', got {use_gpu!r}")

    def _faiss_metric(self):
        return faiss.METRIC_INNER_PRODUCT if self.metric == "ip" else faiss.METRIC_L2

    def _make_cpu_hnsw_index(self):
        index = faiss.IndexHNSWFlat(self.dim, self.M, self._faiss_metric())
        index.hnsw.efConstruction = self.ef_construction
        index.hnsw.efSearch = self.ef_search
        return index

    def _maybe_normalize(self, vectors):
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if self.normalize:
            vectors = vectors.copy()   # faiss.normalize_L2 mutates in place -- never mutate the caller's array
            faiss.normalize_L2(vectors)
        return vectors

    # ------------------------------------------------------------------
    # Add / build
    # ------------------------------------------------------------------
    def add(self, ids, vectors):
        """
        ids: array-like[int], shape [N] -- these are segment_id values from
            indexing.indexes.segment_store.SegmentStore.add(), NOT arbitrary
            IDs (search() will hand these exact values back).
        vectors: array-like[float32], shape [N, dim].

        For the cpu_hnsw backend, vectors are inserted into the live index
        immediately (true incremental add -- safe to call once per video
        while a long build is running). For gpu_cagra, vectors are buffered
        in memory and the actual GPU graph build happens once, in build()
        (CAGRA's GPU speed advantage comes from building the whole graph in
        one batch, not from many small incremental updates) -- call build()
        explicitly once all videos are processed, or rely on search()
        calling it for you.
        """
        ids = np.asarray(ids, dtype=np.int64)
        vectors = self._maybe_normalize(vectors)
        if vectors.shape != (len(ids), self.dim):
            raise ValueError(
                f"vectors must have shape ({len(ids)}, {self.dim}), got {vectors.shape}"
            )

        if self.backend == "cpu_hnsw":
            self._raw_index.add(vectors)
        else:
            self._pending_vectors.append(vectors)
            self._built = False   # a fresh build() is needed to pick up these new vectors

        self._segment_ids.extend(int(x) for x in ids)

    def build(self):
        """
        No-op for cpu_hnsw (already incrementally built by add()). For
        gpu_cagra, performs the actual batched GPU graph construction over
        every vector buffered so far via add(), then converts the result to
        a CPU-searchable index via faiss.index_gpu_to_cpu() -- from that
        point on, self._raw_index is a normal CPU faiss.Index (an
        IndexHNSWCagra), and save()/search() don't need to know or care
        which backend produced it.

        Idempotent: calling this again after add()ing more vectors rebuilds
        from all buffered vectors (cheap no-op if nothing changed since the
        last build).
        """
        if self._built:
            return
        if self.backend == "cpu_hnsw":
            self._built = True
            return

        try:
            self._raw_index = self._build_gpu_cagra_index()
            self._built = True
        except Exception:
            logger.exception(
                "GPU/cuVS CAGRA index build failed -- falling back to CPU IndexHNSWFlat. "
                "See the module docstring: this path is unverified in environments without "
                "real GPU/cuVS hardware, and this fallback is exactly the safety net for that."
            )
            self.backend = "cpu_hnsw"
            self._raw_index = self._make_cpu_hnsw_index()
            if self._pending_vectors:
                self._raw_index.add(np.concatenate(self._pending_vectors, axis=0))
            self._pending_vectors = []
            self._built = True

    def _build_gpu_cagra_index(self):
        """
        See the module docstring's GPU/cuVS section -- tries the officially
        documented recipe (.train() + IndexHNSWCagra + copyTo) first, falls
        back to a simpler (.add() + index_gpu_to_cpu()) variant seen
        elsewhere, and lets the caller (build()) fall back to CPU HNSW if
        both raise. Neither path has been run against real GPU/cuVS
        hardware in this environment.
        """
        if not self._pending_vectors:
            raise RuntimeError("build() called on an empty gpu_cagra index -- nothing was add()ed yet")
        all_vectors = np.concatenate(self._pending_vectors, axis=0)

        try:
            return self._build_gpu_cagra_index_official_recipe(all_vectors)
        except Exception:
            logger.exception(
                "Official GpuIndexCagra recipe (.train() + IndexHNSWCagra.copyTo()) failed -- "
                "trying the .add() + index_gpu_to_cpu() variant instead."
            )
            return self._build_gpu_cagra_index_add_variant(all_vectors)

    def _build_gpu_cagra_index_official_recipe(self, all_vectors):
        """
        Per https://github.com/facebookresearch/faiss/wiki/GPU-Faiss-with-cuVS-usage
        ("CAGRA + HNSW" section):

            faiss::gpu::GpuIndexCagra gpu_cagra_index(res, d);
            gpu_cagra_index.train(n, xb);
            faiss::IndexHNSWCagra* cpu_hnsw_index;
            cpu_hnsw_index->base_level_only = false;   // or true, see below
            gpu_cagra_index.copyTo(cpu_hnsw_index);

        base_level_only=True here (rather than the wiki example's False):
        the wiki notes True gives an immutable-after-build HNSW index that
        can't have vectors added later, which is exactly this class's usage
        pattern -- build() only ever runs once per full batch of buffered
        vectors, never incrementally afterward, so there is nothing to lose
        by taking the (likely faster/more optimized) immutable form.
        """
        res = faiss.StandardGpuResources()
        config = faiss.GpuIndexCagraConfig()
        config.graph_degree = self.graph_degree
        config.use_cuvs = True   # required to actually dispatch to the cuVS backend, not just the class existing

        gpu_index = faiss.GpuIndexCagra(res, self.dim, self._faiss_metric(), config)
        gpu_index.train(all_vectors)

        cpu_index = faiss.IndexHNSWCagra()
        cpu_index.base_level_only = True
        gpu_index.copyTo(cpu_index)

        self._pending_vectors = []
        return cpu_index

    def _build_gpu_cagra_index_add_variant(self, all_vectors):
        """
        Alternate recipe seen in a (non-official, possibly version-specific)
        source: build via .add() directly and convert with the generic
        faiss.index_gpu_to_cpu() rather than manually constructing
        IndexHNSWCagra + copyTo(). Kept as a fallback in case the officially
        documented recipe above doesn't match the installed faiss/cuVS
        version.
        """
        res = faiss.StandardGpuResources()
        config = faiss.GpuIndexCagraConfig()
        config.graph_degree = self.graph_degree
        config.use_cuvs = True

        gpu_index = faiss.GpuIndexCagra(res, self.dim, self._faiss_metric(), config)
        gpu_index.add(all_vectors)
        cpu_index = faiss.index_gpu_to_cpu(gpu_index)

        self._pending_vectors = []
        return cpu_index

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(self, query_vectors, k=5):
        """
        query_vectors: array-like[float32], shape [Q, dim] (a single query
            still needs shape [1, dim]).
        k: number of nearest neighbors per query.

        Returns (distances, segment_ids), both shape [Q, k]. distances are
        inner products (higher = more similar) if metric="ip", or squared
        L2 distances (lower = more similar) if metric="l2". segment_ids
        entries are -1 wherever faiss returned no match (e.g. k > ntotal)
        -- exactly what a plain faiss index returns for the row index in
        that case, translated the same way.
        """
        self.build()   # ensures a gpu_cagra index buffered via add() has actually been built
        if self._raw_index is None or self._raw_index.ntotal == 0:
            raise RuntimeError("search() called on an empty index -- add() some vectors and build() first")

        query_vectors = self._maybe_normalize(query_vectors)
        distances, rows = self._raw_index.search(query_vectors, k)

        id_map = np.array(self._segment_ids, dtype=np.int64)
        segment_ids = np.full(rows.shape, -1, dtype=np.int64)
        valid = rows >= 0
        segment_ids[valid] = id_map[rows[valid]]
        return distances, segment_ids

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path):
        """
        Writes two files: `path` (the faiss index itself, standard
        faiss.write_index format -- works the same whether self._raw_index
        came from the cpu_hnsw or gpu_cagra path, since both are plain CPU
        faiss.Index objects by the time build() finishes) and
        `path + '.ids.npy'` (the segment_id array, position i == row i).
        """
        self.build()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._raw_index, str(path))
        np.save(str(path) + ".ids.npy", np.array(self._segment_ids, dtype=np.int64))

    @classmethod
    def load(cls, path, metric="ip", normalize=True):
        """
        Loads an index previously written by save(). The returned instance
        always reports backend="loaded_cpu" -- once saved, both the
        cpu_hnsw and gpu_cagra backends produce the identical CPU-searchable
        artifact, so there is nothing GPU-specific left to know about at
        load time (no GPU is needed to load or search a saved index, even
        if it was originally built with GPU/cuVS acceleration).
        """
        path = Path(path)
        obj = cls.__new__(cls)
        obj._raw_index = faiss.read_index(str(path))
        obj._segment_ids = np.load(str(path) + ".ids.npy").tolist()
        obj.dim = obj._raw_index.d
        obj.metric = metric
        obj.normalize = normalize and metric == "ip"
        obj.backend = "loaded_cpu"
        obj._pending_vectors = []
        obj._built = True
        obj.M = obj.ef_construction = obj.ef_search = obj.graph_degree = None
        return obj

    def __len__(self):
        return 0 if self._raw_index is None else self._raw_index.ntotal


def recall_at_k(ann_index, vectors, ids, k=10, num_queries=200, seed=0):
    """
    Measures `ann_index`'s recall@k against an EXACT brute-force baseline
    (faiss.IndexFlat, same metric/normalization as ann_index, over the same
    vectors) -- i.e. "of each query's true top-k nearest neighbors, what
    fraction did the ANN index actually return?"

    This exists because HNSWFlat's accuracy is a function of its parameters
    (M, efConstruction, efSearch) AND the data's own intrinsic
    dimensionality -- it is not a fixed property of "HNSW" as an algorithm,
    and it should be measured, not assumed. See faiss's own published
    SIFT1M benchmark (https://github.com/facebookresearch/faiss/wiki/Indexing-1M-vectors):
    recall@1 for HNSWFlat goes from 0.874 at efSearch=16 to 0.992 at
    efSearch=256 -- the same index type, only the ef parameter changed.

    IMPORTANT: a recall number measured on dummy/random vectors (as in this
    repo's tests) is NOT representative of recall on real VL-JEPA
    embeddings -- real embedding distributions have their own intrinsic
    dimensionality and cluster structure that materially affects HNSW
    recall, and that's unknown until a real trained checkpoint exists
    (Week 5). Re-run this against real embeddings once available; treat any
    number from dummy data as a methodology check, not a quality claim.

    Returns a float in [0, 1]: the mean, over `num_queries` random queries,
    of |ANN top-k ∩ exact top-k| / |exact top-k|.
    """
    ids = np.asarray(ids, dtype=np.int64)
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)

    metric = faiss.METRIC_INNER_PRODUCT if ann_index.metric == "ip" else faiss.METRIC_L2
    reference_vectors = vectors.copy()
    if ann_index.normalize:
        faiss.normalize_L2(reference_vectors)

    # A plain IndexFlat wrapped in IndexIDMap2, built from scratch and
    # populated via add_with_ids in one shot -- this is the standard,
    # unambiguous IndexIDMap2 use case (unlike the GPU/cuVS conversion
    # scenario this module avoids IndexIDMap2 for elsewhere).
    brute_force = faiss.IndexIDMap2(faiss.IndexFlat(ann_index.dim, metric))
    brute_force.add_with_ids(reference_vectors, ids)

    rng = np.random.default_rng(seed)
    num_queries = min(num_queries, len(ids))
    query_positions = rng.choice(len(ids), size=num_queries, replace=False)
    queries = vectors[query_positions]

    _, ann_result_ids = ann_index.search(queries, k=k)
    exact_queries = reference_vectors[query_positions] if ann_index.normalize else queries
    _, exact_result_ids = brute_force.search(exact_queries, k)

    per_query_recall = []
    for i in range(num_queries):
        true_neighbors = {int(x) for x in exact_result_ids[i] if x != -1}
        found_neighbors = {int(x) for x in ann_result_ids[i] if x != -1}
        if not true_neighbors:
            continue
        per_query_recall.append(len(true_neighbors & found_neighbors) / len(true_neighbors))

    return float(np.mean(per_query_recall)) if per_query_recall else float("nan")
