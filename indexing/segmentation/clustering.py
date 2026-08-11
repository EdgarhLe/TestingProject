"""
indexing/segmentation/clustering.py

Step 2 of the semantic segmentation pipeline (#69): cut the denoised S_hat_Y
stream (sliding_window.py) into contiguous segments via agglomerative
clustering with a Ward linkage, under a TEMPORAL CONNECTIVITY CONSTRAINT --
a window may only ever be merged with its immediate temporal neighbor, never
with a window elsewhere in the video, no matter how similar their embeddings
are. Without this constraint, plain Ward clustering could merge two
visually-similar-but-far-apart moments (e.g. the same room shown at the
start and end of a video) into one "segment", producing a nonsensical,
non-contiguous [start_frame, end_frame] range.

Implementation note -- sklearn.cluster.AgglomerativeClustering, not raw
scipy.cluster.hierarchy:
scipy.cluster.hierarchy.linkage() (Ward/euclidean) has no notion of a
connectivity constraint -- it always considers every pair of points as
mergeable, which is exactly what the temporal constraint above rules out.
sklearn.cluster.AgglomerativeClustering supports a `connectivity` graph
alongside linkage="ward"/metric="euclidean" -- this is the standard
scikit-learn recipe for connectivity-constrained clustering (the same
mechanism used for e.g. image segmentation with a pixel-adjacency graph). We
use that here so the temporal constraint from the spec is actually enforced,
while keeping the same Ward-linkage/Euclidean-distance criterion the spec
calls for, and the same distance_threshold (not n_clusters, not a
variance-explained cutoff) semantics for picking where to cut the tree.
"""

import numpy as np
import scipy.sparse as sp
from sklearn.cluster import AgglomerativeClustering


# Placeholder -- tune once real, meaningfully-trained checkpoints exist
# (Week 5+, see pipeline.py's docstring). Distance scale depends entirely on
# how spread out real S_hat_Y embeddings from a trained predictor turn out
# to be, which we don't know yet from an undertrained checkpoint.
DEFAULT_WARD_DISTANCE_THRESHOLD = 15.0


def build_temporal_connectivity(num_windows):
    """
    Chain graph: window i is connected only to i-1 and i+1. This is what
    makes Ward's algorithm respect temporal order -- with this connectivity,
    every cluster reachable at any step of the dendrogram is, by
    construction, a contiguous run of windows (a cluster can only ever grow
    from one of its two ends, since growth requires an edge to an
    unclustered window).

    Returns a symmetric scipy.sparse CSR matrix, or None for num_windows < 2
    (AgglomerativeClustering doesn't need a connectivity graph when there's
    at most one point to place).
    """
    if num_windows < 2:
        return None
    rows = np.arange(num_windows - 1)
    cols = rows + 1
    data = np.ones(num_windows - 1)
    upper = sp.coo_matrix((data, (rows, cols)), shape=(num_windows, num_windows))
    return (upper + upper.T).tocsr()


def cluster_embedding_stream(embeddings, distance_threshold=DEFAULT_WARD_DISTANCE_THRESHOLD):
    """
    embeddings: np.ndarray or torch.Tensor, shape [num_windows, shared_dim].

    distance_threshold, not n_clusters: per spec, "So cluster N: khong co
    dinh -- xac dinh bang ward distance threshold, khong phai variance
    threshold" -- the number of segments falls out of cutting the dendrogram
    at this Ward distance; it is not a fixed cluster count and not a
    variance-explained threshold.

    Returns: np.ndarray[int], length num_windows -- cluster label per
    window, in the SAME time order as the input. Label values are otherwise
    arbitrary integers (not necessarily increasing in time order) --
    group_into_segments() below re-derives segment order from label RUNS
    (consecutive equal labels), not from label value.
    """
    if hasattr(embeddings, "detach"):
        embeddings = embeddings.detach().cpu().numpy()
    # float64: Ward's criterion involves variance/sum-of-squares computations
    # that are numerically more stable in float64 than in the float32 (or
    # bfloat16-derived float32) the embeddings arrive in from the model.
    embeddings = np.asarray(embeddings, dtype=np.float64)

    num_windows = embeddings.shape[0]
    if num_windows == 1:
        return np.array([0])

    connectivity = build_temporal_connectivity(num_windows)
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        linkage="ward",
        metric="euclidean",
        connectivity=connectivity,
    )
    labels = clustering.fit_predict(embeddings)
    return labels


def group_into_segments(labels, window_starts, num_frames):
    """
    Collapse per-window cluster labels into contiguous
    (start_frame, end_frame) segments, in time order, along with the window
    index range [window_idx_start, window_idx_end] (inclusive) each segment
    covers -- callers use that range to pool the segment's representative
    embedding from the same windows, without re-deriving frame overlap a
    second time and risking disagreement with the boundaries computed here.

    Boundary placement: because the sliding window's stride is smaller than
    its window_size (spec: ~5-10 frame stride vs. a much larger window),
    consecutive windows OVERLAP. That means a segment cannot be bounded by
    "where its last member window ends" (window_start + window_size) --
    that end point can fall well past where the NEXT segment's first window
    already starts, producing overlapping segments. Instead, each segment's
    end_frame is defined as the start_frame of the first window assigned to
    the following segment (or num_frames, for the last segment) -- i.e. the
    cut point is where the cluster label changes as we scan forward through
    window start times. This guarantees the returned segments are
    gap-free AND overlap-free and jointly cover exactly [0, num_frames).

    Because of the temporal connectivity constraint, windows sharing a label
    are guaranteed contiguous in window-index order -- but this function
    still detects segment breaks by label CHANGE (rather than assuming
    contiguity) as a defensive check: if a label reappears after a different
    one was seen in between, that means the contiguity guarantee was
    violated upstream (e.g. connectivity wasn't actually passed to
    AgglomerativeClustering), and this raises loudly instead of silently
    emitting corrupted, overlapping segments.

    Returns: list[dict] with keys start_frame, end_frame, window_idx_start,
    window_idx_end. end_frame is exclusive.
    """
    if len(labels) != len(window_starts):
        raise ValueError(
            f"labels ({len(labels)}) and window_starts ({len(window_starts)}) "
            "must have the same length"
        )

    segments = []
    seg_start_window_idx = 0
    seen_labels_so_far = set()

    for i in range(1, len(labels) + 1):
        is_boundary = (i == len(labels)) or (labels[i] != labels[i - 1])
        if not is_boundary:
            continue

        label = labels[seg_start_window_idx]
        if label in seen_labels_so_far:
            raise ValueError(
                f"Cluster label {label} reappeared after a different label was seen in "
                "between -- this breaks the contiguous-segment assumption that relies on "
                "the temporal connectivity constraint (see build_temporal_connectivity). "
                "Check that `connectivity` was actually passed to AgglomerativeClustering."
            )
        seen_labels_so_far.add(label)

        start_frame = window_starts[seg_start_window_idx]
        end_frame = window_starts[i] if i < len(labels) else num_frames

        segments.append({
            "start_frame": start_frame,
            "end_frame": end_frame,
            "window_idx_start": seg_start_window_idx,
            "window_idx_end": i - 1,
        })
        seg_start_window_idx = i

    return segments
