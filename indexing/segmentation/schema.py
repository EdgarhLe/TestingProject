"""
indexing/segmentation/schema.py

Segment metadata schema, per system design doc section 6.2:

    {video_id, start_frame, end_frame, midpoint_frame,
     embedding: float32[1536], level: "coarse"}

Kept as a standalone dataclass (not defined inline in pipeline.py) so #70
(the FAISS index builder) and any future reader/consumer of segment metadata
can `from indexing.segmentation.schema import SegmentMetadata` without
pulling in the whole segmentation pipeline (X-Encoder / Predictor
construction, torch model loading, etc.) -- this module only depends on
numpy.
"""

from dataclasses import dataclass, asdict
import json

import numpy as np

# Must match model.vl_jepa.SHARED_EMBED_DIM / configs/model.yaml's top-level
# embedding_dim (1536). Duplicated here rather than importing model.vl_jepa
# so this schema module stays lightweight (no torch / transformers import
# just to read or validate a jsonl file of segment metadata).
EMBEDDING_DIM = 1536


@dataclass
class SegmentMetadata:
    video_id: str
    start_frame: int
    end_frame: int
    midpoint_frame: int
    embedding: np.ndarray          # float32[EMBEDDING_DIM]
    level: str = "coarse"          # only "coarse" exists today (this pipeline's single
                                    # sliding-window + cluster pass). A future finer pass
                                    # that re-segments *inside* a coarse segment would emit
                                    # "fine" records referencing the same video_id.

    def __post_init__(self):
        self.embedding = np.asarray(self.embedding, dtype=np.float32)
        if self.embedding.shape != (EMBEDDING_DIM,):
            raise ValueError(
                f"SegmentMetadata.embedding must have shape ({EMBEDDING_DIM},), got "
                f"{self.embedding.shape} (video_id={self.video_id!r}, "
                f"start_frame={self.start_frame})"
            )
        if self.start_frame >= self.end_frame:
            raise ValueError(
                f"start_frame ({self.start_frame}) must be < end_frame "
                f"({self.end_frame}) (video_id={self.video_id!r})"
            )
        if not (self.start_frame <= self.midpoint_frame < self.end_frame):
            raise ValueError(
                f"midpoint_frame ({self.midpoint_frame}) must satisfy "
                f"start_frame <= midpoint_frame < end_frame "
                f"({self.start_frame} <= x < {self.end_frame}) (video_id={self.video_id!r})"
            )

    def to_json_dict(self):
        """JSON-safe dict -- embedding as a plain list instead of an ndarray."""
        d = asdict(self)
        d["embedding"] = self.embedding.tolist()
        return d

    def to_json_line(self):
        return json.dumps(self.to_json_dict())

    @classmethod
    def from_json_dict(cls, d):
        return cls(**d)


def write_segments_jsonl(segments, path):
    """segments: list[SegmentMetadata] -> one JSON object per line at `path`."""
    with open(path, "w", encoding="utf-8") as f:
        for seg in segments:
            f.write(seg.to_json_line() + "\n")


def read_segments_jsonl(path):
    segments = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            segments.append(SegmentMetadata.from_json_dict(json.loads(line)))
    return segments
