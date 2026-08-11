"""
indexing/indexes/segment_store.py

SegmentStore: the metadata "lookup table" that sits behind the FAISS index
(#70). FAISS only ever returns an integer vector ID -- this store is what
maps that ID back to real video/frame information (and is where OCR/ASR
text, filled in separately by DE, gets attached).

Row schema (see SCHEMA.md for the version shared with DE):
    segment_id      INTEGER PRIMARY KEY AUTOINCREMENT  -- the FAISS vector ID
    video_id        TEXT
    start_frame     INTEGER
    end_frame       INTEGER
    midpoint_frame  INTEGER
    embedding       BLOB   (float32[EMBEDDING_DIM], packed via ndarray.tobytes())
    ocr_text        TEXT, nullable  -- filled in later via update_ocr()
    asr_text        TEXT, nullable  -- filled in later via update_asr()
    level           TEXT   ('coarse' | 'fine')

Why AUTOINCREMENT and not plain INTEGER PRIMARY KEY:
Plain SQLite rowids can be reused after a row is deleted (SQLite may hand
out a previously-freed rowid to a new row). segment_id doubles as the FAISS
vector ID, so a reused ID would silently point a stale FAISS vector at a
different segment's metadata -- a correctness bug that would be very hard to
notice. AUTOINCREMENT enforces monotonically increasing IDs that are never
reused (SQLite tracks the historical maximum in an internal
`sqlite_sequence` table), at the (here irrelevant) cost of being slightly
slower than plain rowids.

Why SQLite (vs. JSON-per-video) at this scale:
Access here is dominated by point lookups (`get(segment_id)`, one row via
the primary key -- exactly what FAISS search results need) and per-video
scans (`get_by_video`, via a secondary index on video_id), with writes
concentrated in one batch (indexer run) plus low-frequency updates
afterward (DE filling in OCR/ASR). SQLite handles millions of rows fine
under that access pattern, needs no server (matches how checkpoints are
already shipped as flat files -- see training/phase1.py's
sync_checkpoints_to_machine2()), and WAL mode lets readers proceed
concurrently with a writer. This does NOT hold up if the store later needs
many concurrent high-throughput writers or serving at very high read QPS
across machines -- at that point the same schema migrates cleanly to
Postgres/DuckDB, but that's not today's requirement.
"""

import sqlite3
from pathlib import Path

import numpy as np

EMBEDDING_DIM = 1536
VALID_LEVELS = ("coarse", "fine")

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS segments (
    segment_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id       TEXT NOT NULL,
    start_frame    INTEGER NOT NULL,
    end_frame      INTEGER NOT NULL,
    midpoint_frame INTEGER NOT NULL,
    embedding      BLOB NOT NULL,
    ocr_text       TEXT,
    asr_text       TEXT,
    level          TEXT NOT NULL CHECK (level IN ('coarse', 'fine')),
    CHECK (start_frame < end_frame),
    CHECK (start_frame <= midpoint_frame AND midpoint_frame < end_frame)
);
CREATE INDEX IF NOT EXISTS idx_segments_video_id ON segments(video_id);
"""


class SegmentStore:
    """
    SQLite-backed segment metadata store.

    Usage:
        with SegmentStore("segments.db") as store:
            segment_id = store.add(segment_record)   # int, also the FAISS vector ID
            record = store.get(segment_id)
            video_records = store.get_by_video("video_001")
            store.update_ocr(segment_id, "some ocr text")
            store.update_asr(segment_id, "some asr text")

    segment_record passed to add() can be EITHER:
      - a indexing.segmentation.schema.SegmentMetadata instance (#68's output;
        ocr_text/asr_text are absent on that class and default to NULL here,
        to be filled in later by DE), or
      - a plain dict with keys video_id, start_frame, end_frame,
        midpoint_frame, embedding, level, and optionally ocr_text/asr_text.
    """

    def __init__(self, db_path, embedding_dim=EMBEDDING_DIM):
        self.db_path = str(db_path)
        self.embedding_dim = embedding_dim

        # check_same_thread=False: this store may be constructed once and
        # handed to multiple worker threads (e.g. one thread streaming
        # segments in while another serves reads) -- callers are still
        # responsible for not interleaving writes from multiple threads
        # without external locking, same as any other sqlite3 connection.
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def close(self):
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __len__(self):
        row = self._conn.execute("SELECT COUNT(*) AS n FROM segments").fetchone()
        return row["n"]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _unpack_record(self, segment_record):
        """Accept either a SegmentMetadata-like object (attribute access) or a
        dict (key access); ocr_text/asr_text default to None either way, since
        #68's SegmentMetadata doesn't carry them and DE fills them in later."""
        if isinstance(segment_record, dict):
            get = segment_record.get
            required = ("video_id", "start_frame", "end_frame", "midpoint_frame", "embedding", "level")
            missing = [k for k in required if k not in segment_record]
            if missing:
                raise ValueError(f"segment_record dict is missing required key(s): {missing}")
        else:
            get = lambda key, default=None: getattr(segment_record, key, default)
            for required_attr in ("video_id", "start_frame", "end_frame", "midpoint_frame", "embedding", "level"):
                if not hasattr(segment_record, required_attr):
                    raise ValueError(
                        f"segment_record is missing required attribute {required_attr!r} "
                        "(expected a SegmentMetadata instance or an equivalent dict)"
                    )

        return {
            "video_id": get("video_id"),
            "start_frame": get("start_frame"),
            "end_frame": get("end_frame"),
            "midpoint_frame": get("midpoint_frame"),
            "embedding": get("embedding"),
            "level": get("level"),
            "ocr_text": get("ocr_text", None),
            "asr_text": get("asr_text", None),
        }

    def _embedding_to_blob(self, embedding):
        arr = np.asarray(embedding, dtype=np.float32)
        if arr.shape != (self.embedding_dim,):
            raise ValueError(
                f"embedding must have shape ({self.embedding_dim},), got {arr.shape}"
            )
        return arr.tobytes()

    @staticmethod
    def _blob_to_embedding(blob):
        # np.frombuffer returns a read-only view over the sqlite3-owned bytes
        # object; .copy() makes it a normal, writable, independently-owned
        # array so callers can't accidentally corrupt driver-internal memory.
        return np.frombuffer(blob, dtype=np.float32).copy()

    def _row_to_record(self, row):
        return {
            "segment_id": row["segment_id"],
            "video_id": row["video_id"],
            "start_frame": row["start_frame"],
            "end_frame": row["end_frame"],
            "midpoint_frame": row["midpoint_frame"],
            "embedding": self._blob_to_embedding(row["embedding"]),
            "ocr_text": row["ocr_text"],
            "asr_text": row["asr_text"],
            "level": row["level"],
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def add(self, segment_record):
        """
        Insert one segment. Returns the newly assigned segment_id (int) --
        this is also the ID that must be used as the corresponding vector's
        ID in the FAISS index (#70), so the two stay in lockstep.
        """
        fields = self._unpack_record(segment_record)
        if fields["level"] not in VALID_LEVELS:
            raise ValueError(f"level must be one of {VALID_LEVELS}, got {fields['level']!r}")
        blob = self._embedding_to_blob(fields["embedding"])

        cur = self._conn.execute(
            """
            INSERT INTO segments
                (video_id, start_frame, end_frame, midpoint_frame, embedding, ocr_text, asr_text, level)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                fields["video_id"], fields["start_frame"], fields["end_frame"],
                fields["midpoint_frame"], blob, fields["ocr_text"], fields["asr_text"],
                fields["level"],
            ),
        )
        self._conn.commit()
        return cur.lastrowid

    def get(self, segment_id):
        """Returns a dict record, or None if segment_id doesn't exist."""
        row = self._conn.execute(
            "SELECT * FROM segments WHERE segment_id = ?", (segment_id,)
        ).fetchone()
        return self._row_to_record(row) if row is not None else None

    def get_by_video(self, video_id):
        """Returns list[dict], all segments for one video, ordered by start_frame."""
        rows = self._conn.execute(
            "SELECT * FROM segments WHERE video_id = ? ORDER BY start_frame", (video_id,)
        ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def update_ocr(self, segment_id, text):
        """DE calls this to fill in OCR text for a segment once it's extracted."""
        cur = self._conn.execute(
            "UPDATE segments SET ocr_text = ? WHERE segment_id = ?", (text, segment_id)
        )
        self._conn.commit()
        if cur.rowcount == 0:
            raise KeyError(f"No segment with segment_id={segment_id}")

    def update_asr(self, segment_id, text):
        """DE calls this to fill in ASR text for a segment once it's transcribed."""
        cur = self._conn.execute(
            "UPDATE segments SET asr_text = ? WHERE segment_id = ?", (text, segment_id)
        )
        self._conn.commit()
        if cur.rowcount == 0:
            raise KeyError(f"No segment with segment_id={segment_id}")
