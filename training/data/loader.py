"""Phase 1 DataLoader — training/data/loader.py.

Reads phase1_combined.jsonl (training/data/preprocess.py's output). Every
row's file is guaranteed to already exist on disk (preprocessing only writes
a row after successfully downloading and decoding it), so this never needs
a cache-miss/download fallback -- a missing file here means a preprocessing
bug, not something to recover from at load time.

One caption per sample -- no prompt ensembling (dropped for Phase 1: single
raw caption, no fixed event/content prompts, no Y-Encoder averaging).

Batches are (video_frames, captions) tuples: video_frames is a
(B, T, C, size, size) uint8 tensor, captions is a list[str] of length B.
"""

import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_image(path: Path) -> torch.Tensor:
    """Decode an image into a (3, H, W) uint8 tensor, RGB."""
    with Image.open(path) as img:
        array = np.array(img.convert("RGB"))
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _load_video(path: Path, num_frames: int) -> torch.Tensor:
    """Decode a video, uniformly sampling num_frames frames, into a
    (num_frames, 3, H, W) uint8 tensor, RGB."""
    capture = cv2.VideoCapture(str(path))
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            raise ValueError(f"video at {path} has no readable frames")
        indices = torch.linspace(0, total - 1, num_frames).round().long().tolist()
        frames = []
        for idx in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame_bgr = capture.read()
            if ok:
                frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            elif frames:
                frames.append(frames[-1])
            else:
                raise ValueError(f"video at {path} has no readable frames")
    finally:
        capture.release()
    array = np.stack(frames)
    return torch.from_numpy(array).permute(0, 3, 1, 2).contiguous()


def _resize(media: torch.Tensor, size: int) -> torch.Tensor:
    """Resize a (T, C, H, W) uint8 tensor's spatial dims to (size, size)."""
    resized = F.interpolate(media.float(), size=(size, size), mode="bilinear", align_corners=False)
    return resized.round().clamp(0, 255).to(torch.uint8)


class Phase1Dataset(Dataset):
    """Map-style dataset over phase1_combined.jsonl.

    T (frames per clip) is mutable for curriculum learning:
    `loader.dataset.T = 4` changes it mid-run without rebuilding the loader.
    Images are repeated to fill T (Bain et al. 2021's image-as-single-frame-
    clip convention, which this project's curriculum is modeled on).
    """

    def __init__(self, jsonl_path, data_root, curriculum_frames, image_size):
        self._samples = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self._samples.append(json.loads(line))
        self._data_root = Path(data_root)
        self.T = curriculum_frames
        self._image_size = image_size

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, index):
        sample = self._samples[index]
        path = self._data_root / sample["data_path"]
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing but listed in phase1_combined.jsonl -- "
                "re-run training/data/preprocess.py"
            )

        if sample["media_type"] == "image":
            frame = _load_image(path)
            frames = frame.unsqueeze(0).repeat(self.T, 1, 1, 1)
        else:
            frames = _load_video(path, self.T)

        frames = _resize(frames, self._image_size)
        return frames, sample["caption"]


def build_phase1_loader(jsonl_path, curriculum_frames, batch_size, num_workers,
                         data_root=None, image_size=None, shuffle=True):
    """Build the Phase 1 DataLoader.

    Yields (video_frames, captions) batches -- video_frames is a
    (B, T, C, size, size) uint8 tensor, captions is a list[str] of length B
    (PyTorch's default collate keeps a list of strings as-is, so no custom
    collate_fn is needed).

    data_root defaults to the DATA_ROOT env var; image_size defaults to
    configs/model.yaml's x_encoder.image_size.
    """
    if data_root is None:
        data_root = os.environ["DATA_ROOT"]

    if image_size is None:
        with open(_REPO_ROOT / "configs" / "model.yaml", encoding="utf-8") as f:
            image_size = yaml.safe_load(f)["x_encoder"]["image_size"]

    dataset = Phase1Dataset(jsonl_path, data_root, curriculum_frames, image_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)