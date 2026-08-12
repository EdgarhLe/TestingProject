"""
scripts/tests/test_build_index.py

Smoke tests for scripts/build_index.py that don't require a real checkpoint,
GPU, or network access -- they exercise the CLI/config plumbing (argument
parsing, video-list loading, level defaults) and confirm the script can be
imported without eagerly pulling in torchcodec/transformers (those are only
imported inside load_checkpoint_model, which is not exercised here).

Running the script for real (segmenting an actual video with a trained
checkpoint) is explicitly out of scope this week -- see the module
docstring in scripts/build_index.py.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.build_index import LEVEL_DEFAULTS, load_video_list


def test_level_defaults_cover_both_levels():
    assert set(LEVEL_DEFAULTS) == {"coarse", "fine"}
    for level, defaults in LEVEL_DEFAULTS.items():
        assert defaults["window_size"] > 0
        assert defaults["stride"] > 0
        assert defaults["stride"] <= defaults["window_size"]


def test_fine_level_has_a_smaller_window_than_coarse():
    assert LEVEL_DEFAULTS["fine"]["window_size"] < LEVEL_DEFAULTS["coarse"]["window_size"]


def test_load_video_list_skips_blank_lines_and_comments(tmp_path):
    list_path = tmp_path / "videos.txt"
    list_path.write_text(
        "video_a.mp4\n"
        "\n"
        "# a comment line\n"
        "  \n"
        "video_b.mp4\n"
    )
    videos = load_video_list(list_path)
    assert videos == ["video_a.mp4", "video_b.mp4"]


def test_load_video_list_strips_whitespace(tmp_path):
    list_path = tmp_path / "videos.txt"
    list_path.write_text("  video_a.mp4  \n")
    assert load_video_list(list_path) == ["video_a.mp4"]


def test_script_help_runs_without_importing_heavy_optional_deps():
    """
    Confirms `python scripts/build_index.py --help` succeeds -- in
    particular, that model.x_encoder.vjepa2 / model.predictor.gwen2_5_1_5b
    (which need transformers/torchcodec and, for the real predictor, a
    configs/model.yaml on disk) are NOT imported at module load time, only
    inside load_checkpoint_model when the script actually runs.
    """
    repo_root = Path(__file__).resolve().parents[2]
    env = {**os.environ, "PYTHONPATH": str(repo_root)}
    result = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "build_index.py"), "--help"],
        cwd=repo_root, capture_output=True, text=True, timeout=30, env=env,
    )
    assert result.returncode == 0, result.stderr
    assert "--level" in result.stdout
    assert "--checkpoint" in result.stdout


def test_script_requires_level_checkpoint_videos_output():
    repo_root = Path(__file__).resolve().parents[2]
    env = {**os.environ, "PYTHONPATH": str(repo_root)}
    result = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "build_index.py")],
        cwd=repo_root, capture_output=True, text=True, timeout=30, env=env,
    )
    assert result.returncode != 0
    assert "required" in result.stderr.lower()
