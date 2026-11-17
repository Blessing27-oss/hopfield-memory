"""Loader for the Optdigits original 32x32 bitmap file (data/optdigits-orig.cv).

This is the same parser as in notebooks/01_dataset_exploration.ipynb, moved here
so later notebooks can reuse it with a single import.

File format:
  - 21 header lines (metadata)
  - then repeated blocks of 33 lines: 32 rows of 32 '0'/'1' characters,
    followed by one label line such as " 5"
"""
from pathlib import Path

import numpy as np

HEADER_LINES = 21
IMG_SIZE = 32
BLOCK = IMG_SIZE + 1  # 32 pixel rows + 1 label line

# Project root = the folder that contains src/ (one level up from this file).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = PROJECT_ROOT / "data" / "optdigits-orig.cv"


def load_optdigits(path=DEFAULT_PATH):
    """Return (images, labels).

    images: int8 array, shape (N, 32, 32), values 0 (background) or 1 (ink)
    labels: int array, shape (N,), values 0-9
    """
    lines = Path(path).read_text().splitlines()
    body = lines[HEADER_LINES:]

    images, labels = [], []
    for start in range(0, len(body), BLOCK):
        block = body[start:start + BLOCK]
        if len(block) < BLOCK:
            break
        pixel_rows, label_line = block[:IMG_SIZE], block[IMG_SIZE]
        for row in pixel_rows:
            if len(row) != IMG_SIZE or set(row) - {"0", "1"}:
                raise ValueError(f"Bad pixel row near body line {start}: {row!r}")
        images.append([[int(ch) for ch in row] for row in pixel_rows])
        labels.append(int(label_line.strip()))

    return np.array(images, dtype=np.int8), np.array(labels)
