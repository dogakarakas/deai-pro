"""Shared settings for every NYC pipeline; environment overrides are optional."""

import os
from pathlib import Path

SAMPLE_FRACTION = float(os.environ.get("NYC_SAMPLE_FRACTION", "0.2"))
SAMPLE_SEED = int(os.environ.get("NYC_SAMPLE_SEED", "42"))
DATA_ROOT = Path(os.environ.get("NYC_DATA_ROOT", Path(__file__).parent / "input")).resolve()

if not 0 < SAMPLE_FRACTION <= 1:
    raise ValueError("SAMPLE_FRACTION must be greater than zero and at most one")
