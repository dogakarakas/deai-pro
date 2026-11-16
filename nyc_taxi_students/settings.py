"""Shared read limit and sampling fraction."""

import os


# Read the first READ_ROWS rows, then take an ordinary seeded random sample.
# TAXI_READ_ROWS=0 reads the full original CSV.
READ_ROWS = int(os.environ.get("TAXI_READ_ROWS", "3000000"))
SAMPLE_FRACTION = float(os.environ.get("TAXI_SAMPLE_FRACTION", "0.2"))

if READ_ROWS < 0:
    raise ValueError("TAXI_READ_ROWS must be nonnegative")
if not 0 < SAMPLE_FRACTION <= 1:
    raise ValueError("TAXI_SAMPLE_FRACTION must be greater than zero and at most one")
if READ_ROWS == 0:
    READ_ROWS = None
