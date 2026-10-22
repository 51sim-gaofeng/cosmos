#!/usr/bin/env python3
"""Generate one synchronized camera view for Sample10."""

import sys
from pathlib import Path


SAMPLE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SAMPLE_DIR.parent))

from common.weather_transfer import SampleConfig, run  # noqa: E402


if __name__ == "__main__":
    run(SampleConfig(
        sample_id="sample10_five_view_heavy_rain",
        sample_dir=SAMPLE_DIR,
        default_output=Path("/root/workspace/51sim/samples/outputs/sample10_five_view_heavy_rain"),
        description=__doc__ or "Generate one camera view for Sample10.",
    ))
