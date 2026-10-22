#!/usr/bin/env python3
"""Generate one synchronized camera view for Sample12 heavy snow."""

import sys
from pathlib import Path


SAMPLE_DIR = Path(__file__).resolve().parent
SNOW_PROMPT_SAMPLE_DIR = SAMPLE_DIR.parent / "11_cosmos3_heavy_snow"
sys.path.insert(0, str(SAMPLE_DIR.parent))

from common.weather_transfer import SampleConfig, run  # noqa: E402


if __name__ == "__main__":
    run(SampleConfig(
        sample_id="sample12_five_view_heavy_snow",
        sample_dir=SNOW_PROMPT_SAMPLE_DIR,
        default_output=Path("/root/workspace/51sim/samples/outputs/sample12_five_view_heavy_snow"),
        description=__doc__ or "Generate one camera view for Sample12 heavy snow.",
    ))
