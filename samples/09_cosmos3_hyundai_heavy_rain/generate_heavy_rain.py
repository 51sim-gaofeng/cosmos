#!/usr/bin/env python3
"""Generate Hyundai heavy rain with sparse lens droplets and RGB comparison."""

import sys
from pathlib import Path


SAMPLE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SAMPLE_DIR.parent))

from common.weather_transfer import SampleConfig, run  # noqa: E402


if __name__ == "__main__":
    run(SampleConfig(
        sample_id="sample09_hyundai_front_heavy_rain",
        sample_dir=SAMPLE_DIR,
        default_output=Path(
            "/root/workspace/cosmos3/outputs/cosmos3_sample09_hyundai_front_heavy_rain"
        ),
        description=__doc__ or "Generate Hyundai heavy rain.",
    ))
