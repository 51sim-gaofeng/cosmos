#!/usr/bin/env python3
"""Generate a front-camera heavy snow scene with edge and RGB blur control."""

import sys
from pathlib import Path


SAMPLE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SAMPLE_DIR.parent))

from common.weather_transfer import SampleConfig, run  # noqa: E402


if __name__ == "__main__":
    run(SampleConfig(
        sample_id="sample11_front_heavy_snow",
        sample_dir=SAMPLE_DIR,
        default_output=Path("/root/workspace/51sim/samples/outputs/sample11_front_heavy_snow"),
        description=__doc__ or "Generate a front-camera heavy snow scene.",
    ))
