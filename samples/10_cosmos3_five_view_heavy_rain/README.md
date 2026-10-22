# Sample10: five-view synchronized heavy rain

This sample transforms the five synchronized videos in
`/root/workspace/51sim/data/camera-5v` into heavy-rain videos. It uses the
first 600 frames of every source (30 seconds at 20 FPS) so all outputs share
the same timeline:

- `camera_front_wide`
- `camera_front_left`
- `camera_front_right`
- `camera_rear_left`
- `camera_rear_right`

Each view uses the same seed and heavy-rain prompts, with edge weight `1.0`
and RGB blur control weight `0.2` at `high` strength. Each view is generated
in 121-frame chunks. The default generates the five views in sequence, using
all eight GPUs for each view with `CP=4` and `CFGP=2`. This minimizes each
individual camera's generation time; it does not make the five final videos
complete simultaneously.

The source directory's `intrinsic.json` and `extrinsic.json` are validated and
recorded in `multiview_manifest.json`. The installed Nano Transfer inference
interface does not use them as a joint multiview condition, so the workflow
provides synchronized time, prompt, seed and per-view controls—not strict
pixel-level cross-camera generation consistency.

## Run

```bash
COSMOS3_PYTHON=/root/miniconda3/envs/cos-sim/bin/python \
bash samples/10_cosmos3_five_view_heavy_rain/run_five_view_heavy_rain.sh \
  --framework-root /root/workspace/51sim/cosmos-framework-main \
  --model /root/workspace/51sim/models/Cosmos3-Nano \
  --torchrun /root/miniconda3/envs/cos-sim/bin/torchrun
```

To prepare and validate RGB/edge control assets without starting inference:

```bash
bash samples/10_cosmos3_five_view_heavy_rain/run_five_view_heavy_rain.sh --prepare-only
```

By default, results are in
`/root/workspace/51sim/samples/outputs/sample10_five_view_heavy_rain/<camera>/`.
Each view contains its own `inputs/` controls, generated `vision.mp4`,
comparison video and manifest. The output root contains
`multiview_manifest.json` with the common 600-frame, 20-FPS synchronization
contract and calibration references.
