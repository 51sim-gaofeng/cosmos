# Sample12: synchronized five-view heavy snow

Sample12 transforms the five synchronized videos in
`/root/workspace/51sim/data/camera-5v` into heavy-snow videos. It uses the
first 600 frames of every source (30 seconds at 20 FPS) so every output shares
one timeline:

- `camera_front_wide`
- `camera_front_left`
- `camera_front_right`
- `camera_rear_left`
- `camera_rear_right`

Each camera runs in sequence with all eight GPUs (`CP=4`, `CFGP=2`), using
the current Sample11 snow prompts, `blur-weight=0.01`, vision-only loading and
the CUDA/NPP blur control. The workflow produces per-view outputs at 1280×720,
20 FPS and records the source calibration files in `multiview_manifest.json`.

This remains a synchronized per-view baseline: Nano Transfer does not consume
the calibration as a joint multiview generation condition, so prompt, seed and
source timing provide only weak cross-view consistency.

## Run

```bash
COSMOS3_PYTHON=/root/miniconda3/envs/cos-sim/bin/python \
bash samples/12_cosmos3_five_view_heavy_snow/run_five_view_heavy_snow.sh \
  --framework-root /root/workspace/51sim/cosmos-framework-main \
  --model /root/workspace/51sim/models/Cosmos3-Nano \
  --torchrun /root/miniconda3/envs/cos-sim/bin/torchrun
```

Use `--prepare-only` to validate all five inputs and prepare their control
assets without starting inference.
