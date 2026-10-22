# Sample11: front-camera heavy snow

Sample11 applies edge and RGB blur controlled `video2video` Transfer to create
a photorealistic heavy daytime snow scene from a front-camera video. The target
road state is an approximately 10 cm compacted snow layer with vehicle tire
ruts. It is an inference-only workflow.

The intended baseline uses:

- source: `/root/workspace/51sim/data/camera_front_wide_120fov.mp4`;
- 121 frames in one chunk;
- 720p, 10 FPS (`--sample-step 3` from the 30 FPS source);
- 50 UniPC steps;
- edge weight `1.0`;
- RGB blur weight `0.01`, strength `high`, so the source asphalt does not overpower the snow-covered road target;
- vision-only mode, which skips the unused AVAE audio tokenizer;
- optional `--gpu-blur-control` uses the CUDA/NPP bilateral filter for on-the-fly blur controls;
- GPU 1 only (`CP=1`, `CFGP=1`).

## Run

```bash
COSMOS3_PYTHON=/root/miniconda3/envs/cos-sim/bin/python \
bash samples/11_cosmos3_heavy_snow/run_heavy_snow.sh \
  --source-video /root/workspace/51sim/data/camera_front_wide_120fov.mp4 \
  --sample-step 3 \
  --max-frames 121 \
  --chunk-frames 121 \
  --framework-root /root/workspace/51sim/cosmos-framework-main \
  --model /root/workspace/51sim/models/Cosmos3-Nano \
  --torchrun /root/miniconda3/envs/cos-sim/bin/torchrun \
  --python /root/miniconda3/envs/cos-sim/bin/python \
  --gpus 1 --dp-shard-size 1 --cp-size 1 --cfgp-size 1 \
  --output-dir /root/workspace/51sim/samples/outputs/sample11_front_heavy_snow_121f_1gpu
```

The output directory stores reusable RGB/edge controls, the actual input spec,
the generated `vision.mp4`, and a source/generated `comparison.mp4`.
