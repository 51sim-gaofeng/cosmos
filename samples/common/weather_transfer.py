#!/usr/bin/env python3
"""Shared Hyundai weather-transfer preparation, inference, and comparison workflow."""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FRAMEWORK = Path(
    os.environ.get("COSMOS_FRAMEWORK", "/root/workspace/cosmos3/cosmos-framework-main")
)
RESOLUTION_SIZES = {
    "256": (320, 192),
    "480": (832, 480),
    "720": (1280, 720),
    "768": (1360, 768),
}


@dataclass(frozen=True)
class SampleConfig:
    sample_id: str
    sample_dir: Path
    default_output: Path
    description: str


def parse_args(config: SampleConfig) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=config.description)
    parser.add_argument("--source-root", type=Path, default=REPO_ROOT / "samples/data/hyundai/seed-5v")
    parser.add_argument(
        "--source-video",
        type=Path,
        help="MP4/video input to sample directly instead of the Hyundai PNG dataset layout.",
    )
    parser.add_argument("--output-dir", type=Path, default=config.default_output)
    parser.add_argument("--framework-root", type=Path, default=DEFAULT_FRAMEWORK)
    parser.add_argument("--model", type=Path, default=REPO_ROOT / "models/nano/Cosmos3-Nano")
    parser.add_argument("--log-root", type=Path, default=Path("/root/workspace/cosmos3/outputs/run_logs"))
    parser.add_argument(
        "--torchrun",
        type=Path,
        help="torchrun executable. Defaults to <framework-root>/.venv/bin/torchrun.",
    )
    parser.add_argument(
        "--python",
        type=Path,
        help="Python executable used to discover CUDA library paths. Defaults to <framework-root>/.venv/bin/python.",
    )
    parser.add_argument("--resolution", choices=tuple(RESOLUTION_SIZES), default="720")
    parser.add_argument("--sample-step", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=93)
    parser.add_argument("--chunk-frames", type=int, default=93)
    parser.add_argument(
        "--camera-name",
        default="camera_front_wide",
        help="Camera name recorded in the output manifest.",
    )
    parser.add_argument("--conditional-frames", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--control-guidance", type=float, default=1.0)
    parser.add_argument(
        "--blur-weight", type=float, default=0.2,
        help="Original-RGB blur control weight; set to 0 for edge-only mode.",
    )
    parser.add_argument(
        "--blur-strength",
        choices=("none", "very_low", "low", "medium", "high", "very_high"),
        default="high",
        help="Blur applied to the RGB control, not the generated output.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--vision-only",
        action="store_true",
        help="Skip the unused audio tokenizer for video-only Transfer inference.",
    )
    parser.add_argument(
        "--gpu-blur-control",
        action="store_true",
        help="Compute the on-the-fly blur control with the CUDA/NPP bilateral filter.",
    )
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--dp-shard-size", type=int, default=1)
    parser.add_argument("--cp-size", type=int, default=4)
    parser.add_argument("--cfgp-size", type=int, choices=(1, 2), default=1)
    parser.add_argument("--master-port", type=int, default=29682)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--comparison-only", action="store_true",
                      help="Join existing videos without preparing inputs or running inference.")
    return parser.parse_args()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def select_frames(source: Path, step: int, limit: int) -> tuple[list[tuple[int, Path]], float]:
    calibration = json.loads(
        (source / "distorted_cameras/intrinsic.json").read_text(encoding="utf-8")
    )["camera_front_wide"]
    if calibration["is_fisheye"] or calibration["image_size"] != [3840, 2160]:
        raise ValueError("Expected the non-fisheye 3840x2160 front-wide camera.")
    directory = source / "distorted_cameras/images/camera_front_wide"
    frames = sorted(
        ((int(path.stem), path) for path in directory.glob("*.png") if path.stem.isdigit()),
        key=lambda item: item[0],
    )
    if len(frames) < 2:
        raise ValueError(f"At least two timestamped PNG frames are required: {directory}")
    intervals = {b[0] - a[0] for a, b in zip(frames, frames[1:])}
    if len(intervals) != 1 or min(intervals) <= 0:
        raise ValueError(f"Source timestamps must have a uniform positive interval: {intervals}")
    return frames[::step][:limit], 1000 / (intervals.pop() * step)


def select_video_frames(
    source: Path, output_dir: Path, step: int, limit: int
) -> tuple[list[tuple[int, Path]], float]:
    import cv2

    if not source.is_file():
        raise FileNotFoundError(f"Video does not exist: {source}")
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"Failed to open video: {source}")

    native_fps = capture.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(native_fps) or native_fps <= 0:
        capture.release()
        raise ValueError(f"Video has an invalid frame rate: {source}")

    output_dir.mkdir(parents=True, exist_ok=True)
    selected: list[tuple[int, Path]] = []
    frame_index = 0
    try:
        while len(selected) < limit:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % step == 0:
                path = output_dir / f"{frame_index:06d}.png"
                if not cv2.imwrite(str(path), frame):
                    raise RuntimeError(f"Failed to write extracted frame: {path}")
                timestamp_ms = int(round(frame_index * 1000 / native_fps))
                selected.append((timestamp_ms, path))
            frame_index += 1
    finally:
        capture.release()

    if len(selected) < 2:
        raise ValueError(f"At least two frames are required from video: {source}")
    return selected, native_fps / step


def build_prompt(
    prompt_dir: Path, frame_count: int, fps: float, width: int, height: int,
) -> dict:
    duration = frame_count / fps
    prompt = json.loads((prompt_dir / "positive_prompt.json").read_text(encoding="utf-8"))
    prompt["temporal_caption"] = prompt["temporal_caption"].replace("{duration}", f"{duration:.2f}")
    prompt.update(resolution={"W": width, "H": height}, duration=f"{duration:.2f}s", fps=fps)
    return prompt


def probe_video(path: Path) -> tuple[int, int, Fraction, int]:
    if not path.is_file():
        raise FileNotFoundError(f"Video does not exist: {path}")
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
            "-show_entries", "stream=width,height,avg_frame_rate,nb_read_frames",
            "-of", "json", str(path),
        ],
        check=True, capture_output=True, text=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    return (
        int(stream["width"]), int(stream["height"]),
        Fraction(stream["avg_frame_rate"]), int(stream["nb_read_frames"]),
    )


def make_rgb_video(
    frames: list[tuple[int, Path]], output: Path, fps: float, width: int, height: int,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError("FFmpeg is required to package RGB frames.")
    if output.is_file():
        try:
            valid_output = probe_video(output) == (width, height, fps, len(frames))
        except (subprocess.CalledProcessError, IndexError, KeyError, ValueError):
            valid_output = False
        if valid_output:
            print(f"Reusing validated RGB video: {output}", flush=True)
            return
        output.unlink()
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".ffconcat") as listing:
        listing.write("ffconcat version 1.0\n")
        for _, path in frames:
            escaped = str(path).replace("'", "'\\''")
            listing.write(f"file '{escaped}'\nduration {1 / fps:.12f}\n")
        escaped = str(frames[-1][1]).replace("'", "'\\''")
        listing.write(f"file '{escaped}'\n")
        listing.flush()
        subprocess.run(
            [
                ffmpeg, "-y", "-v", "error", "-f", "concat", "-safe", "0",
                "-i", listing.name, "-vf",
                f"scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,"
                f"crop={width}:{height}",
                "-vsync", "cfr", "-r", str(fps), "-frames:v", str(len(frames)),
                "-c:v", "libx264", "-preset", "fast", "-crf", "0",
                "-profile:v", "high444", "-pix_fmt", "yuv444p",
                "-movflags", "+faststart", str(output),
            ],
            check=True,
        )


def make_edge_control_video(
    frames: list[tuple[int, Path]], output: Path, fps: float, width: int, height: int,
) -> None:
    import cv2
    import numpy as np

    encoder = subprocess.Popen(
        [
            "ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0",
            "-frames:v", str(len(frames)), "-c:v", "libx264", "-preset", "fast",
            "-crf", "0", "-profile:v", "high444", "-pix_fmt", "yuv444p",
            "-movflags", "+faststart", str(output),
        ],
        stdin=subprocess.PIPE,
    )
    assert encoder.stdin is not None
    try:
        for _, path in frames:
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError(f"Failed to read RGB frame: {path}")
            original_height, original_width = frame.shape[:2]
            scale = max(width / original_width, height / original_height)
            resized_width = int(round(original_width * scale))
            resized_height = int(round(original_height * scale))
            resized = cv2.resize(
                frame, (resized_width, resized_height),
                interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
            )
            x = max(0, (resized_width - width) // 2)
            y = max(0, (resized_height - height) // 2)
            rgb = cv2.cvtColor(resized[y:y + height, x:x + width], cv2.COLOR_BGR2RGB)
            edge = cv2.Canny(rgb, 100, 200)
            encoder.stdin.write(np.repeat(edge[:, :, None], 3, axis=2).tobytes())
    finally:
        encoder.stdin.close()
        return_code = encoder.wait()
    if return_code:
        raise RuntimeError(f"Edge control encoder failed with exit code {return_code}")


def launch(
    *, config: SampleConfig, args: argparse.Namespace, spec_path: Path, frame_count: int,
) -> None:
    framework_root = args.framework_root.resolve()
    torchrun = (args.torchrun or framework_root / ".venv/bin/torchrun").resolve()
    python = (args.python or framework_root / ".venv/bin/python").resolve()
    if not torchrun.is_file():
        raise FileNotFoundError(f"torchrun executable not found: {torchrun}")
    if not python.is_file():
        raise FileNotFoundError(f"Python executable not found: {python}")
    gpu_ids = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not gpu_ids or any(not value.isdigit() for value in gpu_ids):
        raise ValueError("--gpus must be a comma-separated list of GPU IDs.")
    world_size = len(gpu_ids)
    if args.dp_shard_size * args.cp_size * args.cfgp_size != world_size:
        raise ValueError("--dp-shard-size * --cp-size * --cfgp-size must equal the number of GPUs.")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = (
        f"{config.sample_id}_day_{args.resolution}p_"
        f"{args.fps}fps_{frame_count}f_{world_size}gpu"
    )
    args.log_root.mkdir(parents=True, exist_ok=True)
    log_path = args.log_root / f"{timestamp}_{label}.log"
    (args.log_root / f"{timestamp}_{label}.meta").write_text(
        "\n".join((
            f"started_at={datetime.now().astimezone().isoformat()}",
            f"label={label}", f"spec={spec_path}", f"output_dir={args.output_dir}",
            f"gpus={','.join(gpu_ids)}", f"log={log_path}",
        )) + "\n", encoding="utf-8",
    )
    command = [
        str(torchrun), f"--nproc-per-node={world_size}",
        "--master-addr=127.0.0.1", f"--master-port={args.master_port}",
        "-m", "cosmos_framework.scripts.inference", "--parallelism-preset=latency",
        f"--dp-shard-size={args.dp_shard_size}", "--dp-replicate-size=1",
        f"--cp-size={args.cp_size}", f"--cfgp-size={args.cfgp_size}",
    ]
    command.extend((
        "-i", str(spec_path), "-o", str(args.output_dir),
        "--checkpoint-path", str(args.model.resolve()), "--seed", str(args.seed),
        "--no-guardrails",
    ))
    site_packages = subprocess.check_output(
        [str(python), "-c", "import site; print(site.getsitepackages()[0])"],
        text=True,
    ).strip()
    cuda_libs = sorted(Path(site_packages).glob("nvidia/*/lib"))
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": ",".join(gpu_ids),
        "COSMOS_TRAINING": "false", "PYTHONUNBUFFERED": "1",
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "PATH": os.pathsep.join([str(python.parent), env.get("PATH", "")]),
        "LD_LIBRARY_PATH": os.pathsep.join(
            [*(str(path) for path in cuda_libs), env.get("LD_LIBRARY_PATH", "")]
        ).rstrip(os.pathsep),
    })
    if args.vision_only:
        env["COSMOS_TRANSFER_VISION_ONLY"] = "1"
    if args.gpu_blur_control:
        env["COSMOS_TRANSFER_BLUR_GPU"] = "1"
    print(f"Run log: {log_path}", flush=True)
    print("Command:", " ".join(command), flush=True)
    with log_path.open("wb") as log_file:
        process = subprocess.Popen(
            command, cwd=framework_root, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        assert process.stdout is not None
        while chunk := os.read(process.stdout.fileno(), 4096):
            log_file.write(chunk)
            log_file.flush()
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def make_comparison(config: SampleConfig, output_dir: Path) -> Path:
    rgb_path = output_dir / "inputs/front_rgb.mp4"
    generated_path = output_dir / config.sample_id / "vision.mp4"
    output = generated_path.parent / "comparison.mp4"
    streams = [probe_video(path) for path in (rgb_path, generated_path)]
    if streams[0] != streams[1]:
        raise ValueError(f"Comparison requires matching sizes, FPS and frame counts: {streams}")
    if streams[0][2] <= 0 or streams[0][3] <= 0:
        raise ValueError(f"Comparison requires positive FPS and frame count: {streams[0]}")
    print(f"Joining original RGB (left) and generated video (right): {output}", flush=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error", "-i", str(rgb_path), "-i", str(generated_path),
            "-filter_complex",
            "[0:v]setpts=PTS-STARTPTS,setsar=1[left];"
            "[1:v]setpts=PTS-STARTPTS,setsar=1[right];"
            "[left][right]hstack=inputs=2[v]",
            "-map", "[v]", "-an", "-c:v", "libx264", "-preset", "fast",
            "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
        ],
        check=True,
    )
    print(f"Saved comparison video: {output}", flush=True)
    return output


def run(config: SampleConfig) -> None:
    args = parse_args(config)
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.comparison_only:
        make_comparison(config, args.output_dir)
        return
    if min(args.sample_step, args.max_frames, args.num_steps) <= 0:
        raise ValueError("Sample step, frame limit and inference steps must be positive.")
    if args.chunk_frames < 5 or (args.chunk_frames - 1) % 4:
        raise ValueError("Chunk frames must be at least 5 and satisfy 4n+1.")
    if not 0 <= args.conditional_frames < args.chunk_frames:
        raise ValueError("Conditional frames must be within [0, chunk frames).")
    if not math.isfinite(args.blur_weight) or args.blur_weight < 0:
        raise ValueError("Blur weight must be finite and non-negative.")
    width, height = RESOLUTION_SIZES[args.resolution]
    assets = args.output_dir / "inputs"
    assets.mkdir(parents=True, exist_ok=True)
    rgb_path = assets / "front_rgb.mp4"
    edge_path = assets / "front_edge.mp4"
    prompt_path = assets / "rain_prompt.json"
    negative_path = assets / "rain_negative_prompt.json"
    spec_path = assets / "rain_spec.json"
    if args.source_video is not None:
        args.source_video = args.source_video.expanduser().resolve()
        legacy_frames = assets / "source_frames"
        if legacy_frames.exists():
            shutil.rmtree(legacy_frames)
        with tempfile.TemporaryDirectory(prefix="cosmos-weather-transfer-") as temporary_directory:
            frames, args.fps = select_video_frames(
                args.source_video,
                Path(temporary_directory) / "source_frames",
                args.sample_step,
                args.max_frames,
            )
            make_rgb_video(frames, rgb_path, args.fps, width, height)
            make_edge_control_video(frames, edge_path, args.fps, width, height)
    else:
        args.source_root = args.source_root.expanduser().resolve()
        frames, args.fps = select_frames(args.source_root, args.sample_step, args.max_frames)
        make_rgb_video(frames, rgb_path, args.fps, width, height)
        make_edge_control_video(frames, edge_path, args.fps, width, height)
    prompt_dir = config.sample_dir / "prompts"
    write_json(prompt_path, build_prompt(prompt_dir, len(frames), args.fps, width, height))
    write_json(negative_path, json.loads(
        (prompt_dir / "negative_prompt.json").read_text(encoding="utf-8")
    ))
    spec = {
        "name": config.sample_id, "model_mode": "video2video",
        "resolution": args.resolution, "aspect_ratio": "16,9",
        "num_frames": args.chunk_frames, "max_frames": len(frames), "fps": args.fps,
        "shift": 10.0, "num_steps": args.num_steps, "seed": args.seed,
        "num_video_frames_per_chunk": args.chunk_frames,
        "num_conditional_frames": args.conditional_frames,
        "num_first_chunk_conditional_frames": 0,
        "share_vision_temporal_positions": True,
        "negative_metadata_mode": "none", "negative_prompt_keep_metadata": False,
        "guidance": args.guidance, "control_guidance": args.control_guidance,
        "negative_prompt_file": str(negative_path), "prompt_path": str(prompt_path),
        "emphasize_control_in_prompt": False,
        "edge": {"weight": 1.0, "control_path": str(edge_path)},
    }
    if args.blur_weight > 0:
        spec["vision_path"] = str(rgb_path)
        spec["blur"] = {"weight": args.blur_weight, "preset_blur_strength": args.blur_strength}
    write_json(spec_path, spec)
    write_json(assets / "manifest.json", {
        "source_root": str(args.source_root) if args.source_video is None else None,
        "source_video": str(args.source_video) if args.source_video is not None else None,
        "camera": args.camera_name, "source_fps": args.fps * args.sample_step,
        "sample_step": args.sample_step,
        "timestamps_ms": [timestamp for timestamp, _ in frames],
        "frame_count": len(frames), "fps": args.fps, "duration_seconds": len(frames) / args.fps,
        "resolution": [width, height], "rgb_video": str(rgb_path),
        "edge_video": str(edge_path), "spec": str(spec_path),
        "control": "edge-blur" if args.blur_weight > 0 else "edge-only",
        "blur_source": str(rgb_path) if args.blur_weight > 0 else None,
        "blur_weight": args.blur_weight,
        "blur_strength": args.blur_strength if args.blur_weight > 0 else None,
    })
    print(f"Prepared {len(frames)} frames at {args.fps:g} FPS: {spec_path}", flush=True)
    if not args.prepare_only:
        launch(config=config, args=args, spec_path=spec_path, frame_count=len(frames))
        make_comparison(config, args.output_dir)
