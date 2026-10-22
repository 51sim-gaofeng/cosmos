#!/usr/bin/env python3
"""Generate synchronized 20 FPS heavy-rain transfers for five camera videos."""

import argparse
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path


SAMPLE_DIR = Path(__file__).resolve().parent
VIEW_NAMES = (
    "camera_front_wide",
    "camera_front_left",
    "camera_front_right",
    "camera_rear_left",
    "camera_rear_right",
)
DEFAULT_SOURCE_DIR = Path("/root/workspace/51sim/data/camera-5v")
DEFAULT_OUTPUT_DIR = Path("/root/workspace/51sim/samples/outputs/sample10_five_view_heavy_rain")
DEFAULT_FRAMEWORK = Path("/root/workspace/51sim/cosmos-framework-main")
DEFAULT_MODEL = Path("/root/workspace/51sim/models/Cosmos3-Nano")
DEFAULT_PYTHON = Path("/root/miniconda3/envs/cos-sim/bin/python")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--framework-root", type=Path, default=DEFAULT_FRAMEWORK)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--torchrun", type=Path, default=DEFAULT_PYTHON.parent / "torchrun")
    parser.add_argument("--log-root", type=Path, default=Path("/root/workspace/51sim/samples/outputs/run_logs"))
    parser.add_argument(
        "--gpus",
        default="0,1,2,3,4,5,6,7",
        help="Exactly eight GPU IDs used by every camera view in sequence.",
    )
    parser.add_argument("--max-frames", type=int, default=600)
    parser.add_argument("--chunk-frames", type=int, default=121)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--control-guidance", type=float, default=1.0)
    parser.add_argument("--blur-weight", type=float, default=0.2)
    parser.add_argument(
        "--blur-strength",
        choices=("none", "very_low", "low", "medium", "high", "very_high"),
        default="high",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--master-port", type=int, default=29700)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--comparison-only", action="store_true")
    return parser.parse_args()


def probe_video(path: Path) -> tuple[Fraction, int]:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
            "-show_entries", "stream=avg_frame_rate,nb_read_frames", "-of", "json", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    return Fraction(stream["avg_frame_rate"]), int(stream["nb_read_frames"])


def validate_sources(source_dir: Path, max_frames: int) -> dict[str, dict[str, object]]:
    intrinsic_path = source_dir / "intrinsic.json"
    extrinsic_path = source_dir / "extrinsic.json"
    if max_frames <= 1:
        raise ValueError("--max-frames must be at least 2.")
    if not intrinsic_path.is_file() or not extrinsic_path.is_file():
        raise FileNotFoundError("Both intrinsic.json and extrinsic.json are required in --source-dir.")
    intrinsic = json.loads(intrinsic_path.read_text(encoding="utf-8"))
    extrinsic = json.loads(extrinsic_path.read_text(encoding="utf-8"))
    streams: dict[str, dict[str, object]] = {}
    for view_name in VIEW_NAMES:
        video = source_dir / f"{view_name}.mp4"
        if not video.is_file():
            raise FileNotFoundError(f"Missing synchronized source video: {video}")
        if view_name not in intrinsic or view_name not in extrinsic:
            raise ValueError(f"Missing calibration for {view_name}.")
        if "camera_matrix" not in intrinsic[view_name] or "ego_2_camera" not in extrinsic[view_name]:
            raise ValueError(f"Incomplete calibration for {view_name}.")
        fps, frame_count = probe_video(video)
        if fps != 20:
            raise ValueError(f"{video} must be 20 FPS, got {fps}.")
        if frame_count < max_frames:
            raise ValueError(f"{video} has {frame_count} frames; {max_frames} are required.")
        streams[view_name] = {
            "source_video": str(video),
            "source_fps": float(fps),
            "available_frames": frame_count,
            "generated_frames": max_frames,
        }
    return streams


def write_manifest(args: argparse.Namespace, streams: dict[str, dict[str, object]]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "sample": "sample10_five_view_heavy_rain",
        "source_dir": str(args.source_dir),
        "intrinsic_file": str(args.source_dir / "intrinsic.json"),
        "extrinsic_file": str(args.source_dir / "extrinsic.json"),
        "cameras": streams,
        "sync_fps": 20,
        "sync_frame_count": args.max_frames,
        "duration_seconds": args.max_frames / 20,
        "chunk_frames": args.chunk_frames,
        "control": {
            "edge_weight": 1.0,
            "blur_weight": args.blur_weight,
            "blur_strength": args.blur_strength,
        },
        "cross_view_consistency": (
            "Each view uses synchronized source frames, identical weather prompts and seed, "
            "plus edge and RGB blur controls. This single-view Transfer checkpoint does not "
            "consume camera calibration as a joint multiview inference condition."
        ),
    }
    (args.output_dir / "multiview_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_command(args: argparse.Namespace, view_name: str, port: int) -> list[str]:
    command = [
        str(args.python), str(SAMPLE_DIR / "generate_view_heavy_rain.py"),
        "--source-video", str(args.source_dir / f"{view_name}.mp4"),
        "--camera-name", view_name,
        "--output-dir", str(args.output_dir / view_name),
        "--framework-root", str(args.framework_root),
        "--model", str(args.model),
        "--torchrun", str(args.torchrun),
        "--python", str(args.python),
        "--log-root", str(args.log_root),
        "--sample-step", "1",
        "--max-frames", str(args.max_frames),
        "--chunk-frames", str(args.chunk_frames),
        "--num-steps", str(args.num_steps),
        "--guidance", str(args.guidance),
        "--control-guidance", str(args.control_guidance),
        "--blur-weight", str(args.blur_weight),
        "--blur-strength", args.blur_strength,
        "--seed", str(args.seed),
        "--gpus", args.gpus,
        "--dp-shard-size", "1",
        "--cp-size", "4",
        "--cfgp-size", "2",
        "--master-port", str(port),
    ]
    if args.prepare_only:
        command.append("--prepare-only")
    if args.comparison_only:
        command.append("--comparison-only")
    return command


def run_all(args: argparse.Namespace) -> None:
    gpu_ids = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if len(gpu_ids) != 8 or any(not value.isdigit() for value in gpu_ids):
        raise ValueError("--gpus must contain exactly eight numeric GPU IDs.")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("--gpus must not contain duplicate GPU IDs.")
    if args.chunk_frames < 5 or (args.chunk_frames - 1) % 4:
        raise ValueError("--chunk-frames must be at least 5 and satisfy 4n+1.")
    args.source_dir = args.source_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.framework_root = args.framework_root.expanduser().resolve()
    args.model = args.model.expanduser().resolve()
    args.python = args.python.expanduser().resolve()
    args.torchrun = args.torchrun.expanduser().resolve()
    if not args.python.is_file() or not args.torchrun.is_file():
        raise FileNotFoundError("Configured --python or --torchrun executable does not exist.")
    streams = validate_sources(args.source_dir, args.max_frames)
    write_manifest(args, streams)

    for index, view_name in enumerate(VIEW_NAMES):
        print(f"Generating {view_name} with GPUs {args.gpus}.", flush=True)
        subprocess.run(
            build_command(args, view_name, args.master_port + index),
            check=True,
        )


if __name__ == "__main__":
    run_all(parse_args())
