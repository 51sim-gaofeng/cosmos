#!/usr/bin/env python3
"""Generate synchronized five-view heavy-snow transfers with Cosmos3-Nano."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


SAMPLE_DIR = Path(__file__).resolve().parent
SAMPLE10_DIR = SAMPLE_DIR.parent / "10_cosmos3_five_view_heavy_rain"
sys.path.insert(0, str(SAMPLE10_DIR))

from generate_five_view_heavy_rain import VIEW_NAMES, validate_sources  # noqa: E402


DEFAULT_SOURCE_DIR = Path("/root/workspace/51sim/data/camera-5v")
DEFAULT_OUTPUT_DIR = Path("/root/workspace/51sim/samples/outputs/sample12_five_view_heavy_snow")
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
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--max-frames", type=int, default=600)
    parser.add_argument("--chunk-frames", type=int, default=121)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--control-guidance", type=float, default=1.0)
    parser.add_argument("--blur-weight", type=float, default=0.01)
    parser.add_argument(
        "--blur-strength",
        choices=("none", "very_low", "low", "medium", "high", "very_high"),
        default="high",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--master-port", type=int, default=29710)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--comparison-only", action="store_true")
    return parser.parse_args()


def write_manifest(args: argparse.Namespace, streams: dict[str, dict[str, object]]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "sample": "sample12_five_view_heavy_snow",
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
            "vision_only": True,
            "gpu_blur_control": True,
        },
        "prompt_source": str(SAMPLE_DIR.parent / "11_cosmos3_heavy_snow" / "prompts"),
        "cross_view_consistency": (
            "Each view uses synchronized source frames, the same snow prompts and seed, "
            "plus edge and RGB blur controls. This single-view Transfer checkpoint does not "
            "consume calibration as a joint multiview inference condition."
        ),
    }
    (args.output_dir / "multiview_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_command(args: argparse.Namespace, view_name: str, port: int) -> list[str]:
    command = [
        str(args.python), str(SAMPLE_DIR / "generate_view_heavy_snow.py"),
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
        "--vision-only",
        "--gpu-blur-control",
    ]
    if args.prepare_only:
        command.append("--prepare-only")
    if args.comparison_only:
        command.append("--comparison-only")
    return command


def run_all(args: argparse.Namespace) -> None:
    gpu_ids = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if len(gpu_ids) != 8 or any(not value.isdigit() for value in gpu_ids) or len(set(gpu_ids)) != 8:
        raise ValueError("--gpus must contain exactly eight unique numeric GPU IDs.")
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
        subprocess.run(build_command(args, view_name, args.master_port + index), check=True)


if __name__ == "__main__":
    run_all(parse_args())
