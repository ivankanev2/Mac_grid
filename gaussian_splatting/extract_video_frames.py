#!/usr/bin/env python3
"""Extract frames from videos into a COLMAP/2DGS-ready scene folder.

This helper is intentionally compatible with the existing ``convert.py`` flow.
After it runs, the chosen scene folder will contain::

    <scene>/input/*.jpg

which is exactly what ``convert.py -s <scene>`` expects.

It is also tolerant of the layout used in this project archive:

    gaussian_splatting/
      extract_video_frames.py
      2d-gaussian-splatting/
        convert.py
        roblox_pipes/

So you can run, from ``gaussian_splatting/``::

    python ./extract_video_frames.py -s roblox_pipes --fps 2 --overwrite
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract video frames into <scene>/input so the dataset can be "
            "processed by convert.py and then trained with 2D Gaussian Splatting."
        )
    )
    parser.add_argument(
        "-s",
        "--source_path",
        required=True,
        type=Path,
        help=(
            "Scene folder path or scene name, e.g. roblox_pipes. "
            "The script will auto-search common project locations if a relative name is given."
        ),
    )
    parser.add_argument(
        "--ffmpeg_executable",
        default="ffmpeg",
        type=str,
        help="Path to ffmpeg executable (default: ffmpeg on PATH)",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Frames per second to extract (default: 2.0)",
    )
    group.add_argument(
        "--every_nth",
        type=int,
        default=None,
        help="Extract every Nth frame instead of using FPS sampling",
    )
    parser.add_argument(
        "--max_frames_per_video",
        type=int,
        default=None,
        help="Optional cap on extracted frames per video",
    )
    parser.add_argument(
        "--max_dimension",
        type=int,
        default=None,
        help=(
            "Optional resize so the longest image side is at most this many pixels. "
            "Useful if the source video is extremely large."
        ),
    )
    parser.add_argument(
        "--image_ext",
        type=str,
        default="jpg",
        choices=["jpg", "png"],
        help="Frame output format (default: jpg)",
    )
    parser.add_argument(
        "--jpg_quality",
        type=int,
        default=2,
        help="ffmpeg JPEG quality for .jpg output; lower is better quality (default: 2)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing image files in <scene>/input before extracting",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would happen without extracting anything",
    )
    return parser.parse_args()


def require_ffmpeg(ffmpeg_executable: str) -> str:
    exe = shutil.which(ffmpeg_executable) if ffmpeg_executable == "ffmpeg" else ffmpeg_executable
    if not exe:
        raise SystemExit(
            "Could not find ffmpeg. Install ffmpeg or pass --ffmpeg_executable <path>."
        )
    return exe


def sanitize_stem(name: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")
    return clean or "video"


def clear_existing_images(input_dir: Path) -> None:
    for path in input_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            path.unlink()


def existing_image_count(input_dir: Path) -> int:
    if not input_dir.exists():
        return 0
    return sum(1 for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def build_video_filter(args: argparse.Namespace) -> str:
    filters: List[str] = []
    if args.every_nth is not None:
        filters.append(f"select=not(mod(n\\,{args.every_nth}))")
    else:
        filters.append(f"fps={args.fps}")

    if args.max_dimension:
        md = int(args.max_dimension)
        filters.append(
            f"scale='if(gt(iw,ih),min({md},iw),-2)':'if(gt(ih,iw),min({md},ih),-2)'"
        )

    return ",".join(filters)


def run_ffmpeg_extract(
    ffmpeg_exe: str,
    video_path: Path,
    tmp_dir: Path,
    args: argparse.Namespace,
) -> None:
    output_pattern = tmp_dir / f"frame_%06d.{args.image_ext}"
    vf = build_video_filter(args)

    cmd = [
        ffmpeg_exe,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        vf,
    ]

    if args.every_nth is not None:
        cmd.extend(["-vsync", "vfr"])

    if args.max_frames_per_video is not None:
        cmd.extend(["-frames:v", str(args.max_frames_per_video)])

    if args.image_ext == "jpg":
        cmd.extend(["-qscale:v", str(args.jpg_quality)])

    cmd.append(str(output_pattern))

    subprocess.run(cmd, check=True)


def list_extracted_frames(tmp_dir: Path, image_ext: str) -> List[Path]:
    return sorted(tmp_dir.glob(f"*.{image_ext}"))


def move_frames(
    extracted: Iterable[Path],
    input_dir: Path,
    manifest_writer: csv.writer,
    video_name: str,
    start_index: int,
    image_ext: str,
) -> int:
    index = start_index
    for frame_path in extracted:
        dst_name = f"frame_{index:06d}.{image_ext}"
        dst_path = input_dir / dst_name
        shutil.move(str(frame_path), str(dst_path))
        manifest_writer.writerow([dst_name, video_name])
        index += 1
    return index


def find_project_root(script_dir: Path) -> Optional[Path]:
    candidates = [
        script_dir,
        script_dir / "2d-gaussian-splatting",
        script_dir.parent,
        script_dir.parent / "2d-gaussian-splatting",
    ]
    for cand in candidates:
        if (cand / "convert.py").is_file():
            return cand.resolve()
    return None


def scene_candidates(raw_scene: Path, cwd: Path, script_dir: Path, project_root: Optional[Path]) -> List[Path]:
    candidates: List[Path] = []
    if raw_scene.is_absolute():
        return [raw_scene]

    bases: List[Path] = [cwd, script_dir]
    if project_root is not None:
        bases.extend([project_root, project_root.parent])

    # Common archive layout: extractor one level above the real 2dgs project root.
    for base in list(bases):
        bases.append(base / "2d-gaussian-splatting")

    seen = set()
    for base in bases:
        candidate = (base / raw_scene).resolve()
        if candidate not in seen:
            seen.add(candidate)
            candidates.append(candidate)
    return candidates


def suggest_scene_dirs(raw_scene: Path, cwd: Path, script_dir: Path, project_root: Optional[Path]) -> List[Path]:
    suggestions: List[Path] = []
    name = raw_scene.name.lower()
    search_roots: List[Path] = [cwd, script_dir]
    if project_root is not None:
        search_roots.extend([project_root, project_root.parent])

    seen = set()
    for root in search_roots:
        if not root.exists() or not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_dir():
                continue
            if path.name.lower() == name:
                resolved = path.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    suggestions.append(resolved)
    return suggestions[:8]


def resolve_scene_dir(raw_scene: Path, cwd: Path, script_dir: Path, project_root: Optional[Path]) -> Path:
    candidates = scene_candidates(raw_scene, cwd, script_dir, project_root)
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate

    msg = [f"Scene folder does not exist: {raw_scene}"]
    msg.append("Searched these locations:")
    for candidate in candidates:
        msg.append(f"  - {candidate}")

    suggestions = suggest_scene_dirs(raw_scene, cwd, script_dir, project_root)
    if suggestions:
        msg.append("Possible matches:")
        for suggestion in suggestions:
            msg.append(f"  - {suggestion}")

    raise SystemExit("\n".join(msg))


def find_videos(scene_dir: Path) -> List[Path]:
    candidates: List[Path] = []
    for folder in [scene_dir, scene_dir / "videos"]:
        if not folder.exists() or not folder.is_dir():
            continue
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
                candidates.append(path)
    unique = []
    seen = set()
    for path in candidates:
        key = path.resolve()
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def next_step_text(cwd: Path, project_root: Optional[Path], scene_dir: Path) -> str:
    if project_root is None:
        return f"Next step: run convert.py on this scene folder: {scene_dir}"

    try:
        rel_scene = scene_dir.relative_to(project_root)
        rel_scene_str = rel_scene.as_posix()
    except ValueError:
        rel_scene_str = str(scene_dir)

    convert_path = project_root / "convert.py"
    if cwd.resolve() == project_root.resolve():
        return f"Next step: python ./convert.py -s {rel_scene_str}"

    try:
        rel_project = project_root.relative_to(cwd)
        rel_project_str = rel_project.as_posix()
        return (
            "Next step:\n"
            f"  cd {rel_project_str}\n"
            f"  python ./convert.py -s {rel_scene_str}"
        )
    except ValueError:
        return f"Next step: python \"{convert_path}\" -s \"{scene_dir}\""


def main() -> int:
    args = parse_args()
    cwd = Path.cwd().resolve()
    script_dir = Path(__file__).resolve().parent
    project_root = find_project_root(script_dir)
    scene_dir = resolve_scene_dir(args.source_path, cwd, script_dir, project_root)

    ffmpeg_exe = require_ffmpeg(args.ffmpeg_executable) if not args.dry_run else args.ffmpeg_executable
    videos = find_videos(scene_dir)
    if not videos:
        raise SystemExit(
            f"No video files found in {scene_dir} or {scene_dir / 'videos'}. "
            f"Supported extensions: {', '.join(sorted(VIDEO_EXTS))}"
        )

    input_dir = scene_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    old_count = existing_image_count(input_dir)
    if old_count and not args.overwrite:
        raise SystemExit(
            f"{input_dir} already contains {old_count} image(s). "
            "Use --overwrite to replace them."
        )

    if args.overwrite and not args.dry_run:
        clear_existing_images(input_dir)

    manifest_path = scene_dir / "frame_manifest.csv"

    print(f"Scene folder: {scene_dir}")
    if project_root is not None:
        print(f"2DGS project root: {project_root}")
    print(f"Found {len(videos)} video(s):")
    for v in videos:
        print(f"  - {v.name}")
    print(f"Output frames -> {input_dir}")
    if args.every_nth is not None:
        print(f"Sampling mode: every {args.every_nth} frame(s)")
    else:
        print(f"Sampling mode: {args.fps} fps")
    if args.max_dimension:
        print(f"Resize longest side to <= {args.max_dimension}px")
    if args.max_frames_per_video:
        print(f"Cap per video: {args.max_frames_per_video} frames")
    print(f"Manifest -> {manifest_path}")

    if args.dry_run:
        print(next_step_text(cwd, project_root, scene_dir))
        return 0

    next_index = 1
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_name", "source_video"])

        for video in videos:
            with tempfile.TemporaryDirectory(prefix=f"gs_frames_{sanitize_stem(video.stem)}_") as tmp:
                tmp_dir = Path(tmp)
                print(f"Extracting: {video.name}")
                run_ffmpeg_extract(ffmpeg_exe, video, tmp_dir, args)
                frames = list_extracted_frames(tmp_dir, args.image_ext)
                if not frames:
                    raise SystemExit(f"No frames were extracted from {video}")
                next_index = move_frames(
                    frames,
                    input_dir,
                    writer,
                    video.name,
                    next_index,
                    args.image_ext,
                )
                print(f"  -> extracted {len(frames)} frame(s)")

    total = existing_image_count(input_dir)
    print(f"Done. {total} frame(s) are now in {input_dir}")
    print(next_step_text(cwd, project_root, scene_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
