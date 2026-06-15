#!/usr/bin/env python3
"""Build DAVIS anno_0..anno_3 folders from per-expression mask outputs.

The DAVIS text meta used by this project stores four expressions per object:

    exp_id = obj_index * 4 + anno_id

This script converts per-expression binary masks such as
``<pred_root>/<video>/<exp_id>/merge_mask/<frame>.png`` into DAVIS-style
multi-object masks:

    <output_root>/anno_0/<video>/<frame>.png
    <output_root>/anno_1/<video>/<frame>.png
    <output_root>/anno_2/<video>/<frame>.png
    <output_root>/anno_3/<video>/<frame>.png
"""

from __future__ import annotations

import argparse
import json
import shutil
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_exp_path", default="../../dataset/davis/meta_expressions/valid/meta_expressions.json")
    parser.add_argument("--pred_root", required=True, help="Per-expression prediction root.")
    parser.add_argument("--output_root", required=True, help="Destination root that will contain anno_0..anno_3.")
    parser.add_argument("--dataset_root", default="../../dataset/davis/valid", help="Root containing JPEGImages and optional Annotations.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--missing-as-empty",
        action="store_true",
        help="Fill missing expression/frame masks with black masks. Otherwise missing masks are reported and skipped.",
    )
    parser.add_argument(
        "--recursive-search",
        action="store_true",
        help="Search pred_root recursively for <video>/<exp_id> folders. Useful when outputs are grouped by text class.",
    )
    parser.add_argument(
        "--overlap-policy",
        choices=("first", "last"),
        default="first",
        help="How to resolve overlapping object masks within the same anno folder. first matches numpy argmax tie behavior.",
    )
    return parser.parse_args()


def numeric_key(name: str) -> tuple[int, Any]:
    stem = Path(name).stem
    try:
        return (0, int(stem))
    except ValueError:
        return (1, stem)


def read_meta(meta_exp_path: Path) -> dict[str, Any]:
    return json.loads(meta_exp_path.read_text(encoding="utf-8"))["videos"]


def list_video_frames(dataset_root: Path, video: str, meta_frames: list[str]) -> list[str]:
    if meta_frames:
        return [f"{frame}.png" if Path(str(frame)).suffix == "" else str(frame) for frame in meta_frames]

    image_dir = dataset_root / "JPEGImages" / video
    frames = [f"{p.stem}.png" for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES]
    frames.sort(key=numeric_key)
    return frames


def get_frame_size(dataset_root: Path, video: str, frame_name: str) -> tuple[int, int]:
    image_dir = dataset_root / "JPEGImages" / video
    stem = Path(frame_name).stem
    for suffix in (".jpg", ".jpeg", ".png"):
        image_path = image_dir / f"{stem}{suffix}"
        if image_path.exists():
            with Image.open(image_path) as image:
                return image.height, image.width
    raise FileNotFoundError(f"No image found for {video}/{stem} under {image_dir}")


def load_palette(dataset_root: Path) -> list[int] | None:
    anno_root = dataset_root / "Annotations"
    if not anno_root.exists():
        return None
    for path in sorted(anno_root.glob("*/*.png")):
        with Image.open(path) as image:
            palette = image.getpalette()
            if palette:
                return palette
    return None


def resolve_pred_dir(pred_root: Path, video: str, exp_id: str, recursive_search: bool) -> Path | None:
    direct = pred_root / video / exp_id
    direct_merge = direct / "merge_mask"
    if direct_merge.exists():
        return direct_merge
    if direct.exists():
        return direct

    if recursive_search:
        matches = []
        for candidate in pred_root.rglob(exp_id):
            if not candidate.is_dir():
                continue
            try:
                relative_parts = candidate.relative_to(pred_root).parts
            except ValueError:
                relative_parts = candidate.parts
            if video not in relative_parts:
                continue
            merge = candidate / "merge_mask"
            matches.append(merge if merge.exists() else candidate)
        if matches:
            matches.sort(key=lambda path: str(path))
            return matches[0]

    return None


def read_binary_mask(path: Path, size_hw: tuple[int, int]) -> np.ndarray:
    h, w = size_hw
    with Image.open(path) as image:
        mask = image.convert("L")
        if mask.size != (w, h):
            mask = mask.resize((w, h), Image.Resampling.NEAREST)
        return (np.asarray(mask) > 128)


def save_index_mask(path: Path, mask: np.ndarray, palette: list[int] | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(mask.astype(np.uint8), mode="P")
    if palette is not None:
        image.putpalette(palette)
    image.save(path)


def build_video_task(task: dict[str, Any]) -> dict[str, Any]:
    video = task["video"]
    video_data = task["video_data"]
    pred_root = Path(task["pred_root"])
    output_root = Path(task["output_root"])
    dataset_root = Path(task["dataset_root"])
    palette = task["palette"]
    missing_as_empty = task["missing_as_empty"]
    recursive_search = task["recursive_search"]
    overlap_policy = task["overlap_policy"]

    expressions = video_data["expressions"]
    if isinstance(expressions, dict):
        expression_lookup = expressions
        exp_ids = sorted([int(exp_id) for exp_id in expressions.keys()])
    else:
        expression_lookup = {str(index): expression for index, expression in enumerate(expressions)}
        exp_ids = list(range(len(expressions)))
    frames = list_video_frames(dataset_root, video, video_data.get("frames", []))
    anno_ids = range(4)
    num_objects = (max(exp_ids) // 4 + 1) if exp_ids else 0

    missing: list[dict[str, str]] = []
    written = 0
    overlap_pixels = 0

    for anno_id in anno_ids:
        exp_to_obj: list[tuple[str, int]] = []
        for obj_index in range(num_objects):
            exp_id = obj_index * 4 + anno_id
            if str(exp_id) in expression_lookup:
                exp_to_obj.append((str(exp_id), obj_index + 1))

        for frame_name in frames:
            size_hw = get_frame_size(dataset_root, video, frame_name)
            out_mask = np.zeros(size_hw, dtype=np.uint8)

            for exp_id, object_id in exp_to_obj:
                pred_dir = resolve_pred_dir(pred_root, video, exp_id, recursive_search)
                if pred_dir is None:
                    missing.append({"video": video, "exp_id": exp_id, "frame": frame_name, "reason": "missing_exp_dir"})
                    if not missing_as_empty:
                        continue
                    mask = np.zeros(size_hw, dtype=bool)
                else:
                    pred_file = pred_dir / frame_name
                    if not pred_file.exists():
                        missing.append({"video": video, "exp_id": exp_id, "frame": frame_name, "reason": "missing_frame"})
                        if not missing_as_empty:
                            continue
                        mask = np.zeros(size_hw, dtype=bool)
                    else:
                        mask = read_binary_mask(pred_file, size_hw)

                overlap_pixels += int(np.count_nonzero(mask & (out_mask > 0)))
                if overlap_policy == "first":
                    out_mask[mask & (out_mask == 0)] = object_id
                else:
                    out_mask[mask] = object_id

            save_index_mask(output_root / f"anno_{anno_id}" / video / frame_name, out_mask, palette)
            written += 1

    return {"video": video, "written_frames": written, "missing": missing, "overlap_pixels": overlap_pixels}


def main() -> int:
    args = parse_args()
    meta_exp_path = Path(args.meta_exp_path)
    pred_root = Path(args.pred_root)
    output_root = Path(args.output_root)
    dataset_root = Path(args.dataset_root)

    if not meta_exp_path.exists():
        raise FileNotFoundError(f"Meta file not found: {meta_exp_path}")
    if not pred_root.exists():
        raise FileNotFoundError(f"Prediction root not found: {pred_root}")
    if not (dataset_root / "JPEGImages").exists():
        raise FileNotFoundError(f"JPEGImages not found under dataset root: {dataset_root}")

    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    videos = read_meta(meta_exp_path)
    palette = load_palette(dataset_root)
    tasks = [
        {
            "video": video,
            "video_data": video_data,
            "pred_root": str(pred_root),
            "output_root": str(output_root),
            "dataset_root": str(dataset_root),
            "palette": palette,
            "missing_as_empty": bool(args.missing_as_empty),
            "recursive_search": bool(args.recursive_search),
            "overlap_policy": args.overlap_policy,
        }
        for video, video_data in sorted(videos.items())
    ]

    worker_count = max(1, min(args.workers, len(tasks))) if tasks else 1
    with Pool(processes=worker_count) as pool:
        results = pool.map(build_video_task, tasks)

    missing = [item for result in results for item in result["missing"]]
    summary = {
        "meta_exp_path": str(meta_exp_path),
        "pred_root": str(pred_root),
        "output_root": str(output_root),
        "dataset_root": str(dataset_root),
        "videos": len(results),
        "written_frames": sum(result["written_frames"] for result in results),
        "missing_count": len(missing),
        "overlap_policy": args.overlap_policy,
        "overlap_pixels": int(sum(result["overlap_pixels"] for result in results)),
        "missing_examples": missing[:50],
    }
    summary_path = output_root / "_build_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved_summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
