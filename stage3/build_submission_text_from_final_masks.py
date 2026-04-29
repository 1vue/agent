#!/usr/bin/env python3
"""Build submission.zip in sample_submission_text style from final merged masks."""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create submission_text-style folders: <video>/<json_id>/*.png from "
            "final merged masks, filling missing groups/frames with black masks."
        )
    )
    parser.add_argument("--batch-root", default="outputs/batch_ark_video_qa")
    parser.add_argument("--dataset-root", default="../../dataset/mevis/valid")
    parser.add_argument("--final-mask-root", default="mask_output/exp_agent_final1")
    parser.add_argument("--submission-dir", default="submission_text_final1")
    parser.add_argument("--submission-zip", default="submission.zip")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def list_expected_groups(batch_root: Path) -> list[tuple[str, str]]:
    groups = []
    for path in sorted(batch_root.glob("*/*.json")):
        if path.parent.name == "_errors":
            continue
        groups.append((path.parent.name, path.stem))
    return groups


def numeric_key(name: str):
    stem = Path(name).stem
    try:
        return (0, int(stem))
    except ValueError:
        return (1, stem)


def expected_frame_files(video_dir: Path) -> list[str]:
    files = [p.name for p in video_dir.glob("*.jpg")]
    files.sort(key=numeric_key)
    return files


def save_black_mask(path: Path, size_hw: tuple[int, int]) -> None:
    h, w = size_hw
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((h, w), dtype=np.uint8)).save(path)


def main() -> int:
    args = parse_args()
    batch_root = Path(args.batch_root)
    dataset_root = Path(args.dataset_root)
    final_mask_root = Path(args.final_mask_root)
    submission_dir = Path(args.submission_dir)
    submission_zip = Path(args.submission_zip)

    if submission_dir.exists() and args.overwrite:
        shutil.rmtree(submission_dir)
    submission_dir.mkdir(parents=True, exist_ok=True)

    expected_groups = list_expected_groups(batch_root)
    used_black_groups = []
    completed_groups = []

    for video_id, json_id in expected_groups:
        video_dir = dataset_root / "JPEGImages" / video_id
        frame_files = expected_frame_files(video_dir)
        if not frame_files:
            raise FileNotFoundError(f"No frames found for {video_id}")
        first_image = Image.open(video_dir / frame_files[0])
        size_hw = (first_image.height, first_image.width)

        src_merge_dir = final_mask_root / video_id / json_id / "merge_mask"
        dst_dir = submission_dir / video_id / json_id
        dst_dir.mkdir(parents=True, exist_ok=True)

        used_black = False
        for frame_file in frame_files:
            png_name = f"{Path(frame_file).stem}.png"
            src_mask = src_merge_dir / png_name
            dst_mask = dst_dir / png_name
            if src_mask.exists():
                shutil.copy2(src_mask, dst_mask)
            else:
                save_black_mask(dst_mask, size_hw)
                used_black = True

        if used_black:
            used_black_groups.append({"video_id": video_id, "json_id": json_id})
        else:
            completed_groups.append({"video_id": video_id, "json_id": json_id})

    # structure checks
    video_dirs = sorted([p for p in submission_dir.iterdir() if p.is_dir()])
    # if len(video_dirs) != 40:
    #     raise ValueError(f"Expected 40 video folders, found {len(video_dirs)}")
    # bad_video_counts = {}
    # for video_dir in video_dirs:
    #     subdirs = sorted([p for p in video_dir.iterdir() if p.is_dir()])
    #     if len(subdirs) != 10:
    #         bad_video_counts[video_dir.name] = len(subdirs)
    # if bad_video_counts:
    #     raise ValueError(f"Expected 10 subfolders per video, mismatched counts: {bad_video_counts}")
    if len(video_dirs) != 140:
        print(f"⚠️ Warning: Expected 140 videos, but found {len(video_dirs)}.")

        # 2. 移除每视频 10 个子文件夹的硬检查（MeViS 的 expression 数量是动态的）
    print(f"✅ Found {len(video_dirs)} videos and {len(expected_groups)} total groups.")

    summary = {
        "submission_dir": str(submission_dir.resolve()),
        "total_groups": len(expected_groups),
        "video_folder_count": len(video_dirs),
        "groups_using_any_black_mask": len(used_black_groups),
        "fully_from_final_masks": len(completed_groups),
        "used_black_groups": used_black_groups,
    }
    (submission_dir / "_build_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    if submission_zip.exists() and args.overwrite:
        submission_zip.unlink()
    with zipfile.ZipFile(submission_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(submission_dir.rglob("*")):
            if path.name == "_build_summary.json":
                continue
            if path.is_file():
                zf.write(path, arcname=str(path.relative_to(submission_dir)))

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved_zip: {submission_zip.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
