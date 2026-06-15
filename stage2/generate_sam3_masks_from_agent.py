#!/usr/bin/env python3
"""Propagate SAM3-agent seed masks through videos with the official SAM3 tracker."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pycocotools.mask as mask_utils
from PIL import Image

from stage2.generate_sam3_masks_from_bbox import (
    OBJ_SUFFIX_RE,
    DEFAULT_CHECKPOINT,
    build_video_tracker,
    import_sam3_builders,
    propagate_object_masks,
    resolve_video_dir,
    save_binary_mask,
    sorted_jpeg_frame_files,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read SAM3-agent pred.json files, decode their selected seed masks, "
            "and propagate them through the full video with the official SAM3 tracker."
        )
    )
    parser.add_argument(
        "--agent-root",
        default="agent_output/sam3_agent",
        help=(
            "Root directory containing agent outputs. Can point either to "
            "agent_output/sam3_agent or to a specific model subdirectory under it."
        ),
    )
    parser.add_argument(
        "--dataset-root",
        default="../../dataset/mevis/valid",
        help="Dataset root directory containing JPEGImages/",
    )
    parser.add_argument(
        "--mask-root",
        default="outputs/mask_output",
        help="Directory to save propagated masks",
    )
    parser.add_argument(
        "--exp-name",
        default="exp_agent",
        help="Top-level experiment folder name under mask_output/",
    )
    parser.add_argument(
        "--sam3-official-root",
        default="sam3",
        help="Path to the official SAM3 codebase",
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="SAM3 checkpoint path",
    )
    parser.add_argument("--device", default="cuda", help="Torch device, e.g. cuda or cuda:0")
    parser.add_argument(
        "--score-thresh",
        type=float,
        default=0.0,
        help="Threshold applied to propagated mask logits",
    )
    parser.add_argument(
        "--only-video-id",
        action="append",
        help="Optional filter; may be passed multiple times",
    )
    parser.add_argument(
        "--only-json-id",
        action="append",
        help="Optional filter on result json-id stem; may be passed multiple times",
    )
    parser.add_argument(
        "--limit-groups",
        type=int,
        help="Optional cap on the number of <video>/<json_id> groups to process",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run and overwrite mask outputs that already exist",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print planned work without running SAM3",
    )
    return parser.parse_args()


def parse_object_order(name: str) -> tuple[int, str]:
    match = OBJ_SUFFIX_RE.search(name)
    if match:
        return int(match.group(1)), name
    return 10**9, name


def list_agent_groups(
    agent_root: Path,
    only_video_ids: set[str] | None,
    only_json_ids: set[str] | None,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for pred_path in sorted(agent_root.rglob("pred.json")):
        parts = pred_path.relative_to(agent_root).parts
        if len(parts) < 4:
            continue
        video_id = "/".join(parts[:-3])
        json_id = parts[-3]
        if not video_id:
            continue
        if only_video_ids and video_id not in only_video_ids:
            continue
        if only_json_ids and json_id not in only_json_ids:
            continue
        groups[(video_id, json_id)].append(pred_path)

    result = []
    for (video_id, json_id), pred_paths in sorted(groups.items()):
        pred_paths.sort(key=lambda p: parse_object_order(p.parent.name))
        result.append({"video_id": video_id, "json_id": json_id, "pred_paths": pred_paths})
    return result


def decode_agent_masks(pred_json: dict[str, Any]) -> np.ndarray:
    orig_h = int(pred_json["orig_img_h"])
    orig_w = int(pred_json["orig_img_w"])
    rles = pred_json.get("pred_masks", [])
    if not rles:
        raise ValueError("Agent pred.json contains no pred_masks")

    merged = np.zeros((orig_h, orig_w), dtype=np.uint8)
    for counts in rles:
        rle = {"size": [orig_h, orig_w], "counts": counts}
        decoded = mask_utils.decode(rle)
        if decoded.ndim == 3:
            decoded = decoded[..., 0]
        merged[decoded > 0] = 1
    return merged


def load_agent_seed(pred_path: Path) -> dict[str, Any]:
    pred_json = json.loads(pred_path.read_text())
    source_item = pred_json.get("source_result_item", {})
    frame_filename = source_item.get("actual_frame_filename")
    if not frame_filename:
        input_meta_path = pred_path.parent / "input_meta.json"
        if input_meta_path.exists():
            input_meta = json.loads(input_meta_path.read_text())
            frame_filename = input_meta.get("frame_filename")
    if not frame_filename:
        raise ValueError(f"Could not determine seed frame filename from {pred_path}")

    return {
        "frame_filename": frame_filename,
        "mask": decode_agent_masks(pred_json),
        "object_name": pred_json.get("object_name") or pred_path.parent.name,
        "prompt": pred_json.get("input_prompt"),
        "pred_path": pred_path,
    }


def process_group(
    group: dict[str, Any],
    *,
    dataset_root: Path,
    output_root: Path,
    tracker,
    score_thresh: float,
    overwrite: bool,
) -> dict[str, Any]:
    video_id = group["video_id"]
    json_id = group["json_id"]
    video_dir = resolve_video_dir(dataset_root, video_id)
    frame_files = sorted_jpeg_frame_files(video_dir)
    frame_name_to_index = {name: idx for idx, name in enumerate(frame_files)}
    if not frame_files:
        raise FileNotFoundError(f"No JPEG frames found in {video_dir}")

    group_output_root = output_root / video_id / json_id
    merge_dir = group_output_root / "merge_mask"

    object_specs = []
    for pred_path in group["pred_paths"]:
        object_dir = group_output_root / pred_path.parent.name
        first_frame_out = object_dir / f"{Path(frame_files[0]).stem}.png"
        if first_frame_out.exists() and not overwrite:
            object_specs.append({"skip": True, "pred_path": pred_path, "object_dir": object_dir})
            continue

        seed = load_agent_seed(pred_path)
        frame_filename = seed["frame_filename"]
        if frame_filename not in frame_name_to_index:
            raise ValueError(f"Frame {frame_filename} from {pred_path} not found in {video_dir}")

        object_specs.append(
            {
                "skip": False,
                "pred_path": pred_path,
                "object_dir": object_dir,
                "frame_filename": frame_filename,
                "frame_idx": frame_name_to_index[frame_filename],
                "init_mask": seed["mask"],
                "object_name": seed["object_name"],
                "prompt": seed["prompt"],
            }
        )

    per_object_outputs: list[dict[str, Any]] = []
    for obj_idx, spec in enumerate(object_specs, start=1):
        object_name = spec["pred_path"].parent.name
        if spec.get("skip"):
            per_object_outputs.append(
                {
                    "object_name": object_name,
                    "label_id": obj_idx,
                    "mask_dir": spec["object_dir"],
                    "outputs": None,
                    "skipped": True,
                }
            )
            continue

        outputs = propagate_object_masks(
            tracker,
            video_dir=video_dir,
            init_frame_idx=spec["frame_idx"],
            init_mask=spec["init_mask"],
            score_thresh=score_thresh,
        )

        for frame_idx, frame_file in enumerate(frame_files):
            frame_stem = Path(frame_file).stem
            mask = outputs.get(frame_idx)
            if mask is None:
                mask = np.zeros_like(spec["init_mask"], dtype=np.uint8)
            save_binary_mask(spec["object_dir"] / f"{frame_stem}.png", mask)

        meta = {
            "video_id": video_id,
            "json_id": json_id,
            "object_name": spec["object_name"],
            "seed_frame_filename": spec["frame_filename"],
            "seed_frame_index": spec["frame_idx"],
            "prompt": spec["prompt"],
            "source_pred_json": str(spec["pred_path"]),
            "num_frames": len(frame_files),
        }
        (spec["object_dir"] / "propagation_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2)
        )

        per_object_outputs.append(
            {
                "object_name": object_name,
                "label_id": obj_idx,
                "mask_dir": spec["object_dir"],
                "outputs": outputs,
                "skipped": False,
            }
        )

    for frame_idx, frame_file in enumerate(frame_files):
        frame_stem = Path(frame_file).stem
        merged = None
        for obj in per_object_outputs:
            mask_path = obj["mask_dir"] / f"{frame_stem}.png"
            if obj["outputs"] is None and mask_path.is_file():
                mask = (np.array(Image.open(mask_path)) > 0).astype(np.uint8)
            elif obj["outputs"] is not None:
                mask = obj["outputs"].get(frame_idx)
                if mask is None:
                    continue
            else:
                continue

            if merged is None:
                merged = np.zeros_like(mask, dtype=np.uint8)
            merged[mask > 0] = np.uint8(1)
            # merged[mask > 0] = np.uint8(obj["label_id"])

        if merged is None:
            sample_image = Image.open(video_dir / frame_file)
            merged = np.zeros((sample_image.height, sample_image.width), dtype=np.uint8)
        save_binary_mask(merge_dir / f"{frame_stem}.png", merged)

    summary = {
        "video_id": video_id,
        "json_id": json_id,
        "num_objects": len(per_object_outputs),
        "num_frames": len(frame_files),
        "output_root": str(group_output_root),
        "source_agent_root": str(group["pred_paths"][0].parents[3]) if len(group["pred_paths"][0].parents) >= 4 else str(group["pred_paths"][0].parent),
    }
    (group_output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> int:
    args = parse_args()
    agent_root = Path(args.agent_root)
    dataset_root = Path(args.dataset_root)
    mask_root = Path(args.mask_root)
    output_root = mask_root / args.exp_name

    only_video_ids = set(args.only_video_id or [])
    only_json_ids = set(args.only_json_id or [])
    groups = list_agent_groups(agent_root, only_video_ids or None, only_json_ids or None)
    if args.limit_groups is not None:
        groups = groups[: args.limit_groups]

    print(f"planned_groups: {len(groups)}")
    print(f"output_root: {output_root}")
    if groups:
        first = groups[0]
        print(
            "first_group: "
            f"video_id={first['video_id']} json_id={first['json_id']} num_objects={len(first['pred_paths'])}"
        )

    if args.dry_run:
        return 0

    build_sam3_image_model, build_sam3_video_model = import_sam3_builders(Path(args.sam3_official_root))
    tracker = build_video_tracker(build_sam3_video_model, args.checkpoint, args.device)

    success_count = 0
    error_count = 0
    for group in groups:
        prefix = f"{group['video_id']}/{group['json_id']}"
        try:
            summary = process_group(
                group,
                dataset_root=dataset_root,
                output_root=output_root,
                tracker=tracker,
                score_thresh=args.score_thresh,
                overwrite=args.overwrite,
            )
            success_count += 1
            print(f"saved {prefix} -> {summary['output_root']}")
        except Exception as exc:
            error_count += 1
            print(f"error {prefix}: {exc}")

    print(
        json.dumps(
            {
                "planned_groups": len(groups),
                "success_count": success_count,
                "error_count": error_count,
                "output_root": str(output_root.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
