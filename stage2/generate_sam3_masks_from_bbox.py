#!/usr/bin/env python3
"""Generate per-object and merged masks from bbox_output using SAM3."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


OBJ_SUFFIX_RE = re.compile(r"_obj_(\d+)$")
DEFAULT_CHECKPOINT = os.environ.get("SAM3_CHECKPOINT", "checkpoints/sam3.pt")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read bbox txt files from bbox_output, use SAM3 to segment the prompted frame, "
            "propagate masks forward/backward through the whole video, and save per-object "
            "and merged masks under mask_output/<exp_name>/<video>/<json_id>/"
        )
    )
    parser.add_argument("--bbox-root", default="bbox_output", help="Directory containing bbox txt files")
    parser.add_argument(
        "--dataset-root",
        default="../../dataset/mevis/valid",
        help="Dataset root directory containing JPEGImages/",
    )
    parser.add_argument(
        "--mask-root",
        default="mask_output",
        help="Directory to save propagated masks",
    )
    parser.add_argument(
        "--exp-name",
        default="exp_i",
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
        help="Optional filter on bbox json-id stem; may be passed multiple times",
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


def numeric_sort_key(stem: str):
    parts = re.split(r"(\d+)", stem)
    key = [int(part) if part.isdigit() else part for part in parts]
    return (0, key)


def resolve_video_dir(dataset_root: Path, video_id: str) -> Path:
    candidates = [
        dataset_root / "JPEGImages" / video_id,
        dataset_root / video_id,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Video id {video_id!r} not found. Tried: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def sorted_jpeg_frame_files(video_dir: Path) -> list[str]:
    files = [
        p.name
        for p in video_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ]
    files.sort(key=lambda name: numeric_sort_key(Path(name).stem))
    return files


def parse_bbox_txt(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for line in path.read_text().splitlines():
        if not line.strip() or "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip()
    if "frame_filename" not in data or "bbox" not in data:
        raise ValueError(f"Invalid bbox txt format: {path}")
    bbox = [int(x) for x in data["bbox"].split()]
    if len(bbox) != 4:
        raise ValueError(f"Invalid bbox coordinate count in {path}: {bbox}")
    frame_index = int(data.get("frame_index", Path(data["frame_filename"]).stem))
    return {
        "frame_filename": data["frame_filename"],
        "frame_index": frame_index,
        "bbox_xyxy": bbox,
    }


def parse_object_order(stem: str) -> tuple[int, str]:
    match = OBJ_SUFFIX_RE.search(stem)
    if match:
        return int(match.group(1)), stem
    return 10**9, stem


def list_bbox_groups(
    bbox_root: Path,
    only_video_ids: set[str] | None,
    only_json_ids: set[str] | None,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for txt_path in sorted(bbox_root.rglob("*.txt")):
        relative_parts = txt_path.relative_to(bbox_root).parts
        if len(relative_parts) < 3:
            continue
        video_id = "/".join(relative_parts[:-2])
        json_id = relative_parts[-2]
        if only_video_ids and video_id not in only_video_ids:
            continue
        if only_json_ids and json_id not in only_json_ids:
            continue
        groups[(video_id, json_id)].append(txt_path)

    result = []
    for (video_id, json_id), txt_paths in sorted(groups.items()):
        txt_paths.sort(key=lambda p: parse_object_order(p.stem))
        result.append({"video_id": video_id, "json_id": json_id, "txt_paths": txt_paths})
    return result


def import_sam3_builders(sam3_official_root: Path):
    root_str = str(sam3_official_root.resolve())
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    from sam3.model_builder import build_sam3_image_model, build_sam3_video_model

    return build_sam3_image_model, build_sam3_video_model


def build_image_predictor(build_sam3_image_model, checkpoint: str, device: str):
    model = build_sam3_image_model(
        checkpoint_path=checkpoint,
        load_from_HF=False,
        enable_inst_interactivity=True,
        device=device,
        eval_mode=True,
    )
    predictor = model.inst_interactive_predictor
    if predictor.model.backbone is None:
        predictor.model.backbone = model.backbone
    return predictor


def build_video_tracker(build_sam3_video_model, checkpoint: str, device: str):
    model = build_sam3_video_model(
        checkpoint_path=checkpoint,
        load_from_HF=False,
        strict_state_dict_loading=True,
        apply_temporal_disambiguation=True,
        device=device,
        compile=False,
    )
    tracker = model.tracker
    tracker.backbone = model.detector.backbone
    tracker.eval()
    return tracker


def segment_bbox_on_frame(predictor, image_path: Path, bbox_xyxy: list[int]) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    predictor.set_image(image)
    box = np.asarray(bbox_xyxy, dtype=np.float32)
    masks, _, _ = predictor.predict(
        box=box,
        multimask_output=False,
        return_logits=False,
        normalize_coords=True,
    )
    return (masks[0] > 0).astype(np.uint8)


def propagate_object_masks(
    tracker,
    video_dir: Path,
    init_frame_idx: int,
    init_mask: np.ndarray,
    score_thresh: float,
) -> dict[int, np.ndarray]:
    inference_state = tracker.init_state(
        video_path=str(video_dir),
        offload_video_to_cpu=True,
        offload_state_to_cpu=False,
        async_loading_frames=False,
    )
    tracker.clear_all_points_in_video(inference_state)
    tracker.add_new_mask(
        inference_state=inference_state,
        frame_idx=int(init_frame_idx),
        obj_id=1,
        mask=torch.from_numpy(init_mask.astype(np.uint8)),
    )

    outputs: dict[int, np.ndarray] = {int(init_frame_idx): init_mask.astype(np.uint8)}

    for frame_idx, out_obj_ids, _, video_res_masks, _ in tracker.propagate_in_video(
        inference_state=inference_state,
        start_frame_idx=int(init_frame_idx),
        max_frame_num_to_track=None,
        reverse=False,
        tqdm_disable=True,
        propagate_preflight=True,
    ):
        logits = video_res_masks[:, 0].detach().cpu().numpy()
        mask_out = (logits[0] > float(score_thresh)).astype(np.uint8)
        outputs[int(frame_idx)] = mask_out

    if int(init_frame_idx) > 0:
        for frame_idx, out_obj_ids, _, video_res_masks, _ in tracker.propagate_in_video(
            inference_state=inference_state,
            start_frame_idx=int(init_frame_idx),
            max_frame_num_to_track=None,
            reverse=True,
            tqdm_disable=True,
            propagate_preflight=False,
        ):
            logits = video_res_masks[:, 0].detach().cpu().numpy()
            mask_out = (logits[0] > float(score_thresh)).astype(np.uint8)
            outputs[int(frame_idx)] = mask_out

    return outputs

# 获得id
def save_instance_mask(path: Path, mask: np.ndarray) -> None:
    """保存实例分割 mask，其中像素值为 0（背景）或 1..K（实例 ID）"""
    path.parent.mkdir(parents=True, exist_ok=True)
    # 直接保存原始值，不乘以 255
    Image.fromarray(mask.astype(np.uint8)).save(path)

def save_binary_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255)).save(path)


def process_group(
    group: dict[str, Any],
    *,
    dataset_root: Path,
    output_root: Path,
    predictor,
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
    for txt_path in group["txt_paths"]:
        object_dir = group_output_root / txt_path.stem
        first_frame_out = object_dir / f"{Path(frame_files[0]).stem}.png"
        if first_frame_out.exists() and not overwrite:
            object_specs.append({"skip": True, "txt_path": txt_path, "object_dir": object_dir})
            continue
        bbox_info = parse_bbox_txt(txt_path)
        frame_filename = bbox_info["frame_filename"]
        if frame_filename not in frame_name_to_index:
            raise ValueError(f"Frame {frame_filename} from {txt_path} not found in {video_dir}")
        init_frame_idx = frame_name_to_index[frame_filename]
        image_path = video_dir / frame_filename
        object_specs.append(
            {
                "skip": False,
                "txt_path": txt_path,
                "object_dir": object_dir,
                "frame_filename": frame_filename,
                "frame_idx": init_frame_idx,
                "image_path": image_path,
                "bbox_xyxy": bbox_info["bbox_xyxy"],
            }
        )

    per_object_outputs: list[dict[str, Any]] = []
    for obj_idx, spec in enumerate(object_specs, start=1):
        object_name = spec["txt_path"].stem
        if spec.get("skip"):
            per_object_outputs.append({
                "object_name": object_name,
                "label_id": obj_idx,
                "mask_dir": spec["object_dir"],
                "outputs": None,
                "skipped": True,
            })
            continue

        init_mask = segment_bbox_on_frame(predictor, spec["image_path"], spec["bbox_xyxy"])
        outputs = propagate_object_masks(
            tracker,
            video_dir=video_dir,
            init_frame_idx=spec["frame_idx"],
            init_mask=init_mask,
            score_thresh=score_thresh,
        )
        for frame_idx, frame_file in enumerate(frame_files):
            frame_stem = Path(frame_file).stem
            mask = outputs.get(frame_idx)
            if mask is None:
                mask = np.zeros((init_mask.shape[0], init_mask.shape[1]), dtype=np.uint8)
            save_binary_mask(spec["object_dir"] / f"{frame_stem}.png", mask)
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
        if merged is None:
            # If every object was skipped but outputs are absent, create an empty mask based on the frame size.
            sample_image = Image.open(video_dir / frame_file)
            merged = np.zeros((sample_image.height, sample_image.width), dtype=np.uint8)
        save_binary_mask(merge_dir / f"{frame_stem}.png", merged)

    return {
        "video_id": video_id,
        "json_id": json_id,
        "num_objects": len(per_object_outputs),
        "num_frames": len(frame_files),
        "output_root": str(group_output_root),
    }


def main() -> int:
    args = parse_args()
    bbox_root = Path(args.bbox_root)
    dataset_root = Path(args.dataset_root)
    mask_root = Path(args.mask_root)
    output_root = mask_root / args.exp_name
    only_video_ids = set(args.only_video_id or [])
    only_json_ids = set(args.only_json_id or [])
    groups = list_bbox_groups(bbox_root, only_video_ids or None, only_json_ids or None)
    if args.limit_groups is not None:
        groups = groups[: args.limit_groups]

    print(f"planned_groups: {len(groups)}")
    print(f"output_root: {output_root}")
    if groups:
        first = groups[0]
        print(
            "first_group: "
            f"video_id={first['video_id']} json_id={first['json_id']} num_objects={len(first['txt_paths'])}"
        )

    if args.dry_run:
        return 0

    build_sam3_image_model, build_sam3_video_model = import_sam3_builders(Path(args.sam3_official_root))
    predictor = build_image_predictor(build_sam3_image_model, args.checkpoint, args.device)
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
                predictor=predictor,
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
