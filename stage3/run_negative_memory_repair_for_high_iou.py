#!/usr/bin/env python3
"""Repair high-IoU multi-object groups with progressive negative visual memory.

The worker-level unit is one text JSON / group, so groups can run in parallel.
Inside each group, conflicted objects are repaired serially because each repaired
object is added to the negative memory bank for later objects.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import sys
import traceback
from dataclasses import dataclass, field
from functools import partial
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import pycocotools.mask as mask_utils

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api_config import add_api_profile_args, apply_api_profile_defaults
from stage2.generate_sam3_masks_from_agent import decode_agent_masks
from stage2.generate_sam3_masks_from_bbox import (
    build_video_tracker,
    import_sam3_builders,
    propagate_object_masks,
    sorted_jpeg_frame_files,
)
from stage2.run_sam3_agent_from_batch_result import (
    build_processor,
    import_agent_modules,
    resolve_agent_prompt,
    safe_name,
)


@dataclass
class MemoryEntry:
    object_index: int
    object_name: str
    seed_frame_filename: str
    seed_frame_index: int
    seed_mask: np.ndarray
    source_pred_path: str
    propagated_by_frame: dict[str, np.ndarray] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Consume route_badcase_repairs.py high_iou.json and rerun conflicted "
            "objects with progressive instance-occupancy negative memory."
        )
    )
    parser.add_argument("--manifest-json", required=True, help="high_iou.json repair manifest")
    parser.add_argument(
        "--batch-root",
        required=True,
        help="Batch result root used by the initial agent run, e.g. outputs/mevis/aware/initial",
    )
    parser.add_argument(
        "--initial-agent-root",
        required=True,
        help="Initial agent root ending in the model name, e.g. .../sam3_agent/doubao-seed-2.0-pro",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Output root; this script writes under output-root/sam3_agent/<model>/...",
    )
    parser.add_argument("--dataset-root", default="../../dataset/mevis/valid")
    parser.add_argument("--sam3-official-root", default="sam3")
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("SAM3_CHECKPOINT", "checkpoints/sam3.pt"),
        help="Local SAM3 checkpoint path",
    )
    add_api_profile_args(parser)
    parser.add_argument("--base-url", default=os.environ.get("SAM3_AGENT_BASE_URL"))
    parser.add_argument("--model", default=os.environ.get("SAM3_AGENT_MODEL"))
    parser.add_argument("--api-key", default=os.environ.get("SAM3_AGENT_API_KEY"))
    parser.add_argument("--extra-body-json", default=os.environ.get("SAM3_AGENT_EXTRA_BODY_JSON"))
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--score-thresh", type=float, default=0.0)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument(
        "--devices",
        help="Comma-separated devices, e.g. cuda:0,cuda:1. Defaults to cuda:0..cuda:<parallel-1>.",
    )
    parser.add_argument("--only-video-id", action="append")
    parser.add_argument("--only-json-id", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--prompt-source",
        choices=("discriminative_description", "object_name", "target_event", "role_aware"),
        default="role_aware",
    )
    parser.add_argument("--single-object-prompt-hint", action="store_true")
    parser.add_argument("--max-generations", type=int, default=10)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--suppress-threshold",
        type=float,
        default=0.7,
        help="Suppress candidate if area(candidate & memory) / area(candidate) exceeds this value.",
    )
    parser.add_argument(
        "--soft-threshold",
        type=float,
        default=0.4,
        help="Record candidates above this overlap as risky even if not suppressed.",
    )
    parser.add_argument(
        "--min-memory-area",
        type=int,
        default=10,
        help="Do not add masks smaller than this to the memory bank.",
    )
    parser.add_argument(
        "--keep-best-if-all-suppressed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If every SAM3 candidate overlaps memory, keep the least-overlapping candidate.",
    )
    parser.add_argument(
        "--fallback-best-remaining-candidate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If the MLLM returns no mask after negative-memory filtering but filtered "
            "candidates remain, use the highest-score remaining candidate as a fallback."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return apply_api_profile_defaults(args)


def load_manifest_tasks(path: Path, only_video_ids: set[str], only_json_ids: set[str]) -> list[dict[str, Any]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    tasks = []
    for task in manifest.get("tasks", []):
        video_id = str(task["video_id"])
        json_id = str(task["json_id"])
        if only_video_ids and video_id not in only_video_ids:
            continue
        if only_json_ids and json_id not in only_json_ids:
            continue
        tasks.append(task)
    return tasks


def chunk_tasks(tasks: list[dict[str, Any]], num_chunks: int) -> list[list[dict[str, Any]]]:
    if num_chunks <= 0:
        raise ValueError("num_chunks must be positive")
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(num_chunks)]
    for idx, task in enumerate(tasks):
        buckets[idx % num_chunks].append(task)
    return buckets


def resolve_devices(args: argparse.Namespace, task_count: int) -> list[str]:
    if args.devices:
        devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    else:
        devices = [f"cuda:{idx}" for idx in range(max(1, int(args.parallel)))]
    if not devices:
        raise ValueError("No devices configured")
    return devices[: max(1, min(int(args.parallel), len(devices), task_count or 1))]


def output_group_dir_for_task(args: argparse.Namespace, task: dict[str, Any]) -> Path:
    return (
        Path(args.output_root)
        / "sam3_agent"
        / safe_name(args.model)
        / str(task["video_id"])
        / str(task["json_id"])
    )


def summarize_task_completion(args: argparse.Namespace, tasks: list[dict[str, Any]]) -> dict[str, Any]:
    completed: list[dict[str, str]] = []
    pending: list[dict[str, str]] = []
    incomplete_existing: list[dict[str, str]] = []
    for task in tasks:
        video_id = str(task["video_id"])
        json_id = str(task["json_id"])
        group_dir = output_group_dir_for_task(args, task)
        summary_path = group_dir / "negative_memory_repair_summary.json"
        item = {
            "video_id": video_id,
            "json_id": json_id,
            "group_dir": str(group_dir),
            "summary_path": str(summary_path),
        }
        if summary_path.exists() and not args.overwrite:
            completed.append(item)
        else:
            pending.append(item)
            if group_dir.exists() and not summary_path.exists():
                incomplete_existing.append(item)
    return {
        "completed": completed,
        "pending": pending,
        "incomplete_existing": incomplete_existing,
    }


def mask_and_normalize_device(device: str) -> tuple[str | None, str]:
    device = str(device).strip()
    if device == "cpu":
        return None, "cpu"
    if device.startswith("cuda:"):
        return device.split(":", 1)[1], "cuda"
    return None, device


def object_dir_for_index(group_dir: Path, source_item: dict[str, Any], object_index: int) -> Path:
    object_name = source_item.get("object_name") or f"object_{object_index}"
    expected = group_dir / f"{safe_name(object_name)}_obj_{object_index}"
    if expected.exists():
        return expected
    matches = sorted(group_dir.glob(f"*_obj_{object_index}"))
    return matches[0] if matches else expected


def pred_path_for_index(group_dir: Path, source_item: dict[str, Any], object_index: int) -> Path:
    return object_dir_for_index(group_dir, source_item, object_index) / "pred.json"


def seed_frame_from_pred(pred_json: dict[str, Any], pred_path: Path) -> str:
    source_item = pred_json.get("source_result_item", {}) or {}
    frame_filename = source_item.get("actual_frame_filename")
    if frame_filename:
        return str(frame_filename)
    input_meta_path = pred_path.parent / "input_meta.json"
    if input_meta_path.exists():
        input_meta = json.loads(input_meta_path.read_text(encoding="utf-8"))
        frame_filename = input_meta.get("frame_filename")
    if not frame_filename:
        raise ValueError(f"Could not determine seed frame filename from {pred_path}")
    return str(frame_filename)


def load_memory_entry(
    *,
    pred_path: Path,
    object_index: int,
    object_name: str,
    frame_name_to_index: dict[str, int],
    min_memory_area: int,
) -> MemoryEntry | None:
    if not pred_path.exists():
        return None
    pred_json = json.loads(pred_path.read_text(encoding="utf-8"))
    mask = decode_agent_masks(pred_json)
    if int((mask > 0).sum()) < int(min_memory_area):
        return None
    frame_filename = seed_frame_from_pred(pred_json, pred_path)
    if frame_filename not in frame_name_to_index:
        raise ValueError(f"Seed frame {frame_filename!r} is not in video frame list")
    return MemoryEntry(
        object_index=object_index,
        object_name=object_name,
        seed_frame_filename=frame_filename,
        seed_frame_index=frame_name_to_index[frame_filename],
        seed_mask=mask.astype(np.uint8),
        source_pred_path=str(pred_path),
        propagated_by_frame={frame_filename: mask.astype(np.uint8)},
    )


def memory_mask_at_frame(
    entry: MemoryEntry,
    *,
    target_frame_filename: str,
    frame_files: list[str],
    video_dir: Path,
    tracker,
    score_thresh: float,
) -> np.ndarray:
    if target_frame_filename in entry.propagated_by_frame:
        return entry.propagated_by_frame[target_frame_filename]

    outputs = propagate_object_masks(
        tracker,
        video_dir=video_dir,
        init_frame_idx=entry.seed_frame_index,
        init_mask=entry.seed_mask,
        score_thresh=score_thresh,
    )
    for frame_idx, mask in outputs.items():
        if 0 <= int(frame_idx) < len(frame_files):
            entry.propagated_by_frame[frame_files[int(frame_idx)]] = mask.astype(np.uint8)
    return entry.propagated_by_frame.get(
        target_frame_filename,
        np.zeros_like(entry.seed_mask, dtype=np.uint8),
    )


def build_negative_union(
    memory_bank: list[MemoryEntry],
    *,
    target_frame_filename: str,
    frame_files: list[str],
    video_dir: Path,
    tracker,
    score_thresh: float,
) -> tuple[np.ndarray | None, list[dict[str, Any]]]:
    negative_union: np.ndarray | None = None
    used_entries: list[dict[str, Any]] = []
    for entry in memory_bank:
        mask = memory_mask_at_frame(
            entry,
            target_frame_filename=target_frame_filename,
            frame_files=frame_files,
            video_dir=video_dir,
            tracker=tracker,
            score_thresh=score_thresh,
        )
        if negative_union is None:
            negative_union = np.zeros_like(mask, dtype=np.uint8)
        negative_union[mask > 0] = 1
        used_entries.append(
            {
                "object_index": entry.object_index,
                "object_name": entry.object_name,
                "seed_frame_filename": entry.seed_frame_filename,
                "target_frame_filename": target_frame_filename,
                "target_frame_area": int((mask > 0).sum()),
            }
        )
    return negative_union, used_entries


def decode_candidate_mask(counts: str, height: int, width: int) -> np.ndarray:
    decoded = mask_utils.decode({"size": [height, width], "counts": counts})
    if decoded.ndim == 3:
        decoded = decoded[..., 0]
    return (decoded > 0).astype(np.uint8)


def filter_sam_outputs_with_negative_memory(
    outputs: dict[str, Any],
    *,
    negative_union: np.ndarray | None,
    memory_records: list[dict[str, Any]],
    suppress_threshold: float,
    soft_threshold: float,
    keep_best_if_all_suppressed: bool,
) -> dict[str, Any]:
    pred_masks = list(outputs.get("pred_masks") or [])
    if negative_union is None or not pred_masks or int((negative_union > 0).sum()) == 0:
        outputs["negative_memory_filter"] = {
            "enabled": False,
            "reason": "empty_memory_or_no_candidates",
            "memory_records": memory_records,
            "candidate_count_before": len(pred_masks),
            "candidate_count_after": len(pred_masks),
        }
        return outputs

    height = int(outputs["orig_img_h"])
    width = int(outputs["orig_img_w"])
    keep_indices: list[int] = []
    suppressed: list[dict[str, Any]] = []
    risky: list[dict[str, Any]] = []
    scored: list[tuple[float, int]] = []

    for idx, counts in enumerate(pred_masks):
        mask = decode_candidate_mask(counts, height, width)
        candidate_area = int((mask > 0).sum())
        if candidate_area == 0:
            overlap_on_candidate = 1.0
        else:
            intersection = int(np.logical_and(mask > 0, negative_union > 0).sum())
            overlap_on_candidate = float(intersection / max(candidate_area, 1))
        scored.append((overlap_on_candidate, idx))
        record = {
            "candidate_index_1based": idx + 1,
            "candidate_area": candidate_area,
            "overlap_on_candidate": overlap_on_candidate,
        }
        if overlap_on_candidate > float(suppress_threshold):
            suppressed.append(record)
        else:
            keep_indices.append(idx)
            if overlap_on_candidate > float(soft_threshold):
                risky.append(record)

    forced_keep: int | None = None
    if not keep_indices and pred_masks and keep_best_if_all_suppressed:
        _, forced_keep = min(scored, key=lambda item: (item[0], item[1]))
        keep_indices = [forced_keep]
        suppressed = [
            record
            for record in suppressed
            if int(record["candidate_index_1based"]) != forced_keep + 1
        ]

    filtered = dict(outputs)
    filtered["pred_masks"] = [outputs["pred_masks"][idx] for idx in keep_indices]
    filtered["pred_boxes"] = [outputs["pred_boxes"][idx] for idx in keep_indices]
    filtered["pred_scores"] = [outputs["pred_scores"][idx] for idx in keep_indices]
    filtered["negative_memory_filter"] = {
        "enabled": True,
        "suppress_threshold": suppress_threshold,
        "soft_threshold": soft_threshold,
        "memory_records": memory_records,
        "memory_union_area": int((negative_union > 0).sum()),
        "candidate_count_before": len(pred_masks),
        "candidate_count_after": len(keep_indices),
        "kept_indices_1based_original": [idx + 1 for idx in keep_indices],
        "suppressed": suppressed,
        "risky_kept": risky,
        "forced_keep_original_index_1based": None if forced_keep is None else forced_keep + 1,
    }
    return filtered


def make_negative_memory_call_sam_service(
    *,
    original_call_sam_service,
    negative_union: np.ndarray | None,
    memory_records: list[dict[str, Any]],
    suppress_threshold: float,
    soft_threshold: float,
    keep_best_if_all_suppressed: bool,
    filter_state: dict[str, Any] | None = None,
):
    def wrapped_call_sam_service(image_path: str, text_prompt: str, output_folder_path: str = "sam3_output"):
        json_path = Path(
            original_call_sam_service(
                image_path=image_path,
                text_prompt=text_prompt,
                output_folder_path=output_folder_path,
            )
        )
        outputs = json.loads(json_path.read_text(encoding="utf-8"))
        filtered = filter_sam_outputs_with_negative_memory(
            outputs,
            negative_union=negative_union,
            memory_records=memory_records,
            suppress_threshold=suppress_threshold,
            soft_threshold=soft_threshold,
            keep_best_if_all_suppressed=keep_best_if_all_suppressed,
        )
        from sam3.agent.viz import visualize

        filtered_json_path = json_path.with_name(f"{json_path.stem}_negmem.json")
        filtered_image_path = json_path.with_name(f"{json_path.stem}_negmem.png")
        filtered["output_image_path"] = str(filtered_image_path)
        filtered["negative_memory_filter"] = {
            **filtered.get("negative_memory_filter", {}),
            "diagnostic_json_path": str(filtered_json_path),
            "diagnostic_image_path": str(filtered_image_path),
        }
        filtered_json_path.write_text(json.dumps(filtered, ensure_ascii=False, indent=4), encoding="utf-8")
        visualize(filtered).save(filtered_image_path)
        if filter_state is not None:
            filter_state["latest_filtered_outputs"] = filtered
            filter_state["latest_filtered_json_path"] = str(filtered_json_path)
            filter_state["latest_filter"] = filtered.get("negative_memory_filter", {})

        if not filtered.get("negative_memory_filter", {}).get("enabled"):
            json_path.write_text(json.dumps(filtered, ensure_ascii=False, indent=4), encoding="utf-8")
            return str(json_path)

        return str(filtered_json_path)

    return wrapped_call_sam_service


def append_negative_memory_prompt(prompt: str, memory_bank: list[MemoryEntry]) -> str:
    if not memory_bank:
        return prompt
    names = ", ".join(
        f"object {entry.object_index} ({entry.object_name})" for entry in memory_bank
    )
    return (
        f"{prompt}\n"
        "Negative visual memory constraints:\n"
        f"- The following instances have already been selected for other output objects: {names}.\n"
        "- Do not select a candidate mask that corresponds to any already selected instance.\n"
        "- Other same-class instances may still be valid if they are not already selected.\n"
        "- If any remaining candidate plausibly matches the current target after negative-memory filtering, select the best remaining candidate instead of reporting no mask."
    )


def build_best_remaining_fallback_outputs(filter_state: dict[str, Any]) -> dict[str, Any] | None:
    outputs = filter_state.get("latest_filtered_outputs")
    if not isinstance(outputs, dict):
        return None
    pred_masks = list(outputs.get("pred_masks") or [])
    if not pred_masks:
        return None

    scores = list(outputs.get("pred_scores") or [])
    best_idx = 0
    if scores:
        best_idx = max(range(len(pred_masks)), key=lambda idx: float(scores[idx]))

    return {
        "original_image_path": outputs["original_image_path"],
        "output_image_path": outputs.get("output_image_path"),
        "orig_img_h": outputs["orig_img_h"],
        "orig_img_w": outputs["orig_img_w"],
        "pred_boxes": [outputs["pred_boxes"][best_idx]],
        "pred_scores": [outputs["pred_scores"][best_idx]],
        "pred_masks": [outputs["pred_masks"][best_idx]],
        "negative_memory_filter": outputs.get("negative_memory_filter", {}),
        "negative_memory_fallback": {
            "used": True,
            "reason": "mllm_returned_no_mask_with_remaining_candidates",
            "selected_filtered_index_1based": best_idx + 1,
            "selected_score": scores[best_idx] if scores else None,
            "filtered_candidate_count": len(pred_masks),
            "filtered_json_path": filter_state.get("latest_filtered_json_path"),
        },
    }


def save_repaired_object_outputs(
    *,
    item_dir: Path,
    pred_json_path: Path,
    agent_history: list[dict[str, Any]],
    final_output_dict: dict[str, Any],
    rendered_final_output,
    video_id: str,
    json_id: str,
    object_index: int,
    object_name: str,
    prompt: str,
    image_path: Path,
    frame_filename: str,
    target_event: str | None,
    item: dict[str, Any],
    memory_records: list[dict[str, Any]],
    repair_mode: str,
) -> None:
    item_dir.mkdir(parents=True, exist_ok=True)
    final_output_dict = {
        **final_output_dict,
        "video_id": video_id,
        "json_id": str(json_id),
        "object_index": object_index,
        "object_name": object_name,
        "input_prompt": prompt,
        "source_result_item": item,
        "negative_memory_repair": {
            "enabled": True,
            "repair_mode": repair_mode,
            "memory_records": memory_records,
        },
        "role_info": {
            "target_role": item.get("target_role"),
            "target_entity": item.get("target_entity"),
            "action": item.get("action"),
            "segment_target": item.get("segment_target"),
            "target_cardinality": item.get("target_cardinality"),
            "target_set_description": item.get("target_set_description"),
            "requires_instance_enumeration": item.get("requires_instance_enumeration"),
            "reference_entities": item.get("reference_entities", []),
            "excluded_entities": item.get("excluded_entities", []),
        },
    }
    pred_json_path.write_text(json.dumps(final_output_dict, indent=2, ensure_ascii=False), encoding="utf-8")
    (item_dir / "history.json").write_text(json.dumps(agent_history, indent=2, ensure_ascii=False), encoding="utf-8")
    rendered_final_output.save(item_dir / "pred.png")
    (item_dir / "input_meta.json").write_text(
        json.dumps(
            {
                "video_id": video_id,
                "json_id": str(json_id),
                "object_index": object_index,
                "object_name": object_name,
                "image_path": str(image_path),
                "inference_image_path": str(image_path),
                "frame_filename": frame_filename,
                "actual_frame_index": item.get("actual_frame_index"),
                "prompt": prompt,
                "target_event": target_event,
                "negative_memory_repair": final_output_dict["negative_memory_repair"],
                "role_info": final_output_dict["role_info"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def prepare_output_group(initial_group_dir: Path, output_group_dir: Path, overwrite: bool) -> str:
    summary_path = output_group_dir / "negative_memory_repair_summary.json"
    if output_group_dir.exists():
        if summary_path.exists() and not overwrite:
            return "skip_completed"
        if not overwrite:
            shutil.rmtree(output_group_dir)
            if initial_group_dir.exists():
                shutil.copytree(initial_group_dir, output_group_dir)
            else:
                output_group_dir.mkdir(parents=True, exist_ok=True)
            return "prepared_retry_incomplete"
        shutil.rmtree(output_group_dir)
    if initial_group_dir.exists():
        shutil.copytree(initial_group_dir, output_group_dir)
    else:
        output_group_dir.mkdir(parents=True, exist_ok=True)
    return "prepared"


def repair_group(task: dict[str, Any], args: argparse.Namespace, ctx: dict[str, Any]) -> dict[str, Any]:
    from sam3.agent.agent_core import agent_inference

    video_id = str(task["video_id"])
    json_id = str(task["json_id"])
    source_path = Path(task.get("batch_json_path") or Path(args.batch_root) / video_id / f"{json_id}.json")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    results = list(source.get("results") or [])
    conflict_indices = sorted({int(idx) for idx in task.get("conflict_indices", [])})
    if not conflict_indices:
        return {"video_id": video_id, "json_id": json_id, "status": "skip_no_conflicts"}

    manifest_group_dir = task.get("agent_group_dir")
    initial_group_dir = (
        Path(manifest_group_dir)
        if manifest_group_dir
        else Path(args.initial_agent_root) / video_id / json_id
    )
    output_model_root = Path(args.output_root) / "sam3_agent" / safe_name(args.model)
    output_group_dir = output_model_root / video_id / json_id
    prepare_status = prepare_output_group(initial_group_dir, output_group_dir, bool(args.overwrite))
    if prepare_status == "skip_completed":
        return {"video_id": video_id, "json_id": json_id, "status": "skip_completed"}

    video_dir = Path(args.dataset_root) / "JPEGImages" / video_id
    frame_files = sorted_jpeg_frame_files(video_dir)
    frame_name_to_index = {name: idx for idx, name in enumerate(frame_files)}
    target_event = source.get("target_event")

    memory_bank: list[MemoryEntry] = []
    reliable_indices = [
        idx
        for idx in range(1, len(results) + 1)
        if idx not in set(conflict_indices)
    ]
    for idx in reliable_indices:
        item = results[idx - 1]
        object_name = item.get("object_name") or f"object_{idx}"
        pred_path = pred_path_for_index(output_group_dir, item, idx)
        entry = load_memory_entry(
            pred_path=pred_path,
            object_index=idx,
            object_name=object_name,
            frame_name_to_index=frame_name_to_index,
            min_memory_area=args.min_memory_area,
        )
        if entry is not None:
            memory_bank.append(entry)

    repair_records: list[dict[str, Any]] = []
    repair_mode = "memory_seeded" if memory_bank else "bootstrap"
    for idx in conflict_indices:
        if idx < 1 or idx > len(results):
            repair_records.append({"object_index": idx, "status": "invalid_index"})
            continue
        item = results[idx - 1]
        object_name = item.get("object_name") or f"object_{idx}"
        frame_filename = item.get("actual_frame_filename")
        if not frame_filename:
            raise ValueError(f"Missing actual_frame_filename for {video_id}/{json_id} object {idx}")
        image_path = video_dir / frame_filename
        if not image_path.exists():
            raise FileNotFoundError(f"Image frame not found: {image_path}")

        negative_union, memory_records = build_negative_union(
            memory_bank,
            target_frame_filename=frame_filename,
            frame_files=frame_files,
            video_dir=video_dir,
            tracker=ctx["tracker"],
            score_thresh=float(args.score_thresh),
        )
        base_prompt = resolve_agent_prompt(
            item,
            prompt_source=args.prompt_source,
            target_event=target_event,
            single_object_prompt_hint=bool(args.single_object_prompt_hint),
        )
        if not base_prompt:
            raise ValueError(f"Missing prompt for {video_id}/{json_id} object {idx}")
        prompt = append_negative_memory_prompt(base_prompt, memory_bank)
        item_dir = object_dir_for_index(output_group_dir, item, idx)
        pred_json_path = item_dir / "pred.json"
        filter_state: dict[str, Any] = {}
        wrapped_call_sam_service = make_negative_memory_call_sam_service(
            original_call_sam_service=ctx["call_sam_service"],
            negative_union=negative_union,
            memory_records=memory_records,
            suppress_threshold=float(args.suppress_threshold),
            soft_threshold=float(args.soft_threshold),
            keep_best_if_all_suppressed=bool(args.keep_best_if_all_suppressed),
            filter_state=filter_state,
        )
        agent_history, final_output_dict, rendered_final_output = agent_inference(
            str(image_path),
            prompt,
            send_generate_request=ctx["send_generate_request"],
            call_sam_service=wrapped_call_sam_service,
            output_dir=str(item_dir),
            max_generations=int(args.max_generations),
            debug=bool(args.debug),
        )
        fallback_used = False
        if (
            bool(args.fallback_best_remaining_candidate)
            and not final_output_dict.get("pred_masks")
        ):
            fallback_outputs = build_best_remaining_fallback_outputs(filter_state)
            if fallback_outputs is not None:
                from sam3.agent.viz import visualize

                final_output_dict = fallback_outputs
                rendered_final_output = visualize(final_output_dict)
                fallback_used = True
        save_repaired_object_outputs(
            item_dir=item_dir,
            pred_json_path=pred_json_path,
            agent_history=agent_history,
            final_output_dict=final_output_dict,
            rendered_final_output=rendered_final_output,
            video_id=video_id,
            json_id=json_id,
            object_index=idx,
            object_name=object_name,
            prompt=prompt,
            image_path=image_path,
            frame_filename=frame_filename,
            target_event=target_event,
            item=item,
            memory_records=memory_records,
            repair_mode=repair_mode,
        )
        if fallback_used:
            pred_payload = json.loads(pred_json_path.read_text(encoding="utf-8"))
            pred_payload.setdefault("negative_memory_repair", {})
            pred_payload["negative_memory_repair"]["fallback_best_remaining_candidate_used"] = True
            pred_payload["negative_memory_repair"]["fallback_detail"] = final_output_dict.get(
                "negative_memory_fallback"
            )
            pred_json_path.write_text(json.dumps(pred_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        entry = load_memory_entry(
            pred_path=pred_json_path,
            object_index=idx,
            object_name=object_name,
            frame_name_to_index=frame_name_to_index,
            min_memory_area=args.min_memory_area,
        )
        added_to_memory = entry is not None
        if entry is not None:
            memory_bank.append(entry)
        repair_records.append(
            {
                "object_index": idx,
                "object_name": object_name,
                "status": "repaired",
                "memory_count_before": len(memory_records),
                "added_to_memory": added_to_memory,
                "fallback_best_remaining_candidate_used": fallback_used,
                "pred_path": str(pred_json_path),
            }
        )

    summary_path = output_group_dir / "negative_memory_repair_summary.json"
    summary = {
        "video_id": video_id,
        "json_id": json_id,
        "target_event": target_event,
        "source_path": str(source_path),
        "initial_group_dir": str(initial_group_dir),
        "output_group_dir": str(output_group_dir),
        "conflict_indices": conflict_indices,
        "reliable_seed_indices": reliable_indices,
        "repair_mode": repair_mode,
        "repair_records": repair_records,
        "pair_issues": task.get("pair_issues", []),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "video_id": video_id,
        "json_id": json_id,
        "status": "success",
        "summary_path": str(summary_path),
        "repaired_count": len([r for r in repair_records if r.get("status") == "repaired"]),
    }


def build_worker_context(args: argparse.Namespace, device: str) -> dict[str, Any]:
    visible, normalized_device = mask_and_normalize_device(device)
    if visible is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = visible
    elif device.startswith("cuda"):
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    extra_body = json.loads(args.extra_body_json) if args.extra_body_json else None
    (
        build_sam3_image_model,
        Sam3Processor,
        _agent_inference,
        send_generate_request_orig,
        call_sam_service_orig,
    ) = import_agent_modules(Path(args.sam3_official_root))
    processor = build_processor(
        build_sam3_image_model=build_sam3_image_model,
        Sam3Processor=Sam3Processor,
        sam3_official_root=Path(args.sam3_official_root),
        checkpoint=args.checkpoint,
        device=normalized_device,
        confidence_threshold=float(args.confidence_threshold),
    )
    send_generate_request = partial(
        send_generate_request_orig,
        server_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
    )
    call_sam_service = partial(call_sam_service_orig, sam3_processor=processor)
    build_sam3_image_model_v, build_sam3_video_model = import_sam3_builders(Path(args.sam3_official_root))
    tracker = build_video_tracker(build_sam3_video_model, args.checkpoint, normalized_device)
    return {
        "send_generate_request": send_generate_request,
        "call_sam_service": call_sam_service,
        "tracker": tracker,
        "device": device,
        "normalized_device": normalized_device,
        "extra_body": extra_body,
    }


def worker_main(worker_id: int, device: str, tasks: list[dict[str, Any]], args: argparse.Namespace, progress_queue) -> None:
    try:
        ctx = build_worker_context(args, device)
        for task in tasks:
            video_id = str(task["video_id"])
            json_id = str(task["json_id"])
            progress_queue.put(
                {
                    "type": "start",
                    "worker_id": worker_id,
                    "device": device,
                    "video_id": video_id,
                    "json_id": json_id,
                }
            )
            try:
                result = repair_group(task, args, ctx)
                progress_queue.put(
                    {
                        "type": "success",
                        "worker_id": worker_id,
                        "device": device,
                        **result,
                    }
                )
            except Exception as exc:
                progress_queue.put(
                    {
                        "type": "error",
                        "worker_id": worker_id,
                        "device": device,
                        "video_id": video_id,
                        "json_id": json_id,
                        "error": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(limit=80),
                    }
                )
    except Exception as exc:
        progress_queue.put(
            {
                "type": "worker_error",
                "worker_id": worker_id,
                "device": device,
                "error": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(limit=80),
            }
        )


def main() -> int:
    args = parse_args()
    if not args.api_key and not args.dry_run:
        raise ValueError("Missing API key; pass --api-key or set the selected profile api_key_env")
    only_video_ids = set(args.only_video_id or [])
    only_json_ids = set(args.only_json_id or [])
    tasks = load_manifest_tasks(Path(args.manifest_json), only_video_ids, only_json_ids)
    if args.limit is not None:
        tasks = tasks[: int(args.limit)]
    devices = resolve_devices(args, len(tasks))
    completion = summarize_task_completion(args, tasks)

    print(f"planned_groups: {len(tasks)}")
    print(f"completed_groups: {len(completion['completed'])}")
    print(f"pending_groups: {len(completion['pending'])}")
    print(f"incomplete_existing_groups: {len(completion['incomplete_existing'])}")
    print(f"manifest_json: {args.manifest_json}")
    print(f"initial_agent_root: {args.initial_agent_root}")
    print(f"output_root: {args.output_root}")
    print(f"model: {args.model}")
    print(f"parallel: {len(devices)}")
    print(f"devices: {devices}")
    if completion["pending"]:
        first_pending = completion["pending"][0]
        print(f"first_group: {first_pending['video_id']}/{first_pending['json_id']}")
    else:
        print("first_group: none")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "manifest_groups": len(tasks),
                    "completed_groups": len(completion["completed"]),
                    "pending_groups": len(completion["pending"]),
                    "incomplete_existing_groups": len(completion["incomplete_existing"]),
                    "overwrite": bool(args.overwrite),
                    "pending_head": [
                        f"{item['video_id']}/{item['json_id']}"
                        for item in completion["pending"][:20]
                    ],
                    "completed_head": [
                        f"{item['video_id']}/{item['json_id']}"
                        for item in completion["completed"][:20]
                    ],
                    "incomplete_existing_head": [
                        f"{item['video_id']}/{item['json_id']}"
                        for item in completion["incomplete_existing"][:20]
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if not tasks:
        return 0

    runnable_keys = {
        (item["video_id"], item["json_id"]) for item in completion["pending"]
    }
    tasks = [
        task
        for task in tasks
        if (str(task["video_id"]), str(task["json_id"])) in runnable_keys
    ]
    if not tasks:
        print("No pending groups to repair.")
        return 0
    devices = resolve_devices(args, len(tasks))
    chunks = chunk_tasks(tasks, len(devices))
    mp = get_context("spawn")
    progress_queue = mp.Queue()
    workers = []
    for worker_id, device in enumerate(devices):
        proc = mp.Process(
            target=worker_main,
            args=(worker_id, device, chunks[worker_id], args, progress_queue),
            daemon=False,
        )
        proc.start()
        workers.append(proc)

    completed = 0
    success_count = 0
    skip_count = 0
    error_count = 0
    live_workers = len(workers)
    while completed < len(tasks) and live_workers > 0:
        try:
            msg = progress_queue.get(timeout=1.0)
        except queue.Empty:
            live_workers = sum(proc.is_alive() for proc in workers)
            continue

        msg_type = msg["type"]
        if msg_type == "start":
            print(f"[worker {msg['worker_id']} {msg['device']}] start {msg['video_id']}/{msg['json_id']}")
        elif msg_type == "success":
            completed += 1
            status = msg.get("status")
            if status == "success":
                success_count += 1
                print(
                    f"[{completed}/{len(tasks)}] success {msg['video_id']}/{msg['json_id']} "
                    f"repaired={msg.get('repaired_count')} -> {msg.get('summary_path')}"
                )
            else:
                skip_count += 1
                print(f"[{completed}/{len(tasks)}] {status} {msg['video_id']}/{msg['json_id']}")
        elif msg_type == "error":
            completed += 1
            error_count += 1
            print(f"[{completed}/{len(tasks)}] error {msg['video_id']}/{msg['json_id']}: {msg['error']}: {msg['message']}")
            print(msg["traceback"], file=sys.stderr)
        elif msg_type == "worker_error":
            error_count += 1
            print(f"[worker {msg['worker_id']} {msg['device']}] fatal error: {msg['error']}: {msg['message']}")
            print(msg["traceback"], file=sys.stderr)
        live_workers = sum(proc.is_alive() for proc in workers)

    for proc in workers:
        proc.join()

    summary = {
        "planned_groups": len(tasks),
        "success_count": success_count,
        "skip_count": skip_count,
        "error_count": error_count,
        "parallel": len(devices),
        "devices": devices,
        "manifest_json": str(Path(args.manifest_json).resolve()),
        "output_root": str(Path(args.output_root).resolve()),
        "model": args.model,
        "suppress_threshold": args.suppress_threshold,
        "soft_threshold": args.soft_threshold,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
