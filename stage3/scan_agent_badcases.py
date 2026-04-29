#!/usr/bin/env python3
"""Scan SAM3-agent outputs for empty masks and high-overlap multi-object cases."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pycocotools.mask as mask_utils

from stage2.run_sam3_agent_from_batch_result import safe_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan one SAM3-agent output root and flag badcases: missing summaries, "
            "missing/empty masks, and IoU>threshold for multi-object same-frame cases."
        )
    )
    parser.add_argument(
        "--batch-root",
        default="outputs/batch_ark_video_qa",
        help="Source batch result root used to determine expected tasks and objects",
    )
    parser.add_argument(
        "--agent-root",
        required=True,
        help="Agent root like agent_output/sam3_agent/<model>",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Where to save the scan report JSON",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.7,
        help="Flag pairwise IoU greater than this threshold as badcase",
    )
    parser.add_argument(
        "--only-video-id",
        action="append",
        help="Optional filter; may be passed multiple times",
    )
    parser.add_argument(
        "--only-json-id",
        action="append",
        help="Optional filter; may be passed multiple times",
    )
    return parser.parse_args()


def decode_pred_union(pred_json: dict[str, Any]) -> np.ndarray:
    h = int(pred_json["orig_img_h"])
    w = int(pred_json["orig_img_w"])
    merged = np.zeros((h, w), dtype=np.uint8)
    for counts in pred_json.get("pred_masks", []):
        if not counts:
            continue
        decoded = mask_utils.decode({"size": [h, w], "counts": counts})
        if decoded.ndim == 3:
            decoded = decoded[..., 0]
        merged[decoded > 0] = 1
    return merged


def iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = np.logical_and(mask_a > 0, mask_b > 0).sum()
    union = np.logical_or(mask_a > 0, mask_b > 0).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


def list_batch_tasks(batch_root: Path, only_video_ids: set[str] | None, only_json_ids: set[str] | None) -> list[Path]:
    tasks = []
    for path in sorted(batch_root.glob("*/*.json")):
        if path.parent.name == "_errors":
            continue
        if only_video_ids and path.parent.name not in only_video_ids:
            continue
        if only_json_ids and path.stem not in only_json_ids:
            continue
        tasks.append(path)
    return tasks


def main() -> int:
    args = parse_args()
    batch_root = Path(args.batch_root)
    agent_root = Path(args.agent_root)
    output_json = Path(args.output_json)
    only_video_ids = set(args.only_video_id or [])
    only_json_ids = set(args.only_json_id or [])

    group_badcases: list[dict[str, Any]] = []
    object_badcases: list[dict[str, Any]] = []
    overlap_badcases: list[dict[str, Any]] = []
    good_groups: list[dict[str, str]] = []
    terminal_empty_groups: list[dict[str, Any]] = []

    total_groups = 0
    for batch_json_path in list_batch_tasks(batch_root, only_video_ids or None, only_json_ids or None):
        total_groups += 1
        source = json.loads(batch_json_path.read_text())
        video_id = source["video_id"]
        json_id = str(source["expression_id"]) if "expression_id" in source else batch_json_path.stem
        results = source.get("results", [])
        group_dir = agent_root / video_id / json_id
        summary_path = group_dir / "summary.json"

        reasons: list[str] = []
        issue_objects: list[dict[str, Any]] = []
        issue_pairs: list[dict[str, Any]] = []

        if not results:
            terminal_empty_groups.append(
                {
                    "video_id": video_id,
                    "json_id": json_id,
                    "target_event": source.get("target_event"),
                    "batch_json_path": str(batch_json_path),
                }
            )
            continue

        if not summary_path.exists():
            reasons.append("missing_summary")

        valid_entries: list[dict[str, Any]] = []
        for idx, item in enumerate(results, start=1):
            object_name = item.get("object_name") or f"object_{idx}"
            obj_dir = group_dir / f"{safe_name(object_name)}_obj_{idx}"
            pred_path = obj_dir / "pred.json"
            if not pred_path.exists():
                reasons.append("missing_pred")
                bad = {
                    "reason": "missing_pred",
                    "video_id": video_id,
                    "json_id": json_id,
                    "object_index": idx,
                    "object_name": object_name,
                    "batch_json_path": str(batch_json_path),
                    "expected_pred_path": str(pred_path),
                }
                object_badcases.append(bad)
                issue_objects.append(bad)
                continue

            pred_json = json.loads(pred_path.read_text())
            union_mask = decode_pred_union(pred_json)
            area = int((union_mask > 0).sum())
            frame_filename = (
                pred_json.get("source_result_item", {}) or {}
            ).get("actual_frame_filename") or item.get("actual_frame_filename")
            entry = {
                "object_index": idx,
                "object_name": object_name,
                "pred_path": str(pred_path),
                "frame_filename": frame_filename,
                "area": area,
                "mask": union_mask,
            }
            valid_entries.append(entry)
            if area == 0:
                reasons.append("empty_mask")
                bad = {
                    "reason": "empty_mask",
                    "video_id": video_id,
                    "json_id": json_id,
                    "object_index": idx,
                    "object_name": object_name,
                    "batch_json_path": str(batch_json_path),
                    "pred_path": str(pred_path),
                    "frame_filename": frame_filename,
                }
                object_badcases.append(bad)
                issue_objects.append(bad)

        by_frame: dict[str, list[dict[str, Any]]] = {}
        for entry in valid_entries:
            if entry["area"] == 0 or not entry["frame_filename"]:
                continue
            by_frame.setdefault(entry["frame_filename"], []).append(entry)

        for frame_filename, entries in by_frame.items():
            if len(entries) < 2:
                continue
            for a, b in combinations(entries, 2):
                pair_iou = iou(a["mask"], b["mask"])
                if pair_iou > args.iou_threshold:
                    reasons.append("high_iou")
                    bad = {
                        "reason": "high_iou",
                        "video_id": video_id,
                        "json_id": json_id,
                        "frame_filename": frame_filename,
                        "object_a_index": a["object_index"],
                        "object_a_name": a["object_name"],
                        "object_b_index": b["object_index"],
                        "object_b_name": b["object_name"],
                        "iou": pair_iou,
                        "pred_a_path": a["pred_path"],
                        "pred_b_path": b["pred_path"],
                    }
                    overlap_badcases.append(bad)
                    issue_pairs.append(bad)

        reasons = sorted(set(reasons))
        if reasons:
            group_badcases.append(
                {
                    "video_id": video_id,
                    "json_id": json_id,
                    "target_event": source.get("target_event"),
                    "batch_json_path": str(batch_json_path),
                    "agent_group_dir": str(group_dir),
                    "reasons": reasons,
                    "object_issues": issue_objects,
                    "pair_issues": issue_pairs,
                    "num_results": len(results),
                }
            )
        else:
            good_groups.append({"video_id": video_id, "json_id": json_id})

    report = {
        "batch_root": str(batch_root.resolve()),
        "agent_root": str(agent_root.resolve()),
        "iou_threshold": args.iou_threshold,
        "total_groups": total_groups,
        "bad_group_count": len(group_badcases),
        "good_group_count": len(good_groups),
        "terminal_empty_group_count": len(terminal_empty_groups),
        "object_badcase_count": len(object_badcases),
        "overlap_badcase_count": len(overlap_badcases),
        "group_badcases": group_badcases,
        "object_badcases": object_badcases,
        "overlap_badcases": overlap_badcases,
        "good_groups": good_groups,
        "terminal_empty_groups": terminal_empty_groups,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: report[k] for k in ["total_groups", "bad_group_count", "good_group_count", "terminal_empty_group_count", "object_badcase_count", "overlap_badcase_count"]}, ensure_ascii=False, indent=2))
    print(f"saved_scan: {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
