###########################################################################
# Adapted from the official ReasonVOS evaluation script for this project.
###########################################################################

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter
from multiprocessing import Manager, Process, Queue
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from metrics import db_eval_boundary, db_eval_iou


def natural_sort_key(name: str) -> list[Any]:
    stem = Path(name).stem
    parts = re.split(r"(\d+)", stem)
    return [int(part) if part.isdigit() else part for part in parts]


def safe_id_part(value: Any) -> str:
    text = str(value).strip()
    text = re.sub(r"[^\w\-\.]+", "_", text, flags=re.UNICODE)
    return text.strip("._") or "unknown"


def reasonvos_json_id(expressions: list[dict[str, Any]], index: int) -> str:
    """Match stage1's output naming rule for ReasonVOS list-style expressions."""
    base_ids = [str(exp.get("exp_id", i)) for i, exp in enumerate(expressions)]
    duplicated_base_ids = {base_id for base_id, count in Counter(base_ids).items() if count > 1}

    exp = expressions[index]
    base_expression_id = str(exp.get("exp_id", index))
    obj_id = exp.get("obj_id")
    if base_expression_id in duplicated_base_ids and obj_id is not None:
        base_output_id = f"{base_expression_id}_obj_{safe_id_part(obj_id)}"
    else:
        base_output_id = base_expression_id

    same_id_seen = 0
    for prev_index, prev_exp in enumerate(expressions[: index + 1]):
        prev_base_id = str(prev_exp.get("exp_id", prev_index))
        prev_obj_id = prev_exp.get("obj_id")
        if prev_base_id in duplicated_base_ids and prev_obj_id is not None:
            prev_output_id = f"{prev_base_id}_obj_{safe_id_part(prev_obj_id)}"
        else:
            prev_output_id = prev_base_id
        if prev_output_id == base_output_id:
            same_id_seen += 1

    if same_id_seen <= 1:
        return base_output_id
    return f"{base_output_id}__{index}"


def build_tasks(meta_exp_path: Path) -> list[dict[str, Any]]:
    meta_exp = json.loads(meta_exp_path.read_text(encoding="utf-8"))["videos"]
    tasks: list[dict[str, Any]] = []
    for video_id, video in meta_exp.items():
        src_dataset = video["source"]
        expressions = video["expressions"]
        if isinstance(expressions, dict):
            items = sorted(expressions.items(), key=lambda item: natural_sort_key(str(item[0])))
            for exp_id, sample in items:
                obj_id = sample["obj_id"]
                tasks.append(
                    {
                        "video_id": video_id,
                        "source": src_dataset,
                        "obj_id": str(obj_id),
                        "raw_exp_id": str(exp_id),
                        "json_id": str(exp_id),
                        "eval_key": f"{src_dataset}_{video_id}_{obj_id}_{exp_id}",
                    }
                )
        else:
            for index, sample in enumerate(expressions):
                obj_id = sample["obj_id"]
                exp_id = sample.get("exp_id", index)
                json_id = reasonvos_json_id(expressions, index)
                tasks.append(
                    {
                        "video_id": video_id,
                        "source": src_dataset,
                        "obj_id": str(obj_id),
                        "raw_exp_id": str(exp_id),
                        "json_id": json_id,
                        "eval_key": f"{src_dataset}_{video_id}_{obj_id}_{exp_id}",
                    }
                )
    return tasks


def read_mask(path: Path, shape_hw: tuple[int, int] | None = None) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        if shape_hw is None:
            raise FileNotFoundError(f"Mask not found: {path}")
        return np.zeros(shape_hw, dtype=np.uint8)
    return mask


def evaluate_one(
    task: dict[str, Any],
    *,
    mask_path: Path,
    pred_path: Path,
    missing_as_empty: bool,
) -> dict[str, Any]:
    video_id = task["video_id"]
    json_id = task["json_id"]
    vid_key = f"{task['source']}_{video_id}_{task['obj_id']}"
    gt_dir = mask_path / vid_key
    pred_dir = pred_path / video_id / json_id / "merge_mask"

    if not gt_dir.exists():
        return {**task, "status": "missing_gt_dir", "j": None, "f": None, "gt_dir": str(gt_dir), "pred_dir": str(pred_dir)}
    if not pred_dir.exists() and not missing_as_empty:
        return {**task, "status": "missing_pred_dir", "j": None, "f": None, "gt_dir": str(gt_dir), "pred_dir": str(pred_dir)}

    gt_mask_list = [p.name for p in gt_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png"]
    gt_mask_list.sort(key=natural_sort_key)
    if not gt_mask_list:
        return {**task, "status": "empty_gt_dir", "j": None, "f": None, "gt_dir": str(gt_dir), "pred_dir": str(pred_dir)}

    gt_0 = read_mask(gt_dir / gt_mask_list[0])
    h, w = gt_0.shape
    gt_masks = np.zeros((len(gt_mask_list), h, w), dtype=np.uint8)
    pred_masks = np.zeros((len(gt_mask_list), h, w), dtype=np.uint8)
    missing_frames = 0

    for frame_idx, frame_name in enumerate(gt_mask_list):
        gt_masks[frame_idx] = read_mask(gt_dir / frame_name)
        pred_file = pred_dir / frame_name
        if not pred_file.exists():
            missing_frames += 1
            if not missing_as_empty:
                return {
                    **task,
                    "status": "missing_pred_frame",
                    "j": None,
                    "f": None,
                    "missing_frame": frame_name,
                    "gt_dir": str(gt_dir),
                    "pred_dir": str(pred_dir),
                }
        pred_masks[frame_idx] = read_mask(pred_file, shape_hw=(h, w))

    j = float(db_eval_iou(gt_masks, pred_masks).mean())
    f = float(db_eval_boundary(gt_masks, pred_masks).mean())
    return {
        **task,
        "status": "ok" if missing_frames == 0 else "ok_missing_pred_as_empty",
        "j": j,
        "f": f,
        "j_and_f": (j + f) / 2,
        "missing_frames": missing_frames,
        "gt_dir": str(gt_dir),
        "pred_dir": str(pred_dir),
    }


def eval_queue(q: Queue, out_dict, mask_path: str, pred_path: str, missing_as_empty: bool) -> None:
    mask_root = Path(mask_path)
    pred_root = Path(pred_path)
    while not q.empty():
        task = q.get()
        result = evaluate_one(
            task,
            mask_path=mask_root,
            pred_path=pred_root,
            missing_as_empty=missing_as_empty,
        )
        out_dict[task["eval_key"]] = result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_exp_path", type=str, default="../../dataset/ReasonVOS/meta_expressions.json")
    parser.add_argument("--mask_path", type=str, default="../../dataset/ReasonVOS/Annotations")
    parser.add_argument("--pred_path", type=str, required=True)
    parser.add_argument("--save_name", type=str, default="outputs/resonvos/eval_reason_vos.json")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Mimic the official behavior more closely by excluding missing predictions from metrics.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tasks = build_tasks(Path(args.meta_exp_path))
    q: Queue = Queue()
    for task in tasks:
        q.put(task)

    print("Q-Size:", q.qsize())
    start_time = time.time()

    output_dict = Manager().dict()
    processes = []
    worker_count = max(1, min(int(args.workers), len(tasks)))
    for _rank in range(worker_count):
        process = Process(
            target=eval_queue,
            args=(q, output_dict, args.mask_path, args.pred_path, not args.skip_missing),
        )
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    results = dict(output_dict)
    valid = [item for item in results.values() if item.get("j") is not None and item.get("f") is not None]
    missing = [item for item in results.values() if item.get("j") is None or item.get("f") is None]
    j = [item["j"] for item in valid]
    f = [item["f"] for item in valid]

    summary = {
        "total_expressions": len(tasks),
        "evaluated_expressions": len(valid),
        "missing_or_skipped_expressions": len(missing),
        "missing_pred_expression_count": sum(1 for item in results.values() if str(item.get("status", "")).startswith("missing_pred")),
        "missing_pred_frame_count": int(sum(item.get("missing_frames", 0) or 0 for item in results.values())),
        "J": float(np.mean(j)) if j else None,
        "F": float(np.mean(f)) if f else None,
        "J&F": float((np.mean(j) + np.mean(f)) / 2) if j and f else None,
        "missing_policy": "skip" if args.skip_missing else "empty_mask",
        "time_sec": time.time() - start_time,
    }

    save_path = Path(args.save_name)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(
        json.dumps({"summary": summary, "results": results, "missing": missing}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved_eval: {save_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
