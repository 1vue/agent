from __future__ import annotations

import argparse
import json
import re
import time
from multiprocessing import Manager, Process, Queue
from pathlib import Path
from queue import Empty
from typing import Any

import cv2
import numpy as np
import pandas as pd

from metrics import db_eval_boundary, db_eval_iou


def natural_sort_key(name: str) -> list[Any]:
    stem = Path(name).stem
    parts = re.split(r"(\d+)", stem)
    return [int(part) if part.isdigit() else part for part in parts]


def read_mask(path: Path, shape_hw: tuple[int, int] | None = None) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        if shape_hw is None:
            raise FileNotFoundError(f"Mask not found: {path}")
        return np.zeros(shape_hw, dtype=np.uint8)
    return (mask > 128).astype(np.uint8)


def build_tasks(meta_exp_path: Path) -> list[dict[str, Any]]:
    meta_exp = json.loads(meta_exp_path.read_text(encoding="utf-8"))["videos"]
    tasks: list[dict[str, Any]] = []
    for video_id, video in meta_exp.items():
        expressions = video["expressions"]
        frames = video.get("frames", [])
        if isinstance(expressions, dict):
            items = sorted(expressions.items(), key=lambda item: natural_sort_key(str(item[0])))
        else:
            items = [(str(index), exp_info) for index, exp_info in enumerate(expressions)]
        for exp_id, exp_info in items:
            tasks.append(
                {
                    "video": video_id,
                    "exp_id": str(exp_id),
                    "exp_text": exp_info.get("exp", ""),
                    "frames": frames,
                    "eval_key": f"{video_id}_{exp_id}",
                }
            )
    return tasks


def resolve_pred_dir(pred_root: Path, video_id: str, exp_id: str) -> Path:
    pred_exp_dir = pred_root / video_id / exp_id
    merge_mask_dir = pred_exp_dir / "merge_mask"
    if merge_mask_dir.exists():
        return merge_mask_dir
    return pred_exp_dir


def list_gt_frames(gt_dir: Path, meta_frames: list[str]) -> list[str]:
    if meta_frames:
        return [f"{frame}.png" if Path(str(frame)).suffix == "" else str(frame) for frame in meta_frames]

    gt_frames = [p.name for p in gt_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png"]
    gt_frames.sort(key=natural_sort_key)
    return gt_frames


def evaluate_one(
    task: dict[str, Any],
    *,
    mask_path: Path,
    pred_path: Path,
    missing_as_empty: bool,
) -> dict[str, Any]:
    video_id = task["video"]
    exp_id = task["exp_id"]
    gt_dir = mask_path / video_id / exp_id
    pred_dir = resolve_pred_dir(pred_path, video_id, exp_id)

    if not gt_dir.exists():
        return {**task, "status": "missing_gt_dir", "J": None, "F": None, "gt_dir": str(gt_dir), "pred_dir": str(pred_dir)}
    if not pred_dir.exists() and not missing_as_empty:
        return {**task, "status": "missing_pred_dir", "J": None, "F": None, "gt_dir": str(gt_dir), "pred_dir": str(pred_dir)}

    frame_names = list_gt_frames(gt_dir, task.get("frames", []))
    if not frame_names:
        return {**task, "status": "empty_gt_dir", "J": None, "F": None, "gt_dir": str(gt_dir), "pred_dir": str(pred_dir)}

    gt_masks = []
    pred_masks = []
    missing_gt_frames = 0
    missing_pred_frames = 0

    for frame_name in frame_names:
        gt_file = gt_dir / frame_name
        if not gt_file.exists():
            missing_gt_frames += 1
            continue

        gt_mask = read_mask(gt_file)
        pred_file = pred_dir / frame_name
        if not pred_file.exists():
            missing_pred_frames += 1
            if not missing_as_empty:
                return {
                    **task,
                    "status": "missing_pred_frame",
                    "J": None,
                    "F": None,
                    "missing_frame": frame_name,
                    "gt_dir": str(gt_dir),
                    "pred_dir": str(pred_dir),
                }
            pred_mask = np.zeros_like(gt_mask)
        else:
            pred_mask = read_mask(pred_file)
            if pred_mask.shape != gt_mask.shape:
                h, w = gt_mask.shape
                pred_mask = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        gt_masks.append(gt_mask)
        pred_masks.append(pred_mask)

    if not gt_masks:
        return {**task, "status": "empty_gt_frames", "J": None, "F": None, "gt_dir": str(gt_dir), "pred_dir": str(pred_dir)}

    gt_stack = np.stack(gt_masks, axis=0)
    pred_stack = np.stack(pred_masks, axis=0)
    j = float(db_eval_iou(gt_stack, pred_stack).mean())
    f = float(db_eval_boundary(gt_stack, pred_stack).mean())
    return {
        **task,
        "status": "ok" if missing_pred_frames == 0 else "ok_missing_pred_as_empty",
        "J": j,
        "F": f,
        "J&F": (j + f) / 2,
        "missing_gt_frames": missing_gt_frames,
        "missing_pred_frames": missing_pred_frames,
        "gt_dir": str(gt_dir),
        "pred_dir": str(pred_dir),
    }


def eval_queue(q: Queue, out_dict, mask_path: str, pred_path: str, missing_as_empty: bool) -> None:
    mask_root = Path(mask_path)
    pred_root = Path(pred_path)
    while True:
        try:
            task = q.get_nowait()
        except Empty:
            break
        result = evaluate_one(
            task,
            mask_path=mask_root,
            pred_path=pred_root,
            missing_as_empty=missing_as_empty,
        )
        out_dict[task["eval_key"]] = result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_exp_path", type=str, default="../../dataset/ref-youtube/meta_expressions/valid/meta_expressions.json")
    parser.add_argument("--mask_path", type=str, default="../../dataset/ref-youtube/valid/Annotations")
    parser.add_argument("--pred_path", type=str, required=True)
    parser.add_argument("--save_name", type=str, default=None)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Exclude missing predictions from metrics instead of treating them as empty masks.",
    )
    return parser.parse_args()


def default_save_name(pred_path: str) -> Path:
    pred = Path(pred_path)
    return pred.parent / f"eval_ytvos_{pred.name}_results.csv"


def main() -> int:
    args = parse_args()
    meta_exp_path = Path(args.meta_exp_path)
    mask_path = Path(args.mask_path)
    pred_path = Path(args.pred_path)

    if not meta_exp_path.exists():
        raise FileNotFoundError(f"Meta file not found: {meta_exp_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"GT mask path not found: {mask_path}")
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction path not found: {pred_path}")

    tasks = build_tasks(meta_exp_path)
    q: Queue = Queue()
    for task in tasks:
        q.put(task)

    print("Q-Size:", q.qsize())
    print(f"GT path: {mask_path}")
    print(f"Pred path: {pred_path}")
    start_time = time.time()

    output_dict = Manager().dict()
    processes = []
    worker_count = max(1, min(int(args.workers), len(tasks))) if tasks else 1
    for _rank in range(worker_count):
        process = Process(
            target=eval_queue,
            args=(q, output_dict, str(mask_path), str(pred_path), not args.skip_missing),
        )
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    results = dict(output_dict)
    all_results = list(results.values())
    valid = [item for item in all_results if item.get("J") is not None and item.get("F") is not None]
    missing = [item for item in all_results if item.get("J") is None or item.get("F") is None]

    if not valid:
        print("No valid evaluation results were produced.")
        return 1

    df = pd.DataFrame(valid).sort_values(by=["video", "exp_id"])
    global_j = float(df["J"].mean())
    global_f = float(df["F"].mean())
    global_jf = float(df["J&F"].mean())

    avg_row = pd.DataFrame(
        [
            {
                "video": "OVERALL_MEAN",
                "exp_id": "-",
                "exp_text": "-",
                "J": global_j,
                "F": global_f,
                "J&F": global_jf,
                "status": "summary",
                "missing_gt_frames": int(df.get("missing_gt_frames", pd.Series(dtype=int)).sum()),
                "missing_pred_frames": int(df.get("missing_pred_frames", pd.Series(dtype=int)).sum()),
            }
        ]
    )
    save_path = Path(args.save_name) if args.save_name else default_save_name(args.pred_path)
    summary = {
        "total_expressions": len(tasks),
        "evaluated_expressions": len(valid),
        "missing_or_skipped_expressions": len(missing),
        "missing_pred_expression_count": sum(1 for item in all_results if str(item.get("status", "")).startswith("missing_pred")),
        "missing_pred_frame_count": int(sum(item.get("missing_pred_frames", 0) or 0 for item in all_results)),
        "J": global_j,
        "F": global_f,
        "J&F": global_jf,
        "missing_policy": "skip" if args.skip_missing else "empty_mask",
        "time_sec": time.time() - start_time,
        "saved_eval": str(save_path),
    }

    df_final = pd.concat([df, avg_row], ignore_index=True)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path.suffix.lower() == ".json":
        save_path.write_text(
            json.dumps({"summary": summary, "results": results, "missing": missing}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    else:
        df_final.to_csv(save_path, index=False, float_format="%.5f")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
