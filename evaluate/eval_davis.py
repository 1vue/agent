from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from davis2017.evaluation import DAVISEvaluation


NUMERIC_COLUMNS = ["J&F-Mean", "J-Mean", "F-Mean", "J-Recall", "F-Recall"]


def natural_folder_key(path: Path) -> tuple[int, Any]:
    name = path.name
    if name.startswith("anno_"):
        suffix = name.split("anno_", 1)[1]
        if suffix.isdigit():
            return (0, int(suffix))
    return (1, name)


def discover_anno_folders(result_path: Path) -> list[str]:
    folders = [p for p in result_path.iterdir() if p.is_dir() and p.name.startswith("anno_")]
    folders.sort(key=natural_folder_key)
    return [p.name for p in folders]


def normalize_anno_folders(value: list[str] | None) -> list[str] | None:
    if value is None:
        return None
    folders: list[str] = []
    for item in value:
        folders.extend(part.strip() for part in item.split(",") if part.strip())
    return folders


def evaluate_single_folder(folder_info: dict[str, Any]) -> dict[str, Any]:
    folder_name = folder_info["folder_name"]
    davis_path = folder_info["davis_path"]
    result_path = Path(folder_info["result_path"])
    task = folder_info["task"]
    gt_set = folder_info["gt_set"]
    debug = folder_info["debug"]
    folder_path = result_path / folder_name

    if not folder_path.exists():
        return {
            "Folder": folder_name,
            "status": "missing_result_dir",
            "result_dir": str(folder_path),
        }

    try:
        dataset_eval = DAVISEvaluation(davis_root=davis_path, task=task, gt_set=gt_set)
        metrics_res = dataset_eval.evaluate(str(folder_path), debug=debug)
        j_metrics = metrics_res["J"]
        f_metrics = metrics_res["F"]
        j_mean = float(np.mean(j_metrics["M"]))
        f_mean = float(np.mean(f_metrics["M"]))
        return {
            "Folder": folder_name,
            "status": "ok",
            "J&F-Mean": (j_mean + f_mean) / 2.0,
            "J-Mean": j_mean,
            "F-Mean": f_mean,
            "J-Recall": float(np.mean(j_metrics["R"])),
            "F-Recall": float(np.mean(f_metrics["R"])),
            "result_dir": str(folder_path),
        }
    except Exception as exc:
        return {
            "Folder": folder_name,
            "status": "error",
            "error": repr(exc),
            "result_dir": str(folder_path),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--davis_path", type=str, default="../../dataset/davis/DAVIS")
    parser.add_argument("--result_path", "--base_result_path", type=str, default="outputs/davis_mamba/Annotations_new_sam_2")
    parser.add_argument("--save_name", type=str, default=None)
    parser.add_argument("--set", "--gt_set", type=str, default="val")
    parser.add_argument("--task", type=str, default="unsupervised", choices=("unsupervised", "semi-supervised"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--anno-folders",
        nargs="*",
        default=None,
        help="Folders under result_path to evaluate. Defaults to auto-discovered anno_* folders.",
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def default_save_name(result_path: Path) -> Path:
    return result_path / "batch_summary_final.csv"


def build_summary(
    *,
    args: argparse.Namespace,
    result_path: Path,
    save_path: Path,
    folders: list[str],
    results: list[dict[str, Any]],
    elapsed: float,
) -> dict[str, Any]:
    valid = [item for item in results if item.get("status") == "ok"]
    missing_or_error = [item for item in results if item.get("status") != "ok"]

    summary: dict[str, Any] = {
        "davis_path": args.davis_path,
        "result_path": str(result_path),
        "set": args.set,
        "task": args.task,
        "requested_folders": folders,
        "total_folders": len(folders),
        "evaluated_folders": len(valid),
        "missing_or_error_folders": len(missing_or_error),
        "workers": args.workers,
        "time_sec": elapsed,
        "saved_eval": str(save_path),
    }

    if valid:
        for column in NUMERIC_COLUMNS:
            summary[column] = float(np.mean([item[column] for item in valid]))
    return summary


def main() -> int:
    args = parse_args()
    davis_path = Path(args.davis_path)
    result_path = Path(args.result_path)
    save_path = Path(args.save_name) if args.save_name else default_save_name(result_path)

    if not davis_path.exists():
        raise FileNotFoundError(f"DAVIS path not found: {davis_path}")
    if not result_path.exists():
        raise FileNotFoundError(f"Result path not found: {result_path}")

    anno_folders = normalize_anno_folders(args.anno_folders)
    if anno_folders is None:
        anno_folders = discover_anno_folders(result_path)
    if not anno_folders:
        raise FileNotFoundError(f"No anno_* folders found under result path: {result_path}")

    start_time = time.time()
    tasks = [
        {
            "folder_name": folder,
            "davis_path": str(davis_path),
            "result_path": str(result_path),
            "task": args.task,
            "gt_set": args.set,
            "debug": bool(args.debug),
        }
        for folder in anno_folders
    ]

    worker_count = max(1, min(int(args.workers), len(tasks)))
    print(f"Evaluating DAVIS folders: {len(tasks)} | workers: {worker_count}")
    with Pool(processes=worker_count) as pool:
        results = pool.map(evaluate_single_folder, tasks)

    valid = [item for item in results if item.get("status") == "ok"]
    missing_or_error = [item for item in results if item.get("status") != "ok"]
    if not valid:
        print(json.dumps({"results": results}, ensure_ascii=False, indent=2))
        return 1

    df = pd.DataFrame(valid).sort_values(by="Folder")
    avg_values = df[NUMERIC_COLUMNS].mean()
    avg_row = pd.DataFrame(
        [
            {
                "Folder": "OVERALL_AVERAGE",
                "status": "summary",
                **{column: float(avg_values[column]) for column in NUMERIC_COLUMNS},
                "result_dir": "-",
            }
        ]
    )
    df_final = pd.concat([df, avg_row], ignore_index=True)

    summary = build_summary(
        args=args,
        result_path=result_path,
        save_path=save_path,
        folders=anno_folders,
        results=results,
        elapsed=time.time() - start_time,
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path.suffix.lower() == ".json":
        save_path.write_text(
            json.dumps({"summary": summary, "results": results, "missing_or_error": missing_or_error}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    else:
        df_final.to_csv(save_path, index=False, float_format="%.5f")

    print(df_final.to_string(index=False))
    if missing_or_error:
        print(json.dumps({"missing_or_error": missing_or_error}, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
