#!/usr/bin/env python3
"""Propagate masks using the best available agent stage per group."""

from __future__ import annotations

import argparse
import json
import os
import traceback
from multiprocessing import get_context
from pathlib import Path
from typing import Any

DEFAULT_CHECKPOINT = os.environ.get("SAM3_CHECKPOINT", "checkpoints/sam3.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Choose the best available agent stage per group (initial, refine1, fallback) "
            "according to scan reports and propagate masks with the official SAM3 tracker."
        )
    )
    parser.add_argument("--base-batch-root", default="outputs/batch_ark_video_qa")
    parser.add_argument("--initial-agent-root", required=True)
    parser.add_argument("--initial-scan-json", required=True)
    parser.add_argument("--refine1-agent-root", required=True)
    parser.add_argument("--refine1-scan-json", required=True)
    parser.add_argument("--fallback-agent-root", required=True)
    parser.add_argument("--fallback-scan-json", required=True)
    parser.add_argument("--dataset-root", default="../../dataset/mevis/valid")
    parser.add_argument("--mask-root", default="outputs/mask_output")
    parser.add_argument("--exp-name", default="exp_agent_final")
    parser.add_argument("--sam3-official-root", default="sam3")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of worker processes to run in parallel (one tracker per worker).",
    )
    parser.add_argument(
        "--devices",
        help=(
            "Comma-separated CUDA devices for workers, e.g. cuda:0,cuda:1. "
            "If omitted and --parallel>1, defaults to cuda:0..cuda:<parallel-1>."
        ),
    )
    parser.add_argument(
        "--print-groups-only",
        action="store_true",
        help="Print selected group assignment info and exit without running propagation.",
    )
    parser.add_argument(
        "--print-limit",
        type=int,
        default=10,
        help="Max items to print per bucket/stage when --print-groups-only is set.",
    )
    parser.add_argument("--score-thresh", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def load_bad_group_set(path: Path) -> set[tuple[str, str]]:
    data = json.loads(path.read_text())
    return {(item["video_id"], str(item["json_id"])) for item in data.get("group_badcases", [])}


def load_terminal_empty_group_set(path: Path) -> set[tuple[str, str]]:
    data = json.loads(path.read_text())
    return {(item["video_id"], str(item["json_id"])) for item in data.get("terminal_empty_groups", [])}


def list_base_groups(batch_root: Path) -> list[tuple[str, str]]:
    groups = []
    for path in sorted(batch_root.glob("*/*.json")):
        if path.parent.name == "_errors":
            continue
        groups.append((path.parent.name, path.stem))
    return groups


def pred_paths_for_group(agent_root: Path, video_id: str, json_id: str) -> list[Path]:
    group_dir = agent_root / video_id / json_id
    return sorted(group_dir.glob("*_obj_*/pred.json"))

def chunk_tasks(tasks: list[dict[str, Any]], num_chunks: int) -> list[list[dict[str, Any]]]:
    if num_chunks <= 0:
        raise ValueError("num_chunks must be positive")
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(num_chunks)]
    for idx, task in enumerate(tasks):
        buckets[idx % num_chunks].append(task)
    return buckets


def resolve_devices(args: argparse.Namespace) -> list[str]:
    if args.devices:
        devices = [item.strip() for item in args.devices.split(",") if item.strip()]
        if not devices:
            raise ValueError("--devices is provided but empty after parsing")
        return devices
    if int(args.parallel) <= 1:
        return [str(args.device)]
    return [f"cuda:{i}" for i in range(int(args.parallel))]

def _mask_and_normalize_device(device: str) -> tuple[str | None, str]:
    """
    Bind a worker process to a single CUDA device.

    If device is "cuda:<N>", we set CUDA_VISIBLE_DEVICES=<N> so the process sees
    only that GPU (as cuda:0). We then normalize the device string to "cuda:0"
    to avoid internal code accidentally mixing cuda:0/cuda:1 tensors.
    """
    device = str(device).strip()
    if device == "cpu":
        return None, "cpu"
    if device.startswith("cuda:"):
        idx_str = device.split(":", 1)[1].strip()
        try:
            idx = int(idx_str)
        except ValueError as exc:
            raise ValueError(f"Invalid CUDA device spec: {device!r}") from exc
        return str(idx), "cuda:0"
    if device == "cuda":
        return None, "cuda"
    return None, device


def _worker_run(
    worker_id: int,
    device: str,
    groups: list[dict[str, Any]],
    args_dict: dict[str, Any],
    result_queue,
) -> None:
    try:
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        visible, normalized_device = _mask_and_normalize_device(device)
        if visible is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = visible

        # Import torch/CUDA-touching modules only after CUDA_VISIBLE_DEVICES is set.
        from stage2.generate_sam3_masks_from_agent import process_group
        from stage2.generate_sam3_masks_from_bbox import build_video_tracker, import_sam3_builders

        build_sam3_image_model, build_sam3_video_model = import_sam3_builders(Path(args_dict["sam3_official_root"]))
        tracker = build_video_tracker(build_sam3_video_model, args_dict["checkpoint"], normalized_device)
        output_root = Path(args_dict["mask_root"]) / args_dict["exp_name"]

        success_count = 0
        error_count = 0
        selection_summary: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []

        dataset_root = Path(args_dict["dataset_root"])
        score_thresh = float(args_dict["score_thresh"])
        overwrite = bool(args_dict["overwrite"])

        for group in groups:
            group_local = dict(group)
            group_local["pred_paths"] = [Path(p) for p in group_local["pred_paths"]]
            prefix = f"{group_local['video_id']}/{group_local['json_id']}"
            try:
                summary = process_group(
                    group_local,
                    dataset_root=dataset_root,
                    output_root=output_root,
                    tracker=tracker,
                    score_thresh=score_thresh,
                    overwrite=overwrite,
                )
                success_count += 1
                selection_summary.append(
                    {
                        "video_id": group_local["video_id"],
                        "json_id": group_local["json_id"],
                        "selected_stage": group_local["selected_stage"],
                        "output_root": summary["output_root"],
                        "worker_id": worker_id,
                        "device": device,
                        "normalized_device": normalized_device,
                    }
                )
                print(
                    f"[worker {worker_id} {device}->{normalized_device}] saved "
                    f"{prefix} ({group_local['selected_stage']}) -> {summary['output_root']}"
                )
            except Exception as exc:
                error_count += 1
                msg = f"{type(exc).__name__}: {exc}"
                print(f"[worker {worker_id} {device}->{normalized_device}] error {prefix}: {msg}")
                errors.append(
                    {
                        "video_id": group_local["video_id"],
                        "json_id": group_local["json_id"],
                        "selected_stage": group_local.get("selected_stage"),
                        "error": msg,
                        "traceback": traceback.format_exc(limit=50),
                    }
                )

        result_queue.put(
            {
                "worker_id": worker_id,
                "device": device,
                "normalized_device": normalized_device,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "success_count": success_count,
                "error_count": error_count,
                "selection_summary": selection_summary,
                "errors": errors,
            }
        )
    except Exception as exc:
        result_queue.put(
            {
                "worker_id": worker_id,
                "device": device,
                "fatal_error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=80),
            }
        )


def main() -> int:
    args = parse_args()
    base_groups = list_base_groups(Path(args.base_batch_root))
    initial_bad = load_bad_group_set(Path(args.initial_scan_json))
    initial_terminal_empty = load_terminal_empty_group_set(Path(args.initial_scan_json))
    refine1_bad = load_bad_group_set(Path(args.refine1_scan_json))
    fallback_bad = load_bad_group_set(Path(args.fallback_scan_json))

    selected_groups: list[dict[str, Any]] = []
    last_list: list[dict[str, str]] = []
    terminal_empty_groups: list[dict[str, str]] = []
    for video_id, json_id in base_groups:
        key = (video_id, json_id)
        if key in initial_terminal_empty:
            terminal_empty_groups.append({"video_id": video_id, "json_id": json_id})
            continue
        if key not in initial_bad:
            root = Path(args.initial_agent_root)
            stage = "initial"
        elif key not in refine1_bad:
            root = Path(args.refine1_agent_root)
            stage = "refine1"
        elif key not in fallback_bad:
            root = Path(args.fallback_agent_root)
            stage = "fallback"
        else:
            last_list.append({"video_id": video_id, "json_id": json_id})
            continue

        pred_paths = pred_paths_for_group(root, video_id, json_id)
        if not pred_paths:
            last_list.append({"video_id": video_id, "json_id": json_id})
            continue
        selected_groups.append(
            {
                "video_id": video_id,
                "json_id": json_id,
                "pred_paths": [str(p) for p in pred_paths],
                "selected_stage": stage,
            }
        )
    

    output_root = Path(args.mask_root) / args.exp_name

    devices = resolve_devices(args)
    parallel = len(devices)

    if args.print_groups_only:
        from collections import Counter

        def group_label(g: dict[str, Any]) -> str:
            return f"{g['video_id']}/{g['json_id']} ({g['selected_stage']})"

        limit = max(0, int(args.print_limit))
        stage_counts = Counter([g["selected_stage"] for g in selected_groups])
        chunks = chunk_tasks(selected_groups, parallel) if parallel > 0 else []

        summary = {
            "base_batch_root": str(Path(args.base_batch_root).resolve()),
            "base_group_count": len(base_groups),
            "selected_group_count": len(selected_groups),
            "stage_counts": dict(stage_counts),
            "last_list_count": len(last_list),
            "terminal_empty_group_count": len(terminal_empty_groups),
            "parallel": parallel,
            "devices": devices,
            "print_limit": limit,
            "selected_groups_head": [group_label(g) for g in selected_groups[:limit]],
            "last_list_head": [f"{x['video_id']}/{x['json_id']}" for x in last_list[:limit]],
            "terminal_empty_groups_head": [f"{x['video_id']}/{x['json_id']}" for x in terminal_empty_groups[:limit]],
            "per_worker": [
                {
                    "worker_id": worker_id,
                    "device": device,
                    "group_count": len(bucket),
                    "groups_head": [group_label(g) for g in bucket[:limit]],
                }
                for worker_id, (device, bucket) in enumerate(zip(devices, chunks, strict=False))
            ],
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    args_dict = {
        "dataset_root": str(args.dataset_root),
        "mask_root": str(args.mask_root),
        "exp_name": str(args.exp_name),
        "sam3_official_root": str(args.sam3_official_root),
        "checkpoint": str(args.checkpoint),
        "score_thresh": float(args.score_thresh),
        "overwrite": bool(args.overwrite),
    }

    success_count = 0
    error_count = 0
    selection_summary: list[dict[str, Any]] = []
    worker_errors: list[dict[str, Any]] = []

    if parallel <= 1:
        ctx = get_context("spawn")
        q = ctx.Queue()
        _worker_run(0, devices[0], selected_groups, args_dict, q)
        result = q.get()
        if result.get("fatal_error"):
            raise RuntimeError(result["fatal_error"])
        success_count = int(result["success_count"])
        error_count = int(result["error_count"])
        selection_summary = list(result.get("selection_summary", []))
        worker_errors = list(result.get("errors", []))
    else:
        ctx = get_context("spawn")
        q = ctx.Queue()
        chunks = chunk_tasks(selected_groups, parallel)
        procs = []
        for worker_id, device in enumerate(devices):
            p = ctx.Process(
                target=_worker_run,
                args=(worker_id, device, chunks[worker_id], args_dict, q),
            )
            p.daemon = False
            p.start()
            procs.append(p)

        results = [q.get() for _ in procs]
        for p in procs:
            p.join()

        for result in results:
            if result.get("fatal_error"):
                error_count += 1
                worker_errors.append(result)
                continue
            success_count += int(result.get("success_count", 0))
            error_count += int(result.get("error_count", 0))
            selection_summary.extend(result.get("selection_summary", []))
            worker_errors.extend(result.get("errors", []))

    final_summary = {
        "selected_group_count": len(selected_groups),
        "last_list_count": len(last_list),
        "terminal_empty_group_count": len(terminal_empty_groups),
        "success_count": success_count,
        "error_count": error_count,
        "output_root": str(output_root.resolve()),
        "parallel": parallel,
        "devices": devices,
        "selection_summary": selection_summary,
        "last_list": last_list,
        "terminal_empty_groups": terminal_empty_groups,
        "worker_errors": worker_errors,
    }
    summary_path = output_root / "_selection_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(final_summary, ensure_ascii=False, indent=2))
    print(f"saved_selection_summary: {summary_path}")
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
