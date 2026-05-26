#!/usr/bin/env python3
"""Parallel propagation of SAM3-agent seed masks through full videos."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback
from multiprocessing import get_context
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OBJ_SUFFIX_RE = re.compile(r"_obj_(\d+)$")
DEFAULT_CHECKPOINT = os.environ.get("SAM3_CHECKPOINT", "checkpoints/sam3.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read SAM3-agent pred.json files and propagate seed masks through full videos "
            "with one SAM3 tracker per worker process."
        )
    )
    parser.add_argument(
        "--agent-root",
        default="agent_output/sam3_agent",
        help=(
            "Root containing agent outputs. Point this to the concrete model directory, "
            "for example outputs/.../sam3_agent/doubao-seed-2.0-pro."
        ),
    )
    parser.add_argument("--dataset-root", default="../../dataset/mevis/valid")
    parser.add_argument("--mask-root", default="outputs/mask_output")
    parser.add_argument("--exp-name", default="exp_agent")
    parser.add_argument("--sam3-official-root", default="sam3")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--devices",
        help=(
            "Comma-separated worker devices, e.g. cuda:0,cuda:1. If omitted, "
            "--parallel > 1 defaults to cuda:0..cuda:<parallel-1>."
        ),
    )
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--score-thresh", type=float, default=0.0)
    parser.add_argument("--only-video-id", action="append")
    parser.add_argument("--only-json-id", action="append")
    parser.add_argument("--limit-groups", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--error-json",
        default=None,
        help="Optional path for detailed propagation errors; defaults under output_root.",
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
    groups: dict[tuple[str, str], list[Path]] = {}
    for pred_path in sorted(agent_root.rglob("pred.json")):
        parts = pred_path.relative_to(agent_root).parts
        if len(parts) < 4:
            continue
        video_id = parts[-4]
        json_id = parts[-3]
        if only_video_ids and video_id not in only_video_ids:
            continue
        if only_json_ids and json_id not in only_json_ids:
            continue
        groups.setdefault((video_id, json_id), []).append(pred_path)

    result: list[dict[str, Any]] = []
    for (video_id, json_id), pred_paths in sorted(groups.items()):
        pred_paths.sort(key=lambda p: parse_object_order(p.parent.name))
        result.append(
            {
                "video_id": video_id,
                "json_id": json_id,
                "pred_paths": [str(path) for path in pred_paths],
            }
        )
    return result


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
            raise ValueError("--devices was provided but parsed as empty")
        return devices
    if args.parallel <= 1:
        return [str(args.device)]
    return [f"cuda:{idx}" for idx in range(args.parallel)]


def mask_and_normalize_device(device: str) -> tuple[str | None, str]:
    device = str(device).strip()
    if device == "cpu":
        return None, "cpu"
    if device.startswith("cuda:"):
        idx_text = device.split(":", 1)[1].strip()
        try:
            int(idx_text)
        except ValueError as exc:
            raise ValueError(f"Invalid CUDA device spec: {device!r}") from exc
        return idx_text, "cuda:0"
    if device == "cuda":
        return None, "cuda"
    return None, device


def worker_main(
    worker_id: int,
    device: str,
    groups: list[dict[str, Any]],
    args_dict: dict[str, Any],
    result_queue,
) -> None:
    try:
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        visible_device, normalized_device = mask_and_normalize_device(device)
        if visible_device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = visible_device

        # Import SAM3/torch-touching modules only after CUDA_VISIBLE_DEVICES is set.
        from stage2.generate_sam3_masks_from_agent import process_group
        from stage2.generate_sam3_masks_from_bbox import build_video_tracker, import_sam3_builders

        build_sam3_image_model, build_sam3_video_model = import_sam3_builders(
            Path(args_dict["sam3_official_root"])
        )
        _ = build_sam3_image_model
        tracker = build_video_tracker(
            build_sam3_video_model,
            args_dict["checkpoint"],
            normalized_device,
        )

        dataset_root = Path(args_dict["dataset_root"])
        output_root = Path(args_dict["mask_root"]) / args_dict["exp_name"]
        score_thresh = float(args_dict["score_thresh"])
        overwrite = bool(args_dict["overwrite"])

        success_count = 0
        error_count = 0
        summaries: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []

        for group in groups:
            group_local = dict(group)
            group_local["pred_paths"] = [Path(path) for path in group_local["pred_paths"]]
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
                summaries.append(
                    {
                        "video_id": group_local["video_id"],
                        "json_id": group_local["json_id"],
                        "output_root": summary["output_root"],
                        "worker_id": worker_id,
                        "device": device,
                        "normalized_device": normalized_device,
                    }
                )
                print(
                    f"[worker {worker_id} {device}->{normalized_device}] saved "
                    f"{prefix} -> {summary['output_root']}",
                    flush=True,
                )
            except Exception as exc:
                error_count += 1
                message = f"{type(exc).__name__}: {exc}"
                errors.append(
                    {
                        "video_id": group_local["video_id"],
                        "json_id": group_local["json_id"],
                        "device": device,
                        "normalized_device": normalized_device,
                        "error": message,
                        "traceback": traceback.format_exc(limit=80),
                    }
                )
                print(
                    f"[worker {worker_id} {device}->{normalized_device}] error "
                    f"{prefix}: {message}",
                    flush=True,
                )

        result_queue.put(
            {
                "worker_id": worker_id,
                "device": device,
                "normalized_device": normalized_device,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "success_count": success_count,
                "error_count": error_count,
                "summaries": summaries,
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


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> int:
    args = parse_args()
    agent_root = Path(args.agent_root)
    mask_root = Path(args.mask_root)
    output_root = mask_root / args.exp_name

    only_video_ids = set(args.only_video_id or [])
    only_json_ids = set(args.only_json_id or [])
    groups = list_agent_groups(agent_root, only_video_ids or None, only_json_ids or None)
    if args.limit_groups is not None:
        groups = groups[: args.limit_groups]

    devices = resolve_devices(args)
    if args.parallel > 0:
        devices = devices[: args.parallel]
    if not devices:
        raise ValueError("No devices configured")
    num_workers = min(len(devices), max(1, len(groups))) if groups else 0
    devices = devices[:num_workers] if num_workers else devices[:1]

    print(f"planned_groups: {len(groups)}")
    print(f"output_root: {output_root}")
    print(f"parallel: {num_workers if groups else 0}")
    print(f"devices: {devices[:num_workers] if groups else devices}")
    if groups:
        first = groups[0]
        print(
            "first_group: "
            f"video_id={first['video_id']} json_id={first['json_id']} "
            f"num_objects={len(first['pred_paths'])}"
        )

    if args.dry_run or not groups:
        return 0

    task_buckets = chunk_tasks(groups, num_workers)
    args_dict = {
        "dataset_root": str(args.dataset_root),
        "mask_root": str(args.mask_root),
        "exp_name": str(args.exp_name),
        "sam3_official_root": str(args.sam3_official_root),
        "checkpoint": str(args.checkpoint),
        "score_thresh": float(args.score_thresh),
        "overwrite": bool(args.overwrite),
    }

    ctx = get_context("spawn")
    result_queue = ctx.Queue()
    processes = []
    for worker_id, device in enumerate(devices[:num_workers]):
        proc = ctx.Process(
            target=worker_main,
            args=(worker_id, device, task_buckets[worker_id], args_dict, result_queue),
            daemon=False,
        )
        proc.start()
        processes.append(proc)

    results = [result_queue.get() for _ in processes]
    for proc in processes:
        proc.join()

    success_count = 0
    error_count = 0
    fatal_errors: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    worker_results: list[dict[str, Any]] = []

    for result in results:
        worker_results.append(
            {
                key: result.get(key)
                for key in (
                    "worker_id",
                    "device",
                    "normalized_device",
                    "cuda_visible_devices",
                    "success_count",
                    "error_count",
                    "fatal_error",
                )
            }
        )
        if result.get("fatal_error"):
            error_count += 1
            fatal_errors.append(result)
            continue
        success_count += int(result.get("success_count", 0))
        error_count += int(result.get("error_count", 0))
        errors.extend(result.get("errors", []))
        summaries.extend(result.get("summaries", []))

    final_summary = {
        "planned_groups": len(groups),
        "success_count": success_count,
        "error_count": error_count,
        "output_root": str(output_root.resolve()),
        "agent_root": str(agent_root.resolve()),
        "parallel": num_workers,
        "devices": devices[:num_workers],
        "worker_results": worker_results,
        "fatal_errors": fatal_errors,
        "errors": errors,
        "summaries": summaries,
    }
    summary_path = output_root / "_parallel_propagation_summary.json"
    write_summary(summary_path, final_summary)
    if args.error_json:
        write_summary(Path(args.error_json), {"fatal_errors": fatal_errors, "errors": errors})

    print(json.dumps({k: final_summary[k] for k in ["planned_groups", "success_count", "error_count", "output_root", "parallel", "devices"]}, ensure_ascii=False, indent=2))
    print(f"saved_summary: {summary_path}")
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
