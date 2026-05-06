#!/usr/bin/env python3
"""Batch-run the official SAM3 agent on every JSON under outputs/batch_ark_video_qa."""

from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import traceback
from functools import partial
from multiprocessing import get_context
from pathlib import Path
from typing import Any

from run_sam3_agent_from_batch_result import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    build_processor,
    import_agent_modules,
    load_result_json,
    safe_name,
    select_result_items,
    save_json
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan outputs/batch_ark_video_qa/<video>/<json>.json and batch-run "
            "the official SAM3 agent with a configurable OpenAI-compatible VLM backend."
        )
    )
    parser.add_argument(
        "--batch-root",
        default="outputs/batch_ark_video_qa",
        help="Directory containing per-video per-json result files",
    )
    parser.add_argument(
        "--dataset-root",
        default="../../dataset/mevis/valid",
        help="Dataset root directory containing JPEGImages/<video>/",
    )
    parser.add_argument(
        "--output-root",
        default="agent_output",
        help="Directory to save agent outputs",
    )
    parser.add_argument(
        "--sam3-official-root",
        default="sam3",
        help="Path to the official SAM3 codebase",
    )
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("SAM3_CHECKPOINT", "checkpoints/sam3.pt"),
        help="Local SAM3 checkpoint path",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("SAM3_AGENT_BASE_URL", DEFAULT_BASE_URL),
        help="OpenAI-compatible LLM base URL",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("SAM3_AGENT_MODEL", DEFAULT_MODEL),
        help="LLM model name",
    )
    parser.add_argument(
        "--api-key",
        # default=os.environ.get("XIAOMI_API_KEY").strip(),
        default=os.environ.get("VOLCES_API_KEY").strip(),
        # default=os.environ.get("XUNFEI_API_KEY").strip(),
        help="LLM API key; can also come from SAM3_AGENT_API_KEY",
    )
    parser.add_argument(
        "--extra-body-json",
        default=os.environ.get("SAM3_AGENT_EXTRA_BODY_JSON"),
        help="Optional JSON string passed as extra_body to chat.completions.create",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for the SAM3 image processor",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=3,
        help="Number of worker processes to run in parallel",
    )
    parser.add_argument(
        "--devices",
        help=(
            "Comma-separated devices for workers, e.g. cuda:0,cuda:1,cuda:2. "
            "Defaults to cuda:0..cuda:<parallel-1>."
        ),
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
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional cap on number of tasks after filtering",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable SAM3 agent debug outputs",
    )
    parser.add_argument(
        "--prompt-source",
        choices=("discriminative_description", "object_name", "target_event"),
        default="discriminative_description",
        help="Which text source to use as the SAM3 agent prompt",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing pred.json outputs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print planned work without running",
    )
    parser.add_argument(
        "--error-dir",
        default="agent_output/sam3_agent/_errors",
        help="Directory to store task error logs",
    )
    parser.add_argument(
        "--retry-errors-only",
        action="store_true",
        help="If set, only tasks found in the error-dir will be processed",
    )
    return parser.parse_args()

# 保存错误任务
def load_error_tasks(
        error_dir: Path,
        only_video_ids: set[str] | None = None,
        only_json_ids: set[str] | None = None
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    if not error_dir.exists():
        return tasks

    for error_path in sorted(error_dir.glob("*/*.json")):
        try:
            error_info = json.loads(error_path.read_text())
            video_id = error_info["video_id"]
            json_id = str(error_info["json_id"])

            # 过滤 video_id
            if only_video_ids and video_id not in only_video_ids:
                continue
            # 过滤 json_id
            if only_json_ids and json_id not in only_json_ids:
                continue

            tasks.append({
                "video_id": video_id,
                "json_id": json_id,
                "result_json_path": error_info["result_json_path"],
                "error_path": error_path
            })
        except Exception as e:
            print(f"Warning: Failed to load error task {error_path}: {e}")

    return tasks

def list_tasks(
    batch_root: Path,
    only_video_ids: set[str] | None,
    only_json_ids: set[str] | None,
) -> list[dict[str, str]]:
    tasks: list[dict[str, str]] = []
    for json_path in sorted(batch_root.glob("*/*.json")):
        video_id = json_path.parent.name
        json_id = json_path.stem
        if video_id == "_errors":
            continue
        if only_video_ids and video_id not in only_video_ids:
            continue
        if only_json_ids and json_id not in only_json_ids:
            continue
        tasks.append(
            {
                "video_id": video_id,
                "json_id": json_id,
                "result_json_path": str(json_path),
            }
        )
    return tasks


def chunk_tasks(tasks: list[dict[str, str]], num_chunks: int) -> list[list[dict[str, str]]]:
    if num_chunks <= 0:
        raise ValueError("num_chunks must be positive")
    buckets = [[] for _ in range(num_chunks)]
    for idx, task in enumerate(tasks):
        buckets[idx % num_chunks].append(task)
    return buckets


def run_one_task(
    task: dict[str, str],
    *,
    dataset_root: Path,
    output_root: Path,
    send_generate_request,
    call_sam_service,
    selected_indices: set[int] | None = None,
    prompt_source: str = "discriminative_description",
    debug: bool = False,
    overwrite: bool = False,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    video_id = task["video_id"]
    json_id = task["json_id"]
    result_json = load_result_json(Path(task["result_json_path"]).parent.parent, video_id, json_id)
    selected_items = select_result_items(result_json, selected_indices)
    run_root = output_root / "sam3_agent" / safe_name(send_generate_request.keywords["model"]) / video_id / str(json_id)
    run_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "video_id": video_id,
        "json_id": str(json_id),
        "target_event": result_json.get("target_event"),
        "llm_base_url": send_generate_request.keywords["server_url"],
        "llm_model": send_generate_request.keywords["model"],
        "llm_extra_body": extra_body,
        "prompt_source": prompt_source,
        "items": [],
    }

    if not selected_items:
        summary["status"] = "empty_results"
        summary_path = run_root / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        return {
            "video_id": video_id,
            "json_id": json_id,
            "summary_path": str(summary_path),
            "num_items": 0,
            "status": "empty_results",
        }

    from sam3.agent.agent_core import agent_inference

    for obj_idx, item in selected_items:
        object_name = item.get("object_name") or f"object_{obj_idx}"
        if prompt_source == "object_name":
            prompt = item.get("object_name")
        elif prompt_source == "target_event":
            prompt = result_json.get("target_event")
        else:
            prompt = item.get("discriminative_description")
        if not prompt:
            raise ValueError(f"Missing prompt text for object index {obj_idx} in {task['result_json_path']}")

        frame_filename = item.get("actual_frame_filename")
        if not frame_filename:
            raise ValueError(f"Missing actual_frame_filename for object index {obj_idx}")

        image_path = dataset_root / "JPEGImages" / video_id / frame_filename
        if not image_path.exists():
            raise FileNotFoundError(f"Image frame not found: {image_path}")

        item_dir = run_root / f"{safe_name(object_name)}_obj_{obj_idx}"
        pred_json_path = item_dir / "pred.json"
        if pred_json_path.exists() and not overwrite:
            summary["items"].append(
                {
                    "object_index": obj_idx,
                    "object_name": object_name,
                    "prompt": prompt,
                    "image_path": str(image_path),
                    "output_dir": str(item_dir),
                    "status": "skipped_existing",
                }
            )
            continue

        item_dir.mkdir(parents=True, exist_ok=True)
        agent_history, final_output_dict, rendered_final_output = agent_inference(
            str(image_path),
            prompt,
            send_generate_request=send_generate_request,
            call_sam_service=call_sam_service,
            output_dir=str(item_dir),
            max_generations=10,   #最大对话轮数
            debug=debug,
        )

        final_output_dict = {
            **final_output_dict,
            "video_id": video_id,
            "json_id": str(json_id),
            "object_index": obj_idx,
            "object_name": object_name,
            "input_prompt": prompt,
            "source_result_item": item,
        }
        pred_json_path.write_text(json.dumps(final_output_dict, indent=2, ensure_ascii=False))
        (item_dir / "history.json").write_text(json.dumps(agent_history, indent=2, ensure_ascii=False))
        rendered_final_output.save(item_dir / "pred.png")
        (item_dir / "input_meta.json").write_text(
            json.dumps(
                {
                    "video_id": video_id,
                    "json_id": str(json_id),
                    "object_index": obj_idx,
                    "object_name": object_name,
                    "image_path": str(image_path),
                    "frame_filename": frame_filename,
                    "actual_frame_index": item.get("actual_frame_index"),
                    "prompt": prompt,
                    "target_event": result_json.get("target_event"),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        summary["items"].append(
            {
                "object_index": obj_idx,
                "object_name": object_name,
                "prompt": prompt,
                "image_path": str(image_path),
                "output_dir": str(item_dir),
                "num_masks": len(final_output_dict.get("pred_masks", [])),
                "status": "completed",
            }
        )

    summary_path = run_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return {
        "video_id": video_id,
        "json_id": json_id,
        "summary_path": str(summary_path),
        "num_items": len(summary["items"]),
    }


def worker_main(
    worker_id: int,
    device: str,
    tasks: list[dict[str, str]],
    args: argparse.Namespace,
    progress_queue: queue.Queue,
) -> None:
    try:
        process_device = device
        if device.startswith("cuda:"):
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[1]
            process_device = "cuda"
        elif device == "cuda":
            os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

        dataset_root = Path(args.dataset_root)
        output_root = Path(args.output_root)
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
            device=process_device,
            confidence_threshold=args.confidence_threshold,
        )
        send_generate_request = partial(
            send_generate_request_orig,
            server_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
            # extra_body=extra_body,
        )
        call_sam_service = partial(call_sam_service_orig, sam3_processor=processor)

        for task in tasks:
            video_id = task["video_id"]
            json_id = task["json_id"]
            error_dir_root = Path(args.error_dir)
            error_path = error_dir_root / video_id / f"{json_id}.json"
            progress_queue.put(
                {
                    "type": "start",
                    "worker_id": worker_id,
                    "device": device,
                    "video_id": task["video_id"],
                    "json_id": task["json_id"],
                }
            )
            try:
                result = run_one_task(
                    task,
                    dataset_root=dataset_root,
                    output_root=output_root,
                    send_generate_request=send_generate_request,
                    call_sam_service=call_sam_service,
                    prompt_source=args.prompt_source,
                    debug=args.debug,
                    overwrite=args.overwrite,
                    extra_body=extra_body,
                )
                if error_path.exists():
                    error_path.unlink()
                progress_queue.put(
                    {
                        "type": "success",
                        "worker_id": worker_id,
                        "device": device,
                        **result,
                    }
                )
            except Exception as exc:
                error_payload = {
                    "video_id": video_id,
                    "json_id": json_id,
                    "result_json_path": task["result_json_path"],
                    "error": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),  # 建议把堆栈也存了，方便查错
                }
                error_path.parent.mkdir(parents=True, exist_ok=True)
                save_json(error_payload, error_path)
                progress_queue.put(
                    {
                        "type": "error",
                        "worker_id": worker_id,
                        "device": device,
                        "video_id": task["video_id"],
                        "json_id": task["json_id"],
                        "error": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
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
                "traceback": traceback.format_exc(),
            }
        )


def main() -> int:
    args = parse_args()
    if not args.api_key and not args.dry_run:
        raise ValueError("Missing API key; pass --api-key or set SAM3_AGENT_API_KEY")

    batch_root = Path(args.batch_root)
    only_video_ids = set(args.only_video_id or [])
    only_json_ids = set(args.only_json_id or [])
    error_dir = Path(args.error_dir)
    if args.retry_errors_only:
        # 只跑失败的
        tasks = load_error_tasks(error_dir, only_video_ids or None, only_json_ids or None)
    else:
        # 跑全量（list_tasks 内部会自动根据 output 检查是否跳过已完成的）
        tasks = list_tasks(batch_root, only_video_ids or None, only_json_ids or None)
    if args.limit is not None:
        tasks = tasks[: args.limit]

    print(f"planned_tasks: {len(tasks)}")
    print(f"batch_root: {batch_root}")
    print(f"output_root: {Path(args.output_root)}")
    print(f"model: {args.model}")
    print(f"parallel: {args.parallel}")

    if tasks:
        print(f"first_task: {tasks[0]['video_id']}/{tasks[0]['json_id']}")

    if args.dry_run:
        return 0
    if not tasks:
        return 0

    if args.devices:
        devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    else:
        devices = [f"cuda:{idx}" for idx in range(args.parallel)]
    if not devices:
        raise ValueError("No devices configured")

    num_workers = min(args.parallel, len(devices), len(tasks))
    devices = devices[:num_workers]
    task_buckets = chunk_tasks(tasks, num_workers)

    mp = get_context("spawn")
    progress_queue = mp.Queue()
    workers = []
    for worker_id in range(num_workers):
        proc = mp.Process(
            target=worker_main,
            args=(worker_id, devices[worker_id], task_buckets[worker_id], args, progress_queue),
            daemon=False,
        )
        proc.start()
        workers.append(proc)

    completed = 0
    success_count = 0
    error_count = 0
    total_tasks = len(tasks)
    live_workers = len(workers)
    while completed < total_tasks and live_workers > 0:
        try:
            msg = progress_queue.get(timeout=1.0)
        except queue.Empty:
            live_workers = sum(proc.is_alive() for proc in workers)
            continue

        msg_type = msg["type"]
        if msg_type == "start":
            print(
                f"[worker {msg['worker_id']} {msg['device']}] start "
                f"{msg['video_id']}/{msg['json_id']}"
            )
        elif msg_type == "success":
            completed += 1
            success_count += 1
            print(
                f"[{completed}/{total_tasks}] success "
                f"{msg['video_id']}/{msg['json_id']} -> {msg['summary_path']}"
            )
        elif msg_type == "error":
            completed += 1
            error_count += 1
            print(
                f"[{completed}/{total_tasks}] error "
                f"{msg['video_id']}/{msg['json_id']}: {msg['error']}: {msg['message']}"
            )
            print(msg["traceback"], file=sys.stderr)
        elif msg_type == "worker_error":
            error_count += 1
            print(
                f"[worker {msg['worker_id']} {msg['device']}] fatal error: "
                f"{msg['error']}: {msg['message']}"
            )
            print(msg["traceback"], file=sys.stderr)
            live_workers = sum(proc.is_alive() for proc in workers)

        live_workers = sum(proc.is_alive() for proc in workers)

    for proc in workers:
        proc.join()

    summary = {
        "planned_tasks": total_tasks,
        "success_count": success_count,
        "error_count": error_count,
        "parallel": num_workers,
        "devices": devices,
        "batch_root": str(batch_root.resolve()),
        "output_root": str(Path(args.output_root).resolve()),
        "model": args.model,
        "base_url": args.base_url,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
