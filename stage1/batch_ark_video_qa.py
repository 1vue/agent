#!/usr/bin/env python3
"""Run Ark video QA in batch for every event in meta_expressions_text_release.json."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_ark_video_qa import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    build_prompt_text,
    build_result_payload,
    call_api,
    extract_answer,
    normalize_result_records,
    prepare_request,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one inference per video/event pair from meta_expressions_text_release.json "
            "and save one JSON result file per pair."
        )
    )
    parser.add_argument(
        "--meta-json",
        default="../../dataset/mevis/valid/meta_expressions.json",
        help="Path to meta_expressions_text_release.json",
    )
    parser.add_argument(
        "--dataset-root",
        default="../../dataset/mevis/valid",
        help="Dataset root directory containing JPEGImages/",
    )
    parser.add_argument(
        "--error-dir",
        default="outputs/batch_ark_video_qa/_errors",
        help="Directory containing per-task error JSON files",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/batch_ark_video_qa",
        help="Directory to store per-video per-event result JSON files",
    )
    parser.add_argument(
        "--response-format",
        choices=("json", "text"),
        default="json",
        help="json: request and save normalized JSON; text: keep raw model text",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ark model name")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Ark API base URL")
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of concurrent API requests to run",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional cap on how many video/event pairs to run",
    )
    parser.add_argument(
        "--only-video-id",
        action="append",
        help="Optional filter; may be passed multiple times",
    )
    parser.add_argument(
        "--retry-errors-only",
        action="store_true",
        help="Only load and rerun tasks listed in the error directory",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run pairs even if their output JSON already exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print planned work without calling the API",
    )
    return parser.parse_args()


def load_tasks(meta_json_path: Path, only_video_ids: set[str] | None = None) -> list[dict[str, Any]]:
    meta = json.loads(meta_json_path.read_text())
    tasks: list[dict[str, Any]] = []
    for video_id, video_info in meta["videos"].items():
        if only_video_ids and video_id not in only_video_ids:
            continue
        expressions = video_info["expressions"]
        for expression_id, expression_info in sorted(expressions.items(), key=lambda item: int(item[0])):
            tasks.append(
                {
                    "video_id": video_id,
                    "expression_id": str(expression_id),
                    "target_event": expression_info["exp"],
                    "vid_id": video_info.get("vid_id"),
                }
            )
    return tasks


def load_error_tasks(error_dir: Path, only_video_ids: set[str] | None = None) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    if not error_dir.exists():
        return tasks
    for error_path in sorted(error_dir.glob("*.json")):
        error_info = json.loads(error_path.read_text())
        video_id = error_info["video_id"]
        if only_video_ids and video_id not in only_video_ids:
            continue
        tasks.append(
            {
                "video_id": video_id,
                "expression_id": str(error_info["expression_id"]),
                "target_event": error_info["target_event"],
                "vid_id": error_info.get("vid_id"),
                "error_path": error_path,
            }
        )
    return tasks


def output_path_for_task(output_dir: Path, video_id: str, expression_id: str) -> Path:
    return output_dir / video_id / f"{expression_id}.json"


def error_path_for_task(output_dir: Path, video_id: str, expression_id: str) -> Path:
    return output_dir / "_errors" / f"{video_id}__{expression_id}.json"


def build_summary(args: argparse.Namespace, total_tasks: int) -> dict[str, Any]:
    return {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "meta_json": str(Path(args.meta_json).resolve()),
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "output_dir": str(Path(args.output_dir).resolve()),
        "model": args.model,
        "response_format": args.response_format,
        "parallel": args.parallel,
        "total_tasks": total_tasks,
        "dry_run": args.dry_run,
    }


def process_task(
    task: dict[str, Any],
    *,
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[str, str, str, str]:
    video_id = task["video_id"]
    expression_id = task["expression_id"]
    target_event = task["target_event"]
    result_path = output_path_for_task(output_dir, video_id, expression_id)
    error_path = Path(task.get("error_path") or error_path_for_task(output_dir, video_id, expression_id))
    if result_path.exists() and not args.overwrite:
        if error_path.exists():
            error_path.unlink()
        return ("skip", video_id, expression_id, str(result_path))

    prompt_text = build_prompt_text(
        target_event=target_event,
        response_format=args.response_format,
    )
    bundle = prepare_request(
        video_id=video_id,
        dataset_root=args.dataset_root,
        prompt_text=prompt_text,
        model=args.model,
    )
    try:
        response = call_api(bundle.payload, args.base_url)
        answer = extract_answer(response)

        if args.response_format == "json":
            results = normalize_result_records(answer, bundle.selected_frames)
            result_payload = build_result_payload(
                video_id=video_id,
                prompt_label="target_event",
                prompt_value=target_event,
                bundle=bundle,
                results=results,
                raw_response_text=answer,
            )
        else:
            result_payload = build_result_payload(
                video_id=video_id,
                prompt_label="target_event",
                prompt_value=target_event,
                bundle=bundle,
                raw_response_text=answer,
            )

        result_payload["expression_id"] = expression_id
        result_payload["vid_id"] = task["vid_id"]
        save_json(result_payload, result_path)
        if error_path.exists():
            error_path.unlink()
        return ("success", video_id, expression_id, str(result_path))
    except Exception as exc:
        error_payload = {
            "video_id": video_id,
            "expression_id": expression_id,
            "target_event": target_event,
            "vid_id": task.get("vid_id"),
            "error": type(exc).__name__,
            "message": str(exc),
        }
        save_json(error_payload, error_path)
        return ("error", video_id, expression_id, str(exc))


def main() -> int:
    args = parse_args()
    meta_json_path = Path(args.meta_json)
    output_dir = Path(args.output_dir)
    error_dir = Path(args.error_dir)
    only_video_ids = set(args.only_video_id or [])
    if args.retry_errors_only:
        tasks = load_error_tasks(error_dir, only_video_ids or None)
    else:
        tasks = load_tasks(meta_json_path, only_video_ids or None)
    if args.limit is not None:
        tasks = tasks[: args.limit]

    summary = build_summary(args, len(tasks))
    print(f"planned_tasks: {len(tasks)}")
    print(f"output_dir: {output_dir}")
    print(f"parallel: {args.parallel}")
    if tasks:
        print(
            "first_task: "
            f"video_id={tasks[0]['video_id']} expression_id={tasks[0]['expression_id']} "
            f"target_event={tasks[0]['target_event']}"
        )

    if args.dry_run:
        summary["finished_at"] = datetime.now(timezone.utc).isoformat()
        save_json(summary, output_dir / "run_summary.json")
        print(f"saved dry-run summary to {output_dir / 'run_summary.json'}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    success_count = 0
    skip_count = 0
    error_count = 0
    max_workers = max(1, args.parallel)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_meta = {
            executor.submit(process_task, task, args=args, output_dir=output_dir): (
                index,
                task["video_id"],
                task["expression_id"],
                task["target_event"],
            )
            for index, task in enumerate(tasks, start=1)
        }
        for future in as_completed(future_to_meta):
            index, video_id, expression_id, target_event = future_to_meta[future]
            try:
                status, _, _, message = future.result()
            except Exception as exc:  # Defensive guard around worker failures.
                status = "error"
                message = str(exc)
            if status == "skip":
                skip_count += 1
                print(f"[{index}/{len(tasks)}] skip {video_id}/{expression_id} -> {message}")
            elif status == "success":
                success_count += 1
                print(f"[{index}/{len(tasks)}] saved {message}")
            else:
                error_count += 1
                print(f"[{index}/{len(tasks)}] error {video_id}/{expression_id}: {message}")

    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    summary["success_count"] = success_count
    summary["skip_count"] = skip_count
    summary["error_count"] = error_count
    save_json(summary, output_dir / "run_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
