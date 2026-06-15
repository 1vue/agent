#!/usr/bin/env python3
"""Run Ark video QA in batch for every event in meta_expressions_text_release.json."""

from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api_config import add_api_profile_args, apply_api_profile_defaults
from run_ark_video_qa import (
    build_semantic_parse_payload,
    build_prompt_text,
    build_result_payload,
    call_api,
    extract_answer,
    normalize_result_records,
    parse_semantic_answer,
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
        # default="../../dataset/ref-youtube/meta_expressions/valid/valid_filtered.json",
        help="Path to meta_expressions_text_release.json",
    )
    parser.add_argument(
        "--dataset-root",
        default="../../dataset/mevis/valid",
        # default="../../dataset/ref-youtube/valid",
        help="Dataset root directory containing either JPEGImages/<video_id>/ or <video_id>/",
    )
    parser.add_argument(
        "--error-dir",
        # default="outputs/ref-youtube/batch_ark_video_qa2/_errors",
        default="outputs/mevis/batch_video_qa/_errors",
        help="Directory containing per-task error JSON files",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/mevis/batch_ark_video_qa",
        help="Directory to store per-video per-event result JSON files",
    )
    parser.add_argument(
        "--response-format",
        choices=("json", "text"),
        default="json",
        help="json: request and save normalized JSON; text: keep raw model text",
    )
    parser.add_argument(
        "--max-sampled-frames",
        type=int,
        default=10,
        help="Maximum number of uniformly sampled frames to send to the VLM",
    )
    add_api_profile_args(parser)
    parser.add_argument("--model", default=None, help="Model name; defaults to selected API profile")
    parser.add_argument("--base-url", default=None, help="API base URL; defaults to selected API profile")
    parser.add_argument("--api-key", default=None, help="API key; defaults to selected profile api_key_env")
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
    args = parser.parse_args()
    return apply_api_profile_defaults(args)


def _sort_key(value: Any) -> tuple[int, Any]:
    try:
        return (0, int(value))
    except (TypeError, ValueError):
        return (1, str(value))


def _safe_id_part(value: Any) -> str:
    text = str(value).strip()
    text = re.sub(r"[^\w\-\.]+", "_", text, flags=re.UNICODE)
    return text.strip("._") or "unknown"


def _iter_expressions(expressions: Any) -> list[tuple[str, dict[str, Any]]]:
    if isinstance(expressions, dict):
        items = [(str(expression_id), expression_info) for expression_id, expression_info in expressions.items()]
        return sorted(items, key=lambda item: _sort_key(item[0]))
    elif isinstance(expressions, list):
        items = []
        seen_ids: dict[str, int] = {}
        base_ids = [str(item.get("exp_id", index)) for index, item in enumerate(expressions)]
        duplicated_base_ids = {base_id for base_id, count in Counter(base_ids).items() if count > 1}
        for index, expression_info in enumerate(expressions):
            if not isinstance(expression_info, dict):
                raise ValueError(f"Expression at index {index} must be an object")
            base_expression_id = str(expression_info.get("exp_id", index))
            obj_id = expression_info.get("obj_id")
            if base_expression_id in duplicated_base_ids and obj_id is not None:
                base_output_id = f"{base_expression_id}_obj_{_safe_id_part(obj_id)}"
            else:
                base_output_id = base_expression_id
            occurrence = seen_ids.get(base_output_id, 0)
            seen_ids[base_output_id] = occurrence + 1
            expression_id = (
                base_output_id
                if occurrence == 0
                else f"{base_output_id}__{index}"
            )
            items.append((str(expression_id), expression_info))
        return items
    else:
        raise ValueError("Expected expressions to be either a dict or a list")


def _extract_expression_text(expression_info: dict[str, Any], expression_id: str) -> str:
    for key in ("exp", "exp_text", "expression", "sentence", "text"):
        value = expression_info.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    raise ValueError(f"Expression {expression_id} is missing exp/exp_text text")


def load_tasks(meta_json_path: Path, only_video_ids: set[str] | None = None) -> list[dict[str, Any]]:
    meta = json.loads(meta_json_path.read_text(encoding="utf-8"))
    tasks: list[dict[str, Any]] = []
    for video_id, video_info in meta["videos"].items():
        if only_video_ids and video_id not in only_video_ids:
            continue
        expressions = video_info["expressions"]
        for expression_id, expression_info in _iter_expressions(expressions):
            tasks.append(
                {
                    "video_id": video_id,
                    "expression_id": str(expression_id),
                    "raw_exp_id": expression_info.get("exp_id", expression_id),
                    "target_event": _extract_expression_text(expression_info, expression_id),
                    "vid_id": video_info.get("vid_id"),
                    "source": video_info.get("source"),
                    "obj_id": expression_info.get("obj_id"),
                    "is_sent": expression_info.get("is_sent"),
                    "frame_ids": video_info.get("frames"),
                }
            )
    return tasks


def load_error_tasks(error_dir: Path, only_video_ids: set[str] | None = None) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    if not error_dir.exists():
        return tasks
    for error_path in sorted(error_dir.rglob("*.json")):
        error_info = json.loads(error_path.read_text())
        video_id = error_info["video_id"]
        if only_video_ids and video_id not in only_video_ids:
            continue
        tasks.append(
            {
                "video_id": video_id,
                "expression_id": str(error_info["expression_id"]),
                "raw_exp_id": error_info.get("raw_exp_id"),
                "target_event": error_info["target_event"],
                "vid_id": error_info.get("vid_id"),
                "source": error_info.get("source"),
                "obj_id": error_info.get("obj_id"),
                "is_sent": error_info.get("is_sent"),
                "frame_ids": error_info.get("frame_ids"),
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

    semantic_parse: dict[str, Any] | None = None
    raw_semantic_answer: str | None = None
    answer: str | None = None
    response: Any | None = None
    try:
        semantic_response = call_api(
            build_semantic_parse_payload(target_event=target_event, model=args.model),
            args.base_url,
            args.api_key,
        )
        raw_semantic_answer = extract_answer(semantic_response)
        semantic_parse = parse_semantic_answer(raw_semantic_answer, target_event)

        prompt_text = build_prompt_text(
            target_event=target_event,
            semantic_parse=semantic_parse,
            response_format=args.response_format,
        )
        bundle = prepare_request(
            video_id=video_id,
            dataset_root=args.dataset_root,
            prompt_text=prompt_text,
            model=args.model,
            frame_ids=task.get("frame_ids"),
            max_sampled_frames=args.max_sampled_frames,
        )
        response = call_api(bundle.payload, args.base_url, args.api_key)
        answer = extract_answer(response)

        if args.response_format == "json":
            results = normalize_result_records(answer, bundle.selected_frames, semantic_parse)
            result_payload = build_result_payload(
                video_id=video_id,
                prompt_label="target_event",
                prompt_value=target_event,
                bundle=bundle,
                semantic_parse=semantic_parse,
                results=results,
                raw_semantic_response_text=raw_semantic_answer,
                raw_response_text=answer,
            )
        else:
            result_payload = build_result_payload(
                video_id=video_id,
                prompt_label="target_event",
                prompt_value=target_event,
                bundle=bundle,
                semantic_parse=semantic_parse,
                raw_semantic_response_text=raw_semantic_answer,
                raw_response_text=answer,
            )

        result_payload["expression_id"] = expression_id
        result_payload["raw_exp_id"] = task.get("raw_exp_id")
        result_payload["vid_id"] = task["vid_id"]
        result_payload["source"] = task.get("source")
        result_payload["obj_id"] = task.get("obj_id")
        result_payload["is_sent"] = task.get("is_sent")
        save_json(result_payload, result_path)
        if error_path.exists():
            error_path.unlink()
        return ("success", video_id, expression_id, str(result_path))
    except Exception as exc:
        cause = exc.__cause__ or exc.__context__
        error_payload = {
            "video_id": video_id,
            "expression_id": expression_id,
            "raw_exp_id": task.get("raw_exp_id"),
            "target_event": target_event,
            "vid_id": task.get("vid_id"),
            "source": task.get("source"),
            "obj_id": task.get("obj_id"),
            "is_sent": task.get("is_sent"),
            "frame_ids": task.get("frame_ids"),
            "error": type(exc).__name__,
            "message": str(exc),
            "repr": repr(exc),
            "cause_error": type(cause).__name__ if cause else None,
            "cause_message": str(cause) if cause else None,
            "cause_repr": repr(cause) if cause else None,
            "traceback": traceback.format_exc(),
            "base_url": args.base_url,
            "model": args.model,
        }
        if answer is not None:
            error_payload["raw_response_text"] = answer
        if raw_semantic_answer is not None:
            error_payload["raw_semantic_response_text"] = raw_semantic_answer
        if semantic_parse is not None:
            error_payload["semantic_parse"] = semantic_parse
        if response is not None:
            try:
                error_payload["raw_api_response"] = response.model_dump(mode="json")
            except Exception:
                error_payload["raw_api_response_repr"] = repr(response)
            try:
                choice = response.choices[0]
                message = choice.message
                error_payload["response_debug"] = {
                    "finish_reason": getattr(choice, "finish_reason", None),
                    "message_content": getattr(message, "content", None),
                    "message_refusal": getattr(message, "refusal", None),
                    "message_tool_calls": getattr(message, "tool_calls", None),
                }
            except Exception:
                pass
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
