#!/usr/bin/env python3
"""Rerun video QA for badcases using the original QA generation path."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api_config import add_api_profile_args, apply_api_profile_defaults
from stage1.run_ark_video_qa import (
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


NON_EMPTY_RERUN_INSTRUCTION = """## Badcase Rerun Override

This is a badcase rerun. The downstream segmentation pipeline needs at least one candidate target.

Do NOT output an empty array `[]`.
If the target is ambiguous, partially visible, small, blurry, or only weakly supported by the sampled frames, still output the most likely matching physical target instance.
Choose the sampled frame where that most likely target is clearest among the provided frames.
Use the `discriminative_description`, `reference_entities`, and `excluded_entities` fields to make the uncertainty and instance distinction precise enough for downstream segmentation.
Never output `[]` in this rerun path."""


NON_EMPTY_RETRY_INSTRUCTION = """## Final Non-empty Retry

Your previous answer produced no target candidates. For this rerun, you must provide at least one most likely target candidate from the sampled frames.
Return a valid JSON array with one or more objects following the required schema.
Pick the best available sampled frame even if the evidence is uncertain or imperfect."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rerun QA with a refinement prompt for groups listed in a badcase scan JSON.",
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--scan-json", help="Path to badcase scan JSON")
    input_group.add_argument(
        "--manifest-json",
        help=(
            "Path to a route_badcase_repairs.py manifest JSON. "
            "Use repair_manifests/reselect.json to rerun only routed reselect tasks."
        ),
    )
    parser.add_argument(
        "--output-root",
        default="outputs/batch_ark_video_qa_refine1",
        help="Where to save refined QA outputs",
    )
    parser.add_argument(
        "--dataset-root",
        default="../../dataset/mevis/valid",
        help="Dataset root containing JPEGImages/",
    )
    add_api_profile_args(parser)
    parser.add_argument("--base-url", default=None, help="API base URL; defaults to selected API profile")
    parser.add_argument("--model", default=None, help="Model name; defaults to selected API profile")
    parser.add_argument("--api-key", default=None, help="API key; defaults to selected profile api_key_env")
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
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing refined QA files")
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of concurrent API rerun tasks.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned rerun tasks without calling the API or writing outputs.",
    )
    parser.add_argument(
        "--max-sampled-frames",
        type=int,
        default=10,
        help="Maximum number of uniformly sampled frames to send to the VLM",
    )
    parser.add_argument(
        "--include-terminal-empty",
        action="store_true",
        help="Also rerun groups whose initial QA results were an empty result list",
    )
    parser.add_argument(
        "--allow-empty-results",
        action="store_true",
        help="Allow the rerun VLM to return an empty results list. By default reruns force a most-likely candidate.",
    )
    args = parser.parse_args()
    return apply_api_profile_defaults(args)


def collect_groups(scan: dict[str, Any], include_terminal_empty: bool) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for group in scan.get("group_badcases", []):
        item = dict(group)
        item.setdefault("rerun_source", "group_badcase")
        groups.append(item)
    if include_terminal_empty:
        for group in scan.get("terminal_empty_groups", []):
            item = dict(group)
            item.setdefault("reasons", ["terminal_empty"])
            item.setdefault("rerun_source", "terminal_empty")
            groups.append(item)
    return groups


def collect_manifest_groups(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for task in manifest.get("tasks", []):
        item = dict(task)
        item.setdefault("reasons", [item.get("failure_type") or manifest.get("manifest_type")])
        item.setdefault("rerun_source", "repair_manifest")
        item.setdefault("manifest_type", manifest.get("manifest_type"))
        item.setdefault("repair_route", manifest.get("repair_route"))
        groups.append(item)
    return groups


def select_groups(
    groups: list[dict[str, Any]],
    only_video_ids: set[str],
    only_json_ids: set[str],
) -> list[dict[str, Any]]:
    selected = []
    for group in groups:
        video_id = group["video_id"]
        json_id = str(group["json_id"])
        if only_video_ids and video_id not in only_video_ids:
            continue
        if only_json_ids and json_id not in only_json_ids:
            continue
        selected.append(group)
    return selected


def build_rerun_prompt_text(
    *,
    target_event: str,
    semantic_parse: dict[str, Any],
    allow_empty_results: bool,
    retry: bool = False,
) -> str:
    prompt_text = build_prompt_text(
        target_event=target_event,
        semantic_parse=semantic_parse,
        response_format="json",
    )
    if allow_empty_results:
        return prompt_text
    suffix = NON_EMPTY_RETRY_INSTRUCTION if retry else NON_EMPTY_RERUN_INSTRUCTION
    return f"{prompt_text}\n\n{suffix}"


def build_most_likely_fallback_record(
    selected_frames: list[tuple[int, Path]],
    semantic_parse: dict[str, Any],
    target_event: str,
) -> dict[str, Any]:
    best_frame = len(selected_frames) // 2
    actual_frame_index, actual_frame_path = selected_frames[best_frame]
    target_entity = str(semantic_parse.get("target_entity") or "target object").strip() or "target object"
    segment_target = str(semantic_parse.get("segment_target") or target_event or target_entity).strip()
    object_name = f"most_likely_{target_entity}".replace(" ", "_")
    return {
        "object_name": object_name,
        "best_frame": best_frame,
        "actual_frame_filename": actual_frame_path.name,
        "actual_frame_index": actual_frame_index,
        "target_entity": target_entity,
        "target_role": semantic_parse.get("target_role") or "unknown",
        "action": semantic_parse.get("action") or "unknown",
        "segment_target": segment_target,
        "target_cardinality": semantic_parse.get("target_cardinality"),
        "target_set_description": semantic_parse.get("target_set_description") or "",
        "requires_instance_enumeration": semantic_parse.get("requires_instance_enumeration"),
        "reference_entities": list(semantic_parse.get("reference_entities") or []),
        "excluded_entities": list(semantic_parse.get("excluded_entities") or []),
        "discriminative_description": (
            f"Most likely visible {target_entity} matching the referring expression: {target_event}"
        ),
        "fallback_reason": "vlm_returned_empty_after_non_empty_retry",
    }


def process_group(group: dict[str, Any], args: argparse.Namespace, output_root: Path) -> dict[str, Any]:
    video_id = group["video_id"]
    json_id = str(group["json_id"])
    source_path = Path(group["batch_json_path"])
    output_path = output_root / video_id / f"{json_id}.json"

    result_base = {
        "video_id": video_id,
        "json_id": json_id,
        "source_path": str(source_path),
        "output_path": str(output_path),
    }

    if output_path.exists() and not args.overwrite:
        return {**result_base, "status": "skip"}

    if args.dry_run:
        return {**result_base, "status": "dry_run"}

    try:
        source = json.loads(source_path.read_text())
        target_event = source.get("target_event")
        semantic_response = call_api(
            build_semantic_parse_payload(target_event=target_event, model=args.model),
            args.base_url,
            args.api_key,
        )
        raw_semantic_answer = extract_answer(semantic_response)
        semantic_parse = parse_semantic_answer(raw_semantic_answer, target_event)

        prompt_text = build_rerun_prompt_text(
            target_event=target_event,
            semantic_parse=semantic_parse,
            allow_empty_results=args.allow_empty_results,
        )
        bundle = prepare_request(
            video_id=video_id,
            dataset_root=args.dataset_root,
            prompt_text=prompt_text,
            model=args.model,
            frame_ids=source.get("frame_ids"),
            max_sampled_frames=args.max_sampled_frames,
        )
        response = call_api(bundle.payload, args.base_url, args.api_key)
        answer = extract_answer(response)
        results = normalize_result_records(answer, bundle.selected_frames, semantic_parse)
        retry_answer = None
        fallback_used = False
        if not results and not args.allow_empty_results:
            retry_prompt_text = build_rerun_prompt_text(
                target_event=target_event,
                semantic_parse=semantic_parse,
                allow_empty_results=False,
                retry=True,
            )
            retry_bundle = prepare_request(
                video_id=video_id,
                dataset_root=args.dataset_root,
                prompt_text=retry_prompt_text,
                model=args.model,
                frame_ids=source.get("frame_ids"),
                max_sampled_frames=args.max_sampled_frames,
            )
            retry_response = call_api(retry_bundle.payload, args.base_url, args.api_key)
            retry_answer = extract_answer(retry_response)
            retry_results = normalize_result_records(retry_answer, retry_bundle.selected_frames, semantic_parse)
            if retry_results:
                bundle = retry_bundle
                answer = retry_answer
                results = retry_results
            else:
                bundle = retry_bundle
                answer = retry_answer
                results = [
                    build_most_likely_fallback_record(
                        bundle.selected_frames,
                        semantic_parse,
                        target_event,
                    )
                ]
                fallback_used = True
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
        result_payload["expression_id"] = json_id
        result_payload["raw_exp_id"] = source.get("raw_exp_id")
        result_payload["vid_id"] = source.get("vid_id")
        result_payload["source"] = source.get("source")
        result_payload["obj_id"] = source.get("obj_id")
        result_payload["is_sent"] = source.get("is_sent")
        result_payload["frame_ids"] = source.get("frame_ids")
        result_payload["refined_from"] = str(source_path)
        result_payload["rerun_reason_reasons"] = group.get("reasons", [])
        result_payload["rerun_source"] = group.get("rerun_source")
        result_payload["repair_manifest_type"] = group.get("manifest_type")
        result_payload["repair_failure_type"] = group.get("failure_type")
        result_payload["repair_route"] = group.get("repair_route")
        result_payload["force_non_empty_results"] = not args.allow_empty_results
        result_payload["non_empty_fallback_used"] = fallback_used
        if retry_answer is not None:
            result_payload["non_empty_retry_raw_response_text"] = retry_answer
        save_json(result_payload, output_path)
        return {**result_base, "status": "success"}
    except Exception as exc:
        return {
            **result_base,
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def main() -> int:
    args = parse_args()
    input_path = Path(args.manifest_json or args.scan_json)
    source = json.loads(input_path.read_text())
    only_video_ids = set(args.only_video_id or [])
    only_json_ids = set(args.only_json_id or [])
    output_root = Path(args.output_root)

    if args.manifest_json:
        groups = collect_manifest_groups(source)
    else:
        groups = collect_groups(source, args.include_terminal_empty)
    selected_groups = select_groups(groups, only_video_ids, only_json_ids)
    success_count = 0
    skip_count = 0
    dry_run_count = 0
    error_count = 0

    worker_count = max(1, min(args.parallel, len(selected_groups) or 1))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(process_group, group, args, output_root) for group in selected_groups]
        for future in as_completed(futures):
            result = future.result()
            video_id = result["video_id"]
            json_id = result["json_id"]
            output_path = result["output_path"]
            status = result["status"]
            if status == "success":
                success_count += 1
                print(f"saved {video_id}/{json_id} -> {output_path}")
            elif status == "skip":
                skip_count += 1
                print(f"skip {video_id}/{json_id} -> {output_path}")
            elif status == "dry_run":
                dry_run_count += 1
                print(f"dry-run {video_id}/{json_id} -> {output_path}")
            else:
                error_count += 1
                print(
                    f"error {video_id}/{json_id}: "
                    f"{result.get('error_type')}: {result.get('error')}"
                )

    print(
        json.dumps(
            {
                "groups": len(groups),
                "selected_groups": len(selected_groups),
                "include_terminal_empty": args.include_terminal_empty,
                "success_count": success_count,
                "skip_count": skip_count,
                "dry_run_count": dry_run_count,
                "error_count": error_count,
                "parallel": worker_count,
                "output_root": str(output_root.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
