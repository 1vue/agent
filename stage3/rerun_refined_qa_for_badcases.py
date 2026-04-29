#!/usr/bin/env python3
"""Rerun video QA for badcases using the original QA generation path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from stage1.run_ark_video_qa import (
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
        description="Rerun QA with a refinement prompt for groups listed in a badcase scan JSON.",
    )
    parser.add_argument("--scan-json", required=True, help="Path to badcase scan JSON")
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
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible API base URL")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name")
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scan = json.loads(Path(args.scan_json).read_text())
    only_video_ids = set(args.only_video_id or [])
    only_json_ids = set(args.only_json_id or [])
    output_root = Path(args.output_root)

    groups = scan.get("group_badcases", [])
    success_count = 0
    skip_count = 0
    error_count = 0

    for group in groups:
        video_id = group["video_id"]
        json_id = group["json_id"]
        if only_video_ids and video_id not in only_video_ids:
            continue
        if only_json_ids and json_id not in only_json_ids:
            continue

        source_path = Path(group["batch_json_path"])
        source = json.loads(source_path.read_text())
        output_path = output_root / video_id / f"{json_id}.json"
        if output_path.exists() and not args.overwrite:
            skip_count += 1
            print(f"skip {video_id}/{json_id} -> {output_path}")
            continue

        prompt_text = build_prompt_text(
            target_event=source.get("target_event"),
            response_format="json",
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
            results = normalize_result_records(answer, bundle.selected_frames)
            result_payload = build_result_payload(
                video_id=video_id,
                prompt_label="target_event",
                prompt_value=source.get("target_event"),
                bundle=bundle,
                results=results,
                raw_response_text=answer,
            )
            result_payload["expression_id"] = json_id
            result_payload["vid_id"] = source.get("vid_id")
            result_payload["refined_from"] = str(source_path)
            result_payload["rerun_reason_reasons"] = group.get("reasons", [])
            save_json(result_payload, output_path)
            success_count += 1
            print(f"saved {video_id}/{json_id} -> {output_path}")
        except Exception as exc:
            error_count += 1
            print(f"error {video_id}/{json_id}: {type(exc).__name__}: {exc}")

    print(
        json.dumps(
            {
                "groups": len(groups),
                "success_count": success_count,
                "skip_count": skip_count,
                "error_count": error_count,
                "output_root": str(output_root.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
