#!/usr/bin/env python3
"""Build a batch-root subset from a repair manifest.

This is useful when a later stage should scan or rerun only the exact groups
listed in a manifest such as high_iou.json or reselect.json.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read a route_badcase_repairs.py manifest and create a batch-root "
            "containing only the listed task JSON files."
        )
    )
    parser.add_argument("--manifest-json", required=True, help="Input repair manifest JSON")
    parser.add_argument("--output-root", required=True, help="Output batch root to create")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination JSON files if they already exist",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip tasks whose batch_json_path is missing instead of failing",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned copies without writing files",
    )
    return parser.parse_args()


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def build_subset(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest_json)
    output_root = Path(args.output_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    copied: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for task in _as_list(manifest.get("tasks")):
        video_id = str(task["video_id"])
        json_id = str(task["json_id"])
        key = (video_id, json_id)
        if key in seen:
            skipped.append(
                {
                    "video_id": video_id,
                    "json_id": json_id,
                    "reason": "duplicate_task",
                }
            )
            continue
        seen.add(key)

        src_value = task.get("batch_json_path")
        if not src_value:
            errors.append(
                {
                    "video_id": video_id,
                    "json_id": json_id,
                    "reason": "missing_batch_json_path_field",
                }
            )
            continue

        src = Path(src_value)
        dst = output_root / video_id / f"{json_id}.json"
        if not src.exists():
            item = {
                "video_id": video_id,
                "json_id": json_id,
                "source": str(src),
                "reason": "source_missing",
            }
            if args.skip_missing:
                skipped.append(item)
                continue
            errors.append(item)
            continue

        if dst.exists() and not args.overwrite:
            skipped.append(
                {
                    "video_id": video_id,
                    "json_id": json_id,
                    "source": str(src),
                    "destination": str(dst),
                    "reason": "destination_exists",
                }
            )
            continue

        copied.append(
            {
                "video_id": video_id,
                "json_id": json_id,
                "source": str(src),
                "destination": str(dst),
            }
        )
        if not args.dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    summary = {
        "manifest_json": str(manifest_path.resolve()),
        "manifest_type": manifest.get("manifest_type"),
        "source_scan_json": manifest.get("source_scan_json"),
        "output_root": str(output_root.resolve()),
        "task_count": len(_as_list(manifest.get("tasks"))),
        "unique_task_count": len(seen),
        "copied_count": len(copied),
        "skipped_count": len(skipped),
        "error_count": len(errors),
        "dry_run": bool(args.dry_run),
        "copied": copied,
        "skipped": skipped,
        "errors": errors,
    }
    if errors:
        raise RuntimeError(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> int:
    args = parse_args()
    summary = build_subset(args)
    print(
        json.dumps(
            {
                "manifest_type": summary["manifest_type"],
                "task_count": summary["task_count"],
                "unique_task_count": summary["unique_task_count"],
                "copied_count": summary["copied_count"],
                "skipped_count": summary["skipped_count"],
                "error_count": summary["error_count"],
                "output_root": summary["output_root"],
                "dry_run": summary["dry_run"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not args.dry_run:
        summary_path = Path(args.output_root) / "_manifest_subset_summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved_summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
