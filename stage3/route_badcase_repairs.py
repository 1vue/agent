#!/usr/bin/env python3
"""Route scan-agent badcases into repair manifests by failure type."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read a scan_agent_badcases.py report and split badcases into "
            "difficulty-aware repair manifests."
        )
    )
    parser.add_argument("--scan-json", required=True, help="Input scan report JSON")
    parser.add_argument(
        "--output-dir",
        default="repair_manifests",
        help="Directory where manifest JSON files will be written",
    )
    parser.add_argument(
        "--include-empty-mask-as-reselect",
        action="store_true",
        help=(
            "Also route empty_mask object failures to the reselect manifest. "
            "By default this script only routes high_iou, missing_pred, and terminal_empty."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print routing summary without writing manifest files",
    )
    return parser.parse_args()


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _group_key(item: dict[str, Any]) -> tuple[str, str]:
    return (str(item["video_id"]), str(item["json_id"]))


def _base_group_task(group: dict[str, Any]) -> dict[str, Any]:
    return {
        "video_id": str(group["video_id"]),
        "json_id": str(group["json_id"]),
        "target_event": group.get("target_event"),
        "batch_json_path": group.get("batch_json_path"),
        "agent_group_dir": group.get("agent_group_dir"),
        "reasons": list(group.get("reasons") or []),
        "num_results": group.get("num_results"),
    }


def _object_name_map(pair_issues: list[dict[str, Any]]) -> dict[str, str]:
    names: dict[str, str] = {}
    for issue in pair_issues:
        if issue.get("object_a_index") is not None:
            names[str(issue["object_a_index"])] = str(issue.get("object_a_name") or "")
        if issue.get("object_b_index") is not None:
            names[str(issue["object_b_index"])] = str(issue.get("object_b_name") or "")
    return names


def build_high_iou_tasks(scan: dict[str, Any]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for group in _as_list(scan.get("group_badcases")):
        pair_issues = [
            issue
            for issue in _as_list(group.get("pair_issues"))
            if issue.get("reason") == "high_iou"
        ]
        if not pair_issues:
            continue

        conflict_indices = sorted(
            {
                int(index)
                for issue in pair_issues
                for index in (issue.get("object_a_index"), issue.get("object_b_index"))
                if index is not None
            }
        )
        task = {
            **_base_group_task(group),
            "repair_route": "visual_suppression",
            "failure_type": "high_iou",
            "conflict_indices": conflict_indices,
            "conflict_object_names": _object_name_map(pair_issues),
            "pair_issues": pair_issues,
        }
        tasks.append(task)
    return sorted(tasks, key=_group_key)


def build_missing_pred_tasks(scan: dict[str, Any], include_empty_mask: bool) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for group in _as_list(scan.get("group_badcases")):
        routed_reasons = {"missing_pred"}
        if include_empty_mask:
            routed_reasons.add("empty_mask")
        object_issues = [
            issue
            for issue in _as_list(group.get("object_issues"))
            if issue.get("reason") in routed_reasons
        ]
        if not object_issues:
            continue

        by_reason: dict[str, list[dict[str, Any]]] = {}
        for issue in object_issues:
            by_reason.setdefault(str(issue.get("reason")), []).append(issue)

        failure_type = (
            "missing_pred"
            if "missing_pred" in by_reason
            else "empty_mask"
        )
        task = {
            **_base_group_task(group),
            "repair_route": "frame_reselection",
            "failure_type": failure_type,
            "object_issues": object_issues,
            "issues_by_reason": by_reason,
        }
        tasks.append(task)
    return sorted(tasks, key=_group_key)


def build_terminal_empty_tasks(scan: dict[str, Any]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for group in _as_list(scan.get("terminal_empty_groups")):
        task = {
            "video_id": str(group["video_id"]),
            "json_id": str(group["json_id"]),
            "target_event": group.get("target_event"),
            "batch_json_path": group.get("batch_json_path"),
            "agent_group_dir": group.get("agent_group_dir"),
            "reasons": ["terminal_empty"],
            "num_results": 0,
            "repair_route": "candidate_discovery",
            "failure_type": "terminal_empty",
        }
        tasks.append(task)
    return sorted(tasks, key=_group_key)


def build_reselect_tasks(
    missing_pred_tasks: list[dict[str, Any]],
    terminal_empty_tasks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return sorted(
        [
            *missing_pred_tasks,
            *terminal_empty_tasks,
        ],
        key=_group_key,
    )


def build_manifest(
    *,
    manifest_type: str,
    route_name: str,
    scan_json: Path,
    scan: dict[str, Any],
    tasks: list[dict[str, Any]],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary_keys = [
        "total_groups",
        "bad_group_count",
        "good_group_count",
        "terminal_empty_group_count",
        "object_badcase_count",
        "overlap_badcase_count",
        "iou_threshold",
    ]
    manifest = {
        "manifest_type": manifest_type,
        "repair_route": route_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_scan_json": str(scan_json.resolve()),
        "scan_summary": {key: scan.get(key) for key in summary_keys if key in scan},
        "task_count": len(tasks),
        "tasks": tasks,
    }
    if extra:
        manifest.update(extra)
    return manifest


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    scan_json = Path(args.scan_json)
    output_dir = Path(args.output_dir)
    scan = json.loads(scan_json.read_text(encoding="utf-8"))

    high_iou_tasks = build_high_iou_tasks(scan)
    missing_pred_tasks = build_missing_pred_tasks(
        scan,
        include_empty_mask=args.include_empty_mask_as_reselect,
    )
    terminal_empty_tasks = build_terminal_empty_tasks(scan)
    reselect_tasks = build_reselect_tasks(missing_pred_tasks, terminal_empty_tasks)

    manifests = {
        "high_iou.json": build_manifest(
            manifest_type="high_iou",
            route_name="visual_suppression",
            scan_json=scan_json,
            scan=scan,
            tasks=high_iou_tasks,
        ),
        "missing_pred.json": build_manifest(
            manifest_type="missing_pred",
            route_name="frame_reselection",
            scan_json=scan_json,
            scan=scan,
            tasks=missing_pred_tasks,
            extra={"include_empty_mask": args.include_empty_mask_as_reselect},
        ),
        "terminal_empty.json": build_manifest(
            manifest_type="terminal_empty",
            route_name="candidate_discovery",
            scan_json=scan_json,
            scan=scan,
            tasks=terminal_empty_tasks,
        ),
        "reselect.json": build_manifest(
            manifest_type="reselect",
            route_name="frame_reselection_or_candidate_discovery",
            scan_json=scan_json,
            scan=scan,
            tasks=reselect_tasks,
            extra={"include_empty_mask": args.include_empty_mask_as_reselect},
        ),
    }

    routing_summary = {
        "source_scan_json": str(scan_json.resolve()),
        "output_dir": str(output_dir.resolve()),
        "high_iou_tasks": len(high_iou_tasks),
        "missing_pred_tasks": len(missing_pred_tasks),
        "terminal_empty_tasks": len(terminal_empty_tasks),
        "reselect_tasks": len(reselect_tasks),
        "include_empty_mask_as_reselect": args.include_empty_mask_as_reselect,
        "dry_run": args.dry_run,
    }
    print(json.dumps(routing_summary, ensure_ascii=False, indent=2))

    if args.dry_run:
        return 0

    for filename, manifest in manifests.items():
        write_manifest(output_dir / filename, manifest)
    write_manifest(
        output_dir / "routing_summary.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            **routing_summary,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
