#!/usr/bin/env python3
"""Merge only successfully repaired groups from multiple repair agents.

This builds a unified refine-style directory that contains repaired outputs only.
It intentionally does not copy the full initial stage.
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
            "Collect only successfully repaired groups from refine / negative-memory "
            "agent roots into one unified batch root and agent root."
        )
    )
    parser.add_argument("--refine-batch-root")
    parser.add_argument("--refine-agent-root")
    parser.add_argument("--refine-scan-json")
    parser.add_argument(
        "--neg-memory-batch-root",
        help="Batch root for negative-memory groups, usually the initial batch root.",
    )
    parser.add_argument("--neg-memory-agent-root")
    parser.add_argument("--neg-memory-scan-json")
    parser.add_argument("--output-batch-root", required=True)
    parser.add_argument(
        "--output-agent-root",
        required=True,
        help=(
            "Unified repair agent root ending in the model name, e.g. "
            "outputs/.../agent_output/refine_merged/sam3_agent/doubao-seed-2.0-pro"
        ),
    )
    parser.add_argument(
        "--include-bad-repairs",
        action="store_true",
        help="Copy all scanned repair groups, not only good_groups. Not recommended.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_scan_group_set(path: Path, *, include_bad: bool) -> set[tuple[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    selected = {
        (str(item["video_id"]), str(item["json_id"]))
        for item in data.get("good_groups", [])
    }
    if include_bad:
        selected.update(
            (str(item["video_id"]), str(item["json_id"]))
            for item in data.get("group_badcases", [])
        )
        selected.update(
            (str(item["video_id"]), str(item["json_id"]))
            for item in data.get("terminal_empty_groups", [])
        )
    return selected


def ensure_output_root(root: Path, *, overwrite: bool, dry_run: bool) -> None:
    if root.exists() and any(root.iterdir()) and not overwrite:
        raise FileExistsError(f"Output root already exists and is not empty: {root}")
    if root.exists() and overwrite and not dry_run:
        shutil.rmtree(root)
    if not dry_run:
        root.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path, *, dry_run: bool) -> None:
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path, *, dry_run: bool) -> None:
    if dry_run:
        return
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)


def copy_repair_group(
    *,
    stage: str,
    video_id: str,
    json_id: str,
    batch_root: Path,
    agent_root: Path,
    output_batch_root: Path,
    output_agent_root: Path,
    dry_run: bool,
) -> dict[str, Any]:
    src_batch = batch_root / video_id / f"{json_id}.json"
    src_agent = agent_root / video_id / json_id
    dst_batch = output_batch_root / video_id / f"{json_id}.json"
    dst_agent = output_agent_root / video_id / json_id
    errors: list[str] = []
    if not src_batch.exists():
        errors.append(f"missing_batch:{src_batch}")
    if not src_agent.exists():
        errors.append(f"missing_agent:{src_agent}")
    if not errors:
        copy_file(src_batch, dst_batch, dry_run=dry_run)
        copy_tree(src_agent, dst_agent, dry_run=dry_run)
    return {
        "stage": stage,
        "video_id": video_id,
        "json_id": json_id,
        "status": "error" if errors else "copied",
        "source_batch": str(src_batch),
        "source_agent": str(src_agent),
        "output_batch": str(dst_batch),
        "output_agent": str(dst_agent),
        "errors": errors,
    }


def main() -> int:
    args = parse_args()
    output_batch_root = Path(args.output_batch_root)
    output_agent_root = Path(args.output_agent_root)
    ensure_output_root(output_batch_root, overwrite=args.overwrite, dry_run=args.dry_run)
    ensure_output_root(output_agent_root, overwrite=args.overwrite, dry_run=args.dry_run)

    routes: list[dict[str, Any]] = []
    if args.refine_batch_root and args.refine_agent_root and args.refine_scan_json:
        routes.append(
            {
                "stage": "refine",
                "batch_root": Path(args.refine_batch_root),
                "agent_root": Path(args.refine_agent_root),
                "scan_json": Path(args.refine_scan_json),
            }
        )
    if args.neg_memory_batch_root and args.neg_memory_agent_root and args.neg_memory_scan_json:
        routes.append(
            {
                "stage": "neg_memory",
                "batch_root": Path(args.neg_memory_batch_root),
                "agent_root": Path(args.neg_memory_agent_root),
                "scan_json": Path(args.neg_memory_scan_json),
            }
        )
    if not routes:
        raise ValueError("No repair routes configured. Pass refine and/or neg-memory inputs.")

    records: list[dict[str, Any]] = []
    selected: dict[tuple[str, str], str] = {}
    for route in routes:
        group_set = load_scan_group_set(
            route["scan_json"],
            include_bad=bool(args.include_bad_repairs),
        )
        for video_id, json_id in sorted(group_set):
            record = copy_repair_group(
                stage=route["stage"],
                video_id=video_id,
                json_id=json_id,
                batch_root=route["batch_root"],
                agent_root=route["agent_root"],
                output_batch_root=output_batch_root,
                output_agent_root=output_agent_root,
                dry_run=bool(args.dry_run),
            )
            records.append(record)
            if record["status"] == "copied":
                selected[(video_id, json_id)] = route["stage"]

    stage_counts: dict[str, int] = {}
    for stage in selected.values():
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
    errors = [record for record in records if record["status"] == "error"]
    summary = {
        "output_batch_root": str(output_batch_root.resolve()),
        "output_agent_root": str(output_agent_root.resolve()),
        "repair_group_count": len(selected),
        "stage_counts": stage_counts,
        "record_count": len(records),
        "error_count": len(errors),
        "include_bad_repairs": bool(args.include_bad_repairs),
        "dry_run": bool(args.dry_run),
        "routes": [
            {
                "stage": route["stage"],
                "batch_root": str(route["batch_root"]),
                "agent_root": str(route["agent_root"]),
                "scan_json": str(route["scan_json"]),
            }
            for route in routes
        ],
        "records": records,
        "errors": errors,
    }
    print(
        json.dumps(
            {
                "repair_group_count": summary["repair_group_count"],
                "stage_counts": stage_counts,
                "record_count": summary["record_count"],
                "error_count": summary["error_count"],
                "output_batch_root": summary["output_batch_root"],
                "output_agent_root": summary["output_agent_root"],
                "dry_run": summary["dry_run"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not args.dry_run:
        summary_path = output_batch_root / "_repair_only_merge_summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved_summary: {summary_path}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
