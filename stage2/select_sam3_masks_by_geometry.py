#!/usr/bin/env python3
"""Select SAM3 candidate masks with Stage1 geometry hints and global matching."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api_config import add_api_profile_args, apply_api_profile_defaults  # noqa: E402
from stage2.run_sam3_agent_from_batch_result import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    build_processor,
    import_agent_modules,
    safe_name,
)


DEFAULT_BATCH_ROOT = "outputs/mevis/batch_ark_video_qa"
DEFAULT_DATASET_ROOT = "../../dataset/mevis/valid"
DEFAULT_OUTPUT_ROOT = "outputs/agent_output"
DEFAULT_SAM3_OFFICIAL_ROOT = "sam3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read Stage1 JSON outputs with target_point/target_box_hint, generate one "
            "SAM3 candidate pool per frame/entity, and globally assign unique masks."
        )
    )
    parser.add_argument("--batch-root", default=DEFAULT_BATCH_ROOT)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--exp-name", default="sam3_geometry")
    parser.add_argument("--sam3-official-root", default=DEFAULT_SAM3_OFFICIAL_ROOT)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument(
        "--min-match-score",
        type=float,
        default=0.35,
        help="If the best geometry score for a target is below this value, output no mask.",
    )
    parser.add_argument("--only-video-id", action="append")
    parser.add_argument("--only-json-id", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--use-mllm-rerank",
        action="store_true",
        help="Use an API MLLM to choose among top geometry-filtered candidates.",
    )
    parser.add_argument(
        "--mllm-top-k",
        type=int,
        default=6,
        help="Number of candidate masks shown to the MLLM per target.",
    )
    add_api_profile_args(parser)
    parser.add_argument("--base-url", default=os.environ.get("SAM3_AGENT_BASE_URL"))
    parser.add_argument("--model", default=os.environ.get("SAM3_AGENT_MODEL"))
    parser.add_argument("--api-key", default=os.environ.get("SAM3_AGENT_API_KEY"))
    parser.add_argument("--dry-run", action="store_true")
    return apply_api_profile_defaults(parser.parse_args())


def load_stage1_tasks(
    batch_root: Path,
    only_video_ids: set[str] | None,
    only_json_ids: set[str] | None,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for json_path in sorted(batch_root.glob("*/*.json")):
        video_id = json_path.parent.name
        json_id = json_path.stem
        if only_video_ids and video_id not in only_video_ids:
            continue
        if only_json_ids and json_id not in only_json_ids:
            continue
        data = json.loads(json_path.read_text())
        tasks.append(
            {
                "video_id": video_id,
                "json_id": json_id,
                "json_path": json_path,
                "data": data,
            }
        )
    return tasks


def normalize_prompt(entity: Any) -> str:
    text = str(entity or "").strip().lower()
    if not text:
        return "object"

    # Keep simple visual adjectives that reduce obvious distractors. For this
    # task, "long fish" is usually better than broad "fish" because it avoids
    # many short/round/background fish candidates before geometry matching.
    if re.search(r"\bfish\b", text) and any(
        word in text for word in ("long", "elongated", "slender", "thin")
    ):
        return "long fish"

    # SAM3 text prompts work best as simple category nouns. Keep the remaining
    # mapping conservative; Stage1 geometry will handle harder relations.
    category_terms = [
        "fish",
        "person",
        "man",
        "woman",
        "girl",
        "boy",
        "dog",
        "cat",
        "bird",
        "panda",
        "horse",
        "car",
        "truck",
        "bus",
        "bicycle",
        "motorcycle",
    ]
    for term in category_terms:
        if re.search(rf"\b{re.escape(term)}\b", text):
            return term
    return text


def as_point(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, list) or len(value) != 2:
        return None
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        return None
    return x, y


def as_box_xyxy(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in value)
    except (TypeError, ValueError):
        return None
    x1, x2 = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
    y1, y2 = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def pad_box(
    box: tuple[float, float, float, float] | None,
    *,
    pad_x: float = 0.035,
    pad_y: float = 0.025,
) -> tuple[float, float, float, float] | None:
    if box is None:
        return None
    x1, y1, x2, y2 = box
    return (
        max(0.0, x1 - pad_x),
        max(0.0, y1 - pad_y),
        min(1.0, x2 + pad_x),
        min(1.0, y2 + pad_y),
    )


def cxcywh_to_xyxy(box: list[float]) -> tuple[float, float, float, float]:
    cx, cy, w, h = (float(v) for v in box)
    return (
        max(0.0, cx - w / 2.0),
        max(0.0, cy - h / 2.0),
        min(1.0, cx + w / 2.0),
        min(1.0, cy + h / 2.0),
    )


def box_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def box_iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def point_in_box(
    point: tuple[float, float],
    box: tuple[float, float, float, float],
) -> bool:
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def distance_score(
    target: tuple[float, float] | None,
    candidate: tuple[float, float],
    *,
    scale: float = 0.35,
) -> float:
    if target is None:
        return 0.5
    dist = math.hypot(target[0] - candidate[0], target[1] - candidate[1])
    return max(0.0, 1.0 - dist / scale)


def shape_score(item: dict[str, Any], cand_box: tuple[float, float, float, float]) -> float:
    text = " ".join(
        str(item.get(key) or "")
        for key in ("target_entity", "discriminative_description", "ordinal_hint")
    ).lower()
    x1, y1, x2, y2 = cand_box
    w, h = max(1e-6, x2 - x1), max(1e-6, y2 - y1)
    aspect = w / h
    if any(word in text for word in ("long", "elongated", "thin", "slender")):
        return max(0.0, min(1.0, (aspect - 1.0) / 3.0))
    return 0.5


def ordinal_score(
    hint: str,
    cand_idx: int,
    cand_centers: list[tuple[float, float]],
) -> float:
    hint = hint.lower()
    if not hint:
        return 0.5

    n = len(cand_centers)
    if n <= 1:
        return 1.0

    cx, cy = cand_centers[cand_idx]
    scores: list[float] = []

    x_rank = sorted(range(n), key=lambda i: cand_centers[i][0]).index(cand_idx)
    y_rank = sorted(range(n), key=lambda i: cand_centers[i][1]).index(cand_idx)
    denom = max(1, n - 1)

    if "leftmost" in hint:
        scores.append(1.0 - x_rank / denom)
    elif "left" in hint:
        scores.append(max(0.0, 1.0 - cx / 0.75))

    if "rightmost" in hint:
        scores.append(x_rank / denom)
    elif "right" in hint:
        scores.append(max(0.0, 1.0 - (1.0 - cx) / 0.75))

    if "topmost" in hint:
        scores.append(1.0 - y_rank / denom)
    elif "upper" in hint or "top" in hint:
        scores.append(max(0.0, 1.0 - cy / 0.75))

    if "lower" in hint or "bottom" in hint:
        scores.append(max(0.0, 1.0 - (1.0 - cy) / 0.75))

    ordinal_targets = {
        "first": 0,
        "second": 1,
        "third": 2,
        "fourth": 3,
        "fifth": 4,
        "sixth": 5,
    }
    for word, target_rank in ordinal_targets.items():
        if word in hint:
            scores.append(max(0.0, 1.0 - abs(y_rank - target_rank) / denom))
            break

    if "middle" in hint or "center" in hint:
        scores.append(max(0.0, 1.0 - abs(cx - 0.5) / 0.5))

    return sum(scores) / len(scores) if scores else 0.5


def build_candidates(sam_outputs: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = []
    for idx, box in enumerate(sam_outputs.get("pred_boxes", [])):
        xyxy = cxcywh_to_xyxy(box)
        candidates.append(
            {
                "index": idx,
                "mask_number": idx + 1,
                "box_cxcywh": box,
                "box_xyxy": xyxy,
                "center": box_center(xyxy),
                "score": float(sam_outputs.get("pred_scores", [0.0] * (idx + 1))[idx]),
            }
        )
    return candidates


def score_pair(
    item: dict[str, Any],
    candidate: dict[str, Any],
    cand_idx: int,
    cand_centers: list[tuple[float, float]],
) -> dict[str, float]:
    target_point = as_point(item.get("target_point"))
    raw_box = as_box_xyxy(item.get("target_box_hint"))
    target_box = pad_box(raw_box)
    target_box_center = box_center(raw_box) if raw_box else None
    candidate_box = candidate["box_xyxy"]
    candidate_center = candidate["center"]

    point = distance_score(target_point, candidate_center)
    box_center_score = distance_score(target_box_center, candidate_center)
    iou = box_iou(target_box, candidate_box) if target_box else 0.5
    ordinal = ordinal_score(str(item.get("ordinal_hint") or ""), cand_idx, cand_centers)
    shape = shape_score(item, candidate_box)
    sam = max(0.0, min(1.0, candidate["score"]))

    total = (
        0.28 * point
        + 0.22 * box_center_score
        + 0.25 * iou
        + 0.15 * ordinal
        + 0.07 * shape
        + 0.03 * sam
    )
    spatial_valid = True
    if target_box is not None:
        candidate_center_in_hint = point_in_box(candidate_center, target_box)
        target_point_in_candidate = (
            point_in_box(target_point, candidate_box) if target_point is not None else False
        )
        # Hard gate: if the target hint exists and the candidate neither overlaps
        # it nor contains the target point, this is almost certainly a distractor.
        if iou < 0.005 and not candidate_center_in_hint and not target_point_in_candidate:
            spatial_valid = False
            total = 0.0
        elif candidate_center_in_hint or target_point_in_candidate:
            total *= 1.08

    return {
        "total": total,
        "point": point,
        "box_center": box_center_score,
        "box_iou": iou,
        "ordinal": ordinal,
        "shape": shape,
        "sam": sam,
        "spatial_valid": spatial_valid,
    }


def greedy_assign(
    items: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    *,
    min_match_score: float = 0.35,
) -> list[dict[str, Any]]:
    cand_centers = [cand["center"] for cand in candidates]
    pairs = []
    for item_idx, item in enumerate(items):
        for cand_idx, cand in enumerate(candidates):
            score = score_pair(item, cand, cand_idx, cand_centers)
            pairs.append(
                {
                    "item_idx": item_idx,
                    "candidate_idx": cand_idx,
                    "mask_number": cand["mask_number"],
                    "score": score,
                }
            )
    pairs.sort(key=lambda pair: pair["score"]["total"], reverse=True)

    assigned_items: set[int] = set()
    assigned_candidates: set[int] = set()
    assignments: list[dict[str, Any] | None] = [None for _ in items]
    for pair in pairs:
        if not pair["score"].get("spatial_valid", True):
            continue
        if pair["score"]["total"] < min_match_score:
            continue
        if pair["item_idx"] in assigned_items or pair["candidate_idx"] in assigned_candidates:
            continue
        assigned_items.add(pair["item_idx"])
        assigned_candidates.add(pair["candidate_idx"])
        assignments[pair["item_idx"]] = pair
        if len(assigned_items) == len(items):
            break

    return [
        assignment
        or {
            "item_idx": idx,
            "candidate_idx": None,
            "mask_number": None,
            "score": {"total": 0.0},
        }
        for idx, assignment in enumerate(assignments)
    ]


def extract_json_object(text: str | None) -> dict[str, Any]:
    if not text:
        return {}
    text = text.strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def candidate_ranking(
    item: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    used_candidate_indices: set[int],
) -> list[dict[str, Any]]:
    cand_centers = [cand["center"] for cand in candidates]
    ranked = []
    for cand_idx, cand in enumerate(candidates):
        if cand_idx in used_candidate_indices:
            continue
        score = score_pair(item, cand, cand_idx, cand_centers)
        ranked.append(
            {
                "item_idx": 0,
                "candidate_idx": cand_idx,
                "mask_number": cand["mask_number"],
                "score": score,
            }
        )
    ranked.sort(key=lambda pair: pair["score"]["total"], reverse=True)
    return ranked


def save_candidate_views(
    *,
    sam_outputs: dict[str, Any],
    candidate_pairs: list[dict[str, Any]],
    candidate_dir: Path,
    visualize,
) -> list[dict[str, Any]]:
    candidate_dir.mkdir(parents=True, exist_ok=True)
    views = []
    for pair in candidate_pairs:
        cand_idx = pair["candidate_idx"]
        mask_number = pair["mask_number"]
        full_img, zoom_img = visualize(sam_outputs, cand_idx)
        full_path = candidate_dir / f"candidate_{mask_number:02d}_full.png"
        zoom_path = candidate_dir / f"candidate_{mask_number:02d}_zoom.png"
        full_img.save(full_path)
        zoom_img.save(zoom_path)
        views.append(
            {
                "mask_number": mask_number,
                "candidate_idx": cand_idx,
                "full_path": full_path,
                "zoom_path": zoom_path,
                "score": pair["score"],
            }
        )
    return views


def choose_mask_with_mllm(
    *,
    item: dict[str, Any],
    image_path: Path,
    sam_outputs: dict[str, Any],
    candidate_pairs: list[dict[str, Any]],
    candidate_dir: Path,
    send_generate_request,
    visualize,
) -> dict[str, Any] | None:
    if not candidate_pairs:
        return None

    views = save_candidate_views(
        sam_outputs=sam_outputs,
        candidate_pairs=candidate_pairs,
        candidate_dir=candidate_dir,
        visualize=visualize,
    )
    candidate_text = []
    for view in views:
        score = view["score"]
        candidate_text.append(
            "mask_number={mask_number}, geometry_score={total:.3f}, "
            "box_iou={box_iou:.3f}, point_score={point:.3f}, ordinal_score={ordinal:.3f}".format(
                mask_number=view["mask_number"],
                total=float(score.get("total", 0.0)),
                box_iou=float(score.get("box_iou", 0.0)),
                point=float(score.get("point", 0.0)),
                ordinal=float(score.get("ordinal", 0.0)),
            )
        )

    prompt = f"""You are selecting the correct SAM3 candidate mask for one target.

Target object name: {item.get('object_name')}
Target entity: {item.get('target_entity')}
Target description: {item.get('discriminative_description')}
Grid cell: {item.get('grid_cell')}
Target point: {item.get('target_point')}
Target box hint: {item.get('target_box_hint')}
Ordinal hint: {item.get('ordinal_hint')}

Candidate metadata:
{chr(10).join(candidate_text)}

You will see the raw image first, then for each candidate mask you will see a full-image highlight and a zoomed highlight.
Choose the candidate that best matches the target description and spatial hint. Reject distractors, short/round fish, background/reflection/screen fish, and candidates outside the target region.

Output ONLY valid JSON:
{{"selected_mask_number": 3, "confidence": 0.0, "reason": "short reason"}}

If none of the candidates match, output:
{{"selected_mask_number": null, "confidence": 0.0, "reason": "none match"}}
"""
    content: list[dict[str, Any]] = [
        {"type": "text", "text": prompt},
        {"type": "image", "image": str(image_path)},
    ]
    for view in views:
        content.append({"type": "text", "text": f"Candidate mask_number={view['mask_number']} full image:"})
        content.append({"type": "image", "image": str(view["full_path"])})
        content.append({"type": "text", "text": f"Candidate mask_number={view['mask_number']} zoom image:"})
        content.append({"type": "image", "image": str(view["zoom_path"])})

    response_text = send_generate_request(
        [
            {
                "role": "user",
                "content": content,
            }
        ]
    )
    parsed = extract_json_object(response_text)
    selected = parsed.get("selected_mask_number")
    try:
        selected = int(selected) if selected is not None else None
    except (TypeError, ValueError):
        selected = None

    for pair in candidate_pairs:
        if pair["mask_number"] == selected:
            pair = dict(pair)
            pair["mllm_rerank"] = {
                "response_text": response_text,
                "parsed": parsed,
                "candidate_views": [
                    {
                        "mask_number": view["mask_number"],
                        "full_path": str(view["full_path"]),
                        "zoom_path": str(view["zoom_path"]),
                    }
                    for view in views
                ],
            }
            return pair
    return {
        "item_idx": 0,
        "candidate_idx": None,
        "mask_number": None,
        "score": {"total": 0.0},
        "mllm_rerank": {
            "response_text": response_text,
            "parsed": parsed,
            "candidate_views": [
                {
                    "mask_number": view["mask_number"],
                    "full_path": str(view["full_path"]),
                    "zoom_path": str(view["zoom_path"]),
                }
                for view in views
            ],
        },
    }


def mllm_assign(
    items: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    *,
    image_path: Path,
    sam_outputs: dict[str, Any],
    candidate_root: Path,
    top_k: int,
    send_generate_request,
    visualize,
    min_match_score: float,
) -> list[dict[str, Any]]:
    assigned_candidates: set[int] = set()
    assignments: list[dict[str, Any]] = []
    for item_idx, item in enumerate(items):
        ranked = [
            pair
            for pair in candidate_ranking(item, candidates, used_candidate_indices=assigned_candidates)
            if pair["score"].get("spatial_valid", True)
            and pair["score"].get("total", 0.0) >= min_match_score
        ]
        if not ranked:
            assignments.append(
                {
                    "item_idx": item_idx,
                    "candidate_idx": None,
                    "mask_number": None,
                    "score": {"total": 0.0},
                    "mllm_rerank": {"reason": "no geometry-valid candidates"},
                }
            )
            continue
        chosen = choose_mask_with_mllm(
            item=item,
            image_path=image_path,
            sam_outputs=sam_outputs,
            candidate_pairs=ranked[:top_k],
            candidate_dir=candidate_root / safe_name(str(item.get("object_name") or f"object_{item_idx + 1}")),
            send_generate_request=send_generate_request,
            visualize=visualize,
        )
        if chosen is None:
            chosen = {
                "item_idx": item_idx,
                "candidate_idx": None,
                "mask_number": None,
                "score": {"total": 0.0},
                "mllm_rerank": {"reason": "empty MLLM candidate choice"},
            }
        else:
            chosen["item_idx"] = item_idx
        if chosen.get("candidate_idx") is not None:
            assigned_candidates.add(int(chosen["candidate_idx"]))
        assignments.append(chosen)
    return assignments


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def group_items_by_candidate_pool(items: list[dict[str, Any]]) -> dict[tuple[str, str], list[tuple[int, dict[str, Any]]]]:
    grouped: dict[tuple[str, str], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for obj_idx, item in enumerate(items, start=1):
        frame_filename = str(item.get("actual_frame_filename") or "").strip()
        prompt = normalize_prompt(item.get("target_entity") or item.get("object_name"))
        if frame_filename:
            grouped[(frame_filename, prompt)].append((obj_idx, item))
    return grouped


def process_task(
    task: dict[str, Any],
    *,
    dataset_root: Path,
    run_root: Path,
    call_sam_service,
    send_generate_request,
    visualize,
    overwrite: bool,
    min_match_score: float,
    use_mllm_rerank: bool,
    mllm_top_k: int,
) -> dict[str, Any]:
    video_id = task["video_id"]
    json_id = task["json_id"]
    data = task["data"]
    items = data.get("results") or []
    if not isinstance(items, list):
        raise ValueError(f"Invalid Stage1 results in {task['json_path']}")

    task_root = run_root / video_id / str(json_id)
    summary: dict[str, Any] = {
        "video_id": video_id,
        "json_id": str(json_id),
        "target_event": data.get("target_event"),
        "source_json": str(task["json_path"]),
        "method": "sam3_geometry_global_matching",
        "items": [],
        "candidate_pools": [],
    }

    pool_groups = group_items_by_candidate_pool(items)
    for (frame_filename, prompt), grouped_items in sorted(pool_groups.items()):
        image_path = dataset_root / "JPEGImages" / video_id / frame_filename
        if not image_path.exists():
            raise FileNotFoundError(f"Image frame not found: {image_path}")

        pool_dir = task_root / "_candidate_pools" / safe_name(frame_filename) / safe_name(prompt)
        pool_json_path = pool_dir / f"{safe_name(prompt)}.json"
        if pool_json_path.exists() and not overwrite:
            sam_outputs = json.loads(pool_json_path.read_text())
        else:
            sam_json_path = Path(
                call_sam_service(
                    image_path=str(image_path),
                    text_prompt=prompt,
                    output_folder_path=str(pool_dir),
                )
            )
            sam_outputs = json.loads(sam_json_path.read_text())
            pool_json_path = sam_json_path

        candidates = build_candidates(sam_outputs)
        local_items = [item for _, item in grouped_items]
        if use_mllm_rerank:
            assignments = mllm_assign(
                local_items,
                candidates,
                image_path=image_path,
                sam_outputs=sam_outputs,
                candidate_root=task_root / "_mllm_rerank" / safe_name(frame_filename) / safe_name(prompt),
                top_k=mllm_top_k,
                send_generate_request=send_generate_request,
                visualize=visualize,
                min_match_score=min_match_score,
            )
        else:
            assignments = greedy_assign(
                local_items,
                candidates,
                min_match_score=min_match_score,
            )
        summary["candidate_pools"].append(
            {
                "frame_filename": frame_filename,
                "prompt": prompt,
                "pool_json": str(pool_json_path),
                "num_candidates": len(candidates),
                "num_targets": len(local_items),
            }
        )

        for (obj_idx, item), assignment in zip(grouped_items, assignments):
            object_name = item.get("object_name") or f"object_{obj_idx}"
            item_dir = task_root / f"{safe_name(object_name)}_obj_{obj_idx}"
            pred_json_path = item_dir / "pred.json"
            if pred_json_path.exists() and not overwrite:
                summary["items"].append(
                    {
                        "object_index": obj_idx,
                        "object_name": object_name,
                        "status": "skipped_existing",
                        "output_dir": str(item_dir),
                    }
                )
                continue

            if assignment["candidate_idx"] is None:
                final_outputs = {
                    "original_image_path": str(image_path),
                    "orig_img_h": sam_outputs["orig_img_h"],
                    "orig_img_w": sam_outputs["orig_img_w"],
                    "pred_boxes": [],
                    "pred_scores": [],
                    "pred_masks": [],
                }
            else:
                cand_idx = assignment["candidate_idx"]
                final_outputs = {
                    "original_image_path": sam_outputs["original_image_path"],
                    "orig_img_h": sam_outputs["orig_img_h"],
                    "orig_img_w": sam_outputs["orig_img_w"],
                    "pred_boxes": [sam_outputs["pred_boxes"][cand_idx]],
                    "pred_scores": [sam_outputs["pred_scores"][cand_idx]],
                    "pred_masks": [sam_outputs["pred_masks"][cand_idx]],
                }

            pred_payload = {
                **final_outputs,
                "video_id": video_id,
                "json_id": str(json_id),
                "object_index": obj_idx,
                "object_name": object_name,
                "input_prompt": str(item.get("discriminative_description") or object_name),
                "source_result_item": item,
                "geometry_assignment": {
                    "candidate_prompt": prompt,
                    "candidate_pool_json": str(pool_json_path),
                    "selected_mask_number": assignment.get("mask_number"),
                    "score": assignment.get("score"),
                    "mllm_rerank": assignment.get("mllm_rerank"),
                },
                "role_info": {
                    "target_role": item.get("target_role"),
                    "target_entity": item.get("target_entity"),
                    "action": item.get("action"),
                    "reference_entities": item.get("reference_entities", []),
                    "excluded_entities": item.get("excluded_entities", []),
                },
            }

            item_dir.mkdir(parents=True, exist_ok=True)
            save_json(pred_payload, pred_json_path)
            save_json(
                {
                    "video_id": video_id,
                    "json_id": str(json_id),
                    "object_index": obj_idx,
                    "object_name": object_name,
                    "image_path": str(image_path),
                    "inference_image_path": str(image_path),
                    "frame_filename": frame_filename,
                    "actual_frame_index": item.get("actual_frame_index"),
                    "prompt": pred_payload["input_prompt"],
                    "target_event": data.get("target_event"),
                    "role_info": pred_payload["role_info"],
                    "geometry_assignment": pred_payload["geometry_assignment"],
                },
                item_dir / "input_meta.json",
            )

            rendered = visualize(final_outputs)
            rendered.save(item_dir / "pred.png")
            summary["items"].append(
                {
                    "object_index": obj_idx,
                    "object_name": object_name,
                    "frame_filename": frame_filename,
                    "candidate_prompt": prompt,
                    "selected_mask_number": assignment.get("mask_number"),
                    "score": assignment.get("score", {}).get("total"),
                    "output_dir": str(item_dir),
                    "status": "completed",
                }
            )

    save_json(summary, task_root / "summary.json")
    return summary


def main() -> int:
    args = parse_args()
    batch_root = Path(args.batch_root)
    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)
    run_root = output_root / args.exp_name

    tasks = load_stage1_tasks(
        batch_root,
        set(args.only_video_id or []) or None,
        set(args.only_json_id or []) or None,
    )
    if args.limit is not None:
        tasks = tasks[: args.limit]

    print(f"planned_tasks: {len(tasks)}")
    print(f"batch_root: {batch_root}")
    print(f"output_root: {run_root}")
    if tasks:
        print(f"first_task: {tasks[0]['video_id']}/{tasks[0]['json_id']}")
    if args.dry_run:
        return 0

    (
        build_sam3_image_model,
        Sam3Processor,
        _agent_inference,
        _send_generate_request_orig,
        call_sam_service_orig,
    ) = import_agent_modules(Path(args.sam3_official_root))
    processor = build_processor(
        build_sam3_image_model=build_sam3_image_model,
        Sam3Processor=Sam3Processor,
        sam3_official_root=Path(args.sam3_official_root),
        checkpoint=args.checkpoint,
        device=args.device,
        confidence_threshold=args.confidence_threshold,
    )
    call_sam_service = partial(call_sam_service_orig, sam3_processor=processor)

    send_generate_request = None
    if args.use_mllm_rerank:
        if not args.api_key:
            raise ValueError("Missing API key for --use-mllm-rerank")
        from sam3.agent.client_llm import send_generate_request as send_generate_request_orig

        send_generate_request = partial(
            send_generate_request_orig,
            server_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
        )

    root_str = str(Path(args.sam3_official_root).resolve())
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    from sam3.agent.viz import visualize

    success_count = 0
    error_count = 0
    for task in tasks:
        prefix = f"{task['video_id']}/{task['json_id']}"
        try:
            summary = process_task(
                task,
                dataset_root=dataset_root,
                run_root=run_root,
                call_sam_service=call_sam_service,
                send_generate_request=send_generate_request,
                visualize=visualize,
                overwrite=args.overwrite,
                min_match_score=args.min_match_score,
                use_mllm_rerank=args.use_mllm_rerank,
                mllm_top_k=args.mllm_top_k,
            )
            success_count += 1
            print(f"saved {prefix} -> {run_root / task['video_id'] / str(task['json_id'])}")
            print(f"  items: {len(summary['items'])}, pools: {len(summary['candidate_pools'])}")
        except Exception as exc:
            error_count += 1
            print(f"error {prefix}: {type(exc).__name__}: {exc}")

    print(
        json.dumps(
            {
                "planned_tasks": len(tasks),
                "success_count": success_count,
                "error_count": error_count,
                "output_root": str(run_root.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
