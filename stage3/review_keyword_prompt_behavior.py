#!/usr/bin/env python3
"""Review agent outputs whose prompts contain directional/negation keywords using sampled outlined frames."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI
from PIL import Image, ImageFilter

from stage2.generate_sam3_masks_from_agent import load_agent_seed
from stage2.generate_sam3_masks_from_bbox import DEFAULT_CHECKPOINT, build_video_tracker, import_sam3_builders, propagate_object_masks, sorted_jpeg_frame_files
from stage1.run_ark_video_qa import extract_answer, extract_json_block, image_to_data_url
from stage2.run_sam3_agent_from_batch_result import safe_name


REVIEW_PROMPT_TEMPLATE = """你是一个视频目标行为核验专家。下面会给你同一个目标在视频中每隔三帧抽取的一张高亮图像。

说明：
1. 图像中只有目标物体的边缘被高亮，物体内部不会覆盖。
2. 请只围绕被高亮的目标，判断它在这些抽帧中的身份、相对位置、行为、状态是否符合原始 prompt。
3. 如果 prompt 中包含 left/right/without 等约束，请重点核对这些约束是否真的成立，而不是只看类别是否对。
4. 如果 prompt 中涉及物体位移，请严格判断物体移动，排除镜头运动干扰。

原始 prompt：
{target_event}

请只输出一个 JSON 对象，格式必须是：
{{
  "is_match": true/false,
  "reason": "简明说明判断依据"
}}

不要输出任何额外解释、Markdown 或代码块。"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "For prompts containing left/right/without, sample every 3 frames from the video, "
            "save edge-highlighted object frames locally, and ask an MLLM whether the object's "
            "behavior/identity matches the original prompt."
        )
    )
    parser.add_argument("--batch-root", default="outputs/batch_ark_video_qa")
    parser.add_argument("--agent-root", required=True, help="Agent root like agent_output/sam3_agent/<model>")
    parser.add_argument("--output-json", required=True, help="Where to save the review report JSON")
    parser.add_argument(
        "--review-root",
        required=True,
        help="Directory to save highlighted sampled frames and per-object review JSONs",
    )
    parser.add_argument("--dataset-root", default="../../dataset/mevis/valid")
    parser.add_argument("--base-url", default="https://ark.cn-beijing.volces.com/api/v3")
    parser.add_argument("--model", default="doubao-seed-2-0-lite-260215")
    parser.add_argument("--api-key", default=os.environ.get("VOLCES_API_KEY").strip(), help="API key for the review MLLM")
    parser.add_argument(
        "--keywords",
        default="left,right,without,top,bottom",
        help="Comma-separated lowercase substrings to trigger review",
    )
    parser.add_argument("--sample-stride", type=int, default=3, help="Sample every N frames from the original video")
    parser.add_argument("--edge-width", type=int, default=3, help="Edge highlight width in pixels")
    parser.add_argument("--sam3-official-root", default="sam3")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--score-thresh", type=float, default=0.0)
    parser.add_argument("--only-video-id", action="append")
    parser.add_argument("--only-json-id", action="append")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def list_tasks(batch_root: Path, only_video_ids: set[str] | None, only_json_ids: set[str] | None) -> list[Path]:
    tasks = []
    for path in sorted(batch_root.glob("*/*.json")):
        if path.parent.name == "_errors":
            continue
        if only_video_ids and path.parent.name not in only_video_ids:
            continue
        if only_json_ids and path.stem not in only_json_ids:
            continue
        tasks.append(path)
    return tasks


def prompt_needs_review(prompt: str, keywords: list[str]) -> bool:
    prompt_lc = prompt.lower()
    return any(keyword in prompt_lc for keyword in keywords if keyword)


def highlight_edges(image: Image.Image, mask: np.ndarray, edge_width: int) -> Image.Image:
    image_rgb = image.convert("RGB")
    if mask.sum() == 0:
        return image_rgb
    mask_img = Image.fromarray((mask > 0).astype(np.uint8) * 255)
    size = max(3, edge_width * 2 + 1)
    if size % 2 == 0:
        size += 1
    dilated = mask_img.filter(ImageFilter.MaxFilter(size))
    eroded = mask_img.filter(ImageFilter.MinFilter(size))
    edge = (np.array(dilated) > 0) & ~(np.array(eroded) > 0)

    arr = np.array(image_rgb).copy()
    arr[edge] = np.array([255, 255, 0], dtype=np.uint8)
    return Image.fromarray(arr)


def sampled_indices(num_frames: int, stride: int) -> list[int]:
    if stride <= 0:
        raise ValueError("sample stride must be positive")
    indices = list(range(0, num_frames, stride))
    MAX_FRAMES = 24

    if len(indices) > MAX_FRAMES:
        indices = [i * (num_frames - 1) // (MAX_FRAMES - 1) for i in range(MAX_FRAMES)]
    return indices


def build_review_payload(target_event: str, sample_records: list[dict[str, Any]], model: str) -> dict[str, Any]:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": REVIEW_PROMPT_TEMPLATE.format(target_event=target_event),
        }
    ]
    for record in sample_records:
        content.append({"type": "text", "text": f"[frame {record['frame_index']}]"})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_to_data_url(record["highlight_path"])},
                "detail": "low",
            }
        )
    return {"model": model, "messages": [{"role": "user", "content": content}]}


def call_review_api(payload: dict[str, Any], *, base_url: str, api_key: str):
    client = OpenAI(base_url=base_url, api_key=api_key)
    return client.chat.completions.create(**payload)


def review_one_object(
    *,
    batch_json_path: Path,
    agent_root: Path,
    review_root: Path,
    dataset_root: Path,
    tracker,
    keywords: list[str],
    sample_stride: int,
    edge_width: int,
    base_url: str,
    model: str,
    api_key: str,
    score_thresh: float,
    overwrite: bool,
) -> list[dict[str, Any]]:
    source = json.loads(batch_json_path.read_text())
    video_id = source["video_id"]
    json_id = str(source["expression_id"]) if "expression_id" in source else batch_json_path.stem
    target_event = str(source.get("target_event") or "")
    video_dir = dataset_root / "JPEGImages" / video_id
    frame_files = sorted_jpeg_frame_files(video_dir)
    frame_name_to_index = {name: idx for idx, name in enumerate(frame_files)}
    group_dir = agent_root / video_id / json_id

    reviews: list[dict[str, Any]] = []
    for idx, item in enumerate(source.get("results", []), start=1):
        object_name = item.get("object_name") or f"object_{idx}"
        object_dir = group_dir / f"{safe_name(object_name)}_obj_{idx}"
        if not object_dir.exists():
            object_dir = next(group_dir.glob(f"*obj_{idx}"), None) if group_dir.exists() else None
        if object_dir is None or not object_dir.exists():
            continue
        pred_path = object_dir / "pred.json"
        if not pred_path.exists():
            continue
        pred = json.loads(pred_path.read_text())
        agent_prompt = pred.get("input_prompt") or ""
        if not prompt_needs_review(target_event, keywords):
            continue

        review_dir = review_root / video_id / json_id / object_dir.name
        review_json_path = review_dir / "review.json"
        if review_json_path.exists() and not overwrite:
            reviews.append(json.loads(review_json_path.read_text()))
            continue

        seed = load_agent_seed(pred_path)
        frame_filename = seed["frame_filename"]
        init_frame_idx = frame_name_to_index[frame_filename]
        outputs = propagate_object_masks(
            tracker,
            video_dir=video_dir,
            init_frame_idx=init_frame_idx,
            init_mask=seed["mask"],
            score_thresh=score_thresh,
        )
        sample_records: list[dict[str, Any]] = []
        for frame_idx in sampled_indices(len(frame_files), sample_stride):
            frame_filename_i = frame_files[frame_idx]
            image_path = video_dir / frame_filename_i
            image = Image.open(image_path).convert("RGB")
            mask = outputs.get(frame_idx)
            if mask is None:
                mask = np.zeros((image.height, image.width), dtype=np.uint8)
            highlighted = highlight_edges(image, mask, edge_width)
            out_path = review_dir / "frames" / f"{Path(frame_filename_i).stem}.jpg"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            highlighted.save(out_path)
            sample_records.append(
                {
                    "frame_index": frame_idx,
                    "frame_filename": frame_filename_i,
                    "highlight_path": out_path,
                    "mask_area": int((mask > 0).sum()),
                }
            )

        payload = build_review_payload(target_event, sample_records, model)
        response = call_review_api(payload, base_url=base_url, api_key=api_key)
        answer = extract_answer(response)
        parsed = extract_json_block(answer)
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected JSON object from review model, got: {parsed}")
        is_match = bool(parsed.get("is_match"))
        reason = str(parsed.get("reason", "")).strip()

        review = {
            "video_id": video_id,
            "json_id": json_id,
            "object_index": idx,
            "object_name": object_name,
            "target_event": target_event,
            "agent_prompt": agent_prompt,
            "keywords": [kw for kw in keywords if kw and kw in target_event.lower()],
            "is_match": is_match,
            "reason": reason,
            "review_response_raw": answer,
            "pred_path": str(pred_path),
            "review_dir": str(review_dir),
            "sample_stride": sample_stride,
            "sample_records": [
                {
                    "frame_index": record["frame_index"],
                    "frame_filename": record["frame_filename"],
                    "highlight_path": str(record["highlight_path"]),
                    "mask_area": record["mask_area"],
                }
                for record in sample_records
            ],
        }
        review_json_path.write_text(json.dumps(review, ensure_ascii=False, indent=2))
        reviews.append(review)
    return reviews


def main() -> int:
    args = parse_args()
    batch_root = Path(args.batch_root)
    agent_root = Path(args.agent_root)
    review_root = Path(args.review_root)
    output_json = Path(args.output_json)
    dataset_root = Path(args.dataset_root)
    only_video_ids = set(args.only_video_id or [])
    only_json_ids = set(args.only_json_id or [])
    keywords = [item.strip().lower() for item in args.keywords.split(",") if item.strip()]

    build_sam3_image_model, build_sam3_video_model = import_sam3_builders(Path(args.sam3_official_root))
    tracker = build_video_tracker(build_sam3_video_model, args.checkpoint, args.device)

    reviews_all: list[dict[str, Any]] = []
    group_badcases: list[dict[str, Any]] = []
    object_badcases: list[dict[str, Any]] = []

    for batch_json_path in list_tasks(batch_root, only_video_ids or None, only_json_ids or None):
        source = json.loads(batch_json_path.read_text())
        video_id = source["video_id"]
        json_id = str(source["expression_id"]) if "expression_id" in source else batch_json_path.stem
        reviews = review_one_object(
            batch_json_path=batch_json_path,
            agent_root=agent_root,
            review_root=review_root,
            dataset_root=dataset_root,
            tracker=tracker,
            keywords=keywords,
            sample_stride=args.sample_stride,
            edge_width=args.edge_width,
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
            score_thresh=args.score_thresh,
            overwrite=args.overwrite,
        )
        if not reviews:
            continue
        reviews_all.extend(reviews)
        mismatch_reviews = [review for review in reviews if not review.get("is_match", False)]
        if mismatch_reviews:
            reasons = ["behavior_mismatch"]
            object_issues = []
            for review in mismatch_reviews:
                bad = {
                    "reason": "behavior_mismatch",
                    "video_id": review["video_id"],
                    "json_id": review["json_id"],
                    "object_index": review["object_index"],
                    "object_name": review["object_name"],
                    "target_event": review["target_event"],
                    "agent_prompt": review["agent_prompt"],
                    "review_dir": review["review_dir"],
                    "review_json_path": str(Path(review["review_dir"]) / "review.json"),
                    "reason_text": review["reason"],
                }
                object_badcases.append(bad)
                object_issues.append(bad)
            group_badcases.append(
                {
                    "video_id": video_id,
                    "json_id": json_id,
                    "target_event": source.get("target_event"),
                    "batch_json_path": str(batch_json_path),
                    "agent_group_dir": str(agent_root / video_id / json_id),
                    "reasons": reasons,
                    "object_issues": object_issues,
                    "pair_issues": [],
                    "num_results": len(source.get("results", [])),
                }
            )

    report = {
        "batch_root": str(batch_root.resolve()),
        "agent_root": str(agent_root.resolve()),
        "review_root": str(review_root.resolve()),
        "keywords": keywords,
        "sample_stride": args.sample_stride,
        "total_reviewed_objects": len(reviews_all),
        "behavior_badcase_count": len(object_badcases),
        "group_badcases": group_badcases,
        "object_badcases": object_badcases,
        "reviews": reviews_all,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(
        json.dumps(
            {
                "total_reviewed_objects": len(reviews_all),
                "behavior_badcase_count": len(object_badcases),
                "bad_group_count": len(group_badcases),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"saved_behavior_review: {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
