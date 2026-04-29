#!/usr/bin/env python3
"""Run a single multi-image QA request against Ark using dataset video frames."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import io
from PIL import Image

# DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# DEFAULT_MODEL = "qwen3.5-plus-2026-02-15"
DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/coding/v3"
DEFAULT_MODEL = "doubao-seed-2.0-pro"
AUTO_DOWNSAMPLE_THRESHOLD = 70
AUTO_DOWNSAMPLE_STRIDE = 2
JSON_OUTPUT_INSTRUCTION = """## Output Format

You MUST output ONLY a valid JSON array. Do not include any additional explanations, titles, notes, or Markdown code blocks. 
ALL text content within the JSON (e.g., descriptions) MUST be in English.

Each element in the array must be a JSON object with the following structure:
{
  "object_name": "A unique name to identify the entity",
  "best_frame": 3,
  "discriminative_description": "A unique, discriminative description of the subject based on the global view of this specific second."
}

Additional Requirements:
1. `best_frame` must be an integer representing the selected second.
2. If multiple instances match the target event, `object_name` must explicitly distinguish them.
3. If no relevant entity is found, output an empty array `[]`.
4. The output must be valid JSON that can be parsed directly by `json.loads`."""

EVENT_PROMPT_TEMPLATE = """# Role
You are a top-tier expert in video content analysis and visual grounding. Your task is to precisely decompose specific events in a video, extract all relevant objects, and provide unique, discriminative descriptions in both time and space.

# Inputs
- Target Event: {target_event}

## Task Instructions

Please watch the video carefully and strictly follow these steps centered around the "Target Event":

1. Identify independent object instances: Analyze the target event and find the **core subject (head noun)** it refers to. 
   CRITICAL: Perform instance-level differentiation. Extract ONLY the core subject; DO NOT extract reference objects or modifiers. If different entities match the target event, treat them as separate objects. If an entity does not match the modifiers, ignore it.
2. Select the best frame: For each identified entity, evaluate all its appearances. Select ONLY ONE second where the entity's features are most prominent, unobstructed, and easy to recognize (must be an integer).
3. Write a discriminative description: Generate a unique and discriminative description for the entity based on the full scene of the selected second. Use explicit visual attributes (color, shape, material, clothing) and spatial relations (e.g., "top-left", "next to X") to ensure the object can be uniquely identified without ambiguity.
4. Instance Independence: Each discriminative_description MUST be strictly self-contained. NEVER reference other extracted objects (e.g., avoid 'below fish_1'). Instead, strictly rely on absolute spatial coordinates, fixed environmental anchors, or precise ordinal positions (e.g., 'the second fish from the top in the top-left quadrant') to ensure the object can be uniquely identified in isolation.
## Strict Constraints

1. Uniqueness and Instance Differentiation: Output only one record per independent physical entity. Do not duplicate extractions for the same entity.
2. Naming Convention: Use distinct `object_name` values for different instances of the same category.
3. Language: ALL output must be in **English**.
4. Time Format: Must be a unique integer second (e.g., 3, not "3s" or "2-4").
5. Description Requirement: Descriptions must be based on the global scene of the "selected second." Avoid vague terms.
6. Agency Filtering: Extract only the active agent(s) performing the action. In asymmetric interactions (e.g., "A pushing B"), exclude the patient (B). For collective actions (e.g., "Two lambs running"), extract all participants as separate instances.
"""

@dataclass
class RequestBundle:
    video_id: str
    prompt_text: str
    payload: dict[str, Any]
    selected_frames: list[tuple[int, Path]]
    selection_strategy: str
    total_frames: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build one Ark chat-completions request from dataset/JPEGImages/<video_id>. "
            "If a video has more than 70 frames, the script keeps every other frame "
            "while still labeling them [0.0 second], [1.0 second], ..."
        )
    )
    parser.add_argument("--video-id", required=True, help="Video id under dataset/JPEGImages")
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument(
        "--question",
        help="Direct prompt text to ask about the video",
    )
    prompt_group.add_argument(
        "--target-event",
        help="Fill the built-in event-analysis prompt template with this target event",
    )
    parser.add_argument(
        "--dataset-root",
        default="../../dataset/mevis/valid",
        help="Dataset root directory containing JPEGImages/",
    )
    parser.add_argument(
        "--response-format",
        choices=("json", "text"),
        default="json",
        help="json: append a strict JSON output contract; text: leave the prompt unchanged",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ark model name")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Ark API base URL")
    parser.add_argument(
        "--save-payload",
        help="Optional path to save the generated payload JSON before sending",
    )
    parser.add_argument(
        "--save-response",
        help="Optional path to save the normalized response JSON",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only build and preview the messages without calling the API",
    )
    return parser.parse_args()


def build_prompt_text(
    *,
    question: str | None = None,
    target_event: str | None = None,
    response_format: str = "json",
) -> str:
    if question:
        prompt_text = question
    elif target_event:
        prompt_text = EVENT_PROMPT_TEMPLATE.format(target_event=target_event)
    else:
        raise ValueError("Either question or target_event is required")

    if response_format == "json":
        prompt_text = f"{prompt_text}\n\n{JSON_OUTPUT_INSTRUCTION}"
    return prompt_text


def list_frames(video_dir: Path) -> list[Path]:
    frames = sorted(video_dir.glob("*.jpg"), key=lambda p: int(p.stem))
    if not frames:
        raise FileNotFoundError(f"No .jpg frames found in {video_dir}")
    return frames


# def select_frames(frames: list[Path]) -> tuple[list[tuple[int, Path]], str]:
#     if len(frames) > AUTO_DOWNSAMPLE_THRESHOLD:
#         selected = [(idx, frames[idx]) for idx in range(0, len(frames), AUTO_DOWNSAMPLE_STRIDE)]
#         strategy = f"every_{AUTO_DOWNSAMPLE_STRIDE}_frames (total>{AUTO_DOWNSAMPLE_THRESHOLD})"
#         return selected, strategy
#     return list(enumerate(frames)), "all_frames"
def select_frames(frames: list[Path], max_frames: int = 10) -> tuple[list[tuple[int, Path]], str]:
    total_frames = len(frames)
    if total_frames <= max_frames:
        # 如果视频很短，保留所有帧
        return list(enumerate(frames)), f"all_frames (total {total_frames})"

    # 均匀采样 (Uniform Sampling)
    # 比如 192 帧采 10 帧，会自动每隔 (192/10) 帧取一张
    indices = [i * (total_frames - 1) // (max_frames - 1) for i in range(max_frames)]
    selected = [(idx, frames[idx]) for idx in indices]

    strategy = f"uniform_sample_{max_frames}"
    return selected, strategy


def format_timestamp(sample_order: int) -> str:
    return f"[{float(sample_order):.1f} second]"


# def image_to_data_url(image_path: Path) -> str:
#     mime_type, _ = mimetypes.guess_type(image_path.name)
#     mime_type = mime_type or "image/jpeg"
#     encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
#     return f"data:{mime_type};base64,{encoded}"
def image_to_data_url(image_path: Path, max_size: int = 512) -> str:
    """读取图片并将其压缩到最大 max_size 像素，然后转为 base64"""
    with Image.open(image_path) as img:
        # thumbnail 会等比例缩小图片，保证最长边不超过 max_size
        img.thumbnail((max_size, max_size))

        # 将压缩后的图片写入内存
        buffer = io.BytesIO()
        # 适当降低 quality 也能减小体积，对视觉定位任务的语义理解影响不大
        img.save(buffer, format="JPEG", quality=85)

        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

def build_content(prompt_text: str, selected_frames: list[tuple[int, Path]]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt_text}]
    for sample_order, (_, frame_path) in enumerate(selected_frames):
        content.append({"type": "text", "text": format_timestamp(sample_order)})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_to_data_url(frame_path)},
                "detail": "low",
            }
        )
    return content


def prepare_request(
    *,
    video_id: str,
    dataset_root: str | Path,
    prompt_text: str,
    model: str = DEFAULT_MODEL,
) -> RequestBundle:
    dataset_root = Path(dataset_root)
    video_dir = dataset_root / "JPEGImages" / video_id
    if not video_dir.is_dir():
        raise FileNotFoundError(f"Video id {video_id!r} not found under {video_dir}")

    frames = list_frames(video_dir)
    selected_frames, strategy = select_frames(frames)
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": build_content(prompt_text, selected_frames),
            }
        ],
    }
    return RequestBundle(
        video_id=video_id,
        prompt_text=prompt_text,
        payload=payload,
        selected_frames=selected_frames,
        selection_strategy=strategy,
        total_frames=len(frames),
    )


def save_json(data: Any, path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def sanitize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    preview = {
        "model": payload["model"],
        "messages": [
            {
                "role": "user",
                "content": [],
            }
        ],
    }
    for item in payload["messages"][0]["content"]:
        if item["type"] == "image_url":
            preview["messages"][0]["content"].append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"<data_url length={len(item['image_url']['url'])}>"},
                }
            )
        else:
            preview["messages"][0]["content"].append(item)
    return preview


def preview_selected_frames(selected_frames: list[tuple[int, Path]]) -> None:
    print("selected_frames:")
    preview_limit = 8
    if len(selected_frames) <= preview_limit * 2:
        iterable = list(enumerate(selected_frames))
    else:
        iterable = list(enumerate(selected_frames[:preview_limit]))
        iterable.append((-1, (-1, Path("..."))))
        offset = len(selected_frames) - preview_limit
        iterable.extend((offset + i, item) for i, item in enumerate(selected_frames[-preview_limit:]))

    for sample_order, item in iterable:
        if sample_order == -1:
            print("  - ...")
            continue
        frame_index, frame_path = item
        print(f"  - {format_timestamp(sample_order)} original_frame={frame_index} path={frame_path}")


def preview_request(
    *,
    video_id: str,
    prompt_label: str,
    prompt_value: str,
    bundle: RequestBundle,
) -> None:
    print(f"video_id: {video_id}")
    print(f"{prompt_label}: {prompt_value}")
    print("prompt_text:")
    print(bundle.prompt_text)
    print(f"total_frames: {bundle.total_frames}")
    print(f"selection_strategy: {bundle.selection_strategy}")
    print(f"selected_images: {len(bundle.selected_frames)}")
    preview_selected_frames(bundle.selected_frames)
    print("\npayload_preview:")
    print(json.dumps(sanitize_payload(bundle.payload), ensure_ascii=False, indent=2))


def extract_answer(response: Any) -> str:
    choice = response.choices[0]
    message = choice.message
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for item in content:
            text = getattr(item, "text", None)
            if text:
                texts.append(text)
            elif isinstance(item, dict) and item.get("text"):
                texts.append(item["text"])
        return "\n".join(texts)
    return str(content)


def extract_json_block(text: str) -> Any:
    text = text.strip()
    if not text:
        raise ValueError("Empty response")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fence_markers = ("```json", "```JSON", "```")
    for marker in fence_markers:
        if marker in text:
            start = text.find(marker)
            body = text[start + len(marker) :]
            end = body.find("```")
            if end != -1:
                candidate = body[:end].strip()
                if candidate:
                    return json.loads(candidate)

    array_start = text.find("[")
    array_end = text.rfind("]")
    if array_start != -1 and array_end != -1 and array_end > array_start:
        return json.loads(text[array_start : array_end + 1])

    object_start = text.find("{")
    object_end = text.rfind("}")
    if object_start != -1 and object_end != -1 and object_end > object_start:
        return json.loads(text[object_start : object_end + 1])

    raise ValueError("No JSON object/array found in response")


def _coerce_best_frame(record: dict[str, Any]) -> int:
    if "best_frame" in record:
        value = record["best_frame"]
    elif "best_frame_time" in record:
        value = record["best_frame_time"]
    else:
        raise ValueError("Missing best_frame field")
    if isinstance(value, bool):
        raise ValueError("best_frame cannot be bool")
    return int(value)


def normalize_result_records(answer_text: str, selected_frames: list[tuple[int, Path]]) -> list[dict[str, Any]]:
    parsed = extract_json_block(answer_text)
    if isinstance(parsed, dict):
        if "results" in parsed and isinstance(parsed["results"], list):
            parsed = parsed["results"]
        else:
            raise ValueError("Expected a JSON array or an object with a results array")
    if not isinstance(parsed, list):
        raise ValueError("Expected a JSON array")

    normalized: list[dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, dict):
            raise ValueError("Each JSON item must be an object")
        record = dict(item)
        best_frame = _coerce_best_frame(record)
        if best_frame < 0 or best_frame >= len(selected_frames):
            raise ValueError(
                f"best_frame {best_frame} out of range for {len(selected_frames)} selected frames"
            )
        actual_frame_index, actual_frame_path = selected_frames[best_frame]
        record["best_frame"] = best_frame
        record["actual_frame_filename"] = actual_frame_path.name
        record["actual_frame_index"] = actual_frame_index
        normalized.append(record)
    return normalized


def build_result_payload(
    *,
    video_id: str,
    prompt_label: str,
    prompt_value: str,
    bundle: RequestBundle,
    results: list[dict[str, Any]] | None = None,
    raw_response_text: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "video_id": video_id,
        prompt_label: prompt_value,
        "selection_strategy": bundle.selection_strategy,
        "total_frames": bundle.total_frames,
        "selected_images": len(bundle.selected_frames),
    }
    if results is not None:
        payload["results"] = results
    if raw_response_text is not None:
        payload["raw_response_text"] = raw_response_text
    return payload


def format_response_text(
    answer_text: str,
    *,
    response_format: str,
    selected_frames: list[tuple[int, Path]],
) -> str:
    if response_format != "json":
        return answer_text
    parsed = normalize_result_records(answer_text, selected_frames)
    return json.dumps(parsed, ensure_ascii=False, indent=2)


def call_api(payload: dict[str, Any], base_url: str):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai package is required for non-dry-run mode") from exc

    api_key = (
            os.environ.get("ARK_API_KEY", "").strip()
            or os.environ.get("OPENAI_API_KEY", "").strip()
            or os.environ.get("VOLCES_API_KEY", "").strip()
    )
    if not api_key:
        raise RuntimeError("Set ARK_API_KEY, OPENAI_API_KEY, or DASHSCOPE_API_KEY before calling the API")

    client = OpenAI(base_url=base_url, api_key=api_key)
    return client.chat.completions.create(**payload)


def main() -> int:
    args = parse_args()
    prompt_label = "question" if args.question else "target_event"
    prompt_value = args.question if args.question else args.target_event
    prompt_text = build_prompt_text(
        question=args.question,
        target_event=args.target_event,
        response_format=args.response_format,
    )
    bundle = prepare_request(
        video_id=args.video_id,
        dataset_root=args.dataset_root,
        prompt_text=prompt_text,
        model=args.model,
    )

    if args.save_payload:
        save_json(bundle.payload, args.save_payload)
        print(f"saved payload to {args.save_payload}")

    preview_request(
        video_id=args.video_id,
        prompt_label=prompt_label,
        prompt_value=prompt_value,
        bundle=bundle,
    )
    if args.dry_run:
        return 0

    response = call_api(bundle.payload, args.base_url)
    answer = extract_answer(response)
    print("\nassistant_response:")
    try:
        print(
            format_response_text(
                answer,
                response_format=args.response_format,
                selected_frames=bundle.selected_frames,
            )
        )
    except Exception as exc:
        print(answer)
        print(f"\n[warning] failed to normalize JSON response: {exc}")

    if args.save_response:
        if args.response_format == "json":
            results = normalize_result_records(answer, bundle.selected_frames)
            result_payload = build_result_payload(
                video_id=args.video_id,
                prompt_label=prompt_label,
                prompt_value=prompt_value,
                bundle=bundle,
                results=results,
                raw_response_text=answer,
            )
        else:
            result_payload = build_result_payload(
                video_id=args.video_id,
                prompt_label=prompt_label,
                prompt_value=prompt_value,
                bundle=bundle,
                raw_response_text=answer,
            )
        save_json(result_payload, args.save_response)
        print(f"saved response to {args.save_response}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
