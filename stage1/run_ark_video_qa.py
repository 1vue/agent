#!/usr/bin/env python3
"""Run a single multi-image QA request against Ark using dataset video frames."""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import io
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api_config import DEFAULT_BASE_URL, DEFAULT_MODEL, add_api_profile_args, apply_api_profile_defaults

AUTO_DOWNSAMPLE_THRESHOLD = 70
AUTO_DOWNSAMPLE_STRIDE = 2
JSON_OUTPUT_INSTRUCTION = """## Output Format

You MUST output ONLY a valid JSON array. Do not include any additional explanations, titles, notes, or Markdown code blocks. 
ALL text content within the JSON (e.g., descriptions) MUST be in English.

Each element in the array must be a JSON object with the following structure:
{
  "object_name": "A unique name to identify the entity",
  "best_frame": 3,
  "target_entity": "The object category that should be segmented, e.g. panda, bird, person",
  "target_role": "One of: agent, patient, reference_target, unknown",
  "action": "The core action or relation in the target event, or unknown",
  "segment_target": "The current instance to segment, not the whole group when targets are enumerated",
  "reference_entities": ["Objects used only to identify or locate the target"],
  "excluded_entities": ["Visible or mentioned objects that must NOT be segmented"],
  "discriminative_description": "A unique, discriminative description of the subject based on the global view of this specific selected frame."
}

Additional Requirements:
1. `best_frame` must be an integer representing the selected sampled frame.
2. If multiple instances match the target event, `object_name` must explicitly distinguish them.
3. If no relevant entity is found, output an empty array `[]`.
4. Copy `target_entity`, `target_role`, `action`, `reference_entities`, and `excluded_entities` from the provided semantic parse unless image evidence proves a more specific instance-level wording is needed.
5. For enumerated plural targets, `segment_target` must describe only the current physical instance, not the whole group expression. For example, use "the topmost left elongated striped fish" rather than "a group of six fish".
6. The output must be valid JSON that can be parsed directly by `json.loads`."""

SEMANTIC_PARSE_OUTPUT_INSTRUCTION = """You MUST output ONLY a valid JSON object. Do not include Markdown or explanations outside JSON.

The JSON object must use this structure:
{
  "target_entity": "The object category that should be segmented, e.g. panda, bird, person",
  "action": "The core action or relation, or unknown",
  "voice": "active, passive, relational, collective, or unknown",
  "agent": "The actor/causer if present, otherwise unknown",
  "patient": "The acted-upon entity if present, otherwise unknown",
  "target_role": "One of: agent, patient, reference_target, unknown",
  "segment_target": "The exact phrase or concise description of what should be segmented",
  "target_cardinality": {"type": "single, plural, exact, or unknown", "count": null},
  "target_set_description": "For plural/exact targets, describe the whole set to enumerate visually",
  "reference_entities": ["Entities mentioned only for locating or relating to the target"],
  "excluded_entities": ["Entities that are mentioned or likely visible but must NOT be segmented"],
  "requires_instance_enumeration": false,
  "decision_basis": "One short sentence explaining the grammatical cue, especially active/passive voice"
}

Rules:
1. Decide the entity denoted by the referring expression, not the visually most salient actor.
2. In active expressions like "a panda pushing another panda", the target is usually the grammatical subject/agent.
3. In passive or reduced-relative expressions like "a panda being pushed by another panda" or "the man chased by the dog", the target is the patient.
4. In "A holding B", segment A if the expression denotes A; in "B held by A", segment B.
5. For plural or counted targets such as "eight birds", set requires_instance_enumeration=true and target_cardinality accordingly. Do not invent visual instance names in this text-only step.
6. Keep decision_basis short; do not write a long chain of thought."""

SEMANTIC_PARSE_PROMPT_TEMPLATE = """# Role
You are a precise semantic parser for video referring expression segmentation.

# Referring Expression
{target_event}

# Few-shot Examples
Expression: "a panda pushing another panda"
Output: {{"target_entity":"panda","action":"pushing","voice":"active","agent":"a panda pushing","patient":"another panda","target_role":"agent","segment_target":"a panda pushing another panda","target_cardinality":{{"type":"single","count":null}},"target_set_description":"","reference_entities":["another panda"],"excluded_entities":["another panda being pushed"],"requires_instance_enumeration":false,"decision_basis":"Active construction: the referred panda is the pushing agent."}}

Expression: "a panda being pushed by another panda"
Output: {{"target_entity":"panda","action":"pushed","voice":"passive","agent":"another panda","patient":"a panda being pushed","target_role":"patient","segment_target":"a panda being pushed by another panda","target_cardinality":{{"type":"single","count":null}},"target_set_description":"","reference_entities":["another panda"],"excluded_entities":["the panda doing the pushing"],"requires_instance_enumeration":false,"decision_basis":"Passive construction with 'being pushed by': the referred panda is the patient."}}

Expression: "the man chased by the dog"
Output: {{"target_entity":"man","action":"chased","voice":"passive","agent":"the dog","patient":"the man","target_role":"patient","segment_target":"the man chased by the dog","target_cardinality":{{"type":"single","count":null}},"target_set_description":"","reference_entities":["the dog"],"excluded_entities":["the dog"],"requires_instance_enumeration":false,"decision_basis":"Reduced passive relation: the man is the entity being chased."}}

Expression: "eight birds flying in the sky"
Output: {{"target_entity":"bird","action":"flying","voice":"collective","agent":"eight birds","patient":"unknown","target_role":"agent","segment_target":"eight birds flying in the sky","target_cardinality":{{"type":"exact","count":8}},"target_set_description":"all eight birds flying in the sky","reference_entities":["sky"],"excluded_entities":["non-bird objects","birds not flying if visible"],"requires_instance_enumeration":true,"decision_basis":"Counted plural expression: all eight flying birds are targets."}}

Now parse the referring expression above.

{output_instruction}"""

EVENT_PROMPT_TEMPLATE = """# Role
You are a top-tier expert in video content analysis and visual grounding. Your task is to ground a pre-parsed referring expression in sampled video frames.

# Inputs
- Target Event: {target_event}
- Semantic Parse:
{semantic_parse_json}

## Task Instructions

Please watch the video carefully and strictly follow these steps centered around the "Target Event":

1. Ground ONLY the entity or entity set described by the semantic parse. Do not change the parsed target role.
2. If `target_role` is `patient`, do NOT switch to the active agent even if the active agent is more visually salient.
3. If `requires_instance_enumeration` is true, enumerate each visible matching physical instance as a separate JSON item. For an exact count, output up to that many matching instances if visible in the sampled frames. Each item is one physical target only; do not describe or ask downstream segmentation to select the whole group.
4. Select the single best frame for each target instance. It must support the parsed semantic target, but segmentation quality is also critical: prefer a frame where the target is as clear, sharp, large, complete, and unobstructed as possible, with a clean visible outline and minimal overlap with same-class objects. Do not choose a semantically dramatic frame if the target is blurry, tiny, truncated, heavily occluded, or merged with another object.
5. Write a discriminative description based on the selected frame. For plural or counted targets, each description must identify exactly one instance and must not refer to the group as the segmentation target.
6. If the target interacts with another same-class object, explicitly state which one to segment and which one to exclude using relative position, pose, and role, e.g. "the upper panda leaning over and pushing, not the lower panda lying underneath."
## Strict Constraints

1. Uniqueness and Instance Differentiation: Output only one record per independent physical entity. Do not duplicate extractions for the same entity.
2. Naming Convention: Use distinct `object_name` values for different instances of the same category.
3. Language: ALL output must be in **English**.
4. Frame Index Format: `best_frame` must be an integer sampled frame index (e.g., 3, not "3s" or "2-4").
5. Description Requirement: Descriptions must be based on the global scene of the selected frame. Avoid vague terms.
6. Instance Constraint: Each JSON item must correspond to exactly one physical target instance. For counted groups, do not output an item whose `segment_target` is the whole group.
7. Exclusion Constraint: Never segment or describe `excluded_entities` as targets. They may be used only as references for locating the parsed target.
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
    args = parser.parse_args()
    return apply_api_profile_defaults(
        args,
        fallback_key_env=("ARK_API_KEY", "OPENAI_API_KEY", "VOLCES_API_KEY", "DASHSCOPE_API_KEY"),
    )


def build_prompt_text(
    *,
    question: str | None = None,
    target_event: str | None = None,
    semantic_parse: dict[str, Any] | None = None,
    response_format: str = "json",
) -> str:
    if question:
        prompt_text = question
    elif target_event:
        semantic_parse_json = json.dumps(
            semantic_parse or build_fallback_semantic_parse(target_event),
            ensure_ascii=False,
            indent=2,
        )
        prompt_text = EVENT_PROMPT_TEMPLATE.format(
            target_event=target_event,
            semantic_parse_json=semantic_parse_json,
        )
    else:
        raise ValueError("Either question or target_event is required")

    if response_format == "json":
        prompt_text = f"{prompt_text}\n\n{JSON_OUTPUT_INSTRUCTION}"
    return prompt_text


def build_fallback_semantic_parse(target_event: str) -> dict[str, Any]:
    return {
        "target_entity": "unknown",
        "action": "unknown",
        "voice": "unknown",
        "agent": "unknown",
        "patient": "unknown",
        "target_role": "unknown",
        "segment_target": target_event,
        "target_cardinality": {"type": "unknown", "count": None},
        "target_set_description": "",
        "reference_entities": [],
        "excluded_entities": [],
        "requires_instance_enumeration": True,
        "decision_basis": "Fallback parse because no semantic parse was available.",
    }


def build_semantic_parse_prompt(target_event: str) -> str:
    return SEMANTIC_PARSE_PROMPT_TEMPLATE.format(
        target_event=target_event,
        output_instruction=SEMANTIC_PARSE_OUTPUT_INSTRUCTION,
    )


def build_semantic_parse_payload(
    *,
    target_event: str,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_semantic_parse_prompt(target_event)}
                ],
            }
        ],
    }


def _frame_sort_key(frame_path: Path) -> tuple[int, Any]:
    try:
        return (0, int(frame_path.stem))
    except ValueError:
        return (1, frame_path.stem)


def list_frames(video_dir: Path, frame_ids: list[str] | None = None) -> list[Path]:
    if frame_ids:
        frames = []
        for frame_id in frame_ids:
            frame_name = str(frame_id)
            frame_path = video_dir / frame_name
            if frame_path.suffix.lower() != ".jpg":
                frame_path = video_dir / f"{frame_name}.jpg"
            if not frame_path.exists():
                raise FileNotFoundError(f"Frame listed in meta was not found: {frame_path}")
            frames.append(frame_path)
        return frames

    frames = sorted(video_dir.glob("*.jpg"), key=_frame_sort_key)
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

        img = img.convert("RGB")
        buffer = io.BytesIO()
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
    frame_ids: list[str] | None = None,
    max_sampled_frames: int = 10,
) -> RequestBundle:
    dataset_root = Path(dataset_root)
    video_dir = dataset_root / "JPEGImages" / video_id
    if not video_dir.is_dir():
        raise FileNotFoundError(f"Video id {video_id!r} not found under {video_dir}")

    frames = list_frames(video_dir, frame_ids=frame_ids)
    selected_frames, strategy = select_frames(frames, max_frames=max_sampled_frames)
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


def _coerce_string(value: Any, default: str = "") -> str:
    text = str(value if value is not None else default).strip()
    return text or default


def _coerce_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _normalize_target_role(value: Any) -> str:
    role = str(value or "unknown").strip().lower()
    allowed = {"agent", "patient", "reference_target", "unknown"}
    return role if role in allowed else "unknown"


def _normalize_voice(value: Any) -> str:
    voice = str(value or "unknown").strip().lower()
    allowed = {"active", "passive", "relational", "collective", "unknown"}
    return voice if voice in allowed else "unknown"


def _normalize_cardinality(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"type": "unknown", "count": None}
    card_type = str(value.get("type") or "unknown").strip().lower()
    if card_type not in {"single", "plural", "exact", "unknown"}:
        card_type = "unknown"
    count = value.get("count")
    if isinstance(count, bool):
        count = None
    elif count is not None:
        try:
            count = int(count)
        except (TypeError, ValueError):
            count = None
    return {"type": card_type, "count": count}


def normalize_semantic_parse(parsed: Any, target_event: str) -> dict[str, Any]:
    if not isinstance(parsed, dict):
        raise ValueError("Expected semantic parse to be a JSON object")
    fallback = build_fallback_semantic_parse(target_event)
    normalized = {
        "target_entity": _coerce_string(parsed.get("target_entity"), fallback["target_entity"]),
        "action": _coerce_string(parsed.get("action"), "unknown"),
        "voice": _normalize_voice(parsed.get("voice")),
        "agent": _coerce_string(parsed.get("agent"), "unknown"),
        "patient": _coerce_string(parsed.get("patient"), "unknown"),
        "target_role": _normalize_target_role(parsed.get("target_role")),
        "segment_target": _coerce_string(parsed.get("segment_target"), target_event),
        "target_cardinality": _normalize_cardinality(parsed.get("target_cardinality")),
        "target_set_description": _coerce_string(parsed.get("target_set_description"), ""),
        "reference_entities": _coerce_string_list(parsed.get("reference_entities")),
        "excluded_entities": _coerce_string_list(parsed.get("excluded_entities")),
        "requires_instance_enumeration": bool(parsed.get("requires_instance_enumeration")),
        "decision_basis": _coerce_string(parsed.get("decision_basis"), ""),
    }
    if normalized["target_cardinality"]["type"] in {"plural", "exact"}:
        normalized["requires_instance_enumeration"] = True
    if not normalized["target_set_description"] and normalized["requires_instance_enumeration"]:
        normalized["target_set_description"] = normalized["segment_target"]
    return normalized


def parse_semantic_answer(answer_text: str, target_event: str) -> dict[str, Any]:
    return normalize_semantic_parse(extract_json_block(answer_text), target_event)


def apply_semantic_parse_to_record(record: dict[str, Any], semantic_parse: dict[str, Any] | None) -> None:
    if not semantic_parse:
        return
    for key in ("target_entity", "target_role", "action"):
        value = record.get(key)
        if value is None or not str(value).strip():
            record[key] = semantic_parse.get(key)
    if not str(record.get("segment_target") or "").strip():
        if semantic_parse.get("requires_instance_enumeration"):
            record["segment_target"] = (
                record.get("discriminative_description")
                or record.get("object_name")
                or semantic_parse.get("segment_target")
            )
        else:
            record["segment_target"] = semantic_parse.get("segment_target")
    if "target_cardinality" not in record:
        record["target_cardinality"] = semantic_parse.get("target_cardinality")
    if "target_set_description" not in record:
        record["target_set_description"] = semantic_parse.get("target_set_description")
    if "requires_instance_enumeration" not in record:
        record["requires_instance_enumeration"] = semantic_parse.get("requires_instance_enumeration")
    for key in ("reference_entities", "excluded_entities"):
        if not _coerce_string_list(record.get(key)):
            record[key] = list(semantic_parse.get(key) or [])
        else:
            record[key] = _coerce_string_list(record.get(key))
    record["target_role"] = _normalize_target_role(record.get("target_role"))


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


def normalize_result_records(
    answer_text: str,
    selected_frames: list[tuple[int, Path]],
    semantic_parse: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
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
        apply_semantic_parse_to_record(record, semantic_parse)
        normalized.append(record)
    return normalized


def build_result_payload(
    *,
    video_id: str,
    prompt_label: str,
    prompt_value: str,
    bundle: RequestBundle,
    semantic_parse: dict[str, Any] | None = None,
    results: list[dict[str, Any]] | None = None,
    raw_semantic_response_text: str | None = None,
    raw_response_text: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "video_id": video_id,
        prompt_label: prompt_value,
        "selection_strategy": bundle.selection_strategy,
        "total_frames": bundle.total_frames,
        "selected_images": len(bundle.selected_frames),
    }
    if semantic_parse is not None:
        payload["semantic_parse"] = semantic_parse
    if results is not None:
        payload["results"] = results
    if raw_semantic_response_text is not None:
        payload["raw_semantic_response_text"] = raw_semantic_response_text
    if raw_response_text is not None:
        payload["raw_response_text"] = raw_response_text
    return payload


def format_response_text(
    answer_text: str,
    *,
    response_format: str,
    selected_frames: list[tuple[int, Path]],
    semantic_parse: dict[str, Any] | None = None,
) -> str:
    if response_format != "json":
        return answer_text
    parsed = normalize_result_records(answer_text, selected_frames, semantic_parse)
    return json.dumps(parsed, ensure_ascii=False, indent=2)


def call_api(payload: dict[str, Any], base_url: str, api_key: str | None = None):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai package is required for non-dry-run mode") from exc

    resolved_api_key = api_key or (
            os.environ.get("ARK_API_KEY", "").strip()
            or os.environ.get("OPENAI_API_KEY", "").strip()
            or os.environ.get("VOLCES_API_KEY", "").strip()
            or os.environ.get("DASHSCOPE_API_KEY", "").strip()
            or os.environ.get("XIAOMI_API_KEY", "").strip()
            or os.environ.get("XUNFEI_API_KEY", "").strip()
    )
    if not resolved_api_key:
        raise RuntimeError("Pass --api-key or set the api_key_env used by the selected API profile")

    client = OpenAI(base_url=base_url, api_key=resolved_api_key)
    return client.chat.completions.create(**payload)


def main() -> int:
    args = parse_args()
    prompt_label = "question" if args.question else "target_event"
    prompt_value = args.question if args.question else args.target_event
    semantic_parse: dict[str, Any] | None = None
    raw_semantic_answer: str | None = None
    if args.target_event:
        if args.dry_run:
            semantic_parse = build_fallback_semantic_parse(args.target_event)
        else:
            semantic_payload = build_semantic_parse_payload(
                target_event=args.target_event,
                model=args.model,
            )
            semantic_response = call_api(semantic_payload, args.base_url, args.api_key)
            raw_semantic_answer = extract_answer(semantic_response)
            semantic_parse = parse_semantic_answer(raw_semantic_answer, args.target_event)
            print("semantic_parse:")
            print(json.dumps(semantic_parse, ensure_ascii=False, indent=2))

    prompt_text = build_prompt_text(
        question=args.question,
        target_event=args.target_event,
        semantic_parse=semantic_parse,
        response_format=args.response_format,
    )
    bundle = prepare_request(
        video_id=args.video_id,
        dataset_root=args.dataset_root,
        prompt_text=prompt_text,
        model=args.model,
        max_sampled_frames=args.max_sampled_frames,
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

    response = call_api(bundle.payload, args.base_url, args.api_key)
    answer = extract_answer(response)
    print("\nassistant_response:")
    try:
        print(
            format_response_text(
                answer,
                response_format=args.response_format,
                selected_frames=bundle.selected_frames,
                semantic_parse=semantic_parse,
            )
        )
    except Exception as exc:
        print(answer)
        print(f"\n[warning] failed to normalize JSON response: {exc}")

    if args.save_response:
        if args.response_format == "json":
            results = normalize_result_records(answer, bundle.selected_frames, semantic_parse)
            result_payload = build_result_payload(
                video_id=args.video_id,
                prompt_label=prompt_label,
                prompt_value=prompt_value,
                bundle=bundle,
                semantic_parse=semantic_parse,
                results=results,
                raw_semantic_response_text=raw_semantic_answer,
                raw_response_text=answer,
            )
        else:
            result_payload = build_result_payload(
                video_id=args.video_id,
                prompt_label=prompt_label,
                prompt_value=prompt_value,
                bundle=bundle,
                semantic_parse=semantic_parse,
                raw_semantic_response_text=raw_semantic_answer,
                raw_response_text=answer,
            )
        save_json(result_payload, args.save_response)
        print(f"saved response to {args.save_response}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
