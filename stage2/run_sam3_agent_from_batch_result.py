#!/usr/bin/env python3
"""Run the official SAM3 image agent on objects from a batch result JSON."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from functools import partial
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api_config import DEFAULT_BASE_URL, DEFAULT_MODEL, add_api_profile_args, apply_api_profile_defaults


DEFAULT_BATCH_ROOT = "outputs/batch_ark_video_qa"
DEFAULT_DATASET_ROOT = "../../dataset/mevis/valid"
# DEFAULT_DATASET_ROOT = "../../dataset/ref-youtube/valid"
DEFAULT_OUTPUT_ROOT = "agent_output"
DEFAULT_SAM3_OFFICIAL_ROOT = "sam3"
DEFAULT_CHECKPOINT = os.environ.get("SAM3_CHECKPOINT", "checkpoints/sam3.pt")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read one batch result JSON, locate its selected frame(s), and run the "
            "official SAM3 agent with an external OpenAI-compatible multimodal LLM."
        )
    )
    parser.add_argument("--video-id", required=True, help="Video id, e.g. 214dec75cbd0")
    parser.add_argument("--json-id", required=True, help="Batch-result JSON stem, e.g. 10")
    parser.add_argument(
        "--batch-root",
        default=DEFAULT_BATCH_ROOT,
        help="Directory containing outputs/batch_ark_video_qa/<video>/<json>.json",
    )
    parser.add_argument(
        "--dataset-root",
        default=DEFAULT_DATASET_ROOT,
        help="Dataset root directory containing JPEGImages/<video>/",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory to save agent outputs",
    )
    parser.add_argument(
        "--sam3-official-root",
        default=DEFAULT_SAM3_OFFICIAL_ROOT,
        help="Path to the official SAM3 codebase",
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="Path to the local SAM3 checkpoint",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device for SAM3, e.g. cuda, cuda:0, cpu",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for the SAM3 image processor",
    )
    add_api_profile_args(parser)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("SAM3_AGENT_BASE_URL"),
        help="OpenAI-compatible LLM base URL; defaults to selected API profile",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("SAM3_AGENT_MODEL"),
        help="LLM model name; defaults to selected API profile",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("SAM3_AGENT_API_KEY"),
        help="LLM API key; defaults to selected profile api_key_env",
    )
    parser.add_argument(
        "--extra-body-json",
        default=os.environ.get("SAM3_AGENT_EXTRA_BODY_JSON"),
        help="Optional JSON string passed to chat.completions.create as extra_body",
    )
    parser.add_argument(
        "--object-index",
        action="append",
        type=int,
        help="Optional 1-based object index from the JSON results list; may be passed multiple times",
    )
    parser.add_argument(
        "--prompt-source",
        choices=("discriminative_description", "object_name", "target_event"),
        default="discriminative_description",
        help="Which text source to use as the SAM3 agent prompt",
    )
    parser.add_argument(
        "--use-object-name",
        action="store_true",
        help="Deprecated alias for --prompt-source object_name",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable SAM3 agent debug outputs",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing pred.json outputs",
    )
    parser.add_argument(
        "--max-generations",
        type=int,
        default=10,
        help="Maximum number of MLLM generation rounds allowed inside the SAM3 agent",
    )
    args = parser.parse_args()
    return apply_api_profile_defaults(args)

def save_json(data: Any, path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

def safe_name(text: str) -> str:
    text = re.sub(r"[^\w\-\.]+", "_", text.strip(), flags=re.UNICODE)
    return text.strip("._") or "item"


def import_agent_modules(sam3_official_root: Path):
    root_str = str(sam3_official_root.resolve())
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    from sam3 import build_sam3_image_model
    from sam3.agent.agent_core import agent_inference
    from sam3.agent.client_llm import send_generate_request as send_generate_request_orig
    from sam3.agent.client_sam3 import call_sam_service as call_sam_service_orig
    from sam3.model.sam3_image_processor import Sam3Processor

    return (
        build_sam3_image_model,
        Sam3Processor,
        agent_inference,
        send_generate_request_orig,
        call_sam_service_orig,
    )


def load_result_json(batch_root: Path, video_id: str, json_id: str) -> dict[str, Any]:
    json_path = batch_root / video_id / f"{json_id}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Result JSON not found: {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)
    if "results" not in data or not isinstance(data["results"], list):
        raise ValueError(f"Invalid result JSON format: {json_path}")
    return data


def build_processor(
    build_sam3_image_model,
    Sam3Processor,
    sam3_official_root: Path,
    checkpoint: str,
    device: str,
    confidence_threshold: float,
):
    bpe_path = sam3_official_root / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    model = build_sam3_image_model(
        bpe_path=str(bpe_path),
        checkpoint_path=checkpoint,
        load_from_HF=False,
        device=device,
        eval_mode=True,
    )
    return Sam3Processor(model, confidence_threshold=confidence_threshold)


def select_result_items(
    result_json: dict[str, Any],
    selected_indices: set[int] | None,
) -> list[tuple[int, dict[str, Any]]]:
    selected: list[tuple[int, dict[str, Any]]] = []
    for idx, item in enumerate(result_json.get("results", []), start=1):
        if selected_indices and idx not in selected_indices:
            continue
        selected.append((idx, item))
    return selected


def main() -> int:
    args = parse_args()
    if not args.api_key:
        raise ValueError("Missing API key; pass --api-key or set the selected profile api_key_env")
    extra_body = json.loads(args.extra_body_json) if args.extra_body_json else None
    prompt_source = "object_name" if args.use_object_name else args.prompt_source

    batch_root = Path(args.batch_root)
    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)
    sam3_official_root = Path(args.sam3_official_root)

    result_json = load_result_json(batch_root, args.video_id, args.json_id)
    selected_indices = set(args.object_index) if args.object_index else None
    selected_items = select_result_items(result_json, selected_indices)
    if not selected_items:
        raise ValueError("No matching objects selected from the result JSON")

    (
        build_sam3_image_model,
        Sam3Processor,
        agent_inference,
        send_generate_request_orig,
        call_sam_service_orig,
    ) = import_agent_modules(sam3_official_root)

    processor = build_processor(
        build_sam3_image_model=build_sam3_image_model,
        Sam3Processor=Sam3Processor,
        sam3_official_root=sam3_official_root,
        checkpoint=args.checkpoint,
        device=args.device,
        confidence_threshold=args.confidence_threshold,
    )

    send_generate_request = partial(
        send_generate_request_orig,
        server_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        # extra_body=extra_body,
    )
    call_sam_service = partial(call_sam_service_orig, sam3_processor=processor)

    run_root = output_root / "sam3_agent" / safe_name(args.model) / args.video_id / str(args.json_id)
    run_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "video_id": args.video_id,
        "json_id": str(args.json_id),
        "target_event": result_json.get("target_event"),
        "llm_base_url": args.base_url,
        "llm_model": args.model,
        "llm_extra_body": extra_body,
        "prompt_source": prompt_source,
        "items": [],
    }

    for obj_idx, item in selected_items:
        object_name = item.get("object_name") or f"object_{obj_idx}"
        if prompt_source == "object_name":
            prompt = item.get("object_name")
        elif prompt_source == "target_event":
            prompt = result_json.get("target_event")
        else:
            prompt = item.get("discriminative_description")
        if not prompt:
            raise ValueError(f"Missing prompt text for object index {obj_idx}")

        frame_filename = item.get("actual_frame_filename")
        if not frame_filename:
            raise ValueError(f"Missing actual_frame_filename for object index {obj_idx}")

        image_path = dataset_root / "JPEGImages" / args.video_id / frame_filename
        if not image_path.exists():
            raise FileNotFoundError(f"Image frame not found: {image_path}")

        item_dir = run_root / f"{safe_name(object_name)}_obj_{obj_idx}"
        pred_json_path = item_dir / "pred.json"
        if pred_json_path.exists() and not args.overwrite:
            print(f"[skip] {pred_json_path} already exists")
            summary["items"].append(
                {
                    "object_index": obj_idx,
                    "object_name": object_name,
                    "prompt": prompt,
                    "image_path": str(image_path),
                    "output_dir": str(item_dir),
                    "status": "skipped_existing",
                }
            )
            continue

        item_dir.mkdir(parents=True, exist_ok=True)
        print(f"[run] object {obj_idx}: {object_name}")
        print(f"      image: {image_path}")
        print(f"      prompt: {prompt}")

        agent_history, final_output_dict, rendered_final_output = agent_inference(
            str(image_path),
            prompt,
            send_generate_request=send_generate_request,
            call_sam_service=call_sam_service,
            output_dir=str(item_dir),
            max_generations=args.max_generations,
            debug=args.debug,
        )

        final_output_dict = {
            **final_output_dict,
            "video_id": args.video_id,
            "json_id": str(args.json_id),
            "object_index": obj_idx,
            "object_name": object_name,
            "input_prompt": prompt,
            "source_result_item": item,
        }

        with open(pred_json_path, "w") as f:
            json.dump(final_output_dict, f, indent=2, ensure_ascii=False)
        with open(item_dir / "history.json", "w") as f:
            json.dump(agent_history, f, indent=2, ensure_ascii=False)
        rendered_final_output.save(item_dir / "pred.png")
        with open(item_dir / "input_meta.json", "w") as f:
            json.dump(
                {
                    "video_id": args.video_id,
                    "json_id": str(args.json_id),
                    "object_index": obj_idx,
                    "object_name": object_name,
                    "image_path": str(image_path),
                    "frame_filename": frame_filename,
                    "actual_frame_index": item.get("actual_frame_index"),
                    "prompt": prompt,
                    "target_event": result_json.get("target_event"),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        summary["items"].append(
            {
                "object_index": obj_idx,
                "object_name": object_name,
                "prompt": prompt,
                "image_path": str(image_path),
                "output_dir": str(item_dir),
                "num_masks": len(final_output_dict.get("pred_masks", [])),
                "status": "completed",
            }
        )

    summary_path = run_root / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[done] summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
