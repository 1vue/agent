#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import mimetypes
from pathlib import Path

from openai import OpenAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Call a local OpenAI-compatible model service for text or multimodal testing."
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000/v1",
        help="OpenAI-compatible base URL",
    )
    parser.add_argument(
        "--api-key",
        default="EMPTY",
        help="API key for the local service",
    )
    parser.add_argument(
        "--model",
        help="Model name. If omitted, the script uses the first model returned by /v1/models.",
    )
    parser.add_argument(
        "--prompt",
        default="你好，请用一句话介绍你自己。",
        help="User prompt",
    )
    parser.add_argument(
        "--image",
        help="Optional local image path for multimodal testing",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum completion tokens",
    )
    return parser.parse_args()


def image_to_data_url(image_path: str | Path) -> str:
    path = Path(image_path)
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def resolve_model_name(client: OpenAI, model: str | None) -> str:
    if model:
        return model
    models = client.models.list()
    if not models.data:
        raise RuntimeError("No models were returned by /v1/models")
    return models.data[0].id


def build_messages(prompt: str, image_path: str | None) -> list[dict]:
    if not image_path:
        return [{"role": "user", "content": prompt}]

    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": image_to_data_url(image_path), "detail": "high"},
                },
            ],
        }
    ]


def main() -> int:
    args = parse_args()
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    model_name = resolve_model_name(client, args.model)
    messages = build_messages(args.prompt, args.image)

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=args.max_tokens,
    )
    print(f"model: {model_name}")
    print(response.choices[0].message.content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
