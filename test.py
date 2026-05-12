#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import mimetypes
import socket
from pathlib import Path
from urllib.parse import urlparse

from openai import OpenAI

from api_config import add_api_profile_args, apply_api_profile_defaults
import os

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Call a local OpenAI-compatible model service for text or multimodal testing."
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

    args = parser.parse_args()
    return apply_api_profile_defaults(args)



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
    parsed = urlparse(args.base_url or "")
    host = parsed.hostname
    print(f"api_profile: {args.api_profile}")
    print(f"base_url: {args.base_url!r}")
    print(f"model: {args.model!r}")
    print(f"api_key_present: {bool(args.api_key)}")
    print(f"parsed_host: {host!r}")
    if not parsed.scheme or not host:
        raise ValueError(f"Invalid --base-url: {args.base_url!r}")
    try:
        print(f"dns_result: {socket.getaddrinfo(host, 443)[0]}")
    except OSError as exc:
        print(f"dns_error: {type(exc).__name__}: {exc}")
        raise

    client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=60.0)

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
