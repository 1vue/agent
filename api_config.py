from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_PATH = Path(__file__).resolve().with_name("api_profiles.json")


@dataclass(frozen=True)
class ApiProfile:
    name: str
    base_url: str
    model: str
    api_key_env: tuple[str, ...]


def _coerce_env_names(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list):
        return tuple(str(item) for item in value if str(item).strip())
    raise TypeError("api_key_env must be a string or a list of strings")


def resolve_api_key(
    profile: ApiProfile,
    explicit_api_key: str | None = None,
    fallback_env: tuple[str, ...] = (),
) -> str:
    if explicit_api_key:
        return explicit_api_key.strip()
    for env_name in (*profile.api_key_env, *fallback_env):
        value = os.environ.get(env_name, "").strip()
        if value:
            return value
    return ""


def add_api_profile_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--api-config",
        default=os.environ.get("API_CONFIG") or str(DEFAULT_CONFIG_PATH),
        help="Path to API profile config JSON",
    )
    parser.add_argument(
        "--api-profile",
        default=os.environ.get("API_PROFILE"),
        help="Profile name from API config; defaults to default_profile",
    )


def apply_api_profile_defaults(
    args: argparse.Namespace,
    *,
    fallback_key_env: tuple[str, ...] = (),
    require_api_key: bool = False,
) -> argparse.Namespace:
    profile = resolve_api_profile(args.api_profile, args.api_config)
    args.api_profile = profile.name
    if getattr(args, "base_url", None) is None:
        args.base_url = profile.base_url
    if getattr(args, "model", None) is None:
        args.model = profile.model
    if hasattr(args, "api_key"):
        args.api_key = resolve_api_key(profile, getattr(args, "api_key", None), fallback_key_env)
        if require_api_key and not args.api_key:
            env_names = ", ".join((*profile.api_key_env, *fallback_key_env))
            raise ValueError(f"Missing API key; pass --api-key or set one of: {env_names}")
    return args

def load_api_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path or os.environ.get("API_CONFIG") or DEFAULT_CONFIG_PATH)
    if not path.is_absolute():
        path = DEFAULT_CONFIG_PATH.parent / path
    if not path.exists():
        raise FileNotFoundError(f"API config file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))

def resolve_api_profile(
    profile_name: str | None = None,
    config_path: str | Path | None = None,
) -> ApiProfile:
    config = load_api_config(config_path)
    profiles = config.get("profiles", {})
    name = profile_name or os.environ.get("API_PROFILE") or config.get("default_profile")
    if not name:
        raise ValueError("Missing API profile name and default_profile in API config")
    if name not in profiles:
        available = ", ".join(sorted(profiles))
        raise ValueError(f"Unknown API profile {name!r}. Available profiles: {available}")

    item = profiles[name]
    return ApiProfile(
        name=name,
        base_url=str(item.get("base_url", "")).strip(),
        model=str(item.get("model", "")).strip(),
        api_key_env=_coerce_env_names(item.get("api_key_env")),
    )


DEFAULT_PROFILE = resolve_api_profile()
DEFAULT_BASE_URL = DEFAULT_PROFILE.base_url
DEFAULT_MODEL = DEFAULT_PROFILE.model
