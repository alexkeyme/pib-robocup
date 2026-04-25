"""LangChain tool: call local MolmoPoint `POST /point` (image_path + prompt -> points)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
from langchain_core.tools import tool

MOLMO_BASE_URL = os.environ.get("MOLMO_BASE_URL", "http://127.0.0.1:8010").rstrip("/")
MOLMO_TIMEOUT = float(os.environ.get("MOLMO_TIMEOUT_SECONDS", "120"))


def _allowed_prefix() -> str | None:
    raw = os.environ.get("MOLMO_ALLOWED_PATH_PREFIX", "").strip()
    return raw or None


def _validate_image_path(image_path: str) -> Path | str:
    """Return resolved Path if valid, else an error string for the model."""
    try:
        p = Path(image_path).expanduser().resolve(strict=False)
    except (OSError, ValueError) as e:
        return f"Invalid image_path: {e!s}"

    # Symlink/traversal: resolution handles most cases; require absolute-looking input after expand
    prefix = _allowed_prefix()
    if prefix is not None:
        base = Path(prefix).expanduser().resolve(strict=False)
        pr = p.resolve(strict=False)
        try:
            pr.relative_to(base.resolve(strict=False))
        except ValueError:
            return (
                f"image_path must be under MOLMO_ALLOWED_PATH_PREFIX={prefix!r} "
                f"(got {pr!s})"
            )

    if not p.is_file():
        return f"Not a file or not found: {p}"

    return p


def call_molmo_point(image_path: str, prompt: str) -> dict | str:
    """POST /point. Returns JSON dict on success, or error str."""
    v = _validate_image_path(image_path)
    if isinstance(v, str):
        return v

    body = {
        "image_path": str(v),
        "prompt": prompt,
    }
    try:
        with httpx.Client(timeout=MOLMO_TIMEOUT) as client:
            r = client.post(f"{MOLMO_BASE_URL}/point", json=body)
    except httpx.RequestError as e:
        return f"MolmoPoint unreachable ({MOLMO_BASE_URL}): {e!s}"

    if r.status_code >= 400:
        try:
            detail = r.json()
        except Exception:
            detail = r.text
        return f"MolmoPoint error HTTP {r.status_code}: {detail!s}"

    try:
        return r.json()
    except Exception as e:
        return f"Invalid JSON from MolmoPoint: {e!s}"


@tool
def molmo_point_localize(image_path: str, prompt: str) -> str:
    """Run MolmoPoint on a host-local image file to locate objects from a text description.

    Use this when the user needs 2D coordinates of objects in an image. The file must
    exist on the same machine as MolmoPoint; pass an absolute or resolvable path (e.g. a
    camera frame under /data). Returns JSON with a ``points`` list: object_id, image_index, x, y.
    If MolmoPoint is not running or the path is not allowed, the string explains the error.
    """
    out = call_molmo_point(image_path, prompt)
    if isinstance(out, str):
        return out
    # Compact payload for the LLM
    return json.dumps(
        {
            "points": out.get("points", []),
            "generated_text": out.get("generated_text", ""),
            "device": out.get("device", ""),
        },
        ensure_ascii=False,
    )


def molmo_service_reachable() -> bool:
    """True if GET /health returns 200 (MolmoPoint model is ready to serve /point)."""
    try:
        with httpx.Client(timeout=2.0) as c:
            r = c.get(f"{MOLMO_BASE_URL}/health")
            return r.status_code == 200
    except httpx.RequestError:
        return False
