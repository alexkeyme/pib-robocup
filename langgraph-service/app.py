"""FastAPI: health, /chat, /chat/stream (SSE) — create_agent + local OpenAI-compatible Gemma (llama-server)."""

from __future__ import annotations

import json
import mimetypes
import os
import urllib.error
import urllib.request
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    ToolMessageChunk,
)
from pydantic import BaseModel, Field

from molmo_tool import call_molmo_point, molmo_service_reachable
from state_graph import build_agent, prepare_for_model

GEMMA_HEALTH_URLS = [
    f"http://127.0.0.1:{os.environ.get('GEMMA_PORT', '8080')}/health",
    f"http://127.0.0.1:{os.environ.get('GEMMA_PORT', '8080')}/",
]


def _cors_origins() -> list[str]:
    o = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
    for part in os.environ.get("CORS_EXTRA_ORIGINS", "").split(","):
        p = part.strip()
        if p:
            o.append(p)
    return o


app = FastAPI(title="LangGraph + local Gemma", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_origin_regex=None,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_agent: Any = None


def get_agent():
    global _agent
    if _agent is None:
        _agent = build_agent()
    return _agent


class Msg(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Msg] = Field(min_length=1)


MOLMO_MAX_UPLOAD_BYTES = int(os.environ.get("MOLMO_MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))
MOLMO_ALLOWED_FILE_TYPES: dict[str, str] = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}


def _molmo_upload_dir() -> Path:
    raw = os.environ.get("MOLMO_UPLOAD_DIR", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (Path(__file__).resolve().parent.parent / "run" / "molmo-uploads").resolve()


def _gemma_reachable() -> bool:
    for url in GEMMA_HEALTH_URLS:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=2) as r:
                if 200 <= r.getcode() < 500:
                    return True
        except (urllib.error.URLError, OSError, TimeoutError):
            continue
    return False


def _to_lc_messages(items: list[Msg]) -> list[BaseMessage]:
    out: list[BaseMessage] = []
    for m in items:
        if m.role == "user":
            out.append(HumanMessage(content=m.content))
        elif m.role == "assistant":
            out.append(AIMessage(content=m.content))
        elif m.role == "system":
            out.append(SystemMessage(content=m.content))
        else:
            raise HTTPException(status_code=400, detail=f"Unknown role: {m.role!r}")
    return out


MOLMO_TOOL_NAME = "molmo_point_localize"


def _parse_molmo_tool_content(content: Any) -> dict[str, Any] | None:
    """Parse JSON or plain error string from `molmo_point_localize` tool return."""
    if not isinstance(content, str):
        return None
    s = content.strip()
    if s.startswith("{"):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return {"error": s}
    return {"error": s}


def _molmo_from_tool_message(m: ToolMessage) -> dict[str, Any] | None:
    if m.name != MOLMO_TOOL_NAME:
        return None
    return _parse_molmo_tool_content(m.content)


def _is_molmo_tool_result_dict(p: dict[str, Any]) -> bool:
    """Distinguish molmo_point_localize JSON from other tools when `name` is missing."""
    if "points" in p and isinstance(p.get("points"), list):
        return True
    if "generated_text" in p:
        return True
    if "error" in p and len(p) == 1:
        return True
    return False


def _try_complete_molmo_json(s: str) -> dict[str, Any] | None:
    s = s.strip()
    if not s.startswith("{"):
        if s:
            return _parse_molmo_tool_content(s)
        return None
    try:
        json.loads(s)
    except json.JSONDecodeError:
        return None
    return _parse_molmo_tool_content(s)


def _collect_molmo_results(messages: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if isinstance(m, ToolMessage) and m.name == MOLMO_TOOL_NAME:
            p = _molmo_from_tool_message(m)
            if p is not None:
                out.append(p)
    return out


def _last_assistant_text(messages: list[Any]) -> str:
    """Last AIMessage with user-visible text, skipping tool-only assistant turns if present."""
    for m in reversed(messages):
        if not isinstance(m, AIMessage):
            continue
        if getattr(m, "tool_calls", None):
            continue
        c = m.content
        if isinstance(c, str) and c:
            return c
        if c is not None and c not in ("",):
            return str(c)
    for m in reversed(messages):
        if isinstance(m, AIMessage) and m.content is not None:
            return m.content if isinstance(m.content, str) else str(m.content)
    raise ValueError("No assistant content in result")


@app.get("/health")
def health():
    return {
        "ok": True,
        "gemma": _gemma_reachable(),
        "molmo": molmo_service_reachable(),
    }


@app.post("/molmo/localize")
async def molmo_localize(
    file: UploadFile = File(...),
    prompt: str = Form(...),
):
    """Save an uploaded image to disk, run MolmoPoint, return points (and delete the temp file)."""
    p = (prompt or "").strip()
    if not p:
        raise HTTPException(status_code=400, detail="prompt is required")
    if len(p) > 4000:
        raise HTTPException(status_code=400, detail="prompt too long")

    body = b""
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        body += chunk
        if len(body) > MOLMO_MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="file too large")

    if not body:
        raise HTTPException(status_code=400, detail="empty file")

    ct = (file.content_type or "").split(";")[0].strip() or mimetypes.guess_type(
        file.filename or ""
    )[0]
    if not ct or ct not in MOLMO_ALLOWED_FILE_TYPES:
        raise HTTPException(
            status_code=400,
            detail="unsupported or missing image type; use jpeg, png, or webp",
        )
    ext = MOLMO_ALLOWED_FILE_TYPES[ct]
    dest_dir = _molmo_upload_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{uuid.uuid4().hex}{ext}"
    try:
        dest.write_bytes(body)
        out = call_molmo_point(str(dest), p)
    finally:
        try:
            dest.unlink(missing_ok=True)
        except OSError:
            pass

    if isinstance(out, str):
        raise HTTPException(status_code=502, detail=out)

    return {
        "ok": True,
        "points": out.get("points", []),
        "generated_text": out.get("generated_text", ""),
        "device": out.get("device", ""),
        "model_id": out.get("model_id", ""),
    }


@app.post("/chat")
def chat_post(req: ChatRequest):
    try:
        lc = _to_lc_messages(req.messages)
    except HTTPException:
        raise
    msgs = prepare_for_model(lc)
    try:
        out = get_agent().invoke({"messages": msgs})
    except Exception as e:
        raise HTTPException(
            status_code=502, detail=f"Gemma/LangGraph error: {e!s}"
        ) from e
    if not isinstance(out, dict) or "messages" not in out:
        raise HTTPException(status_code=502, detail="Unexpected agent result shape")
    try:
        text = _last_assistant_text(out["messages"])
    except ValueError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    molmo_results = _collect_molmo_results(out["messages"])
    base: dict[str, Any] = {"message": {"role": "assistant", "content": text}}
    if molmo_results:
        base["molmo_results"] = molmo_results
    return base


def _token_text(chunk: Any) -> str | None:
    c = getattr(chunk, "content", None)
    if isinstance(c, str) and c:
        return c
    return None


async def _stream_body(req: ChatRequest) -> AsyncIterator[str]:
    try:
        lc = _to_lc_messages(req.messages)
    except HTTPException as e:
        err = json.dumps({"error": e.detail})
        yield f"data: {err}\n\n"
        return
    msgs = prepare_for_model(lc)
    agent = get_agent()
    molmo_tmc_merged: dict[str, ToolMessageChunk] = {}
    molmo_sent: set[str] = set()
    try:
        async for item in agent.astream(
            {"messages": msgs},
            stream_mode="messages",
        ):
            chunk: Any
            if isinstance(item, tuple) and len(item) == 2:
                chunk, _ = item
            else:
                chunk = item
            if isinstance(chunk, ToolMessageChunk):
                if chunk.name and chunk.name != MOLMO_TOOL_NAME:
                    pass
                else:
                    tid = chunk.tool_call_id
                    prev = molmo_tmc_merged.get(tid)
                    merged = (prev + chunk) if prev is not None else chunk
                    molmo_tmc_merged[tid] = merged
                    nm = merged.name
                    c = merged.content
                    if (nm == MOLMO_TOOL_NAME or (nm is None and isinstance(c, str))):
                        p = _try_complete_molmo_json(c) if isinstance(c, str) else None
                        if (
                            p is not None
                            and (nm == MOLMO_TOOL_NAME or _is_molmo_tool_result_dict(p))
                            and tid not in molmo_sent
                        ):
                            molmo_sent.add(tid)
                            line = json.dumps({"molmo_result": p})
                            yield f"data: {line}\n\n"
            if isinstance(chunk, ToolMessage) and not isinstance(
                chunk, ToolMessageChunk
            ):
                tid = chunk.tool_call_id
                if tid in molmo_sent:
                    pass
                else:
                    molmo = _molmo_from_tool_message(chunk)
                    if molmo is not None:
                        molmo_sent.add(tid)
                        line = json.dumps({"molmo_result": molmo})
                        yield f"data: {line}\n\n"
            if isinstance(chunk, (ToolMessage, ToolMessageChunk)):
                continue
            text = _token_text(chunk)
            if text:
                line = json.dumps({"token": text})
                yield f"data: {line}\n\n"
    except Exception as e:
        err = json.dumps({"error": str(e)})
        yield f"data: {err}\n\n"
        return
    yield f"data: {json.dumps({'done': True})}\n\n"


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    return StreamingResponse(
        _stream_body(req),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )
