"""FastAPI: health, /chat, /chat/stream (SSE) — create_agent + local OpenAI-compatible Gemma (llama-server)."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from molmo_tool import molmo_service_reachable
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
    return {"message": {"role": "assistant", "content": text}}


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
