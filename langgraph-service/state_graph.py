"""Agent runtime via langchain.agents.create_agent (LangGraph under the hood; tools optional)."""

import json
import os
from collections.abc import Sequence

from langchain.agents import create_agent
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from molmo_tool import call_molmo_point, molmo_point_localize


def get_llm() -> ChatOpenAI:
    base_url = os.environ.get("GEMMA_BASE_URL", "http://127.0.0.1:8080/v1")
    model = os.environ.get("GEMMA_OPENAI_MODEL", "gpt-3.5-turbo")
    api_key = os.environ.get("OPENAI_API_KEY", "not-needed")
    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=0.7,
    )


def _default_system() -> str | None:
    s = os.environ.get("CHAT_SYSTEM_PROMPT", "").strip()
    return s or None


def _uploaded_image_system(uploaded_image_path: str) -> SystemMessage:
    return SystemMessage(
        content=(
            "A user image is attached for this turn and stored on the server at "
            f"{uploaded_image_path}. When image understanding is needed, call tool "
            "`molmo_point_localize_uploaded` with a precise prompt that describes what to locate. "
            "Do not invent coordinates or tool outputs."
        )
    )


def prepare_for_model(
    msgs: Sequence[BaseMessage], uploaded_image_path: str | None = None
) -> list[BaseMessage]:
    """Prepend optional system prompt; same semantics for invoke and /chat/stream.

    `create_agent` is built with `system_prompt=None` so this remains the only place
    we merge `CHAT_SYSTEM_PROMPT` (unless the request already includes a system message).
    """
    for m in msgs:
        if m.type == "system":
            out = list(msgs)
            if uploaded_image_path:
                out.insert(0, _uploaded_image_system(uploaded_image_path))
            return out
    system = _default_system()
    out: list[BaseMessage] = []
    if system:
        out.append(SystemMessage(content=system))
    if uploaded_image_path:
        out.append(_uploaded_image_system(uploaded_image_path))
    out.extend(msgs)
    return out


def _build_uploaded_image_tool(uploaded_image_path: str):
    @tool("molmo_point_localize_uploaded")
    def molmo_point_localize_uploaded(prompt: str) -> str:
        """Run MolmoPoint on the image uploaded in this chat turn."""
        out = call_molmo_point(uploaded_image_path, prompt)
        if isinstance(out, str):
            return out
        return json.dumps(
            {
                "points": out.get("points", []),
                "generated_text": out.get("generated_text", ""),
                "device": out.get("device", ""),
                "model_id": out.get("model_id", ""),
            },
            ensure_ascii=False,
        )

    return molmo_point_localize_uploaded


def build_agent(uploaded_image_path: str | None = None):
    """Return a compiled agent graph: Gemma + MolmoPoint localization tool (HTTP to :8010)."""
    tools = [molmo_point_localize]
    if uploaded_image_path:
        tools.append(_build_uploaded_image_tool(uploaded_image_path))
    return create_agent(
        get_llm(),
        tools=tools,
        system_prompt=None,
    )
