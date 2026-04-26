"""Agent runtime via langchain.agents.create_agent (LangGraph under the hood; tools optional)."""

import os
from collections.abc import Sequence

from langchain.agents import create_agent
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI


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


def prepare_for_model(msgs: Sequence[BaseMessage]) -> list[BaseMessage]:
    """Prepend optional system prompt; same semantics for invoke and /chat/stream.

    `create_agent` is built with `system_prompt=None` so this remains the only place
    we merge `CHAT_SYSTEM_PROMPT` (unless the request already includes a system message).
    """
    for m in msgs:
        if m.type == "system":
            return list(msgs)
    system = _default_system()
    if system:
        return [SystemMessage(content=system), *msgs]
    return list(msgs)


def build_agent():
    """Return a compiled agent graph: model loop with optional tools; currently no tools.

    `tools=None` yields an agent with a model node and no tool-calling loop, matching
    the previous single-node StateGraph.
    """
    return create_agent(get_llm(), tools=None, system_prompt=None)
