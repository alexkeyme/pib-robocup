"""Minimal LangGraph: one LLM node over MessagesState (add_messages)."""

import os
from collections.abc import Sequence
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


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


class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def _default_system() -> str | None:
    s = os.environ.get("CHAT_SYSTEM_PROMPT", "").strip()
    return s or None


def prepare_for_model(msgs: Sequence[BaseMessage]) -> list[BaseMessage]:
    """Prepend optional system prompt; use in graph and /chat/stream."""
    for m in msgs:
        if m.type == "system":
            return list(msgs)
    system = _default_system()
    if system:
        return [SystemMessage(content=system), *msgs]
    return list(msgs)


def build_graph():
    """Compile and return the chat graph (single model node)."""
    llm = get_llm()

    def call_model(state: State) -> dict:
        messages = prepare_for_model(state["messages"])
        response: AIMessage = llm.invoke(messages)  # type: ignore[assignment]
        return {"messages": [response]}

    graph = StateGraph(State)
    graph.add_node("model", call_model)
    graph.add_edge(START, "model")
    graph.add_edge("model", END)
    return graph.compile()
