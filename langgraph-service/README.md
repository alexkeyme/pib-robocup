# LangGraph service

FastAPI app (`app.py`) runs an agent built with **`langchain.agents.create_agent`**, which compiles to a LangGraph `CompiledStateGraph`. The chat model is **`langchain_openai.ChatOpenAI`** against the local **llama.cpp** `llama-server` (Gemma 4) OpenAI-compatible API.

- **No tools today:** `create_agent(get_llm(), tools=None, system_prompt=None)` — same effective behavior as the old single-node `StateGraph` (model-only, no tool loop).
- **Later:** pass a list of tools to `create_agent(..., tools=[...])` in `state_graph.py` for ReAct-style tool use without replacing the runtime.

**System prompt:** `CHAT_SYSTEM_PROMPT` is applied in `prepare_for_model()` in `state_graph.py` (only if the request has no `system` message). `create_agent` is called with `system_prompt=None` so this logic stays in one place.

**Streaming:** `POST /chat/stream` uses the **agent**’s `astream(..., stream_mode="messages")` (not a bare `ChatOpenAI.astream`), so behavior stays aligned when tools are enabled later.

Environment variables (optional):

| Variable | Default | Meaning |
|----------|---------|--------|
| `GEMMA_BASE_URL` | `http://127.0.0.1:8080/v1` | OpenAI API root |
| `GEMMA_OPENAI_MODEL` | `gpt-3.5-turbo` | `model` field sent to the server (many local servers map this to the loaded GGUF) |
| `OPENAI_API_KEY` | `not-needed` | Placeholder; required non-empty for the client |
| `CHAT_SYSTEM_PROMPT` | _(empty)_ | Optional system line prepended if the request has no `system` message |
| `GEMMA_PORT` | `8080` | Used by `/health` to probe Gemma reachability |
| `CORS_EXTRA_ORIGINS` | _(empty)_ | Comma-separated browser origins for the Next.js app (e.g. `http://192.168.1.5:3000` on a LAN) |

The repository uses [setup/setup-langgraph.sh](../setup/setup-langgraph.sh) to create `langgraph-venv`, install dependencies, start `gemma4.service`, and install/start this service via [langgraph.service](langgraph.service).

API:

- `GET /health` — service up; `gemma: true` if the llama `health` URL responds
- `POST /chat` — non-streaming JSON: `{ "messages": [ { "role", "content" } ] }`
- `POST /chat/stream` — SSE, lines `data: {"token": "..."} ` and `data: {"done": true}`
