# LangGraph service

FastAPI app (`app.py`) compiled graph in `state_graph.py`. The model is reached via `langchain_openai.ChatOpenAI` to the local **llama.cpp** `llama-server` (Gemma 4) OpenAI-compatible API.

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
