# LangGraph service

FastAPI app (`app.py`) runs an agent built with **`langchain.agents.create_agent`**, which compiles to a LangGraph `CompiledStateGraph`. The chat model is **`langchain_openai.ChatOpenAI`** against the local **llama.cpp** `llama-server` (Gemma 4) OpenAI-compatible API.

## Tools (MolmoPoint)

- **`molmo_point_localize`**: calls the local **MolmoPoint** HTTP service `POST /point` ([molmopoint/molmopoint-server.py](../molmopoint/molmopoint-server.py)) with a host-local `image_path` and a `prompt` describing what to find. The tool returns JSON with a `points` list: `object_id`, `image_index`, `x`, `y` (and optional `generated_text` from the model).
- MolmoPoint must be running separately (e.g. `systemctl status molmopoint.service` on port **8010**), typically after [setup/setup-pib.sh](../setup/setup-pib.sh). [setup/setup-langgraph.sh](../setup/setup-langgraph.sh) does **not** start MolmoPoint.
- **Local image files only:** the chat API does not upload images. The model must be given a **filesystem path** the host can read (e.g. a frame written under `/data/...`). Set **`MOLMO_ALLOWED_PATH_PREFIX`** in production to restrict which paths the tool will forward to MolmoPoint.

## System prompt

`CHAT_SYSTEM_PROMPT` is applied in `prepare_for_model()` in [state_graph.py](state_graph.py) (only if the request has no `system` message). `create_agent` is called with `system_prompt=None` so that logic stays in one place.

## Streaming

`POST /chat/stream` uses the agent’s `astream(..., stream_mode="messages")` so streaming stays consistent with tool rounds (you may only see final answer tokens, not the full tool phase).

## OpenAI / Gemma: tool calling

Gemma/llama.cpp must expose **OpenAI-style tool / function calls** in the chat API, or the agent will not invoke `molmo_point_localize`. If the model never issues `tool_calls`, check server support, `GEMMA_OPENAI_MODEL`, and logs.

## Environment variables (optional)

| Variable | Default | Meaning |
|----------|---------|--------|
| `GEMMA_BASE_URL` | `http://127.0.0.1:8080/v1` | OpenAI API root for Gemma |
| `GEMMA_OPENAI_MODEL` | `gpt-3.5-turbo` | `model` field sent to the server |
| `OPENAI_API_KEY` | `not-needed` | Placeholder for the OpenAI client |
| `CHAT_SYSTEM_PROMPT` | _(empty)_ | Optional system line if the request has no `system` message |
| `GEMMA_PORT` | `8080` | Used by `/health` to probe Gemma |
| `MOLMO_BASE_URL` | `http://127.0.0.1:8010` | MolmoPoint base URL (no path suffix) |
| `MOLMO_TIMEOUT_SECONDS` | `120` | HTTP timeout for `POST /point` |
| `MOLMO_ALLOWED_PATH_PREFIX` | _(empty)_ | If set, `image_path` must resolve under this directory (security) |
| `CORS_EXTRA_ORIGINS` | _(empty)_ | Comma-separated browser origins for the Next.js app |

The repository uses [setup/setup-langgraph.sh](../setup/setup-langgraph.sh) to create `langgraph-venv`, install dependencies, start `gemma4.service`, and install/start this service via [langgraph.service](langgraph.service).

## API

- `GET /health` — `ok`, `gemma` (llama up), `molmo` (MolmoPoint `/health` or `/` responds)
- `POST /chat` — JSON `{ "messages": [ { "role", "content" } ] }`
- `POST /chat/stream` — SSE: `data: {"token": "..."} `, then `data: {"done": true}`
