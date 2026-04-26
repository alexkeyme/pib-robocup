# pib-robocup

## LangGraph + local Gemma 4 + Next.js chat

One script installs Python/Node dependencies, enables **Gemma 4** (`llama-server` in Docker on port **8080**), starts the **LangGraph** FastAPI service on **8008**, and runs the **Next.js** chat dev server on **3000**.

```bash
sudo ./setup/setup-langgraph.sh
```

- **Gemma** (OpenAI-compatible HTTP): `http://127.0.0.1:8080` — used only from the host; the browser talks to Next.js and the LangGraph API.
- **LangGraph API**: `http://127.0.0.1:8008` — `GET /health`, `POST /chat`, `POST /chat/stream` (SSE).
- **Chat UI**: `http://127.0.0.1:3000` (binds on `0.0.0.0`, so other machines on the LAN can use `http://<host-ip>:3000`).

Configuration and env vars for the Python service: [langgraph-service/README.md](langgraph-service/README.md).

**Model name:** the client defaults to `GEMMA_OPENAI_MODEL=gpt-3.5-turbo`, which many local OpenAI-compatible servers map to the loaded model. If responses fail, set `GEMMA_OPENAI_MODEL` in a systemd override to match what `llama-server` expects (see server logs).

**CORS:** if you open the UI from another host, add its origin via `CORS_EXTRA_ORIGINS` (comma-separated), e.g. a drop-in for `langgraph.service`:

`Environment=CORS_EXTRA_ORIGINS=http://192.168.1.10:3000`

**Stop** (without stopping Gemma):

```bash
sudo ./setup/setup-langgraph-stop.sh
```

Stop including Gemma:

```bash
sudo ./setup/setup-langgraph-stop.sh --gemma
```

Full PIB stack (Gemma, MolmoPoint, etc.) is installed with [setup/setup-pib.sh](setup/setup-pib.sh); it is **not** required for the LangGraph chat path.
