"use client";

import { FormEvent, useRef, useState } from "react";

type Role = "user" | "assistant" | "system";

type Msg = { role: Role; content: string };

type MolmoPoint = {
  object_id: number;
  image_index: number;
  x: number;
  y: number;
};

type MolmoChatResult = {
  points?: MolmoPoint[];
  generated_text?: string;
  device?: string;
  model_id?: string;
  error?: string;
};

const API_BASE =
  process.env.NEXT_PUBLIC_LANGGRAPH_API_URL || "http://127.0.0.1:8008";

function formatPointCell(v: number) {
  if (Number.isNaN(v)) return "—";
  return v.toFixed(4);
}

export function Chat() {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [molmoResults, setMolmoResults] = useState<MolmoChatResult[]>([]);
  const listRef = useRef<HTMLDivElement>(null);

  function scrollToBottom() {
    requestAnimationFrame(() => {
      listRef.current?.lastElementChild?.scrollIntoView({ behavior: "smooth" });
    });
  }

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    const text = input.trim();
    if (!text || sending) return;

    setError(null);
    setInput("");

    const userMsg: Msg = { role: "user", content: text };
    const history: Msg[] = [...messages, userMsg];
    setMessages([...history, { role: "assistant", content: "" }]);
    setMolmoResults([]);
    setSending(true);
    scrollToBottom();

    try {
      const res = await fetch(`${API_BASE}/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: history }),
      });

      if (!res.ok) {
        const t = await res.text();
        throw new Error(t || res.statusText);
      }

      const reader = res.body?.getReader();
      if (!reader) throw new Error("No response body");

      const dec = new TextDecoder();
      let acc = "";
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += dec.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed.startsWith("data:")) continue;
          const payload = trimmed.slice(5).trim();
          if (!payload) continue;
          let data: {
            token?: string;
            error?: string;
            done?: boolean;
            molmo_result?: MolmoChatResult;
          };
          try {
            data = JSON.parse(payload);
          } catch {
            continue;
          }
          if (data.error) throw new Error(data.error);
          if (data.molmo_result) {
            const mr = data.molmo_result;
            setMolmoResults((prev) => [...prev, mr]);
            scrollToBottom();
            continue;
          }
          if (data.token) {
            acc += data.token;
            setMessages((prev) => {
              const next = [...prev];
              const last = next[next.length - 1];
              if (last?.role === "assistant") {
                last.content = acc;
              }
              return next;
            });
            scrollToBottom();
          }
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
      setMessages((prev) => {
        if (prev.length < 2) return prev;
        if (prev[prev.length - 1].role === "assistant" && !prev[prev.length - 1].content) {
          return prev.slice(0, -1);
        }
        return prev;
      });
    } finally {
      setSending(false);
    }
  }

  return (
    <div className="mx-auto flex h-[min(90vh,720px)] max-w-2xl flex-col gap-3 px-3 py-6">
      <header>
        <h1 className="text-xl font-semibold tracking-tight">Gemma 4 (local)</h1>
        <p className="text-sm text-foreground/70">
          LangGraph backend: <code className="text-xs opacity-80">{API_BASE}</code>
        </p>
      </header>

      <div
        ref={listRef}
        className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto rounded-lg border border-foreground/10 p-3"
      >
        {messages.length === 0 && !error && (
          <p className="text-sm text-foreground/60">Send a message to start.</p>
        )}
        {messages.map((m, i) => (
          <div
            key={i}
            className={
              m.role === "user"
                ? "ml-8 self-end rounded-lg bg-foreground/10 px-3 py-2"
                : "max-w-[95%] self-start rounded-lg bg-foreground/5 px-3 py-2"
            }
          >
            <div className="text-xs font-medium text-foreground/50">{m.role}</div>
            <div className="whitespace-pre-wrap text-sm leading-relaxed">{m.content}</div>
          </div>
        ))}
        {error && (
          <div className="rounded border border-red-500/30 bg-red-500/10 px-3 py-2 text-sm text-red-300">
            {error}
          </div>
        )}
        {molmoResults.length > 0 && (
          <div className="space-y-3 rounded-lg border border-foreground/15 bg-foreground/5 p-3">
            <h2 className="text-sm font-medium text-foreground/90">
              MolmoPoint (tool) — 0–1 in image space; not from Gemma
            </h2>
            <p className="text-xs text-foreground/55">
              Absolute (px) columns are only filled when the API still returns pixel coords; for 0–1
              only, use the <strong className="text-foreground/70">Molmo (local) upload</strong> panel
              to see <strong>px</strong> from the known image size.
            </p>
            {molmoResults.map((m, i) => (
              <div key={i} className="space-y-1.5 text-sm">
                {m.error && (
                  <p className="text-red-300">
                    {m.error}
                  </p>
                )}
                {(m.model_id || m.device) && (
                  <p className="text-xs text-foreground/60">
                    {m.model_id ? (
                      <>
                        <span className="text-foreground/80">Model:</span>{" "}
                        <code className="break-all">{m.model_id}</code>
                        {m.device ? " · " : null}
                      </>
                    ) : null}
                    {m.device ? (
                      <>
                        <span className="text-foreground/80">Device:</span>{" "}
                        <code>{m.device}</code>
                      </>
                    ) : null}
                  </p>
                )}
                {m.points && m.points.length > 0 && (
                  <div className="overflow-x-auto rounded border border-foreground/10">
                    <table className="w-full min-w-[18rem] text-left text-xs">
                      <thead>
                        <tr className="border-b border-foreground/10 text-foreground/50">
                          <th className="p-1.5 pr-2 font-medium">#</th>
                          <th className="p-1.5 pr-2 font-medium">object_id</th>
                          <th className="p-1.5 pr-2 font-medium">image</th>
                          <th className="p-1.5 pr-2 font-medium">x (0–1)</th>
                          <th className="p-1.5 pr-2 font-medium">y (0–1)</th>
                          <th className="p-1.5 pr-2 font-medium" title="From API if pixel coords, else em dash">
                            x (px)
                          </th>
                          <th className="p-1.5 font-medium" title="From API if pixel coords, else em dash">
                            y (px)
                          </th>
                        </tr>
                      </thead>
                      <tbody>
                        {m.points.map((p, j) => {
                          const looksPixel = p.x > 1 || p.y > 1;
                          return (
                          <tr key={j} className="border-b border-foreground/5 last:border-0">
                            <td className="p-1.5 pr-2 tabular-nums text-foreground/80">
                              {j + 1}
                            </td>
                            <td className="p-1.5 pr-2 tabular-nums">{p.object_id}</td>
                            <td className="p-1.5 pr-2 tabular-nums">{p.image_index}</td>
                            <td className="p-1.5 pr-2 tabular-nums">
                              {looksPixel ? "—" : formatPointCell(p.x)}
                            </td>
                            <td className="p-1.5 pr-2 tabular-nums">
                              {looksPixel ? "—" : formatPointCell(p.y)}
                            </td>
                            <td className="p-1.5 pr-2 tabular-nums text-foreground/70">
                              {looksPixel
                                ? p.x.toFixed(1)
                                : "—"}
                            </td>
                            <td className="p-1.5 tabular-nums text-foreground/70">
                              {looksPixel
                                ? p.y.toFixed(1)
                                : "—"}
                            </td>
                          </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                )}
                {m.generated_text && (
                  <p className="whitespace-pre-wrap break-words text-xs text-foreground/70">
                    <span className="text-foreground/50">raw text: </span>
                    {m.generated_text}
                  </p>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      <form onSubmit={onSubmit} className="flex gap-2">
        <input
          className="min-w-0 flex-1 rounded-md border border-foreground/15 bg-background px-3 py-2 text-sm outline-none ring-0 focus:border-foreground/30"
          placeholder="Message…"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={sending}
        />
        <button
          type="submit"
          disabled={sending || !input.trim()}
          className="rounded-md bg-foreground px-4 py-2 text-sm font-medium text-background disabled:opacity-40"
        >
          {sending ? "…" : "Send"}
        </button>
      </form>
    </div>
  );
}
