"use client";

import { FormEvent, useRef, useState } from "react";

type Role = "user" | "assistant" | "system";

type Msg = { role: Role; content: string };

const API_BASE =
  process.env.NEXT_PUBLIC_LANGGRAPH_API_URL || "http://127.0.0.1:8008";

export function Chat() {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
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
          let data: { token?: string; error?: string; done?: boolean };
          try {
            data = JSON.parse(payload);
          } catch {
            continue;
          }
          if (data.error) throw new Error(data.error);
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
