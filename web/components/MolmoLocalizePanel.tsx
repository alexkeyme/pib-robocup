"use client";

import { FormEvent, useEffect, useState } from "react";
import { ImageWithPointOverlay, type MolmoPoint } from "./ImageWithPointOverlay";

const API_BASE =
  process.env.NEXT_PUBLIC_LANGGRAPH_API_URL || "http://127.0.0.1:8008";

type LocalizeResult = {
  ok: boolean;
  points: MolmoPoint[];
  generated_text?: string;
  device?: string;
  model_id?: string;
};

export function MolmoLocalizePanel() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [prompt, setPrompt] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<LocalizeResult | null>(null);

  useEffect(() => {
    if (!file) {
      setPreviewUrl((prev) => {
        if (prev) URL.revokeObjectURL(prev);
        return null;
      });
      return;
    }
    const u = URL.createObjectURL(file);
    setPreviewUrl((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return u;
    });
    return () => {
      URL.revokeObjectURL(u);
    };
  }, [file]);

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    if (!file || !prompt.trim() || loading) return;
    setError(null);
    setResult(null);
    setLoading(true);
    try {
      const fd = new FormData();
      fd.append("file", file);
      fd.append("prompt", prompt.trim());
      const res = await fetch(`${API_BASE}/molmo/localize`, {
        method: "POST",
        body: fd,
      });
      const text = await res.text();
      if (!res.ok) {
        let msg = res.statusText;
        try {
          const err = JSON.parse(text) as { detail?: string | { msg?: string } };
          if (typeof err.detail === "string") msg = err.detail;
        } catch {
          if (text) msg = text;
        }
        throw new Error(msg || "Request failed");
      }
      setResult(JSON.parse(text) as LocalizeResult);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setLoading(false);
    }
  }

  const points = result?.points ?? [];

  return (
    <div className="mb-6 rounded-lg border border-foreground/10 bg-foreground/5 p-3">
      <h2 className="mb-1 text-sm font-semibold">MolmoPoint (local)</h2>
      <p className="mb-3 text-xs text-foreground/65">
        This section uses the <strong>MolmoPoint</strong> service on port <strong>8010</strong> (Molmo-8B)
        for localization. It is <strong>not</strong> the Gemma chat model (port 8080) used below in the
        text chat.
      </p>
      <p className="mb-3 text-xs text-foreground/65">
        Upload an image and describe what to find. Markers on the image match the “#” column in the
        table (same order as returned by MolmoPoint). Coordinates are normalized 0–1 in image space.
      </p>
      <form onSubmit={onSubmit} className="mb-3 flex flex-col gap-2 sm:flex-row sm:flex-wrap sm:items-end">
        <label className="block min-w-0 text-xs">
          <span className="text-foreground/60">Image</span>
          <input
            type="file"
            accept="image/jpeg,image/png,image/webp"
            className="mt-0.5 block w-full text-sm"
            onChange={(e) => {
              setFile(e.target.files?.[0] ?? null);
              setResult(null);
              setError(null);
            }}
            disabled={loading}
          />
        </label>
        <input
          className="min-w-0 flex-1 rounded-md border border-foreground/15 bg-background px-2 py-1.5 text-sm"
          placeholder="e.g. red cup, leftmost door handle"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          disabled={loading}
        />
        <button
          type="submit"
          disabled={loading || !file || !prompt.trim()}
          className="rounded-md bg-foreground px-3 py-2 text-sm font-medium text-background disabled:opacity-40"
        >
          {loading ? "…" : "Find objects"}
        </button>
      </form>
      {error && (
        <div className="mb-2 rounded border border-red-500/30 bg-red-500/10 px-2 py-1.5 text-sm text-red-300">
          {error}
        </div>
      )}
      {previewUrl && (
        <div className="mt-2 space-y-3">
          <ImageWithPointOverlay
            imageUrl={previewUrl}
            points={points}
            alt="Selected upload"
          />
          {result && (result.model_id || result.device) && (
            <p className="text-xs text-foreground/60">
              {result.model_id ? (
                <>
                  <span className="font-medium text-foreground/80">MolmoPoint model:</span>{" "}
                  <code className="break-all">{result.model_id}</code>
                </>
              ) : null}
              {result.device ? (
                <>
                  {result.model_id ? " · " : null}
                  <span className="font-medium text-foreground/80">Device:</span>{" "}
                  <code>{result.device}</code>
                </>
              ) : null}
            </p>
          )}
          {result && points.length > 0 && (
            <div className="overflow-x-auto rounded border border-foreground/10">
              <table className="w-full min-w-[18rem] border-collapse text-left text-xs">
                <thead>
                  <tr className="border-b border-foreground/10 bg-foreground/5">
                    <th className="px-2 py-1.5 font-medium">#</th>
                    <th className="px-2 py-1.5 font-medium">object_id</th>
                    <th className="px-2 py-1.5 font-medium">image_index</th>
                    <th className="px-2 py-1.5 font-medium">x</th>
                    <th className="px-2 py-1.5 font-medium">y</th>
                  </tr>
                </thead>
                <tbody>
                  {points.map((pt, i) => (
                    <tr key={`row-${pt.object_id}-${pt.image_index}-${i}`} className="border-b border-foreground/5">
                      <td className="px-2 py-1.5 font-medium">{i + 1}</td>
                      <td className="px-2 py-1.5 tabular-nums">{pt.object_id}</td>
                      <td className="px-2 py-1.5 tabular-nums">{pt.image_index}</td>
                      <td className="px-2 py-1.5 tabular-nums">{pt.x.toFixed(4)}</td>
                      <td className="px-2 py-1.5 tabular-nums">{pt.y.toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {result && points.length === 0 && (
            <p className="text-xs text-foreground/60">No points returned (empty list).</p>
          )}
          {result?.generated_text ? (
            <details className="text-xs text-foreground/70">
              <summary className="cursor-pointer">
                MolmoPoint – raw text (model output before point extraction; debugging)
              </summary>
              <p className="mb-1 mt-1 text-foreground/55">
                From the same Molmo-8B run as the table above — not from Gemma.
              </p>
              <pre className="mt-1 max-h-32 overflow-auto whitespace-pre-wrap break-words rounded border border-foreground/10 bg-foreground/5 p-2">
                {result.generated_text}
              </pre>
            </details>
          ) : null}
        </div>
      )}
    </div>
  );
}
