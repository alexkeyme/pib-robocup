"use client";

import { FormEvent, useEffect, useState } from "react";
import { normalizeMolmoXY } from "@/lib/molmoDisplay";
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
  const [imageNatural, setImageNatural] = useState({ w: 0, h: 0 });

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

  useEffect(() => {
    if (!previewUrl) {
      setImageNatural({ w: 0, h: 0 });
      return;
    }
    const img = new window.Image();
    img.onload = () =>
      setImageNatural({ w: img.naturalWidth, h: img.naturalHeight });
    img.src = previewUrl;
    return () => {
      img.onload = null;
    };
  }, [previewUrl]);

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
        Upload an image and <strong>ask where something is</strong> (e.g. &quot;Where is the red
        cup?&quot; or &quot;Point to the left door handle&quot;). Generic prompts like
        &quot;describe the image&quot; return <strong>no</strong> <code>points</code> because
        the model does not emit Molmo’s point tags—only a caption. Markers and the table use the
        same order as MolmoPoint; the API may return x/y as pixel coords; the UI normalizes to
        0–1 for display.
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
                    <th className="px-2 py-1.5 font-medium" title="0–1 in image width">
                      x (0–1)
                    </th>
                    <th className="px-2 py-1.5 font-medium" title="0–1 in image height">
                      y (0–1)
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {points.map((pt, i) => {
                    const { nx, ny } = normalizeMolmoXY(
                      pt.x,
                      pt.y,
                      imageNatural.w,
                      imageNatural.h
                    );
                    return (
                    <tr key={`row-${pt.object_id}-${pt.image_index}-${i}`} className="border-b border-foreground/5">
                      <td className="px-2 py-1.5 font-medium">{i + 1}</td>
                      <td className="px-2 py-1.5 tabular-nums">{pt.object_id}</td>
                      <td className="px-2 py-1.5 tabular-nums">{pt.image_index}</td>
                      <td className="px-2 py-1.5 tabular-nums">{nx.toFixed(4)}</td>
                      <td className="px-2 py-1.5 tabular-nums">{ny.toFixed(4)}</td>
                    </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
          {result && points.length === 0 && (
            <div className="space-y-1.5 text-xs text-foreground/60">
              <p>
                <strong className="text-foreground/75">No coordinates in the response.</strong> A
                non-empty <code>points</code> list is only returned when the model’s answer includes
                Molmo <strong>pointing</strong> markup (e.g. after a &quot;where / point to&quot;
                style prompt). <strong>Caption or chat-style prompts</strong> usually yield{" "}
                <code>points: []</code> with plain text in the raw output—try rephrasing to ask for a
                location. See the raw model text below; if the model did not output Molmo
                <code className="mx-0.5">&lt;points …&gt;</code>-style point markup, the extractor
                has nothing to turn into <code>points</code>.
              </p>
            </div>
          )}
          {result && (
            <details
              className="text-xs text-foreground/70"
              open={points.length === 0}
            >
              <summary className="cursor-pointer text-foreground/80">
                MolmoPoint – raw model text (before point extraction; not Gemma)
              </summary>
              <p className="mb-1 mt-1 text-foreground/55">
                This is the same string Molmo-8B generated on port 8010. Point extraction is applied to
                this text; if the model did not output pointable tokens, you get an empty{" "}
                <code>points</code> list even when this block has content.
              </p>
              <pre className="mt-1 max-h-40 overflow-auto whitespace-pre-wrap break-words rounded border border-foreground/10 bg-foreground/5 p-2 text-foreground/85">
                {(result.generated_text ?? "").length > 0
                  ? result.generated_text
                  : "(empty — model returned no text, or the field was missing)"}
              </pre>
            </details>
          )}
        </div>
      )}
    </div>
  );
}
