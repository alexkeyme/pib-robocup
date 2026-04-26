"use client";

import { useCallback, useEffect, useRef, useState } from "react";

type WebcamCaptureButtonProps = {
  onCapture: (file: File) => void;
  disabled?: boolean;
};

function stopStream(stream: MediaStream | null) {
  stream?.getTracks().forEach((t) => t.stop());
}

export function WebcamCaptureButton({ onCapture, disabled }: WebcamCaptureButtonProps) {
  const [open, setOpen] = useState(false);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const close = useCallback(() => {
    setOpen(false);
    setErr(null);
    setStream((prev) => {
      stopStream(prev);
      return null;
    });
  }, []);

  useEffect(() => {
    if (!open) return;
    let alive = true;
    setErr(null);
    (async () => {
      if (!navigator.mediaDevices?.getUserMedia) {
        if (alive) setErr("This browser does not support camera access.");
        return;
      }
      try {
        const s = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 1280 },
            height: { ideal: 720 },
            facingMode: { ideal: "environment" },
          },
          audio: false,
        });
        if (!alive) {
          stopStream(s);
          return;
        }
        setStream(s);
      } catch {
        if (alive) setErr("Could not open the camera. Allow access when prompted, or try HTTPS / localhost.");
      }
    })();
    return () => {
      alive = false;
      setStream((prev) => {
        stopStream(prev);
        return null;
      });
    };
  }, [open]);

  useEffect(() => {
    const v = videoRef.current;
    if (!v || !stream) return;
    v.srcObject = stream;
    v.playsInline = true;
    v.muted = true;
    void v.play().catch(() => {
      setErr("Video preview could not start.");
    });
  }, [stream]);

  const streamRef = useRef<MediaStream | null>(null);
  useEffect(() => {
    streamRef.current = stream;
  }, [stream]);
  useEffect(
    () => () => {
      stopStream(streamRef.current);
    },
    []
  );

  const capture = useCallback(() => {
    const v = videoRef.current;
    const c = canvasRef.current;
    if (!v || !c || v.readyState < 2) {
      setErr("Camera is not ready yet. Wait a moment and try again.");
      return;
    }
    const w = v.videoWidth;
    const h = v.videoHeight;
    if (w < 2 || h < 2) {
      setErr("Invalid frame size. Try again.");
      return;
    }
    c.width = w;
    c.height = h;
    const ctx = c.getContext("2d");
    if (!ctx) {
      setErr("Could not read image from camera.");
      return;
    }
    ctx.drawImage(v, 0, 0, w, h);
    c.toBlob(
      (blob) => {
        if (!blob) {
          setErr("Could not encode image.");
          return;
        }
        const file = new File([blob], "webcam-capture.jpg", { type: "image/jpeg" });
        onCapture(file);
        close();
      },
      "image/jpeg",
      0.92
    );
  }, [close, onCapture]);

  return (
    <>
      <button
        type="button"
        disabled={disabled}
        onClick={() => setOpen(true)}
        className="shrink-0 rounded-md border border-foreground/20 px-3 py-2 text-sm text-foreground/85 hover:bg-foreground/5 disabled:opacity-40"
      >
        Camera
      </button>
      {open && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
          role="dialog"
          aria-modal="true"
          aria-label="Take a photo with the camera"
          onClick={close}
        >
          <div
            className="max-h-[min(90dvh,720px)] w-full max-w-lg overflow-hidden rounded-lg border border-foreground/20 bg-background p-3 shadow-lg"
            onClick={(e) => e.stopPropagation()}
          >
            <h2 className="mb-2 text-sm font-medium text-foreground/90">Camera</h2>
            {err && (
              <p className="mb-2 rounded border border-red-500/30 bg-red-500/10 px-2 py-1.5 text-sm text-red-300">
                {err}
              </p>
            )}
            <div className="relative aspect-video w-full overflow-hidden rounded border border-foreground/15 bg-black/80">
              <video
                ref={videoRef}
                className="h-full w-full object-contain"
                playsInline
                muted
                autoPlay
              />
            </div>
            <canvas ref={canvasRef} className="hidden" />
            <div className="mt-3 flex flex-wrap items-center justify-end gap-2">
              <button
                type="button"
                onClick={close}
                className="rounded-md border border-foreground/20 px-3 py-2 text-sm text-foreground/85 hover:bg-foreground/5"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={capture}
                disabled={!stream || !!err}
                className="rounded-md bg-foreground px-3 py-2 text-sm font-medium text-background disabled:opacity-40"
              >
                Take photo
              </button>
            </div>
            <p className="mt-2 text-[11px] text-foreground/50">
              Uses your default camera. Works on this machine over HTTP only for localhost; use HTTPS
              for other hosts.
            </p>
          </div>
        </div>
      )}
    </>
  );
}
