"use client";

type Props = {
  onTrigger: () => void;
  disabled?: boolean;
};

export function OakCaptureButton({ onTrigger, disabled }: Props) {
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={onTrigger}
      className="shrink-0 rounded-md border border-foreground/20 px-3 py-2 text-sm text-foreground/85 hover:bg-foreground/5 disabled:opacity-40"
      title="Capture from the OAK-D (RGB + depth) and ask the assistant about it"
    >
      OAK-D
    </button>
  );
}
