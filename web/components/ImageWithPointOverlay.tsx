"use client";

export type MolmoPoint = {
  object_id: number;
  image_index: number;
  x: number;
  y: number;
};

type Props = {
  imageUrl: string;
  points: MolmoPoint[];
  alt?: string;
};

const PALETTE = [
  "bg-rose-500",
  "bg-amber-500",
  "bg-emerald-500",
  "bg-sky-500",
  "bg-violet-500",
  "bg-fuchsia-500",
] as const;

function chipColorForObjectId(objectId: number): string {
  const i = ((objectId % PALETTE.length) + PALETTE.length) % PALETTE.length;
  return PALETTE[i]!;
}

/**
 * Renders the image and overlays markers; assumes x,y are normalized 0..1 in image space
 * (top-left origin) as returned by MolmoPoint. Display numbers (#) match 1..n in `points` order.
 */
export function ImageWithPointOverlay({ imageUrl, points, alt = "Upload" }: Props) {
  return (
    <div className="relative inline-block max-w-full align-top">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={imageUrl} alt={alt} className="max-h-[min(50vh,480px)] w-auto max-w-full rounded border border-foreground/15" />
      {points.map((pt, i) => {
        const displayNum = i + 1;
        const bg = chipColorForObjectId(pt.object_id);
        const label = `Detection ${displayNum}: object_id=${pt.object_id}, image_index=${pt.image_index}, x=${pt.x.toFixed(4)}, y=${pt.y.toFixed(4)}`;
        return (
          <div
            key={`${pt.object_id}-${pt.image_index}-${i}`}
            className="pointer-events-none absolute -translate-x-1/2 -translate-y-1/2"
            style={{
              left: `${pt.x * 100}%`,
              top: `${pt.y * 100}%`,
            }}
            title={label}
            role="img"
            aria-label={label}
          >
            <div className="flex flex-col items-center gap-0.5">
              <div
                className={`flex h-7 w-7 items-center justify-center rounded-full border-2 border-white text-xs font-bold text-white shadow-md ${bg}`}
              >
                {displayNum}
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}
