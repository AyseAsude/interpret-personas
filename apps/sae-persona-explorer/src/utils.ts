import type { FeaturePoint } from "./types";

const ROLE_PALETTE: [number, number, number][] = [
  [31, 119, 180],
  [255, 127, 14],
  [44, 160, 44],
  [214, 39, 40],
  [148, 103, 189],
  [140, 86, 75],
  [227, 119, 194],
  [127, 127, 127],
  [188, 189, 34],
  [23, 190, 207],
  [57, 106, 177],
  [218, 124, 48],
  [62, 150, 81],
  [204, 37, 41],
  [107, 76, 154],
  [146, 36, 40],
  [148, 139, 61],
  [118, 183, 178],
  [166, 118, 29],
  [83, 81, 84],
];

export type OrthoViewState = {
  target: [number, number, number];
  zoom: number;
};

export function colorForRoleIndex(idx: number): [number, number, number] {
  return ROLE_PALETTE[idx % ROLE_PALETTE.length];
}

export function interpolateColor(
  t: number,
  low: [number, number, number],
  high: [number, number, number],
): [number, number, number] {
  const clamped = Math.max(0, Math.min(1, t));
  return [
    Math.round(low[0] + (high[0] - low[0]) * clamped),
    Math.round(low[1] + (high[1] - low[1]) * clamped),
    Math.round(low[2] + (high[2] - low[2]) * clamped),
  ];
}

export function colorForFeature(
  point: FeaturePoint,
  mode: "preferred_role" | "bridge_entropy" | "active_frac",
): [number, number, number] {
  if (mode === "preferred_role") {
    return colorForRoleIndex(point.preferred_role_idx);
  }
  if (mode === "bridge_entropy") {
    return interpolateColor(point.metrics.bridge_entropy, [253, 187, 45], [12, 130, 138]);
  }
  return interpolateColor(point.metrics.active_frac, [237, 248, 177], [44, 127, 184]);
}

export function fitOrthographicView(
  points: [number, number][],
  width: number,
  height: number,
): OrthoViewState {
  if (!points.length) {
    return { target: [0, 0, 0], zoom: 0 };
  }

  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;

  for (const [x, y] of points) {
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
  }

  const spanX = Math.max(maxX - minX, 1e-6);
  const spanY = Math.max(maxY - minY, 1e-6);
  const padding = 1.2;
  const safeWidth = Math.max(width, 320);
  const safeHeight = Math.max(height, 320);

  const zoomX = Math.log2(safeWidth / (spanX * padding));
  const zoomY = Math.log2(safeHeight / (spanY * padding));
  const zoom = Math.min(zoomX, zoomY);

  return {
    target: [(minX + maxX) / 2, (minY + maxY) / 2, 0],
    zoom,
  };
}

export function sanitizeExternalUrl(rawUrl: string | null | undefined): string | null {
  if (!rawUrl) {
    return null;
  }

  let parsed: URL;
  try {
    parsed = new URL(rawUrl);
  } catch {
    return null;
  }

  if (parsed.protocol !== "https:") {
    return null;
  }

  if (parsed.hostname !== "neuronpedia.org") {
    return null;
  }

  return parsed.toString();
}
