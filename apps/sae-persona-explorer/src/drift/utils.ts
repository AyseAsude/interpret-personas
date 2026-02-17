export function clamp(value: number, minValue: number, maxValue: number): number {
  return Math.min(maxValue, Math.max(minValue, value));
}

export function formatSigned(value: number, digits = 3): string {
  const out = value.toFixed(digits);
  return value >= 0 ? `+${out}` : out;
}

export function formatPct(value: number, digits = 1): string {
  return `${(value * 100).toFixed(digits)}%`;
}

export function shortenText(input: string, maxChars: number): string {
  const cleaned = input.replace(/\s+/g, " ").trim();
  if (cleaned.length <= maxChars) {
    return cleaned;
  }
  return `${cleaned.slice(0, Math.max(0, maxChars - 1)).trimEnd()}…`;
}

function toHex(v: number): string {
  const clamped = clamp(Math.round(v), 0, 255);
  return clamped.toString(16).padStart(2, "0");
}

export function interpolateHexColor(low: [number, number, number], high: [number, number, number], t: number): string {
  const x = clamp(t, 0, 1);
  const rgb: [number, number, number] = [
    low[0] + (high[0] - low[0]) * x,
    low[1] + (high[1] - low[1]) * x,
    low[2] + (high[2] - low[2]) * x,
  ];
  return `#${toHex(rgb[0])}${toHex(rgb[1])}${toHex(rgb[2])}`;
}

export function interpolateMultiStopHexColor(
  stops: Array<[number, number, number]>,
  t: number,
): string {
  if (stops.length === 0) {
    return "#ffffff";
  }
  if (stops.length === 1) {
    return interpolateHexColor(stops[0], stops[0], 0);
  }

  const x = clamp(t, 0, 1) * (stops.length - 1);
  const idx = Math.min(stops.length - 2, Math.floor(x));
  const localT = x - idx;
  return interpolateHexColor(stops[idx], stops[idx + 1], localT);
}

function parseHexColor(hex: string): [number, number, number] {
  const m = /^#([0-9a-f]{6})$/i.exec(hex);
  if (!m) {
    return [255, 255, 255];
  }
  const v = m[1];
  return [
    Number.parseInt(v.slice(0, 2), 16),
    Number.parseInt(v.slice(2, 4), 16),
    Number.parseInt(v.slice(4, 6), 16),
  ];
}

export function textColorForBackground(hex: string): "#111111" | "#ffffff" {
  const [r, g, b] = parseHexColor(hex);
  // W3C relative luminance proxy; threshold tuned for compact token chips.
  const luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;
  return luminance >= 0.56 ? "#111111" : "#ffffff";
}

export function tokenDisplay(raw: string): string {
  return raw
    .replace(/<0x0A>/g, "\n")
    .replace(/▁/g, " ")
    .replace(/Ġ/g, " ");
}

export function deriveExplorerUrl(baseUrl: string): string {
  const trimmed = baseUrl.endsWith("/") ? baseUrl.slice(0, -1) : baseUrl;
  const slash = trimmed.lastIndexOf("/");
  if (slash <= 0) {
    return "/";
  }
  return `${trimmed.slice(0, slash + 1)}`;
}
