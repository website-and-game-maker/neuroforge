/** Colour helpers shared by every renderer, kept pure for unit testing. */
export interface RGBA {
  r: number;
  g: number;
  b: number;
  a: number;
}

/** Class A (label 0) — cool blue. Class B (label 1) — warm orange. */
export const CLASS_A: RGBA = { r: 91, g: 156, b: 255, a: 1 };
export const CLASS_B: RGBA = { r: 255, g: 122, b: 89, a: 1 };

export function rgba(c: RGBA): string {
  return `rgba(${Math.round(c.r)}, ${Math.round(c.g)}, ${Math.round(c.b)}, ${c.a})`;
}

function clamp01(t: number): number {
  return t < 0 ? 0 : t > 1 ? 1 : t;
}

/** Linear interpolation between two colours; `t` is clamped to [0,1]. */
export function mix(a: RGBA, b: RGBA, t: number): RGBA {
  const k = clamp01(t);
  return {
    r: a.r + (b.r - a.r) * k,
    g: a.g + (b.g - a.g) * k,
    b: a.b + (b.b - a.b) * k,
    a: a.a + (b.a - a.a) * k,
  };
}

/** Heatmap colour for a model probability p∈[0,1] (0→class A, 1→class B). */
export function probColor(p: number, alpha = 0.5): RGBA {
  const c = mix(CLASS_A, CLASS_B, p);
  return { ...c, a: alpha };
}

/** Solid colour for a discrete class label. */
export function classColor(label: number): RGBA {
  return label >= 0.5 ? { ...CLASS_B } : { ...CLASS_A };
}

/**
 * Colour for a network weight: positive weights warm (orange), negative cool (blue),
 * with opacity scaled by magnitude relative to the largest |weight| in the diagram.
 */
export function weightColor(w: number, maxAbs: number): RGBA {
  const scale = maxAbs > 0 ? clamp01(Math.abs(w) / maxAbs) : 0;
  const alpha = 0.15 + 0.85 * scale;
  return w >= 0 ? { ...CLASS_B, a: alpha } : { ...CLASS_A, a: alpha };
}
