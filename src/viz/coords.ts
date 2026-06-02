/**
 * Mapping between "world" coordinates (the data plane) and "screen" coordinates
 * (canvas pixels). Screen Y grows downward, so the Y axis is flipped.
 */
export interface Viewport {
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
  width: number; // canvas width in px
  height: number; // canvas height in px
  pad: number; // inner padding in px
}

export function makeViewport(
  width: number,
  height: number,
  pad: number,
  xMin = -1.3,
  xMax = 1.3,
  yMin = -1.3,
  yMax = 1.3,
): Viewport {
  return { width, height, pad, xMin, xMax, yMin, yMax };
}

export function worldToScreen(vp: Viewport, wx: number, wy: number): { sx: number; sy: number } {
  const innerW = vp.width - 2 * vp.pad;
  const innerH = vp.height - 2 * vp.pad;
  const sx = vp.pad + ((wx - vp.xMin) / (vp.xMax - vp.xMin)) * innerW;
  const sy = vp.pad + (1 - (wy - vp.yMin) / (vp.yMax - vp.yMin)) * innerH;
  return { sx, sy };
}

export function screenToWorld(vp: Viewport, sx: number, sy: number): { wx: number; wy: number } {
  const innerW = vp.width - 2 * vp.pad;
  const innerH = vp.height - 2 * vp.pad;
  const wx = vp.xMin + ((sx - vp.pad) / innerW) * (vp.xMax - vp.xMin);
  const wy = vp.yMin + (1 - (sy - vp.pad) / innerH) * (vp.yMax - vp.yMin);
  return { wx, wy };
}
