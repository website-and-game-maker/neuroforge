/** Size a canvas's backing store for the device pixel ratio and return a ctx scaled to
 * logical (CSS) pixels, plus the logical width/height to draw within. Call on resize. */
export function fitCanvas(canvas: HTMLCanvasElement): {
  ctx: CanvasRenderingContext2D;
  w: number;
  h: number;
} {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || 1;
  const h = canvas.clientHeight || 1;
  canvas.width = Math.max(1, Math.round(w * dpr));
  canvas.height = Math.max(1, Math.round(h * dpr));
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('2D canvas context unavailable');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}
