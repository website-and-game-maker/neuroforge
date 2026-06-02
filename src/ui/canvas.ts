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
  const dw = Math.max(1, Math.round(w * dpr));
  const dh = Math.max(1, Math.round(h * dpr));
  // Only reallocate the backing store when the size actually changed — resizing a
  // canvas clears it and is costly, so doing it every animation frame is wasteful.
  if (canvas.width !== dw || canvas.height !== dh) {
    canvas.width = dw;
    canvas.height = dh;
  }
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('2D canvas context unavailable');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}
