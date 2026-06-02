/**
 * Draw a loss sparkline. The series is downsampled to the available width and scaled
 * so the visible window fills the vertical space (min..max of the window).
 */
export function drawLossChart(
  ctx: CanvasRenderingContext2D,
  history: number[],
  width: number,
  height: number,
): void {
  ctx.clearRect(0, 0, width, height);
  if (history.length < 2) return;

  const pad = 6;
  const innerW = width - 2 * pad;
  const innerH = height - 2 * pad;

  // Downsample to at most innerW points.
  const maxPoints = Math.max(2, Math.floor(innerW));
  const stride = Math.max(1, Math.floor(history.length / maxPoints));
  const series: number[] = [];
  for (let i = 0; i < history.length; i += stride) series.push(history[i]!);
  if (series[series.length - 1] !== history[history.length - 1]) {
    series.push(history[history.length - 1]!);
  }

  let lo = Infinity;
  let hi = -Infinity;
  for (const v of series) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  const span = hi - lo || 1;

  ctx.beginPath();
  for (let i = 0; i < series.length; i++) {
    const x = pad + (i / (series.length - 1)) * innerW;
    const y = pad + (1 - (series[i]! - lo) / span) * innerH;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.lineWidth = 2;
  ctx.strokeStyle = 'rgba(120, 200, 160, 0.95)';
  ctx.stroke();
}
