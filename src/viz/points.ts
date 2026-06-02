import type { Matrix } from '../engine/matrix';
import { worldToScreen, type Viewport } from './coords';
import { classColor, rgba } from './colors';

/** Scatter the dataset points, coloured by class label, with a dark outline. */
export function drawPoints(
  ctx: CanvasRenderingContext2D,
  vp: Viewport,
  X: Matrix,
  Y: Matrix,
  radius = 4.5,
): void {
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = 'rgba(8, 10, 18, 0.85)';
  for (let i = 0; i < X.rows; i++) {
    const { sx, sy } = worldToScreen(vp, X.get(i, 0), X.get(i, 1));
    ctx.beginPath();
    ctx.arc(sx, sy, radius, 0, Math.PI * 2);
    ctx.fillStyle = rgba(classColor(Y.get(i, 0)));
    ctx.fill();
    ctx.stroke();
  }
}
