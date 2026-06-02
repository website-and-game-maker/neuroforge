import { Matrix } from '../engine/matrix';
import { worldToScreen, type Viewport } from './coords';
import { rgba, CLASS_B } from './colors';
import type { Predictor } from './types';

/**
 * Regression view: scatter the (x, y) samples and overlay the model's predicted curve
 * sampled densely across the x domain.
 */
export function drawRegression(
  ctx: CanvasRenderingContext2D,
  vp: Viewport,
  X: Matrix,
  Y: Matrix,
  predict: Predictor,
  samples = 160,
): void {
  // Sample points.
  ctx.fillStyle = 'rgba(150, 170, 210, 0.85)';
  for (let i = 0; i < X.rows; i++) {
    const { sx, sy } = worldToScreen(vp, X.get(i, 0), Y.get(i, 0));
    ctx.beginPath();
    ctx.arc(sx, sy, 3.5, 0, Math.PI * 2);
    ctx.fill();
  }

  // Predicted curve.
  const xs = new Float64Array(samples);
  for (let i = 0; i < samples; i++) {
    xs[i] = vp.xMin + (i / (samples - 1)) * (vp.xMax - vp.xMin);
  }
  const ys = predict(new Matrix(samples, 1, xs));

  ctx.beginPath();
  for (let i = 0; i < samples; i++) {
    const { sx, sy } = worldToScreen(vp, xs[i]!, ys.data[i]!);
    if (i === 0) ctx.moveTo(sx, sy);
    else ctx.lineTo(sx, sy);
  }
  ctx.lineWidth = 3;
  ctx.strokeStyle = rgba({ ...CLASS_B, a: 1 });
  ctx.stroke();
}
