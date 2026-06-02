import { Matrix } from '../engine/matrix';
import type { Viewport } from './coords';
import { probColor, rgba } from './colors';
import type { Predictor } from './types';

/**
 * Fill the plane with the model's predicted-probability heatmap by evaluating it on a
 * grid of world points (one batched predict call) and painting each cell.
 */
export function drawBoundary(
  ctx: CanvasRenderingContext2D,
  vp: Viewport,
  predict: Predictor,
  resolution = 64,
): void {
  const innerW = vp.width - 2 * vp.pad;
  const innerH = vp.height - 2 * vp.pad;
  const cols = resolution;
  const rows = resolution;

  const pts = new Float64Array(rows * cols * 2);
  let k = 0;
  for (let j = 0; j < rows; j++) {
    const wy = vp.yMax - ((j + 0.5) / rows) * (vp.yMax - vp.yMin);
    for (let i = 0; i < cols; i++) {
      const wx = vp.xMin + ((i + 0.5) / cols) * (vp.xMax - vp.xMin);
      pts[k * 2] = wx;
      pts[k * 2 + 1] = wy;
      k++;
    }
  }

  const probs = predict(new Matrix(rows * cols, 2, pts));
  const cellW = innerW / cols;
  const cellH = innerH / rows;

  k = 0;
  for (let j = 0; j < rows; j++) {
    for (let i = 0; i < cols; i++) {
      const p = probs.data[k++]!;
      ctx.fillStyle = rgba(probColor(p, 0.55));
      const sx = vp.pad + i * cellW;
      const sy = vp.pad + j * cellH;
      // +1 to avoid hairline seams between cells.
      ctx.fillRect(sx, sy, cellW + 1, cellH + 1);
    }
  }
}
