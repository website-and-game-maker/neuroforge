import type { Network } from '../engine/network';
import { Dense } from '../engine/layers';
import { weightColor, rgba } from './colors';

const MAX_NODES = 12;

/** Pick up to `max` evenly-spaced indices from [0, size). */
function sampleIndices(size: number, max: number): number[] {
  if (size <= max) return Array.from({ length: size }, (_, i) => i);
  const out: number[] = [];
  for (let i = 0; i < max; i++) {
    out.push(Math.round((i / (max - 1)) * (size - 1)));
  }
  return out;
}

/**
 * Draw the network as columns of neurons connected by edges whose thickness and colour
 * encode each weight's magnitude and sign. Layers wider than MAX_NODES are sampled so
 * the diagram stays readable.
 */
export function drawNetwork(
  ctx: CanvasRenderingContext2D,
  network: Network,
  width: number,
  height: number,
): void {
  ctx.clearRect(0, 0, width, height);
  const denses = network.layers.filter((l) => l instanceof Dense) as Dense[];
  if (denses.length === 0) return;

  const sizes = [denses[0]!.W.rows, ...denses.map((d) => d.W.cols)];
  const nLayers = sizes.length;
  const padX = 30;
  const padY = 24;
  const innerW = width - 2 * padX;
  const innerH = height - 2 * padY;

  const colX = (c: number): number => (nLayers === 1 ? width / 2 : padX + (c / (nLayers - 1)) * innerW);
  const nodeY = (k: number, count: number): number =>
    count <= 1 ? height / 2 : padY + (k / (count - 1)) * innerH;

  const displayed = sizes.map((s) => sampleIndices(s, MAX_NODES));

  let maxAbs = 0;
  for (const d of denses) for (const w of d.W.data) maxAbs = Math.max(maxAbs, Math.abs(w));

  // Edges first, so nodes sit on top.
  for (let c = 0; c < denses.length; c++) {
    const d = denses[c]!;
    const inIdx = displayed[c]!;
    const outIdx = displayed[c + 1]!;
    const x1 = colX(c);
    const x2 = colX(c + 1);
    for (let a = 0; a < inIdx.length; a++) {
      const y1 = nodeY(a, inIdx.length);
      for (let b = 0; b < outIdx.length; b++) {
        const w = d.W.get(inIdx[a]!, outIdx[b]!);
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, nodeY(b, outIdx.length));
        ctx.lineWidth = 0.4 + 2.6 * (maxAbs > 0 ? Math.abs(w) / maxAbs : 0);
        ctx.strokeStyle = rgba(weightColor(w, maxAbs));
        ctx.stroke();
      }
    }
  }

  // Nodes.
  for (let c = 0; c < nLayers; c++) {
    const idx = displayed[c]!;
    const x = colX(c);
    for (let k = 0; k < idx.length; k++) {
      ctx.beginPath();
      ctx.arc(x, nodeY(k, idx.length), 6, 0, Math.PI * 2);
      ctx.fillStyle = '#0e1422';
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = 'rgba(180, 200, 235, 0.9)';
      ctx.stroke();
    }
  }
}
