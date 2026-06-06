import type { Network } from '../engine/network';
import { Dense } from '../engine/layers';
import { Matrix } from '../engine/matrix';
import { weightColor, rgba, mix, CLASS_A, CLASS_B } from './colors';

const MAX_SHOW = 16;

export interface NeuronRef {
  /** 0 = input column, 1..L = post-activation columns (last = output). */
  col: number;
  /** Actual neuron index within the layer (not the display slot). */
  idx: number;
}

interface ColLayout {
  x: number;
  header: string;
  size: number;
  sampled: boolean;
  nodes: Array<{ y: number; idx: number }>;
}

export interface InspectorLayout {
  cols: ColLayout[];
  denses: Dense[];
  r: number;
  w: number;
  h: number;
}

function sampleIndices(size: number, max: number): number[] {
  if (size <= max) return Array.from({ length: size }, (_, i) => i);
  const out: number[] = [];
  for (let i = 0; i < max; i++) out.push(Math.round((i / (max - 1)) * (size - 1)));
  return out;
}

/** Compute node/column positions for the current network at a given canvas size. */
export function computeInspectorLayout(
  net: Network,
  w: number,
  h: number,
  task: 'classification' | 'regression',
): InspectorLayout {
  const denses = net.layers.filter((l): l is Dense => l instanceof Dense);
  const sizes = denses.length ? [denses[0]!.W.rows, ...denses.map((d) => d.W.cols)] : [task === 'classification' ? 2 : 1, 1];
  const nLayers = sizes.length;

  const padX = 54;
  const padTop = 34;
  const padBottom = 22;
  const innerW = Math.max(1, w - 2 * padX);
  const innerH = Math.max(1, h - padTop - padBottom);
  const r = 7;

  const cols: ColLayout[] = sizes.map((size, c) => {
    const x = nLayers === 1 ? w / 2 : padX + (c / (nLayers - 1)) * innerW;
    const idxs = sampleIndices(size, MAX_SHOW);
    const nodes = idxs.map((idx, k) => ({
      y: idxs.length <= 1 ? padTop + innerH / 2 : padTop + (k / (idxs.length - 1)) * innerH,
      idx,
    }));
    let header: string;
    if (c === 0) header = 'INPUT';
    else if (c === nLayers - 1) header = 'OUTPUT';
    else header = `HIDDEN ${c}`;
    return { x, header, size, sampled: size > MAX_SHOW, nodes };
  });

  return { cols, denses, r, w, h };
}

export interface InspectorState {
  task: 'classification' | 'regression';
  /** Per-column activations for a single probed point (from net.forwardVerbose). */
  probe?: Matrix[] | null;
  hover?: NeuronRef | null;
}

/** Logistic squash so any activation maps to a blue↔orange display colour. */
function squash(v: number): number {
  return 1 / (1 + Math.exp(-v));
}

function nodeFill(state: InspectorState, col: number, idx: number, isOutput: boolean): string {
  if (!state.probe || !state.probe[col]) return '#0e1422';
  const v = state.probe[col]!.data[idx] ?? 0;
  const t = isOutput ? Math.max(0, Math.min(1, v)) : squash(v);
  const c = mix(CLASS_A, CLASS_B, t);
  return rgba({ ...c, a: 0.92 });
}

export function drawInspector(
  ctx: CanvasRenderingContext2D,
  layout: InspectorLayout,
  state: InspectorState,
): void {
  const { cols, denses, r, w, h } = layout;
  ctx.clearRect(0, 0, w, h);
  if (cols.length === 0) return;

  let maxAbs = 0;
  for (const d of denses) for (const wv of d.W.data) maxAbs = Math.max(maxAbs, Math.abs(wv));

  const hover = state.hover ?? null;
  const dimmed = hover ? 0.12 : 1;

  // --- Edges (under nodes) ---
  for (let c = 0; c < denses.length; c++) {
    const d = denses[c]!;
    const inCol = cols[c]!;
    const outCol = cols[c + 1]!;
    for (const a of inCol.nodes) {
      for (const b of outCol.nodes) {
        const wv = d.W.get(a.idx, b.idx);
        const touchesHover =
          !hover ||
          (hover.col === c && hover.idx === a.idx) ||
          (hover.col === c + 1 && hover.idx === b.idx);
        const base = rgba(weightColor(wv, maxAbs));
        ctx.strokeStyle = touchesHover ? base : rgba({ ...weightColor(wv, maxAbs), a: 0.12 * dimmed + 0.03 });
        ctx.lineWidth = (0.4 + 2.8 * (maxAbs > 0 ? Math.abs(wv) / maxAbs : 0)) * (touchesHover ? 1 : 0.8);
        ctx.beginPath();
        ctx.moveTo(inCol.x, a.y);
        ctx.lineTo(outCol.x, b.y);
        ctx.stroke();
      }
    }
  }

  // --- Column headers ---
  ctx.font = '600 9px "IBM Plex Mono", monospace';
  ctx.textAlign = 'center';
  ctx.fillStyle = 'rgba(150,170,215,0.55)';
  for (const col of cols) {
    ctx.fillText(col.header, col.x, 14);
    if (col.size > 1) ctx.fillText(`${col.size}${col.sampled ? '↓' : ''}`, col.x, 24);
  }

  // --- Nodes ---
  for (let c = 0; c < cols.length; c++) {
    const col = cols[c]!;
    const isOutput = c === cols.length - 1;
    for (const node of col.nodes) {
      const isHover = hover && hover.col === c && hover.idx === node.idx;
      ctx.beginPath();
      ctx.arc(col.x, node.y, isHover ? r + 2 : r, 0, Math.PI * 2);
      ctx.fillStyle = nodeFill(state, c, node.idx, isOutput);
      ctx.fill();
      ctx.lineWidth = isHover ? 2.5 : 1.6;
      ctx.strokeStyle = isHover ? 'rgba(92,225,176,0.95)' : 'rgba(180,200,235,0.8)';
      ctx.stroke();
    }
  }

  // --- Input / output sublabels ---
  ctx.font = '600 11px "IBM Plex Mono", monospace';
  const first = cols[0]!;
  const inLabels = state.task === 'classification' ? ['x', 'y'] : ['x'];
  ctx.textAlign = 'right';
  ctx.fillStyle = 'rgba(231,237,248,0.85)';
  first.nodes.forEach((node, k) => {
    if (inLabels[k]) ctx.fillText(inLabels[k]!, first.x - r - 6, node.y + 4);
  });
  const last = cols[cols.length - 1]!;
  ctx.textAlign = 'left';
  const outLabel = state.task === 'classification' ? 'P(orange)' : 'y';
  if (last.nodes[0]) ctx.fillText(outLabel, last.x + r + 6, last.nodes[0]!.y + 4);
}

/** Map a mouse position to the nearest neuron within hit radius, or null. */
export function hitTestNeuron(layout: InspectorLayout, mx: number, my: number): NeuronRef | null {
  const hit = layout.r + 6;
  for (let c = 0; c < layout.cols.length; c++) {
    const col = layout.cols[c]!;
    for (const node of col.nodes) {
      if (Math.hypot(col.x - mx, node.y - my) <= hit) return { col: c, idx: node.idx };
    }
  }
  return null;
}

/**
 * A predictor that outputs a single hidden neuron's activation over a batch of inputs,
 * so the arena can show "the line this neuron draws". `col` is in forwardVerbose terms
 * (1 = first hidden column); the input column (0) has no field.
 */
export function neuronField(net: Network, ref: NeuronRef): (X: Matrix) => Matrix {
  return (X: Matrix): Matrix => {
    const { columns } = net.forwardVerbose(X);
    const col = columns[ref.col];
    if (!col) return new Matrix(X.rows, 1, new Float64Array(X.rows));
    const out = new Float64Array(col.rows);
    for (let i = 0; i < col.rows; i++) out[i] = squash(col.get(i, ref.idx));
    return new Matrix(col.rows, 1, out);
  };
}
