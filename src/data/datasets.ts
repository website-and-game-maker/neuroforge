import { Matrix } from '../engine/matrix';
import { Rng } from '../engine/rng';

export type TaskKind = 'classification' | 'regression';

export interface Dataset {
  kind: TaskKind;
  /** Human-readable name (used in the UI). */
  name: string;
  /** Input dimensionality: 2 for classification (plane), 1 for regression. */
  inDim: number;
  X: Matrix; // [n × inDim]
  Y: Matrix; // [n × 1]  (label in {0,1} for classification, target value for regression)
}

/** Build a Dataset from arrays of input rows and scalar targets. */
function makeDataset(
  kind: TaskKind,
  name: string,
  inDim: number,
  xs: number[][],
  ys: number[],
): Dataset {
  return {
    kind,
    name,
    inDim,
    X: Matrix.fromRows(xs),
    Y: Matrix.fromRows(ys.map((v) => [v])),
  };
}

// ---------------------------------------------------------------------------
// Classification datasets (2-D input, binary label)
// ---------------------------------------------------------------------------

/** Two well-separated gaussian blobs — linearly separable. */
export function gaussianBlobs(n: number, seed: number): Dataset {
  const rng = new Rng(seed);
  const xs: number[][] = [];
  const ys: number[] = [];
  for (let i = 0; i < n; i++) {
    const cls = i % 2;
    const cx = cls === 0 ? -0.6 : 0.6;
    const cy = cls === 0 ? -0.6 : 0.6;
    xs.push([cx + rng.gaussian() * 0.22, cy + rng.gaussian() * 0.22]);
    ys.push(cls);
  }
  return makeDataset('classification', 'Two Blobs', 2, xs, ys);
}

/** Clustered XOR: four blobs at (±0.5, ±0.5); opposite signs → 1, same → 0. */
export function xorData(n: number, seed: number): Dataset {
  const rng = new Rng(seed);
  const centers: Array<[number, number, number]> = [
    [0.5, 0.5, 0],
    [-0.5, -0.5, 0],
    [0.5, -0.5, 1],
    [-0.5, 0.5, 1],
  ];
  const xs: number[][] = [];
  const ys: number[] = [];
  for (let i = 0; i < n; i++) {
    const c = centers[i % 4]!;
    xs.push([c[0] + rng.gaussian() * 0.16, c[1] + rng.gaussian() * 0.16]);
    ys.push(c[2]);
  }
  return makeDataset('classification', 'XOR', 2, xs, ys);
}

/** Concentric rings: inner ring → 0, outer ring → 1 (not linearly separable). */
export function circlesData(n: number, seed: number): Dataset {
  const rng = new Rng(seed);
  const xs: number[][] = [];
  const ys: number[] = [];
  for (let i = 0; i < n; i++) {
    const cls = i % 2;
    const radius = (cls === 0 ? 0.32 : 0.82) + rng.gaussian() * 0.05;
    const angle = rng.range(0, 2 * Math.PI);
    xs.push([radius * Math.cos(angle), radius * Math.sin(angle)]);
    ys.push(cls);
  }
  return makeDataset('classification', 'Circles', 2, xs, ys);
}

/** Two interleaving half-moons. */
export function moonsData(n: number, seed: number): Dataset {
  const rng = new Rng(seed);
  const xs: number[][] = [];
  const ys: number[] = [];
  for (let i = 0; i < n; i++) {
    const cls = i % 2;
    const t = rng.range(0, Math.PI);
    let px: number;
    let py: number;
    if (cls === 0) {
      px = Math.cos(t);
      py = Math.sin(t);
    } else {
      px = 1 - Math.cos(t);
      py = 0.5 - Math.sin(t);
    }
    // Centre and scale into roughly [-1, 1], then add noise.
    const x = (px - 0.5) * 0.9 + rng.gaussian() * 0.05;
    const y = (py - 0.25) * 0.9 + rng.gaussian() * 0.05;
    xs.push([x, y]);
    ys.push(cls);
  }
  return makeDataset('classification', 'Moons', 2, xs, ys);
}

/** Two intertwined spirals — the classic hard, depth-demanding dataset. */
export function spiralsData(n: number, seed: number): Dataset {
  const rng = new Rng(seed);
  const xs: number[][] = [];
  const ys: number[] = [];
  const perClass = Math.ceil(n / 2);
  let count = 0;
  for (let cls = 0; cls < 2 && count < n; cls++) {
    for (let j = 0; j < perClass && count < n; j++) {
      const frac = j / perClass;
      const radius = 0.15 + 0.85 * frac;
      const theta = frac * 3 * Math.PI + cls * Math.PI + rng.gaussian() * 0.08;
      xs.push([radius * Math.cos(theta), radius * Math.sin(theta)]);
      ys.push(cls);
      count++;
    }
  }
  return makeDataset('classification', 'Spirals', 2, xs, ys);
}

// ---------------------------------------------------------------------------
// Regression datasets (1-D input → scalar target)
// ---------------------------------------------------------------------------

/** Noisy linear relationship. */
export function regLinear(n: number, seed: number): Dataset {
  const rng = new Rng(seed);
  const xs: number[][] = [];
  const ys: number[] = [];
  for (let i = 0; i < n; i++) {
    const x = rng.range(-1, 1);
    xs.push([x]);
    ys.push(0.7 * x + 0.1 + rng.gaussian() * 0.05);
  }
  return makeDataset('regression', 'Linear', 1, xs, ys);
}

/** One period of a sine wave — needs a nonlinear model to fit. */
export function regSine(n: number, seed: number): Dataset {
  const rng = new Rng(seed);
  const xs: number[][] = [];
  const ys: number[] = [];
  for (let i = 0; i < n; i++) {
    const x = rng.range(-1, 1);
    xs.push([x]);
    ys.push(Math.sin(Math.PI * x) * 0.7 + rng.gaussian() * 0.04);
  }
  return makeDataset('regression', 'Sine', 1, xs, ys);
}

// ---------------------------------------------------------------------------
// Registry (used by the sandbox dataset selector)
// ---------------------------------------------------------------------------

export interface DatasetGen {
  id: string;
  label: string;
  kind: TaskKind;
  make: (n: number, seed: number) => Dataset;
}

export const DATASET_GENERATORS: DatasetGen[] = [
  { id: 'blobs', label: 'Two Blobs', kind: 'classification', make: gaussianBlobs },
  { id: 'xor', label: 'XOR', kind: 'classification', make: xorData },
  { id: 'circles', label: 'Circles', kind: 'classification', make: circlesData },
  { id: 'moons', label: 'Moons', kind: 'classification', make: moonsData },
  { id: 'spirals', label: 'Spirals', kind: 'classification', make: spiralsData },
  { id: 'linear', label: 'Linear', kind: 'regression', make: regLinear },
  { id: 'sine', label: 'Sine', kind: 'regression', make: regSine },
];
