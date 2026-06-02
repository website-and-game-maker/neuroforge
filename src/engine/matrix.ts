/**
 * Minimal dense matrix over a flat row-major Float64Array.
 *
 * The whole ML engine is built on this — it is intentionally tiny, total, and
 * dependency-free. Operations return new matrices (cheap enough at the sizes we
 * train on) so callers never accidentally alias buffers.
 */
export class Matrix {
  readonly rows: number;
  readonly cols: number;
  readonly data: Float64Array;

  constructor(rows: number, cols: number, data?: Float64Array) {
    if (rows < 0 || cols < 0 || !Number.isInteger(rows) || !Number.isInteger(cols)) {
      throw new Error(`Matrix: invalid shape ${rows}x${cols}`);
    }
    if (data) {
      if (data.length !== rows * cols) {
        throw new Error(
          `Matrix: data length ${data.length} does not match shape ${rows}x${cols}`,
        );
      }
      this.data = data;
    } else {
      this.data = new Float64Array(rows * cols);
    }
    this.rows = rows;
    this.cols = cols;
  }

  static zeros(rows: number, cols: number): Matrix {
    return new Matrix(rows, cols);
  }

  static fromRows(rows: number[][]): Matrix {
    const r = rows.length;
    const c = r > 0 ? rows[0]!.length : 0;
    const data = new Float64Array(r * c);
    for (let i = 0; i < r; i++) {
      const row = rows[i]!;
      if (row.length !== c) {
        throw new Error('Matrix.fromRows: ragged rows are not allowed');
      }
      for (let j = 0; j < c; j++) {
        data[i * c + j] = row[j]!;
      }
    }
    return new Matrix(r, c, data);
  }

  get(r: number, c: number): number {
    return this.data[r * this.cols + c]!;
  }

  set(r: number, c: number, v: number): void {
    this.data[r * this.cols + c] = v;
  }

  clone(): Matrix {
    return new Matrix(this.rows, this.cols, this.data.slice());
  }

  toRows(): number[][] {
    const out: number[][] = [];
    for (let i = 0; i < this.rows; i++) {
      const row: number[] = [];
      for (let j = 0; j < this.cols; j++) {
        row.push(this.data[i * this.cols + j]!);
      }
      out.push(row);
    }
    return out;
  }
}

/** Matrix product [m×k]·[k×n] → [m×n]. */
export function matmul(a: Matrix, b: Matrix): Matrix {
  if (a.cols !== b.rows) {
    throw new Error(`matmul: shape mismatch ${a.rows}x${a.cols} · ${b.rows}x${b.cols}`);
  }
  const m = a.rows;
  const k = a.cols;
  const n = b.cols;
  const out = new Float64Array(m * n);
  const ad = a.data;
  const bd = b.data;
  for (let i = 0; i < m; i++) {
    const aRow = i * k;
    const oRow = i * n;
    for (let p = 0; p < k; p++) {
      const aip = ad[aRow + p]!;
      if (aip === 0) continue;
      const bRow = p * n;
      for (let j = 0; j < n; j++) {
        out[oRow + j]! += aip * bd[bRow + j]!;
      }
    }
  }
  return new Matrix(m, n, out);
}

/** Transpose [m×n] → [n×m]. */
export function transpose(a: Matrix): Matrix {
  const out = new Float64Array(a.rows * a.cols);
  for (let i = 0; i < a.rows; i++) {
    for (let j = 0; j < a.cols; j++) {
      out[j * a.rows + i] = a.data[i * a.cols + j]!;
    }
  }
  return new Matrix(a.cols, a.rows, out);
}

/** Add a [1×cols] row vector to every row of `a`. */
export function addRowVector(a: Matrix, row: Matrix): Matrix {
  if (row.rows !== 1 || row.cols !== a.cols) {
    throw new Error(
      `addRowVector: expected [1x${a.cols}] row, got ${row.rows}x${row.cols}`,
    );
  }
  const out = new Float64Array(a.data.length);
  for (let i = 0; i < a.rows; i++) {
    for (let j = 0; j < a.cols; j++) {
      out[i * a.cols + j] = a.data[i * a.cols + j]! + row.data[j]!;
    }
  }
  return new Matrix(a.rows, a.cols, out);
}

/** Sum down columns, producing a [1×cols] row. */
export function sumRows(a: Matrix): Matrix {
  const out = new Float64Array(a.cols);
  for (let i = 0; i < a.rows; i++) {
    for (let j = 0; j < a.cols; j++) {
      out[j]! += a.data[i * a.cols + j]!;
    }
  }
  return new Matrix(1, a.cols, out);
}

function requireSameShape(a: Matrix, b: Matrix, op: string): void {
  if (a.rows !== b.rows || a.cols !== b.cols) {
    throw new Error(
      `${op}: shape mismatch ${a.rows}x${a.cols} vs ${b.rows}x${b.cols}`,
    );
  }
}

/** Elementwise product. */
export function hadamard(a: Matrix, b: Matrix): Matrix {
  requireSameShape(a, b, 'hadamard');
  const out = new Float64Array(a.data.length);
  for (let i = 0; i < out.length; i++) out[i] = a.data[i]! * b.data[i]!;
  return new Matrix(a.rows, a.cols, out);
}

/** Multiply every element by a scalar. */
export function scale(a: Matrix, s: number): Matrix {
  const out = new Float64Array(a.data.length);
  for (let i = 0; i < out.length; i++) out[i] = a.data[i]! * s;
  return new Matrix(a.rows, a.cols, out);
}

/** Apply a function to every element. */
export function mapMatrix(a: Matrix, fn: (x: number) => number): Matrix {
  const out = new Float64Array(a.data.length);
  for (let i = 0; i < out.length; i++) out[i] = fn(a.data[i]!);
  return new Matrix(a.rows, a.cols, out);
}
