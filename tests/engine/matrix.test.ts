import { describe, it, expect } from 'vitest';
import {
  Matrix,
  matmul,
  transpose,
  addRowVector,
  sumRows,
  mapMatrix,
  hadamard,
  scale,
} from '../../src/engine/matrix';

describe('Matrix', () => {
  it('builds from rows and round-trips via toRows', () => {
    const m = Matrix.fromRows([
      [1, 2],
      [3, 4],
    ]);
    expect(m.rows).toBe(2);
    expect(m.cols).toBe(2);
    expect(m.get(1, 0)).toBe(3);
    expect(m.toRows()).toEqual([
      [1, 2],
      [3, 4],
    ]);
  });

  it('zeros creates a zero-filled matrix of the right shape', () => {
    const m = Matrix.zeros(2, 3);
    expect(m.rows).toBe(2);
    expect(m.cols).toBe(3);
    expect(m.toRows()).toEqual([
      [0, 0, 0],
      [0, 0, 0],
    ]);
  });

  it('set/get write and read individual cells', () => {
    const m = Matrix.zeros(2, 2);
    m.set(0, 1, 7);
    expect(m.get(0, 1)).toBe(7);
    expect(m.get(0, 0)).toBe(0);
  });

  it('clone is an independent copy', () => {
    const a = Matrix.fromRows([[1, 2]]);
    const b = a.clone();
    b.set(0, 0, 99);
    expect(a.get(0, 0)).toBe(1);
    expect(b.get(0, 0)).toBe(99);
  });
});

describe('matmul', () => {
  it('multiplies a row vector by a column vector', () => {
    const a = Matrix.fromRows([[1, 2, 3]]); // 1x3
    const b = Matrix.fromRows([[4], [5], [6]]); // 3x1
    expect(matmul(a, b).toRows()).toEqual([[32]]);
  });

  it('multiplies 2x3 by 3x2 with correct shape and values', () => {
    const a = Matrix.fromRows([
      [1, 2, 3],
      [4, 5, 6],
    ]); // 2x3
    const b = Matrix.fromRows([
      [7, 8],
      [9, 10],
      [11, 12],
    ]); // 3x2
    const c = matmul(a, b);
    expect(c.rows).toBe(2);
    expect(c.cols).toBe(2);
    expect(c.toRows()).toEqual([
      [58, 64],
      [139, 154],
    ]);
  });

  it('throws on inner-dimension mismatch', () => {
    const a = Matrix.zeros(2, 3);
    const b = Matrix.zeros(2, 2);
    expect(() => matmul(a, b)).toThrow();
  });
});

describe('elementwise + structural ops', () => {
  it('transpose swaps rows and cols', () => {
    const a = Matrix.fromRows([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    expect(transpose(a).toRows()).toEqual([
      [1, 4],
      [2, 5],
      [3, 6],
    ]);
  });

  it('addRowVector broadcasts a [1xN] row across all rows', () => {
    const a = Matrix.fromRows([
      [1, 2],
      [3, 4],
    ]);
    const v = Matrix.fromRows([[10, 20]]);
    expect(addRowVector(a, v).toRows()).toEqual([
      [11, 22],
      [13, 24],
    ]);
  });

  it('addRowVector throws if the vector width mismatches', () => {
    const a = Matrix.zeros(2, 2);
    const v = Matrix.fromRows([[1, 2, 3]]);
    expect(() => addRowVector(a, v)).toThrow();
  });

  it('sumRows sums down columns into a [1xN] row', () => {
    const a = Matrix.fromRows([
      [1, 2],
      [3, 4],
    ]);
    expect(sumRows(a).toRows()).toEqual([[4, 6]]);
  });

  it('hadamard multiplies elementwise', () => {
    const a = Matrix.fromRows([[1, 2, 3]]);
    const b = Matrix.fromRows([[4, 5, 6]]);
    expect(hadamard(a, b).toRows()).toEqual([[4, 10, 18]]);
  });

  it('hadamard throws on shape mismatch', () => {
    expect(() => hadamard(Matrix.zeros(2, 2), Matrix.zeros(2, 3))).toThrow();
  });

  it('scale multiplies by a scalar', () => {
    const a = Matrix.fromRows([[1, -2, 3]]);
    expect(scale(a, 2).toRows()).toEqual([[2, -4, 6]]);
  });

  it('mapMatrix applies a function elementwise', () => {
    const a = Matrix.fromRows([[1, 2, 3]]);
    expect(mapMatrix(a, (x) => x * x).toRows()).toEqual([[1, 4, 9]]);
  });
});
