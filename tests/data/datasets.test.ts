import { describe, it, expect } from 'vitest';
import {
  gaussianBlobs,
  xorData,
  circlesData,
  moonsData,
  spiralsData,
  regLinear,
  regSine,
  DATASET_GENERATORS,
} from '../../src/data/datasets';
import type { Dataset } from '../../src/data/datasets';

const classification = [gaussianBlobs, xorData, circlesData, moonsData, spiralsData];
const regression = [regLinear, regSine];

function labelsOf(ds: Dataset): number[] {
  return ds.Y.toRows().map((r) => r[0]!);
}

describe('classification datasets', () => {
  for (const gen of classification) {
    const ds = gen(120, 1);
    it(`${ds.name}: shape, kind, and binary labels`, () => {
      expect(ds.kind).toBe('classification');
      expect(ds.inDim).toBe(2);
      expect(ds.X.rows).toBe(120);
      expect(ds.X.cols).toBe(2);
      expect(ds.Y.rows).toBe(120);
      expect(ds.Y.cols).toBe(1);
      for (const l of labelsOf(ds)) expect(l === 0 || l === 1).toBe(true);
    });

    it(`${ds.name}: both classes present`, () => {
      const labels = labelsOf(ds);
      expect(labels.includes(0)).toBe(true);
      expect(labels.includes(1)).toBe(true);
    });

    it(`${ds.name}: points are within the plotting range`, () => {
      for (const v of ds.X.data) expect(Math.abs(v)).toBeLessThanOrEqual(1.5);
    });
  }
});

describe('regression datasets', () => {
  for (const gen of regression) {
    const ds = gen(120, 1);
    it(`${ds.name}: shape and kind`, () => {
      expect(ds.kind).toBe('regression');
      expect(ds.inDim).toBe(1);
      expect(ds.X.cols).toBe(1);
      expect(ds.Y.cols).toBe(1);
      expect(ds.X.rows).toBe(120);
      for (const v of ds.X.data) expect(Math.abs(v)).toBeLessThanOrEqual(1.5);
    });
  }
});

describe('determinism', () => {
  it('same seed produces identical data; different seed differs', () => {
    expect(spiralsData(50, 5).X.toRows()).toEqual(spiralsData(50, 5).X.toRows());
    expect(spiralsData(50, 5).X.toRows()).not.toEqual(spiralsData(50, 6).X.toRows());
  });
});

describe('xorData labels follow the sign-XOR rule', () => {
  it('at least 95% of points match (x>0) XOR (y>0)', () => {
    const ds = xorData(400, 2);
    const rows = ds.X.toRows();
    const labels = labelsOf(ds);
    let match = 0;
    for (let i = 0; i < rows.length; i++) {
      const x = rows[i]![0]!;
      const y = rows[i]![1]!;
      const expected = x > 0 !== y > 0 ? 1 : 0;
      if (expected === labels[i]) match++;
    }
    expect(match / rows.length).toBeGreaterThanOrEqual(0.95);
  });
});

describe('DATASET_GENERATORS registry', () => {
  it('has unique ids and correct kinds', () => {
    const ids = DATASET_GENERATORS.map((g) => g.id);
    expect(new Set(ids).size).toBe(ids.length);
    for (const g of DATASET_GENERATORS) {
      const ds = g.make(20, 1);
      expect(ds.kind).toBe(g.kind);
    }
  });
});
