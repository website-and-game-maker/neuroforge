import { describe, it, expect } from 'vitest';
import { LESSONS, CONCEPTS, TUTORIAL, ioExplainer } from '../../src/content/curriculum';
import { DATASET_GENERATORS } from '../../src/data/datasets';
import { activationByName } from '../../src/engine/activations';

const datasetIds = new Set(DATASET_GENERATORS.map((g) => g.id));

describe('LESSONS', () => {
  it('every lesson references a real dataset and a sane config', () => {
    expect(LESSONS.length).toBeGreaterThan(3);
    for (const l of LESSONS) {
      expect(datasetIds.has(l.dataset)).toBe(true);
      expect(['classification', 'regression']).toContain(l.task);
      expect(() => activationByName(l.config.activation)).not.toThrow();
      expect(l.config.hidden.every((w) => w >= 1)).toBe(true);
      expect(l.config.lr).toBeGreaterThan(0);
      expect(l.body.length).toBeGreaterThan(0);
      expect(l.doThis.length).toBeGreaterThan(0);
    }
  });

  it('has unique lesson ids', () => {
    const ids = LESSONS.map((l) => l.id);
    expect(new Set(ids).size).toBe(ids.length);
  });
});

describe('CONCEPTS', () => {
  it('has well-formed glossary entries', () => {
    expect(CONCEPTS.length).toBeGreaterThan(5);
    for (const c of CONCEPTS) {
      expect(c.term.length).toBeGreaterThan(0);
      expect(c.short.length).toBeGreaterThan(0);
      expect(c.long.length).toBeGreaterThan(0);
    }
  });
});

describe('TUTORIAL', () => {
  it('has steps with titles and bodies', () => {
    expect(TUTORIAL.length).toBeGreaterThan(3);
    for (const s of TUTORIAL) {
      expect(s.title.length).toBeGreaterThan(0);
      expect(s.body.length).toBeGreaterThan(0);
    }
  });
});

describe('ioExplainer', () => {
  it('describes 2 inputs for classification and 1 for regression', () => {
    expect(ioExplainer('classification').lines.join(' ')).toContain('2 inputs');
    expect(ioExplainer('regression').lines.join(' ')).toContain('1 input');
  });
});
