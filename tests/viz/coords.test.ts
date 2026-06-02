import { describe, it, expect } from 'vitest';
import { makeViewport, worldToScreen, screenToWorld } from '../../src/viz/coords';

describe('coords', () => {
  const vp = makeViewport(400, 400, 20, -1, 1, -1, 1);

  it('maps the world domain corners to the padded screen box', () => {
    // (xMin, yMax) → top-left padded corner
    const tl = worldToScreen(vp, -1, 1);
    expect(tl.sx).toBeCloseTo(20, 6);
    expect(tl.sy).toBeCloseTo(20, 6);
    // (xMax, yMin) → bottom-right padded corner
    const br = worldToScreen(vp, 1, -1);
    expect(br.sx).toBeCloseTo(380, 6);
    expect(br.sy).toBeCloseTo(380, 6);
  });

  it('flips the Y axis (higher world y → smaller screen y)', () => {
    const low = worldToScreen(vp, 0, -0.5);
    const high = worldToScreen(vp, 0, 0.5);
    expect(high.sy).toBeLessThan(low.sy);
  });

  it('round-trips world → screen → world', () => {
    for (const [wx, wy] of [
      [0.3, -0.7],
      [-0.9, 0.2],
      [0, 0],
    ]) {
      const s = worldToScreen(vp, wx!, wy!);
      const w = screenToWorld(vp, s.sx, s.sy);
      expect(w.wx).toBeCloseTo(wx!, 9);
      expect(w.wy).toBeCloseTo(wy!, 9);
    }
  });
});
