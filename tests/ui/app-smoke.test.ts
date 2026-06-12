// @vitest-environment jsdom
import { describe, it, expect, beforeAll } from 'vitest';
import { App } from '../../src/ui/app';

/** A no-op 2D context so the canvas renderers run in jsdom without a real canvas. */
function stubCanvas(): void {
  const noop = (): void => {};
  HTMLCanvasElement.prototype.getContext = function (): CanvasRenderingContext2D {
    return new Proxy({} as Record<string, unknown>, {
      get: () => noop,
    }) as unknown as CanvasRenderingContext2D;
  } as never;
}

function mountApp(): { app: App; root: HTMLElement } {
  const root = document.createElement('div');
  document.body.appendChild(root);
  const app = new App();
  app.mount(root);
  return { app, root };
}

function clickMode(root: HTMLElement, label: string): void {
  const btns = Array.from(root.querySelectorAll('[data-tour="modes"] .seg-btn')) as HTMLButtonElement[];
  const btn = btns.find((b) => b.textContent === label);
  if (!btn) throw new Error(`mode button "${label}" not found`);
  btn.dispatchEvent(new MouseEvent('click', { bubbles: true }));
}

function clickByText(root: HTMLElement, contains: string): void {
  const btns = Array.from(root.querySelectorAll('button')) as HTMLButtonElement[];
  const btn = btns.find((b) => (b.textContent ?? '').includes(contains));
  if (!btn) throw new Error(`button containing "${contains}" not found`);
  btn.dispatchEvent(new MouseEvent('click', { bubbles: true }));
}

beforeAll(() => {
  (globalThis as { requestAnimationFrame?: unknown }).requestAnimationFrame = () => 0;
  (window as { devicePixelRatio?: number }).devicePixelRatio = 1;
  stubCanvas();
  try {
    localStorage.setItem('neuroforge.tutorialSeen', '1');
  } catch {
    /* ignore */
  }
});

describe('App integration (jsdom)', () => {
  it('mounts the shell with all three zones', () => {
    const { root } = mountApp();
    expect(root.querySelector('.topbar')).toBeTruthy();
    expect(root.querySelector('.stage')).toBeTruthy();
    expect(root.querySelector('.arena-panel')).toBeTruthy();
    expect(root.querySelectorAll('[data-tour="modes"] .seg-btn').length).toBe(4);
  });

  it('renders the Learn course and steps the playground without error', () => {
    const { app, root } = mountApp();
    clickMode(root, 'Learn');
    expect(root.querySelector('.lesson-card')).toBeTruthy();
    expect(() => {
      (app as never as { stepOnce(): void }).stepOnce();
      (app as never as { stepOnce(): void }).stepOnce();
    }).not.toThrow();
  });

  it('renders Challenges and trains', () => {
    const { app, root } = mountApp();
    clickMode(root, 'Challenges');
    expect(root.querySelector('.challenges')).toBeTruthy();
    const a = app as never as { stepOnce(): void };
    expect(() => {
      for (let i = 0; i < 5; i++) a.stepOnce();
    }).not.toThrow();
  });

  it('supports draw-your-own with painted points', () => {
    const { app, root } = mountApp();
    clickMode(root, 'Sandbox');
    const anyApp = app as never as {
      sandboxDatasetId: string;
      loadSandbox(): void;
      placeAt(x: number, y: number): void;
      studio: { trainData: { X: { rows: number } }; trainer: unknown };
      customPoints: unknown[];
    };
    anyApp.sandboxDatasetId = 'draw';
    anyApp.loadSandbox();
    anyApp.placeAt(-0.4, -0.4);
    anyApp.placeAt(0.4, 0.4);
    anyApp.placeAt(0.3, -0.3);
    expect(anyApp.customPoints.length).toBe(3);
    expect(anyApp.studio.trainData.X.rows).toBe(3);
    expect(anyApp.studio.trainer).toBeTruthy();
  });

  interface VsHandle {
    vsSetup: { humanRole: string; aiVsAi: boolean; difficulty: string };
    vs: {
      match: { points: unknown[]; elapsedMs: number; budgetLeft: number; done: boolean };
      finished: boolean;
      winner: string | null;
      saboClass: number;
    } | null;
    placeAt(x: number, y: number): void;
    vsStart(): void;
    vsTick(dtMs: number): void;
  }

  it('plays live as the human Trainer: AI saboteur drips points in while training runs', () => {
    const { app, root } = mountApp();
    clickMode(root, 'Versus');
    expect(root.querySelector('.rules-grid')).toBeTruthy();

    const anyApp = app as never as VsHandle;
    anyApp.vsSetup.humanRole = 'trainer';
    anyApp.vsSetup.aiVsAi = false;
    clickByText(root, 'Start match');

    expect(anyApp.vs).toBeTruthy();
    expect(anyApp.vs!.match.points.length).toBe(0); // no base pattern — empty board

    // ~5 seconds of live play: the AI saboteur should have landed several points.
    for (let i = 0; i < 320; i++) anyApp.vsTick(16);
    expect(anyApp.vs!.match.points.length).toBeGreaterThan(1);
    expect(anyApp.vs!.finished).toBe(false);
  });

  it('lets a human Saboteur paint anywhere, any class, while the clock runs', () => {
    const { app, root } = mountApp();
    clickMode(root, 'Versus');
    const anyApp = app as never as VsHandle;
    anyApp.vsSetup.humanRole = 'saboteur';
    anyApp.vsSetup.aiVsAi = false;
    clickByText(root, 'Start match');

    anyApp.vsTick(16);
    anyApp.placeAt(0.5, 0.5);
    anyApp.vs!.saboClass = 1;
    anyApp.placeAt(0.51, 0.52); // right next to an opposite-class point — allowed
    anyApp.placeAt(-0.8, -0.8);
    expect(anyApp.vs!.match.points.length).toBe(3);
    expect(anyApp.vs!.match.budgetLeft).toBeLessThan(60);
  });

  it('runs an AI-vs-AI live match to the final bell and declares a winner', () => {
    const { app, root } = mountApp();
    clickMode(root, 'Versus');
    const anyApp = app as never as VsHandle;
    anyApp.vsSetup.aiVsAi = true;
    anyApp.vsSetup.difficulty = 'easy';
    anyApp.vsStart();

    let guard = 0;
    expect(() => {
      // 100ms ticks → a 120s match completes in ~1200 ticks.
      while (!anyApp.vs!.finished && guard++ < 5000) anyApp.vsTick(100);
    }).not.toThrow();
    expect(anyApp.vs!.finished).toBe(true);
    expect(anyApp.vs!.match.done).toBe(true);
    expect(['trainer', 'saboteur']).toContain(anyApp.vs!.winner);
    expect(anyApp.vs!.match.points.length).toBeGreaterThan(10);
  });

  it('routes modes to named paths and back', () => {
    const { root } = mountApp();
    clickMode(root, 'Sandbox');
    expect(window.location.pathname.endsWith('/sandbox')).toBe(true);
    clickMode(root, 'Challenges');
    expect(window.location.pathname.endsWith('/challenges')).toBe(true);
    // The brand link routes home (Learn).
    const brand = root.querySelector('a.brand') as HTMLAnchorElement;
    brand.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    expect(window.location.pathname.endsWith('/learn')).toBe(true);
  });
});
