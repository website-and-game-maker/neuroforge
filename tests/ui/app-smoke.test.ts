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

  it('plays a Versus turn cycle as the human Trainer vs an AI Saboteur', () => {
    const { app, root } = mountApp();
    clickMode(root, 'Versus');
    expect(root.querySelector('.rules-grid')).toBeTruthy();

    const anyApp = app as never as {
      vsSetup: { humanRole: string; aiVsAi: boolean; difficulty: string };
      vs: { match: { phase: string; sabotage: unknown[] } } | null;
      stepOnce(): void;
      vsEndTrainerTurn(): void;
      vsTick(): void;
    };
    anyApp.vsSetup.humanRole = 'trainer';
    anyApp.vsSetup.aiVsAi = false;
    clickByText(root, 'Start match');

    expect(anyApp.vs).toBeTruthy();
    expect(anyApp.vs!.match.phase).toBe('trainer');

    anyApp.stepOnce();
    anyApp.vsEndTrainerTurn(); // hand off to the AI saboteur
    expect(anyApp.vs!.match.phase).toBe('saboteur');

    // Let the AI saboteur drop its points and hand back to the trainer.
    let guard = 0;
    while (anyApp.vs!.match.phase === 'saboteur' && guard++ < 2000) anyApp.vsTick();
    expect(anyApp.vs!.match.sabotage.length).toBeGreaterThan(0);
    expect(anyApp.vs!.match.phase).toBe('trainer');
  });

  it('runs an AI-vs-AI match to completion', () => {
    const { app, root } = mountApp();
    clickMode(root, 'Versus');
    const anyApp = app as never as {
      vsSetup: { aiVsAi: boolean; difficulty: string };
      vs: { match: { phase: string }; finished: boolean } | null;
      vsStart(): void;
      vsTick(): void;
    };
    anyApp.vsSetup.aiVsAi = true;
    anyApp.vsSetup.difficulty = 'easy';
    anyApp.vsStart();

    let guard = 0;
    expect(() => {
      while (anyApp.vs!.match.phase !== 'done' && guard++ < 20000) anyApp.vsTick();
    }).not.toThrow();
    expect(anyApp.vs!.match.phase).toBe('done');
  });
});
