import { el, clear } from './dom';
import { renderRich } from './rich';
import { fitCanvas } from './canvas';
import { Matrix } from '../engine/matrix';
import { Rng } from '../engine/rng';
import { DATASET_GENERATORS, type Dataset } from '../data/datasets';
import { CHALLENGES, type Target } from '../game/challenges';
import {
  emptyProgress,
  isCompleted,
  isUnlocked,
  markComplete,
  type Progress,
} from '../game/progress';
import { loadProgress, saveProgress, clearProgress } from '../state/persistence';
import { makeViewport, screenToWorld, worldToScreen } from '../viz/coords';
import { drawBoundary } from '../viz/boundary';
import { drawPoints } from '../viz/points';
import { drawRegression } from '../viz/regression';
import { drawLossChart } from '../viz/chart';
import { CLASS_A, CLASS_B, rgba, classColor } from '../viz/colors';
import {
  computeInspectorLayout,
  drawInspector,
  hitTestNeuron,
  neuronField,
  type InspectorLayout,
  type NeuronRef,
} from '../viz/inspector';
import { Studio, defaultConfig, type StudioConfig } from './studio';
import { coach, type CoachTip } from '../game/coach';
import { LESSONS, CONCEPTS, ioExplainer, TUTORIAL } from '../content/curriculum';
import {
  LiveMatch,
  LIVE_RULES,
  AI_DROP_INTERVAL,
  AI_TUNE_INTERVAL,
  pickSabotagePoint,
  liveTrainerPlan,
  liveTrainerHint,
  neuronsUsed,
  type Role,
  type Difficulty,
  type HintLevel,
} from '../game/versus';
import { Tutorial, hasSeenTutorial } from './tutorial';
import { modeFromPath, pathForMode, type Mode } from './router';

interface Pt {
  x: number;
  y: number;
  label: number;
}

const STEPS_PER_FRAME = 12;
const MAX_LAYERS = 4;
const MAX_WIDTH = 48;
const MIN_WIDTH = 1;
const DRAW_ID = 'draw';
const SANDBOX_N = 240;
const LESSONS_KEY = 'neuroforge.lessons';
const VERSUS_NUDGE_KEY = 'neuroforge.versusNudged';
const TELEMETRY_EVERY = 4; // frames between telemetry refreshes while training

/** In-place-updated HUD elements for a live match (built once per match — no flicker). */
interface VsHudEls {
  acc: HTMLElement;
  fill: HTMLElement;
  timer: HTMLElement;
  pts: HTMLElement;
  /** Optional mirror of the points-left readout in the left panel. */
  panelPts: HTMLElement | null;
}

interface VsState {
  match: LiveMatch;
  studio: Studio;
  humanRole: Role;
  aiVsAi: boolean;
  difficulty: Difficulty;
  hintLevel: HintLevel;
  saboClass: number;
  rng: Rng;
  /** Match-time (elapsedMs) thresholds for the next AI actions. */
  nextAiDropAt: number;
  nextAiTuneAt: number;
  finished: boolean;
  winner: Role | null;
  hud: VsHudEls | null;
}

export class App {
  private mode: Mode = 'learn';
  private studio = new Studio();
  private running = false;
  private frame = 0;

  private challengeIndex = 0;
  private sandboxDatasetId = 'circles';
  private drawClass = 0;
  private customPoints: Pt[] = [];
  private dataNudge = 0;

  private learnIndex = 0;
  private lessonsDone = new Set<string>();

  private progress: Progress = emptyProgress();

  private probe: { wx: number; wy: number } | null = null;
  private hoverNeuron: NeuronRef | null = null;
  private inspectorLayout: InspectorLayout | null = null;

  private vs: VsState | null = null;

  /** rAF timestamp of the previous frame (0 = first frame). */
  private lastFrameT = 0;
  /** Caches so per-frame UI refreshes only touch the DOM when content changes. */
  private statLabels: string[] = [];
  private statValueEls: HTMLElement[] = [];
  private statValues: string[] = [];
  private coachKey = '';
  /** Consecutive adversarial placements detected in sandbox draw mode (→ Versus nudge). */
  private adversarialStreak = 0;
  /** Live refs for the neuron-budget bar (when shown). */
  private budgetEls: { n: HTMLElement; fill: HTMLElement; budget: number } | null = null;
  /** A staged (not yet rebuilt) architecture change exists — flushed once per frame. */
  private archDirty = false;

  private refs: {
    runBtn: HTMLButtonElement;
    status: HTMLElement;
    solvedFlag: HTMLElement;
    arenaTitle: HTMLElement;
    arenaSub: HTMLElement;
    note: HTMLElement;
    legend: HTMLElement;
    coach: HTMLElement;
    arenaCanvas: HTMLCanvasElement;
    lossCanvas: HTMLCanvasElement;
    inspectorCanvas: HTMLCanvasElement;
    inspectorCaption: HTMLElement;
    ioBox: HTMLElement;
    statGrid: HTMLElement;
    leftPanel: HTMLElement;
    actionRow: HTMLElement;
    vsHud: HTMLElement;
    modeSeg: HTMLElement;
  } = {} as never;

  // ===== Mount ============================================================
  /** The app's base path ('/neuroforge/' in prod). Always ends with '/'. */
  private base(): string {
    return import.meta.env.BASE_URL;
  }

  mount(root: HTMLElement): void {
    this.progress = loadProgress();
    this.lessonsDone = loadLessons();
    const firstOpen = CHALLENGES.findIndex((c) => !isCompleted(this.progress, c.id));
    this.challengeIndex = firstOpen === -1 ? 0 : firstOpen;

    // Mode comes from the URL path when present (/learn, /challenges, /sandbox,
    // /versus); otherwise land new players in Learn and returning ones in Challenges.
    const fromUrl = modeFromPath(window.location.pathname, this.base());
    this.mode = fromUrl ?? (hasSeenTutorial() && this.lessonsDone.size > 0 ? 'challenge' : 'learn');

    clear(root);
    root.append(this.buildTopbar(), this.buildStage(), this.buildFooter());

    // Canonicalise the URL (strips index.html, lands on the named path).
    this.enterMode(this.mode, 'replace');
    window.addEventListener('popstate', () => {
      const m = modeFromPath(window.location.pathname, this.base()) ?? 'learn';
      if (m !== this.mode) this.enterMode(m, 'none');
    });
    window.addEventListener('resize', () => this.handleResize());
    requestAnimationFrame((t) => {
      this.handleResize();
      this.loop(t);
      if (!hasSeenTutorial()) new Tutorial(TUTORIAL).start();
    });
  }

  // ===== Top bar ==========================================================
  private buildTopbar(): HTMLElement {
    const seg = el('div', { class: 'seg', attrs: { role: 'tablist' }, dataset: { tour: 'modes' } }, [
      this.modeBtn('Learn', 'learn'),
      this.modeBtn('Challenges', 'challenge'),
      this.modeBtn('Sandbox', 'sandbox'),
      this.modeBtn('Versus', 'versus'),
    ]);
    this.refs.modeSeg = seg;

    const help = el(
      'button',
      { class: 'icon-btn', attrs: { title: 'Replay the tour' }, on: { click: () => new Tutorial(TUTORIAL).start() } },
      ['?'],
    );
    const concepts = el(
      'button',
      { class: 'icon-btn wide', attrs: { title: 'Glossary' }, on: { click: () => this.openConcepts() } },
      ['Concepts'],
    );

    // The logo is a real link to the app's home (the Learn course) — clicking it
    // also normalises any stale /index.html URL to the named path.
    const brand = el(
      'a',
      {
        class: 'brand',
        attrs: { href: pathForMode('learn', this.base()), 'aria-label': 'NeuroForge home' },
        on: {
          click: (e) => {
            e.preventDefault();
            this.enterMode('learn', 'push');
          },
        },
      },
      [
        el('div', { class: 'brand-mark' }, [neuronGlyph()]),
        el('div', { class: 'brand-titles' }, [
          el('div', { class: 'brand-title', html: 'Neuro<em>Forge</em>' }),
          el('div', { class: 'brand-tag' }, ['build · train · understand']),
        ]),
      ],
    );

    return el('header', { class: 'topbar' }, [
      brand,
      el('div', { class: 'topbar-actions' }, [seg, el('div', { class: 'seg ghost-seg' }, [concepts, help])]),
    ]);
  }

  private modeBtn(label: string, mode: Mode): HTMLButtonElement {
    return el(
      'button',
      {
        class: this.mode === mode ? 'seg-btn active' : 'seg-btn',
        on: { click: () => this.setMode(mode) },
      },
      [label],
    ) as HTMLButtonElement;
  }

  // ===== Stage ============================================================
  private buildStage(): HTMLElement {
    return el('main', { class: 'stage' }, [
      this.buildLeftPanel(),
      this.buildArenaPanel(),
      this.buildTelemetryPanel(),
    ]);
  }

  private buildLeftPanel(): HTMLElement {
    const left = el('section', { class: 'panel left-panel' });
    this.refs.leftPanel = left;
    return left;
  }

  // ----- Arena ------------------------------------------------------------
  private buildArenaPanel(): HTMLElement {
    const arenaCanvas = el('canvas', { attrs: { 'aria-label': 'training arena' } }) as HTMLCanvasElement;
    const solvedFlag = el('div', { class: 'solved-flag' }, ['Solved ✓']);
    this.refs.arenaCanvas = arenaCanvas;
    this.refs.solvedFlag = solvedFlag;

    const screen = el('div', { class: 'screen', dataset: { tour: 'arena' } }, [arenaCanvas, solvedFlag]);
    arenaCanvas.addEventListener('pointermove', (e) => this.onArenaPointer(e, 'move'));
    arenaCanvas.addEventListener('pointerdown', (e) => this.onArenaPointer(e, 'down'));
    arenaCanvas.addEventListener('pointerup', () => (this.painting = false));
    arenaCanvas.addEventListener('pointerleave', () => {
      this.painting = false;
      this.probe = null;
      this.redrawInspector();
      this.redrawArena();
    });

    const status = el('span', { class: 'status' }, [el('span', { class: 'dot' }), 'idle']);
    const arenaTitle = el('h2', {}, ['—']);
    const arenaSub = el('div', { class: 'eyebrow' }, ['—']);
    this.refs.status = status;
    this.refs.arenaTitle = arenaTitle;
    this.refs.arenaSub = arenaSub;

    const note = el('div', { class: 'note' });
    const legend = el('div', { class: 'legend' });
    const coachBox = el('div', { class: 'coach', dataset: { tour: 'coach' } });
    const vsHud = el('div', { class: 'vs-hud' });
    this.refs.note = note;
    this.refs.legend = legend;
    this.refs.coach = coachBox;
    this.refs.vsHud = vsHud;

    return el('section', { class: 'panel arena-panel' }, [
      el('div', { class: 'arena-head' }, [
        el('div', { class: 'arena-title' }, [arenaSub, arenaTitle]),
        status,
      ]),
      vsHud,
      el('div', { class: 'screen-wrap' }, [screen]),
      legend,
      note,
      coachBox,
    ]);
  }

  // ----- Telemetry + inspector -------------------------------------------
  private buildTelemetryPanel(): HTMLElement {
    const ioBox = el('div', { class: 'io-box' });
    this.refs.ioBox = ioBox;

    const statGrid = el('div', { class: 'stat-grid' });
    this.refs.statGrid = statGrid;

    const lossCanvas = el('canvas', { attrs: { 'aria-label': 'loss over time' } }) as HTMLCanvasElement;
    const inspectorCanvas = el('canvas', { attrs: { 'aria-label': 'network inspector' } }) as HTMLCanvasElement;
    this.refs.lossCanvas = lossCanvas;
    this.refs.inspectorCanvas = inspectorCanvas;
    inspectorCanvas.addEventListener('pointermove', (e) => this.onInspectorHover(e));
    inspectorCanvas.addEventListener('pointerleave', () => {
      this.hoverNeuron = null;
      this.redrawInspector();
      this.redrawArena();
      this.refreshInspectorCaption();
    });

    const inspectorCaption = el('div', { class: 'inspector-caption' });
    this.refs.inspectorCaption = inspectorCaption;

    return el('section', { class: 'panel' }, [
      el('div', { class: 'panel-head' }, [
        el('div', { class: 'eyebrow' }, ['telemetry']),
        el('h2', {}, ['Inside the network']),
      ]),
      ioBox,
      statGrid,
      el('div', { class: 'subpanel' }, [
        el('div', { class: 'subpanel-head' }, [el('div', { class: 'eyebrow' }, ['loss trace'])]),
        el('div', { class: 'canvas-frame trace' }, [lossCanvas]),
      ]),
      el('div', { class: 'subpanel', dataset: { tour: 'inspector' } }, [
        el('div', { class: 'subpanel-head' }, [
          el('div', { class: 'eyebrow' }, ['inspector']),
          el('span', { class: 'hint' }, ['hover a neuron · or the arena']),
        ]),
        el('div', { class: 'canvas-frame inspector' }, [inspectorCanvas]),
        inspectorCaption,
      ]),
      el('div', { class: 'hint' }, [
        'Every weight is updated by hand-derived backpropagation — gradient-checked, no ML libraries.',
      ]),
    ]);
  }

  private buildFooter(): HTMLElement {
    return el('footer', { class: 'footer' }, [
      el('span', {}, ['NeuroForge — a neural net built from scratch in TypeScript · ']),
      el(
        'a',
        { attrs: { href: 'https://github.com/website-and-game-maker/neuroforge', target: '_blank', rel: 'noopener' } },
        ['source on GitHub'],
      ),
    ]);
  }

  // ===== Mode routing =====================================================
  private setMode(mode: Mode): void {
    if (this.mode === mode) return;
    this.enterMode(mode, 'push');
  }

  /** Switch modes; `nav` controls how the URL reflects it (path-name routing). */
  private enterMode(mode: Mode, nav: 'push' | 'replace' | 'none'): void {
    this.running = false;
    this.mode = mode;
    this.probe = null;
    this.hoverNeuron = null;
    this.adversarialStreak = 0; // a new context is not a continuation of saboteur play
    document.querySelector('.toast')?.remove(); // a stale nudge shouldn't follow the user around
    Array.from(this.refs.modeSeg.children).forEach((c, i) =>
      c.classList.toggle('active', i === ['learn', 'challenge', 'sandbox', 'versus'].indexOf(mode)),
    );

    const target = pathForMode(mode, this.base());
    try {
      if (nav === 'push' && window.location.pathname !== target) {
        window.history.pushState({}, '', target);
      } else if (nav === 'replace' && window.location.pathname !== target) {
        window.history.replaceState({}, '', target);
      }
    } catch {
      /* history may be unavailable (e.g. sandboxed iframe) — the app still works */
    }

    // Entering or leaving Versus always abandons any in-progress match (start at setup).
    this.vs = null;

    if (mode === 'learn') this.loadLesson(this.learnIndex, true);
    else if (mode === 'challenge') this.loadChallenge(this.challengeIndex, true);
    else if (mode === 'sandbox') this.loadSandbox();
    else this.renderVersus();

    this.refs.vsHud.classList.toggle('show', mode === 'versus');
    if (mode !== 'versus') {
      this.refreshArenaHeader();
      this.redrawAll();
      this.refreshRunButton();
    }
  }

  private current(): Studio {
    return this.mode === 'versus' && this.vs ? this.vs.studio : this.studio;
  }

  // ===== Challenge / Learn / Sandbox loading =============================
  private loadChallenge(i: number, applyStarter: boolean): void {
    this.challengeIndex = i;
    const ch = CHALLENGES[i]!;
    this.studio.task = ch.task;
    if (applyStarter) this.studio.config = { ...ch.starter };
    this.studio.setData(ch.makeData(ch.trainSeed + this.dataNudge), ch.makeData(ch.testSeed + this.dataNudge));
    this.studio.rebuild(true);
    this.renderChallengePanel();
    this.refreshArenaHeader();
    this.redrawAll();
    this.refreshRunButton();
  }

  private loadLesson(i: number, applyConfig: boolean): void {
    this.learnIndex = i;
    const lesson = LESSONS[i]!;
    this.studio.task = lesson.task;
    if (applyConfig) this.studio.config = { ...lesson.config };
    const gen = DATASET_GENERATORS.find((g) => g.id === lesson.dataset)!;
    this.studio.setData(gen.make(SANDBOX_N, 100 + this.dataNudge), gen.make(SANDBOX_N, 200 + this.dataNudge));
    this.studio.rebuild(true);
    // Exploratory lessons (no pass condition) count as "got it" once visited, so the
    // course can be completed end to end.
    if (!lesson.check && !this.lessonsDone.has(lesson.id)) {
      this.lessonsDone.add(lesson.id);
      saveLessons(this.lessonsDone);
    }
    this.renderLearnPanel();
    this.refreshArenaHeader();
    this.redrawAll();
    this.refreshRunButton();
  }

  private loadSandbox(): void {
    const drawing = this.sandboxDatasetId === DRAW_ID;
    this.studio.task = drawing ? 'classification' : DATASET_GENERATORS.find((g) => g.id === this.sandboxDatasetId)!.kind;
    this.rebuildSandboxData();
    this.studio.rebuild(true);
    this.renderSandboxPanel();
    this.refreshArenaHeader();
    this.redrawAll();
    this.refreshRunButton();
  }

  private rebuildSandboxData(): void {
    if (this.sandboxDatasetId === DRAW_ID) {
      const ds = this.datasetFromPoints();
      this.studio.setData(ds, ds);
    } else {
      const gen = DATASET_GENERATORS.find((g) => g.id === this.sandboxDatasetId)!;
      this.studio.setData(gen.make(SANDBOX_N, 100 + this.dataNudge), gen.make(SANDBOX_N, 200 + this.dataNudge));
    }
  }

  private datasetFromPoints(): Dataset {
    const pts = this.customPoints;
    const X = pts.length ? Matrix.fromRows(pts.map((p) => [p.x, p.y])) : Matrix.zeros(0, 2);
    const Y = pts.length ? Matrix.fromRows(pts.map((p) => [p.label])) : Matrix.zeros(0, 1);
    return { kind: 'classification', name: 'Your points', inDim: 2, X, Y };
  }

  // ===== Shared control widgets ==========================================
  /** Architecture editor + hyperparameter sliders + action buttons, bound to a studio. */
  private buildControls(opts: { neuronBudget?: number; allowNewData?: boolean } = {}): HTMLElement {
    const arch = el('div', { class: 'group', dataset: { tour: 'arch' } }, [
      el('div', { class: 'group-label' }, ['Architecture']),
      this.buildArchEditor(opts.neuronBudget),
      this.buildSelectField('Hidden activation', ['relu', 'tanh', 'sigmoid'], this.studioConfig().activation, (v) => {
        this.current().setConfig({ activation: v });
        this.applyChange(true);
      }, (a) => a.toUpperCase()),
    ]);

    const hyper = el('div', { class: 'group' }, [
      el('div', { class: 'group-label' }, ['Hyperparameters']),
      this.buildSlider('Learning rate', 'lr', 0.005, 0.6, 0.005, (v) => v.toFixed(3)),
      this.buildSlider('Momentum', 'momentum', 0, 0.95, 0.05, (v) => v.toFixed(2)),
      this.buildSlider('L2 (weight decay)', 'l2', 0, 0.02, 0.0005, (v) => v.toFixed(4)),
      this.buildSelectField('Batch size', ['4', '8', '16', '32', '64'], String(this.studioConfig().batchSize), (v) => {
        this.current().setConfig({ batchSize: Number(v) });
        this.applyChange(false);
      }),
    ]);

    const runBtn = el('button', { class: 'btn primary', dataset: { tour: 'run' }, on: { click: () => this.toggleRun() } }, ['Train']);
    this.refs.runBtn = runBtn as HTMLButtonElement;
    const actionRow = el('div', { class: 'btns' }, [
      el('button', { class: 'btn ghost', on: { click: () => this.stepOnce() } }, ['Step']),
      el('button', { class: 'btn ghost', on: { click: () => this.resetWeights() } }, ['Reset']),
      opts.allowNewData !== false &&
        el('button', { class: 'btn ghost', on: { click: () => this.newData() } }, ['New data']),
    ]);
    this.refs.actionRow = actionRow;

    return el('div', { class: 'controls-block' }, [arch, hyper, el('div', { class: 'group' }, [runBtn, actionRow])]);
  }

  private studioConfig(): StudioConfig {
    return this.current().config;
  }

  private buildArchEditor(neuronBudget?: number): HTMLElement {
    const rows = el('div', { class: 'arch-rows' });
    const cfg = this.studioConfig();
    const inDim = this.current().inDim;
    rows.append(this.endcap(String(inDim), inDim === 2 ? 'inputs · x,y' : 'input · x'));
    cfg.hidden.forEach((width, idx) => {
      rows.append(el('span', { class: 'arrow' }, ['→']));
      rows.append(this.layerChip(width, idx));
    });
    rows.append(el('span', { class: 'arrow' }, ['→']));
    rows.append(this.endcap('1', this.current().task === 'classification' ? 'output · P' : 'output · y'));

    const used = neuronsUsed(cfg);
    const addBtn = el(
      'button',
      {
        class: 'add-layer',
        attrs: cfg.hidden.length >= MAX_LAYERS ? { disabled: 'true' } : {},
        on: { click: () => this.addLayer() },
      },
      ['+ add hidden layer'],
    );

    let budgetBar: HTMLElement | null = null;
    this.budgetEls = null;
    if (neuronBudget !== undefined) {
      const n = el('span', { class: used > neuronBudget ? 'budget-n over' : 'budget-n' }, [`${used} / ${neuronBudget}`]);
      const fill = el('div', {
        class: used > neuronBudget ? 'budget-fill over' : 'budget-fill',
        style: { width: `${Math.min(100, (used / neuronBudget) * 100)}%` },
      });
      budgetBar = el('div', { class: 'budget' }, [
        el('div', { class: 'budget-row' }, [el('span', {}, ['neuron budget']), n]),
        el('div', { class: 'budget-track' }, [fill]),
      ]);
      this.budgetEls = { n, fill, budget: neuronBudget };
    }

    return el('div', { class: 'arch' }, [rows, addBtn, budgetBar].filter(Boolean) as Node[]);
  }

  /** Live update of the neuron-budget readout (used mid-drag; no panel re-render). */
  private refreshBudgetBar(): void {
    const b = this.budgetEls;
    if (!b || !b.n.isConnected) return;
    const used = neuronsUsed(this.studioConfig());
    b.n.textContent = `${used} / ${b.budget}`;
    b.n.className = used > b.budget ? 'budget-n over' : 'budget-n';
    b.fill.className = used > b.budget ? 'budget-fill over' : 'budget-fill';
    b.fill.style.width = `${Math.min(100, (used / b.budget) * 100)}%`;
  }

  private endcap(n: string, label: string): HTMLElement {
    return el('div', { class: 'endcap' }, [
      el('span', { class: 'cap-n' }, [n]),
      el('span', { class: 'cap-l' }, [label]),
    ]);
  }

  private layerChip(width: number, idx: number): HTMLElement {
    // Cap the slider's reachable range by the (versus) neuron budget up front.
    const others = this.studioConfig().hidden.reduce((s, w) => s + w, 0) - width;
    const max = Math.min(32, this.neuronBudget() - others);
    const label = el('span', { class: 'chip-w' }, [String(width)]);
    const slider = el('input', {
      class: 'chip-range',
      attrs: { type: 'range', min: String(MIN_WIDTH), max: String(Math.max(MIN_WIDTH, max)), step: '1', value: String(width), 'aria-label': `layer ${idx + 1} width` },
      on: {
        // While dragging: update config + readouts IN PLACE and mark the engine
        // dirty — the rebuild happens at most once per frame in loop(). Re-rendering
        // the panel here would destroy the slider mid-drag and kill the gesture.
        input: (e) => {
          this.stageWidth(idx, Number((e.target as HTMLInputElement).value));
          label.textContent = String(this.studioConfig().hidden[idx]);
          this.refreshBudgetBar();
        },
        // Interaction committed (note: keyboard arrows fire this per press): flush,
        // resync the panel, and hand focus back to the rebuilt slider so keyboard
        // adjustment keeps working.
        change: (e) => {
          const hadFocus = document.activeElement === e.target;
          this.flushArch();
          this.rerenderLeft();
          if (hadFocus) {
            this.refs.leftPanel.querySelectorAll<HTMLInputElement>('.chip-range')[idx]?.focus();
          }
        },
      },
    });
    return el('div', { class: 'layer-chip' }, [
      label,
      slider,
      el('div', { class: 'chip-controls' }, [
        el('button', { class: 'chip-btn', on: { click: () => this.changeWidth(idx, -1) } }, ['−']),
        el('button', { class: 'chip-btn', on: { click: () => this.changeWidth(idx, +1) } }, ['+']),
        el('button', { class: 'chip-remove', on: { click: () => this.removeLayer(idx) } }, ['✕']),
      ]),
    ]);
  }

  private buildSlider(
    label: string,
    key: 'lr' | 'momentum' | 'l2',
    min: number,
    max: number,
    step: number,
    fmt: (v: number) => string,
  ): HTMLElement {
    const value = el('span', { class: 'field-value' }, [fmt(this.studioConfig()[key])]);
    const input = el('input', {
      attrs: { type: 'range', min: String(min), max: String(max), step: String(step), value: String(this.studioConfig()[key]), 'aria-label': label },
      on: {
        input: (e) => {
          const v = Number((e.target as HTMLInputElement).value);
          this.current().setConfig({ [key]: v } as Partial<StudioConfig>);
          value.textContent = fmt(v);
          this.applyChange(false);
        },
      },
    });
    return el('div', { class: 'field' }, [
      el('div', { class: 'field-row' }, [el('span', { class: 'field-label' }, [label]), value]),
      input,
    ]);
  }

  private buildSelectField(
    label: string,
    options: string[],
    selected: string,
    onChange: (v: string) => void,
    fmt: (v: string) => string = (v) => v,
  ): HTMLElement {
    const sel = el(
      'select',
      { attrs: { 'aria-label': label }, on: { change: (e) => onChange((e.target as HTMLSelectElement).value) } },
      options.map((o) => el('option', { attrs: { value: o } }, [fmt(o)])),
    ) as HTMLSelectElement;
    sel.value = selected;
    return el('div', { class: 'field' }, [
      el('div', { class: 'field-row' }, [el('span', { class: 'field-label' }, [label])]),
      sel,
    ]);
  }

  // ===== Left-panel renderers ============================================
  private renderChallengePanel(): void {
    const list = el('div', { class: 'challenges' });
    CHALLENGES.forEach((ch, i) => {
      const unlocked = isUnlocked(this.progress, CHALLENGES, i);
      const done = isCompleted(this.progress, ch.id);
      const cls = ['challenge'];
      if (i === this.challengeIndex) cls.push('active');
      if (!unlocked) cls.push('locked');
      if (done) cls.push('done');
      const targetTxt = ch.target.kind === 'accuracy' ? `≥ ${(ch.target.min * 100).toFixed(0)}% accuracy` : `≤ ${ch.target.max} MSE`;
      list.append(
        el(
          'button',
          {
            class: cls.join(' '),
            attrs: unlocked ? {} : { disabled: 'true' },
            on: { click: () => unlocked && this.loadChallenge(i, true) },
          },
          [
            el('span', { class: 'challenge-idx' }, [done ? '✓' : String(i + 1)]),
            el('span', { class: 'challenge-meta' }, [
              el('span', { class: 'challenge-name' }, [ch.title]),
              el('span', { class: 'challenge-target' }, [targetTxt]),
            ]),
            el('span', { class: 'challenge-state' }, [unlocked ? (done ? '' : '○') : lockGlyph()]),
          ],
        ),
      );
    });

    this.setLeft([
      this.panelHead('control deck', 'Build & tune'),
      el('div', { class: 'group' }, [
        el('div', { class: 'group-label' }, ['Challenge']),
        list,
        el('button', { class: 'reset-progress', attrs: { title: 'Clear saved progress' }, on: { click: () => this.resetProgress() } }, ['reset progress']),
      ]),
      this.buildControls(),
    ]);
  }

  private resetProgress(): void {
    clearProgress();
    this.progress = emptyProgress();
    this.challengeIndex = 0;
    this.loadChallenge(0, true);
  }

  private renderLearnPanel(): void {
    const lesson = LESSONS[this.learnIndex]!;
    const done = this.lessonsDone.has(lesson.id);

    const dots = el(
      'div',
      { class: 'lesson-dots' },
      LESSONS.map((l, i) =>
        el('button', {
          class: ['ldot', i === this.learnIndex ? 'active' : '', this.lessonsDone.has(l.id) ? 'done' : ''].join(' '),
          attrs: { title: l.title, 'aria-label': l.title },
          on: { click: () => this.loadLesson(i, true) },
        }),
      ),
    );

    const body = el('div', { class: 'lesson-body' }, lesson.body.map((p) => renderRich('p', 'lesson-p', p)));

    const card = el('div', { class: 'lesson-card' }, [
      dots,
      el('div', { class: 'lesson-eyebrow' }, [`Lesson ${this.learnIndex + 1} / ${LESSONS.length}${done ? ' · ✓ got it' : ''}`]),
      el('h3', { class: 'lesson-title' }, [lesson.title]),
      el('div', { class: 'lesson-idea' }, [lesson.bigIdea]),
      body,
      el('div', { class: 'lesson-do' }, [el('span', { class: 'lesson-do-tag' }, ['Try it']), renderRich('div', 'lesson-do-txt', lesson.doThis)]),
      el('div', { class: 'lesson-nav' }, [
        this.learnIndex > 0 && el('button', { class: 'btn ghost', on: { click: () => this.loadLesson(this.learnIndex - 1, true) } }, ['← Prev']),
        this.learnIndex < LESSONS.length - 1
          ? el('button', { class: 'btn primary', on: { click: () => this.loadLesson(this.learnIndex + 1, true) } }, ['Next →'])
          : el('button', { class: 'btn primary', on: { click: () => this.setMode('challenge') } }, ['To Challenges →']),
      ].filter(Boolean) as Node[]),
    ]);

    this.setLeft([this.panelHead('guided course', 'Learn by doing'), card, this.buildControls()]);
  }

  private renderSandboxPanel(): void {
    const select = el(
      'select',
      {
        attrs: { 'aria-label': 'Dataset' },
        on: {
          change: (e) => {
            this.sandboxDatasetId = (e.target as HTMLSelectElement).value;
            this.dataNudge = 0;
            this.loadSandbox();
          },
        },
      },
      [
        ...DATASET_GENERATORS.map((g) =>
          el('option', { attrs: { value: g.id } }, [`${g.label} (${g.kind === 'classification' ? 'classify' : 'regress'})`]),
        ),
        el('option', { attrs: { value: DRAW_ID } }, ['Draw your own (classify)']),
      ],
    ) as HTMLSelectElement;
    select.value = this.sandboxDatasetId;

    const items: Node[] = [
      el('div', { class: 'field' }, [el('div', { class: 'field-row' }, [el('span', { class: 'field-label' }, ['Dataset'])]), select]),
    ];
    if (this.sandboxDatasetId === DRAW_ID) {
      items.push(
        el('div', { class: 'field' }, [
          el('div', { class: 'field-row' }, [el('span', { class: 'field-label' }, ['Click or drag on the arena to paint'])]),
          this.classToggle(this.drawClass, (c) => {
            this.drawClass = c;
            this.renderSandboxPanel();
          }),
        ]),
        el('button', { class: 'btn ghost', on: { click: () => { this.customPoints = []; this.adversarialStreak = 0; this.rebuildSandboxData(); this.studio.rebuild(true); this.redrawAll(); } } }, ['Clear points']),
      );
    }

    this.setLeft([
      this.panelHead('control deck', 'Free play'),
      el('div', { class: 'group' }, [el('div', { class: 'group-label' }, ['Dataset']), ...items]),
      this.buildControls({ allowNewData: this.sandboxDatasetId !== DRAW_ID }),
    ]);
  }

  private classToggle(value: number, onChange: (c: number) => void): HTMLElement {
    return el('div', { class: 'seg class-seg' }, [
      el('button', { class: value === 0 ? 'seg-btn active' : 'seg-btn', on: { click: () => onChange(0) } }, ['● Class A']),
      el('button', { class: value === 1 ? 'seg-btn active' : 'seg-btn', on: { click: () => onChange(1) } }, ['● Class B']),
    ]);
  }

  private setLeft(children: Node[]): void {
    clear(this.refs.leftPanel);
    this.refs.leftPanel.append(...children);
  }

  private panelHead(eyebrow: string, title: string): HTMLElement {
    return el('div', { class: 'panel-head' }, [el('div', { class: 'eyebrow' }, [eyebrow]), el('h2', {}, [title])]);
  }

  // ===== Architecture mutations ==========================================
  /** In Versus the human Trainer is capped to the neuron budget; otherwise unconstrained. */
  private neuronBudget(): number {
    return this.mode === 'versus' && this.vs && this.humanIsTrainer() ? LIVE_RULES.neuronBudget : Infinity;
  }

  private addLayer(): void {
    const cfg = this.studioConfig();
    if (cfg.hidden.length >= MAX_LAYERS) return;
    const remaining = this.neuronBudget() - neuronsUsed(cfg);
    if (remaining < 1) return; // out of budget
    this.current().setConfig({ hidden: [...cfg.hidden, Math.min(8, remaining)] });
    this.applyChange(true);
    this.rerenderLeft();
  }
  private removeLayer(idx: number): void {
    const cfg = this.studioConfig();
    this.current().setConfig({ hidden: cfg.hidden.filter((_, i) => i !== idx) });
    this.applyChange(true);
    this.rerenderLeft();
  }
  private changeWidth(idx: number, delta: number): void {
    this.setWidth(idx, (this.studioConfig().hidden[idx] ?? 0) + delta);
  }
  /**
   * Stage a width change: clamp it into the (versus) budget, write the config, and
   * mark the engine dirty. The actual rebuild is coalesced to one per frame because
   * pointer 'input' events can outrun the display's frame rate.
   */
  private stageWidth(idx: number, width: number): void {
    const next = [...this.studioConfig().hidden];
    const others = next.reduce((s, w) => s + w, 0) - (next[idx] ?? 0);
    const budgetCap = this.neuronBudget() - others;
    const maxForThis = Math.min(MAX_WIDTH, budgetCap);
    next[idx] = Math.max(MIN_WIDTH, Math.min(maxForThis, width));
    this.current().setConfig({ hidden: next });
    this.archDirty = true;
  }

  /** Apply any staged architecture change immediately (loop() also calls this per frame). */
  private flushArch(): void {
    if (!this.archDirty) return;
    this.archDirty = false;
    this.applyChange(true);
  }

  /** Set a layer's width synchronously. `rerender=false` keeps the panel DOM. */
  private setWidth(idx: number, width: number, rerender = true): void {
    this.stageWidth(idx, width);
    this.flushArch();
    if (rerender) this.rerenderLeft();
  }

  private rerenderLeft(): void {
    if (this.mode === 'learn') this.renderLearnPanel();
    else if (this.mode === 'challenge') this.renderChallengePanel();
    else if (this.mode === 'sandbox') this.renderSandboxPanel();
    else this.renderVersus();
    this.refreshRunButton();
  }

  // ===== Engine actions ==================================================
  private applyChange(structural: boolean): void {
    // In a live Versus match the duel keeps running while you retune — pausing on
    // every tweak would fight the real-time format. Everywhere else, pause.
    if (!(this.mode === 'versus' && this.vs && !this.vs.finished)) this.running = false;
    this.current().rebuild(structural);
    this.hideSolved();
    this.redrawAll();
    this.refreshRunButton();
  }

  private toggleRun(): void {
    if (!this.current().trainer) return;
    this.running = !this.running;
    this.refreshRunButton();
    if (!this.running) this.redrawAll(); // settle the final (throttled) telemetry on pause
  }
  private stepOnce(): void {
    if (!this.current().trainer) return;
    this.current().step(40);
    this.redrawAll();
    if (this.mode === 'versus') this.refreshVsHud();
  }
  private resetWeights(): void {
    this.running = false;
    this.current().reseed();
    this.current().rebuild(true);
    this.hideSolved();
    this.redrawAll();
    this.refreshRunButton();
    if (this.mode === 'versus') this.refreshVsHud();
  }
  private newData(): void {
    this.running = false;
    this.adversarialStreak = 0; // fresh data = fresh context for the versus nudge
    if (this.mode === 'sandbox' && this.sandboxDatasetId === DRAW_ID) this.customPoints = [];
    else this.dataNudge++;
    this.current().reseed();
    if (this.mode === 'learn') this.loadLesson(this.learnIndex, false);
    else if (this.mode === 'challenge') this.loadChallenge(this.challengeIndex, false);
    else this.loadSandbox();
  }

  // ===== Run loop ========================================================
  private loop(t: number): void {
    // Real frame delta (clamped: a backgrounded tab must not teleport match time).
    const dt = this.lastFrameT === 0 ? 16 : Math.min(250, Math.max(0, t - this.lastFrameT));
    this.lastFrameT = t;

    // Apply at most one staged architecture rebuild per frame (slider drags).
    this.flushArch();

    // Apply at most one coalesced probe redraw per frame (set by pointermove).
    if (this.probeDirty) {
      this.probeDirty = false;
      this.redrawInspector();
      this.refreshInspectorCaption();
      if (!this.running) this.redrawArena();
    }

    if (this.mode === 'versus' && this.vs) this.vsTick(dt);
    else if (this.running && this.current().trainer) {
      this.current().step(STEPS_PER_FRAME);
      this.frame++;
      this.redrawArena();
      if (this.frame % TELEMETRY_EVERY === 0) this.refreshTelemetry();
      if (this.frame % 6 === 0) this.redrawInspector();
    }
    requestAnimationFrame((t2) => this.loop(t2));
  }

  private redrawAll(): void {
    this.redrawArena();
    this.refreshTelemetry();
    this.redrawInspector();
    this.refreshInspectorCaption();
    this.refreshIoBox();
    // Crosshair cursor whenever the arena accepts point placement.
    this.refs.arenaCanvas.parentElement?.classList.toggle('paintable', this.placementMode() !== null);
  }

  // ===== Arena rendering =================================================
  private arenaViewport(w: number, h: number) {
    return this.current().task === 'classification'
      ? makeViewport(w, h, 12, -1.3, 1.3, -1.3, 1.3)
      : makeViewport(w, h, 16, -1.15, 1.15, -1.15, 1.15);
  }

  private redrawArena(): void {
    if (this.mode === 'versus' && !this.vs) {
      this.clearCanvas(this.refs.arenaCanvas);
      return;
    }
    const studio = this.current();
    const { ctx, w, h } = fitCanvas(this.refs.arenaCanvas);
    ctx.clearRect(0, 0, w, h);
    if (!studio.net) return;
    const vp = this.arenaViewport(w, h);

    if (studio.task === 'classification') {
      // Hovering a hidden neuron replaces the boundary with that neuron's own field.
      const hov = this.hoverNeuron;
      if (hov && hov.col > 0 && hov.col < (studio.config.hidden.length + 1)) {
        drawBoundary(ctx, vp, neuronField(studio.net, hov), 52);
      } else {
        drawBoundary(ctx, vp, studio.forward, 52);
      }
      if (this.mode === 'versus' && this.vs) {
        this.drawVersusPoints(ctx, vp);
      } else if (studio.trainData.X.rows > 0) {
        drawPoints(ctx, vp, studio.trainData.X, studio.trainData.Y);
      }
      this.drawProbeMarker(ctx, vp);
    } else {
      this.drawRegressionGrid(ctx, w, h);
      drawRegression(ctx, vp, studio.trainData.X, studio.trainData.Y, studio.forward);
    }
  }

  /** Draw every placed point; AI drops from the last ~2.5s get a fading warn ring. */
  private drawVersusPoints(ctx: CanvasRenderingContext2D, vp: ReturnType<typeof makeViewport>): void {
    const v = this.vs;
    if (!v) return;
    const now = v.match.elapsedMs;
    for (const p of v.match.points) {
      const { sx, sy } = worldToScreen(vp, p.x, p.y);
      ctx.beginPath();
      ctx.arc(sx, sy, 4.5, 0, Math.PI * 2);
      ctx.fillStyle = rgba(classColor(p.label));
      ctx.fill();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = 'rgba(10,12,20,0.9)';
      ctx.stroke();
      if (p.byAi) {
        const age = now - p.atMs;
        if (age < 2500) {
          const a = 0.95 * (1 - age / 2500);
          ctx.beginPath();
          ctx.arc(sx, sy, 8.5 + (age / 2500) * 4, 0, Math.PI * 2);
          ctx.strokeStyle = `rgba(255,206,107,${a.toFixed(3)})`;
          ctx.lineWidth = 1.5;
          ctx.stroke();
        }
      }
    }
  }

  private drawProbeMarker(ctx: CanvasRenderingContext2D, vp: ReturnType<typeof makeViewport>): void {
    if (!this.probe) return;
    const { sx, sy } = worldToScreen(vp, this.probe.wx, this.probe.wy);
    ctx.beginPath();
    ctx.arc(sx, sy, 7, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(92,225,176,0.95)';
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(sx - 11, sy);
    ctx.lineTo(sx + 11, sy);
    ctx.moveTo(sx, sy - 11);
    ctx.lineTo(sx, sy + 11);
    ctx.strokeStyle = 'rgba(92,225,176,0.4)';
    ctx.lineWidth = 1;
    ctx.stroke();
  }

  private drawRegressionGrid(ctx: CanvasRenderingContext2D, w: number, h: number): void {
    ctx.strokeStyle = 'rgba(150,170,215,0.10)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, h / 2);
    ctx.lineTo(w, h / 2);
    ctx.moveTo(w / 2, 0);
    ctx.lineTo(w / 2, h);
    ctx.stroke();
  }

  // ===== Telemetry =======================================================
  private refreshTelemetry(): void {
    const studio = this.current();
    const m = studio.metrics();

    // Loss trace.
    const lc = fitCanvas(this.refs.lossCanvas);
    drawLossChart(lc.ctx, studio.trainer?.lossHistory ?? [], lc.w, lc.h);

    // Stat tiles.
    if (this.mode === 'versus') {
      // On the setup screen (no live match) the panel already rendered its own stats —
      // nothing to refresh, and versusCoachTip would dereference a null match.
      if (!this.vs) return;
      const v = this.vs;
      const acc = v.studio.net ? v.match.score(v.studio.forward) : 1;
      this.refreshVsHud();
      this.renderStats([
        { label: 'accuracy · all points', value: `${(acc * 100).toFixed(1)}%`, good: acc >= LIVE_RULES.winThreshold },
        { label: 'neurons used', value: `${neuronsUsed(v.studio.config)} / ${LIVE_RULES.neuronBudget}` },
        { label: 'time left', value: fmtClock(v.match.remainingMs) },
        { label: 'points left', value: String(v.match.budgetLeft) },
      ]);
      this.renderCoach(this.versusCoachTip());
      return;
    }

    const isClass = m.task === 'classification';
    const target = this.mode === 'challenge' || this.mode === 'learn' ? this.currentTarget() : null;
    let passed = false;
    if (m.hasData) {
      if (isClass) passed = target?.kind === 'accuracy' ? m.testScore >= target.min : false;
      else passed = target?.kind === 'mse' ? m.testScore <= target.max : false;
    }

    this.renderStats([
      {
        label: isClass ? 'test accuracy' : 'test MSE',
        value: m.hasData ? (isClass ? `${(m.testScore * 100).toFixed(1)}%` : m.testScore.toFixed(4)) : '—',
        good: passed,
      },
      {
        label: isClass ? 'train accuracy' : 'train MSE',
        value: m.hasData ? (isClass ? `${(m.trainScore * 100).toFixed(1)}%` : m.trainScore.toFixed(4)) : '—',
      },
      { label: 'loss', value: Number.isFinite(m.loss) ? m.loss.toFixed(4) : '—' },
      { label: 'steps', value: studio.steps.toLocaleString() },
    ]);

    // Coach.
    if (m.hasData) {
      this.renderCoach(
        coach({
          task: m.task,
          steps: m.steps,
          trainScore: m.trainScore,
          testScore: m.testScore,
          lossHistory: studio.trainer?.lossHistory ?? [],
          config: studio.config,
          target: target ?? undefined,
          pointCount: studio.trainData.X.rows,
        }),
      );
    } else {
      this.renderCoach({ tone: 'idle', title: 'Paint some points', body: 'Click or drag on the arena to add data, then press Train.' });
    }

    // Completion (challenge + learn).
    if ((this.mode === 'challenge' || this.mode === 'learn') && target && m.hasData) {
      this.handleCompletion(passed, m.steps > 0);
    }
  }

  /**
   * Update the stat tiles. Called every few frames during training, so it diffs:
   * if the labels are unchanged it only patches the values that actually changed —
   * rebuilding the DOM each refresh made the panel visibly flicker.
   */
  private renderStats(stats: Array<{ label: string; value: string; good?: boolean }>): void {
    const labels = stats.map((s) => s.label);
    const sameShape =
      labels.length === this.statLabels.length && labels.every((l, i) => l === this.statLabels[i]);

    if (!sameShape || this.statValueEls.some((e) => !e.isConnected)) {
      clear(this.refs.statGrid);
      this.statLabels = labels;
      this.statValueEls = [];
      this.statValues = [];
      for (const s of stats) {
        const value = el('div', { class: s.good ? 'stat-value good' : 'stat-value' }, [s.value]);
        this.statValueEls.push(value);
        this.statValues.push(s.value);
        this.refs.statGrid.append(
          el('div', { class: 'stat' }, [el('div', { class: 'stat-label' }, [s.label]), value]),
        );
      }
      return;
    }
    stats.forEach((s, i) => {
      const elV = this.statValueEls[i]!;
      if (this.statValues[i] !== s.value) {
        elV.textContent = s.value;
        this.statValues[i] = s.value;
      }
      elV.classList.toggle('good', !!s.good);
    });
  }

  /** Update the coach callout — skipped entirely when the tip hasn't changed (no flicker). */
  private renderCoach(tip: CoachTip): void {
    const key = `${tip.tone}|${tip.title}|${tip.body}`;
    if (key === this.coachKey && this.refs.coach.childElementCount > 0) return;
    this.coachKey = key;
    clear(this.refs.coach);
    this.refs.coach.className = `coach tone-${tip.tone}`;
    this.refs.coach.append(
      el('div', { class: 'coach-head' }, [el('span', { class: 'coach-dot' }), el('span', { class: 'coach-title' }, [tip.title])]),
      renderRich('div', 'coach-body', tip.body),
    );
  }

  private currentTarget(): Target | null {
    if (this.mode === 'challenge') return CHALLENGES[this.challengeIndex]!.target;
    if (this.mode === 'learn') return LESSONS[this.learnIndex]!.check ?? null;
    return null;
  }

  private handleCompletion(passed: boolean, trained: boolean): void {
    if (!passed || !trained) {
      this.hideSolved();
      return;
    }
    this.showSolved();
    if (this.mode === 'challenge') {
      const ch = CHALLENGES[this.challengeIndex]!;
      if (!isCompleted(this.progress, ch.id)) {
        const score = ch.target.kind === 'accuracy' ? this.current().metrics().testScore : 1;
        this.progress = markComplete(this.progress, ch.id, score);
        saveProgress(this.progress);
        this.renderChallengePanel();
      }
    } else if (this.mode === 'learn') {
      const lesson = LESSONS[this.learnIndex]!;
      if (!this.lessonsDone.has(lesson.id)) {
        this.lessonsDone.add(lesson.id);
        saveLessons(this.lessonsDone);
        this.renderLearnPanel();
      }
    }
  }

  // ===== Inspector =======================================================
  private redrawInspector(): void {
    if (this.mode === 'versus' && !this.vs) {
      this.clearCanvas(this.refs.inspectorCanvas);
      return;
    }
    const studio = this.current();
    if (!studio.net) return;
    const { ctx, w, h } = fitCanvas(this.refs.inspectorCanvas);
    const layout = computeInspectorLayout(studio.net, w, h, studio.task);
    this.inspectorLayout = layout;
    let probeCols: Matrix[] | null = null;
    if (this.probe) probeCols = studio.net.forwardVerbose(this.probeInput(this.probe.wx, this.probe.wy)).columns;
    drawInspector(ctx, layout, { task: studio.task, probe: probeCols, hover: this.hoverNeuron });
  }

  private probeInput(wx: number, wy: number): Matrix {
    return this.current().task === 'classification'
      ? new Matrix(1, 2, new Float64Array([wx, wy]))
      : new Matrix(1, 1, new Float64Array([wx]));
  }

  private refreshInspectorCaption(): void {
    const cap = this.refs.inspectorCaption;
    clear(cap);
    if (this.mode === 'versus' && !this.vs) return;
    const studio = this.current();
    if (!studio.net) return;
    if (this.hoverNeuron) {
      const h = this.hoverNeuron;
      const last = studio.config.hidden.length + 1;
      if (h.col === 0) {
        cap.append(renderRich('span', '', `**Input ${studio.task === 'classification' ? (h.idx === 0 ? 'x' : 'y') : 'x'}** — the ${h.idx === 0 ? 'horizontal' : 'vertical'} coordinate fed into the network.`));
      } else if (h.col === last) {
        cap.append(renderRich('span', '', studio.task === 'classification' ? '**Output** — probability the point is orange. The arena shades by this.' : '**Output** — the predicted value y.'));
      } else {
        cap.append(renderRich('span', '', `**Hidden ${h.col}·${h.idx + 1}** — one learned feature. The arena now shows the region *this neuron* responds to (orange = it fires, blue = it doesn’t).`));
      }
      return;
    }
    if (this.probe) {
      const out = studio.net.forward(this.probeInput(this.probe.wx, this.probe.wy)).data[0]!;
      const txt =
        studio.task === 'classification'
          ? `Probe (${this.probe.wx.toFixed(2)}, ${this.probe.wy.toFixed(2)}) → output **${out.toFixed(3)}** = ${(out * 100).toFixed(0)}% orange.`
          : `Probe x=${this.probe.wx.toFixed(2)} → predicted **${out.toFixed(3)}**.`;
      cap.append(renderRich('span', '', txt));
      return;
    }
    cap.append(renderRich('span', '', 'Hover the **arena** to watch a point flow through every neuron, or hover a **neuron** to see its role.'));
  }

  private refreshIoBox(): void {
    const box = this.refs.ioBox;
    clear(box);
    const io = ioExplainer(this.current().task);
    box.append(
      el('div', { class: 'io-title' }, [io.title]),
      el('ul', { class: 'io-list' }, io.lines.map((l) => renderRich('li', '', l))),
    );
  }

  // ===== Pointer interaction =============================================
  private painting = false;
  private lastPaint: { wx: number; wy: number } | null = null;
  private probeDirty = false;

  private onArenaPointer(e: PointerEvent, kind: 'move' | 'down'): void {
    const rect = this.refs.arenaCanvas.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return; // panel collapsed/transitioning
    const vp = this.arenaViewport(rect.width, rect.height);
    const { wx, wy } = screenToWorld(vp, e.clientX - rect.left, e.clientY - rect.top);
    if (!Number.isFinite(wx) || !Number.isFinite(wy)) return;

    const placing = this.placementMode();
    if (kind === 'down' && placing) {
      this.painting = true;
      this.lastPaint = null;
    }
    if (placing && (this.painting || kind === 'down')) {
      if (!this.lastPaint || Math.hypot(this.lastPaint.wx - wx, this.lastPaint.wy - wy) > 0.05) {
        this.placeAt(wx, wy);
        this.lastPaint = { wx, wy };
      }
    }

    // No probing on the Versus setup screen (the shared studio is unrelated there).
    if (this.mode === 'versus' && !this.vs) return;

    // Probe lives + draws are coalesced to one redraw per animation frame (see loop()).
    this.probe = { wx, wy };
    this.probeDirty = true;
  }

  private placementMode(): 'draw' | 'sabotage' | null {
    if (this.mode === 'sandbox' && this.sandboxDatasetId === DRAW_ID) return 'draw';
    // Live versus: a human saboteur can paint ANY class ANYWHERE, ANYTIME.
    if (this.mode === 'versus' && this.vs && !this.vs.finished && this.humanIsSaboteur()) return 'sabotage';
    return null;
  }

  private placeAt(wx: number, wy: number): void {
    const mode = this.placementMode();
    if (mode === 'draw') {
      this.detectAdversarialPlay(wx, wy, this.drawClass);
      this.customPoints.push({ x: wx, y: wy, label: this.drawClass });
      this.rebuildSandboxData();
      this.studio.rebuild(false);
      this.redrawAll();
      this.refreshRunButton();
    } else if (mode === 'sabotage' && this.vs) {
      if (!this.vs.match.addPoint(wx, wy, this.vs.saboClass, false)) return; // budget/clock
      this.syncVersusData(false);
      if (!this.running) this.redrawArena();
      this.refreshVsHud();
    }
  }

  /**
   * Sandbox → Versus nudge: notice when the player is *deliberately* contradicting a
   * trained model (dropping a point deep inside the opposite-coloured region). Guards
   * against false positives: the model must be trained on a real dataset, the
   * prediction must be confident, and it takes 4 strikes — while clearly cooperative
   * placements walk the streak back down. Shown once, ever.
   */
  private detectAdversarialPlay(wx: number, wy: number, label: number): void {
    if (this.versusNudgeSeen()) return;
    const s = this.studio;
    if (!s.net || s.steps < 150 || s.trainData.X.rows < 12) return;
    const p = s.forward(new Matrix(1, 2, new Float64Array([wx, wy]))).data[0]!;
    const adversarial = (p >= 0.7 && label === 0) || (p <= 0.3 && label === 1);
    const cooperative = (p >= 0.7 && label === 1) || (p <= 0.3 && label === 0);
    if (adversarial) this.adversarialStreak++;
    else if (cooperative) this.adversarialStreak = Math.max(0, this.adversarialStreak - 1);
    if (this.adversarialStreak >= 4) {
      this.markVersusNudgeSeen();
      this.showToast(
        'You’re playing saboteur',
        'Dropping points where the model is confidently wrong is a whole game here — with an AI on the other side, scoring, and a clock.',
        'Play Versus',
        () => this.setMode('versus'),
      );
    }
  }

  private versusNudgeSeen(): boolean {
    try {
      return localStorage.getItem(VERSUS_NUDGE_KEY) === '1';
    } catch {
      return false;
    }
  }
  private markVersusNudgeSeen(): void {
    try {
      localStorage.setItem(VERSUS_NUDGE_KEY, '1');
    } catch {
      /* ignore */
    }
  }

  /** Small dismissible toast (bottom-right), used for the Versus nudge. */
  private showToast(title: string, body: string, ctaLabel: string, onCta: () => void): void {
    document.querySelector('.toast')?.remove();
    const toast = el('div', { class: 'toast', attrs: { role: 'status' } }, [
      el('div', { class: 'toast-title' }, [title]),
      el('div', { class: 'toast-body' }, [body]),
      el('div', { class: 'toast-actions' }, [
        el('button', { class: 'btn ghost toast-btn', on: { click: () => toast.remove() } }, ['Dismiss']),
        el('button', { class: 'btn primary toast-btn', on: { click: () => { toast.remove(); onCta(); } } }, [ctaLabel]),
      ]),
    ]);
    // Auto-dismiss after a generous read window, paused while hovered.
    let timer = window.setTimeout(() => toast.remove(), 20_000);
    toast.addEventListener('pointerenter', () => window.clearTimeout(timer));
    toast.addEventListener('pointerleave', () => {
      timer = window.setTimeout(() => toast.remove(), 10_000);
    });
    document.body.append(toast);
  }

  private onInspectorHover(e: PointerEvent): void {
    if (!this.inspectorLayout) return;
    const rect = this.refs.inspectorCanvas.getBoundingClientRect();
    const hit = hitTestNeuron(this.inspectorLayout, e.clientX - rect.left, e.clientY - rect.top);
    if ((hit?.col ?? -1) !== (this.hoverNeuron?.col ?? -2) || (hit?.idx ?? -1) !== (this.hoverNeuron?.idx ?? -2)) {
      this.hoverNeuron = hit;
      this.redrawInspector();
      this.redrawArena();
      this.refreshInspectorCaption();
    }
  }

  // ===== Header / legend / status ========================================
  private refreshArenaHeader(): void {
    const studio = this.current();
    if (this.mode === 'challenge') {
      const ch = CHALLENGES[this.challengeIndex]!;
      this.refs.arenaSub.textContent = `challenge ${this.challengeIndex + 1} / ${CHALLENGES.length} · ${ch.task}`;
      this.refs.arenaTitle.textContent = ch.title;
      const goal = ch.target.kind === 'accuracy' ? `Goal: ≥ ${(ch.target.min * 100).toFixed(0)}% test accuracy.` : `Goal: test MSE ≤ ${ch.target.max}.`;
      clear(this.refs.note);
      this.refs.note.append(el('span', { class: 'note-icon' }, ['¶']), el('div', { class: 'note-body' }, [renderRich('span', '', `**${goal}** ${ch.why}`)]));
    } else if (this.mode === 'learn') {
      const lesson = LESSONS[this.learnIndex]!;
      this.refs.arenaSub.textContent = `learn · ${lesson.task}`;
      this.refs.arenaTitle.textContent = lesson.title;
      clear(this.refs.note);
      this.refs.note.append(el('span', { class: 'note-icon' }, ['¶']), el('div', { class: 'note-body' }, [renderRich('span', '', `**Why it matters.** ${lesson.reflect}`)]));
    } else if (this.mode === 'sandbox') {
      this.refs.arenaSub.textContent = `sandbox · ${studio.task}`;
      const name = this.sandboxDatasetId === DRAW_ID ? 'Draw your own' : DATASET_GENERATORS.find((g) => g.id === this.sandboxDatasetId)?.label ?? '—';
      this.refs.arenaTitle.textContent = name;
      clear(this.refs.note);
      this.refs.note.append(el('span', { class: 'note-icon' }, ['¶']), el('div', { class: 'note-body' }, ['Free play — change the architecture and hyperparameters and watch how the model copes. Hover the arena to probe it.']));
    }
    this.refreshLegend();
    this.refreshStatus();
  }

  private refreshLegend(): void {
    const lg = this.refs.legend;
    clear(lg);
    if (this.current().task === 'classification') {
      lg.append(legendKey(rgba(CLASS_A), 'Class A'), legendKey(rgba(CLASS_B), 'Class B'), el('span', { class: 'key' }, ['shaded = model’s guess']));
    } else {
      lg.append(legendKey('rgba(150,170,210,0.9)', 'samples'), legendKey(rgba(CLASS_B), 'predicted curve'));
    }
  }

  private showSolved(): void {
    this.refs.solvedFlag.classList.add('show');
    this.refs.status.className = 'status solved';
    clear(this.refs.status);
    this.refs.status.append(el('span', { class: 'dot' }), 'target reached');
  }
  private hideSolved(): void {
    this.refs.solvedFlag.classList.remove('show');
    this.refreshStatus();
  }
  private refreshStatus(): void {
    const s = this.refs.status;
    // A live match is its own state — the duel runs even while the human pauses training.
    if (this.mode === 'versus' && this.vs && !this.vs.finished) {
      s.className = 'status live';
      clear(s);
      s.append(el('span', { class: 'dot' }), 'match live');
      return;
    }
    if (this.mode === 'versus' && this.vs?.finished) {
      s.className = 'status solved';
      clear(s);
      s.append(el('span', { class: 'dot' }), 'match over');
      return;
    }
    s.className = this.running ? 'status live' : 'status';
    clear(s);
    s.append(el('span', { class: 'dot' }), this.running ? 'training' : 'idle');
  }
  private refreshRunButton(): void {
    const btn = this.refs.runBtn;
    // The run button only exists on panels with controls; skip if it's been detached.
    if (!btn || !btn.isConnected) return;
    btn.disabled = !this.current().trainer;
    btn.classList.toggle('running', this.running);
    btn.textContent = this.running ? 'Pause' : 'Train';
    this.refreshStatus();
  }

  private handleResize(): void {
    this.redrawAll();
  }

  // ===== Concepts drawer =================================================
  private openConcepts(): void {
    const overlay = el('div', { class: 'modal-overlay' });
    overlay.addEventListener('click', (e) => {
      if (e.target === overlay) overlay.remove();
    });
    const list = el('div', { class: 'concept-list' }, CONCEPTS.map((c) =>
      el('div', { class: 'concept' }, [
        el('div', { class: 'concept-term' }, [c.term]),
        el('div', { class: 'concept-short' }, [c.short]),
        el('div', { class: 'concept-long' }, [c.long]),
      ]),
    ));
    overlay.append(
      el('div', { class: 'modal' }, [
        el('div', { class: 'modal-head' }, [
          el('h2', {}, ['Concepts']),
          el('button', { class: 'icon-btn', on: { click: () => overlay.remove() } }, ['✕']),
        ]),
        list,
      ]),
    );
    document.body.append(overlay);
  }

  // ===== VERSUS (live) ====================================================
  private vsSetup = { humanRole: 'trainer' as Role, aiVsAi: false, difficulty: 'medium' as Difficulty, hintLevel: 1 as HintLevel };

  private renderVersus(): void {
    if (!this.vs) this.renderVersusSetup();
    else this.renderVersusMatchPanel();
  }

  private renderVersusSetup(): void {
    const s = this.vsSetup;
    const roleSeg = el('div', { class: 'seg' }, [
      el('button', { class: !s.aiVsAi && s.humanRole === 'trainer' ? 'seg-btn active' : 'seg-btn', on: { click: () => { s.aiVsAi = false; s.humanRole = 'trainer'; this.renderVersusSetup(); } } }, ['Be the Trainer']),
      el('button', { class: !s.aiVsAi && s.humanRole === 'saboteur' ? 'seg-btn active' : 'seg-btn', on: { click: () => { s.aiVsAi = false; s.humanRole = 'saboteur'; this.renderVersusSetup(); } } }, ['Be the Saboteur']),
      el('button', { class: s.aiVsAi ? 'seg-btn active' : 'seg-btn', on: { click: () => { s.aiVsAi = true; this.renderVersusSetup(); } } }, ['Watch AI vs AI']),
    ]);

    const diff = el('div', { class: 'seg' }, (['easy', 'medium', 'hard'] as Difficulty[]).map((d) =>
      el('button', { class: s.difficulty === d ? 'seg-btn active' : 'seg-btn', on: { click: () => { s.difficulty = d; this.renderVersusSetup(); } } }, [d]),
    ));

    const hint =
      s.aiVsAi || s.humanRole === 'saboteur'
        ? null
        : el('div', { class: 'field' }, [
            el('div', { class: 'field-row' }, [el('span', { class: 'field-label' }, ['Trainer hints'])]),
            el('div', { class: 'seg' }, ([['Full', 1], ['Some', 2], ['Minimal', 3], ['Off', 0]] as Array<[string, HintLevel]>).map(([label, lvl]) =>
              el('button', { class: s.hintLevel === lvl ? 'seg-btn active' : 'seg-btn', on: { click: () => { s.hintLevel = lvl; this.renderVersusSetup(); } } }, [label]),
            )),
          ]);

    const roleExplain =
      s.aiVsAi
        ? 'Watch an AI Saboteur paint a hostile dataset, live, while an AI Trainer races to keep fitting it. The accuracy meter is the scoreboard.'
        : s.humanRole === 'trainer'
          ? 'The board starts EMPTY. An AI Saboteur paints points live — your network trains continuously and you retune it on the fly. Be above the win line when the clock hits zero.'
          : 'The board starts EMPTY and the clock runs. Paint points anywhere, any class, anytime — the AI’s network adapts live. Drag to paint shapes its neuron budget can’t fit.';

    this.setLeft([
      this.panelHead('duel', 'Versus mode'),
      el('div', { class: 'group' }, [el('div', { class: 'group-label' }, ['Who are you?']), roleSeg, el('div', { class: 'mini-note' }, [roleExplain])]),
      el('div', { class: 'group' }, [el('div', { class: 'group-label' }, ['AI difficulty']), diff]),
      el('div', { class: 'group' }, [
        el('div', { class: 'group-label' }, ['Match rules']),
        hint,
        el('div', { class: 'rules-grid' }, [
          rule('Clock', fmtClock(LIVE_RULES.durationMs)),
          rule('Point budget', String(LIVE_RULES.pointBudget)),
          rule('Neuron budget', String(LIVE_RULES.neuronBudget)),
          rule('Trainer wins ≥', `${(LIVE_RULES.winThreshold * 100).toFixed(0)}%`),
        ]),
      ].filter(Boolean) as Node[]),
      el('button', { class: 'btn primary', on: { click: () => this.vsStart() } }, ['Start match ▶']),
    ]);

    // Neutral arena while setting up.
    this.refs.arenaSub.textContent = 'versus · setup';
    this.refs.arenaTitle.textContent = 'Trainer vs Saboteur';
    clear(this.refs.note);
    this.refs.note.append(el('span', { class: 'note-icon' }, ['¶']), el('div', { class: 'note-body' }, [renderRich('span', '', '**The duel, live.** The saboteur paints the dataset point by point; the trainer’s network fits it in real time under a neuron budget. No turns — whoever holds the accuracy line when the clock ends wins.')]));
    this.refreshLegend();
    clear(this.refs.vsHud);
    clear(this.refs.coach);
    this.coachKey = '';
    this.renderStats([{ label: 'status', value: 'ready' }, { label: 'difficulty', value: this.vsSetup.difficulty }, { label: 'clock', value: fmtClock(LIVE_RULES.durationMs) }, { label: 'mode', value: this.vsSetup.aiVsAi ? 'AI vs AI' : 'you' }]);
    this.refreshIoBox();
    this.clearCanvas(this.refs.arenaCanvas);
    this.clearCanvas(this.refs.inspectorCanvas);
    this.clearCanvas(this.refs.lossCanvas);
    clear(this.refs.inspectorCaption);
    // No run button exists on this panel, so reset the status pill directly (it could
    // otherwise be stuck showing 'training' from the previous mode).
    this.refreshStatus();
  }

  private clearCanvas(canvas: HTMLCanvasElement): void {
    const { ctx, w, h } = fitCanvas(canvas);
    ctx.clearRect(0, 0, w, h);
  }

  private vsStart(): void {
    const s = this.vsSetup;
    const studio = new Studio({ ...defaultConfig(), hidden: [8], activation: 'tanh', lr: 0.15 });
    studio.task = 'classification';
    const match = new LiveMatch(LIVE_RULES);
    studio.setData(match.dataset(), match.dataset());
    studio.rebuild(true);

    this.vs = {
      match,
      studio,
      humanRole: s.humanRole,
      aiVsAi: s.aiVsAi,
      difficulty: s.difficulty,
      hintLevel: s.hintLevel,
      saboClass: 0,
      rng: new Rng(20260611),
      nextAiDropAt: 800, // first AI drop lands fast so the match feels alive
      nextAiTuneAt: 0, // AI trainer configures itself immediately
      finished: false,
      winner: null,
      hud: null,
    };
    // Training runs from the opening whistle — this is a live duel.
    this.running = true;
    this.buildVsHud();
    this.renderVersusMatchPanel();
    this.refreshArenaHeaderVersus();
    this.refreshRunButton();
    this.redrawAll();
  }

  private humanIsTrainer(): boolean {
    return !!this.vs && !this.vs.aiVsAi && this.vs.humanRole === 'trainer';
  }
  private humanIsSaboteur(): boolean {
    return !!this.vs && !this.vs.aiVsAi && this.vs.humanRole === 'saboteur';
  }
  private aiIsTrainer(): boolean {
    return !!this.vs && (this.vs.aiVsAi || this.vs.humanRole === 'saboteur');
  }
  private aiIsSaboteur(): boolean {
    return !!this.vs && (this.vs.aiVsAi || this.vs.humanRole === 'trainer');
  }

  /** Point set changed → retrain over the new data, KEEPING the learned weights. */
  private syncVersusData(resetWeights: boolean): void {
    const v = this.vs;
    if (!v) return;
    const ds = v.match.dataset();
    v.studio.setData(ds, ds);
    v.studio.rebuild(resetWeights);
    // The first point of a match brings the trainer to life — unlock Train/Pause.
    this.refreshRunButton();
  }

  /**
   * One animation-frame of live match time. Everything happens concurrently:
   * the clock runs, the AI saboteur drips points in on its cadence, the AI trainer
   * retunes on its cadence, and whichever side is a network trains every frame.
   */
  private vsTick(dtMs: number): void {
    const v = this.vs;
    if (!v || v.finished) return;
    v.match.advance(dtMs);

    if (v.match.done) {
      this.vsFinish();
      return;
    }

    // --- AI saboteur: one point per cadence interval ---
    if (this.aiIsSaboteur() && v.match.elapsedMs >= v.nextAiDropAt && v.match.budgetLeft > 0) {
      const predict = v.studio.net && v.studio.steps > 0 ? v.studio.forward : null;
      const p = pickSabotagePoint(predict, v.match.points, v.difficulty, v.rng);
      if (p && v.match.addPoint(p.x, p.y, p.label, true)) {
        this.syncVersusData(false);
      }
      v.nextAiDropAt = v.match.elapsedMs + AI_DROP_INTERVAL[v.difficulty];
    }

    // --- AI trainer: periodic live retune (warm-keeps weights unless arch changes) ---
    if (this.aiIsTrainer() && v.match.elapsedMs >= v.nextAiTuneAt) {
      const plan = liveTrainerPlan(v.match.points.length, v.difficulty, LIVE_RULES.neuronBudget);
      const archChanged =
        plan.hidden.join(',') !== v.studio.config.hidden.join(',') ||
        plan.activation !== v.studio.config.activation;
      v.studio.config = { ...plan };
      if (archChanged) {
        v.studio.reseed();
        v.studio.rebuild(true);
      } else {
        v.studio.rebuild(false);
      }
      v.nextAiTuneAt = v.match.elapsedMs + AI_TUNE_INTERVAL;
    }

    // --- Training: AI trainers always run; a human trainer runs unless paused ---
    if (v.studio.trainer && (this.aiIsTrainer() || this.running)) {
      v.studio.step(STEPS_PER_FRAME);
    }

    this.frame++;
    this.redrawArena();
    if (this.frame % TELEMETRY_EVERY === 0) this.refreshTelemetry();
    if (this.frame % 6 === 0) this.redrawInspector();
  }

  /** Clock hit zero: freeze the duel and show the verdict. */
  private vsFinish(): void {
    const v = this.vs;
    if (!v || v.finished) return;
    v.finished = true;
    this.running = false;
    v.winner = v.studio.net ? v.match.winner(v.studio.forward) : 'trainer';
    this.renderVersusMatchPanel();
    this.refreshArenaHeaderVersus();
    this.refreshVsHud();
    this.redrawAll(); // also clears the paint cursor now that placement is closed
  }

  private renderVersusMatchPanel(): void {
    const v = this.vs;
    if (!v) return this.renderVersusSetup();

    const children: Node[] = [this.panelHead('duel', 'Versus mode')];

    if (v.finished) {
      const acc = v.studio.net ? v.match.score(v.studio.forward) : 1;
      const winner = v.winner ?? 'trainer';
      const forfeit = v.match.points.length < LIVE_RULES.minPoints;
      children.push(
        el('div', { class: `vs-result ${winner}` }, [
          el('div', { class: 'vs-result-tag' }, [winner === 'trainer' ? 'Trainer wins' : 'Saboteur wins']),
          el('div', { class: 'vs-result-acc' }, [forfeit ? `only ${v.match.points.length} points placed` : `${(acc * 100).toFixed(1)}% accuracy at the bell`]),
          el('div', { class: 'mini-note' }, [
            forfeit
              ? 'Too few points landed to contest the board — the trainer holds it by default.'
              : winner === 'trainer'
                ? 'The network kept absorbing every poisoned point in real time. Capacity and tuning held.'
                : 'The saboteur painted a shape the neuron budget couldn’t bend around before the clock ran out.',
          ]),
        ]),
        el('button', { class: 'btn primary', on: { click: () => { this.vs = null; this.renderVersusSetup(); } } }, ['New match']),
      );
      this.setLeft(children);
      return;
    }

    if (this.humanIsTrainer()) {
      children.push(
        el('div', { class: 'turn-banner trainer' }, ['LIVE — defend the accuracy line']),
        el('div', { class: 'mini-note' }, ['The AI is painting hostile points in real time. Retune anything below — training never stops unless you pause it.']),
        this.buildControls({ neuronBudget: LIVE_RULES.neuronBudget, allowNewData: false }),
      );
    } else if (this.humanIsSaboteur()) {
      const pts = el('div', { class: 'vs-budget' }, [`${v.match.points.length} / ${LIVE_RULES.pointBudget} points placed`]);
      if (v.hud) v.hud.panelPts = pts;
      children.push(
        el('div', { class: 'turn-banner saboteur' }, ['LIVE — confuse the network']),
        el('div', { class: 'mini-note' }, ['Click or **drag** on the arena to paint points — any class, anywhere, anytime. The model adapts live, so paint shapes its neuron budget can’t fit (stripes, checkers, islands).']),
        el('div', { class: 'field' }, [el('div', { class: 'field-row' }, [el('span', { class: 'field-label' }, ['Painting'])]), this.classToggle(v.saboClass, (c) => { v.saboClass = c; this.renderVersusMatchPanel(); })]),
        pts,
      );
    } else {
      children.push(
        el('div', { class: 'turn-banner trainer' }, ['LIVE — AI vs AI']),
        el('div', { class: 'mini-note' }, ['The Saboteur paints a hostile dataset while the Trainer retunes and trains nonstop. Watch the meter wrestle around the win line.']),
        el('button', { class: 'btn ghost', on: { click: () => { this.vs = null; this.renderVersus(); } } }, ['Stop · new setup']),
      );
    }

    this.setLeft(children);
    this.refreshRunButton();
  }

  private refreshArenaHeaderVersus(): void {
    const v = this.vs;
    if (!v) return;
    this.refs.arenaSub.textContent = `versus · ${v.aiVsAi ? 'AI vs AI' : `you are the ${v.humanRole}`} · ${v.difficulty}`;
    this.refs.arenaTitle.textContent = 'Trainer vs Saboteur';
    this.refreshLegend();
    this.refs.legend.append(legendKey('rgba(255,206,107,0.95)', 'fresh AI drop'));
    this.refreshStatus();
  }

  /** Build the HUD skeleton once per match; refreshVsHud then only patches values. */
  private buildVsHud(): void {
    const v = this.vs;
    if (!v) return;
    const acc = el('span', { class: 'vs-acc' }, ['—']);
    const fill = el('div', { class: 'vs-fill' });
    const timer = el('span', { class: 'vs-timer' }, [fmtClock(LIVE_RULES.durationMs)]);
    const pts = el('span', { class: 'vs-pts' }, [`${LIVE_RULES.pointBudget} pts`]);
    clear(this.refs.vsHud);
    this.refs.vsHud.append(
      el('div', { class: 'vs-meter' }, [
        el('div', { class: 'vs-meter-head' }, [
          el('span', { class: 'vs-side trainer' }, ['Trainer']),
          acc,
          el('span', { class: 'vs-side saboteur' }, ['Saboteur']),
        ]),
        el('div', { class: 'vs-track' }, [
          fill,
          el('div', { class: 'vs-threshold', style: { left: `${LIVE_RULES.winThreshold * 100}%` } }),
        ]),
        el('div', { class: 'vs-rounds' }, [
          timer,
          el('span', { class: 'vs-sep' }, ['·']),
          pts,
          el('span', { class: 'vs-sep' }, ['·']),
          el('span', {}, [`win line ${(LIVE_RULES.winThreshold * 100).toFixed(0)}%`]),
        ]),
      ]),
    );
    v.hud = { acc, fill, timer, pts, panelPts: null };
  }

  /** Patch the live HUD in place — no DOM rebuilds, no flicker. */
  private refreshVsHud(): void {
    const v = this.vs;
    if (!v || !v.hud) return;
    const acc = v.studio.net ? v.match.score(v.studio.forward) : 1;
    const pct = Math.max(0, Math.min(100, acc * 100));
    v.hud.acc.textContent = `${pct.toFixed(0)}%`;
    v.hud.fill.style.width = `${pct}%`;
    const remaining = v.match.remainingMs;
    v.hud.timer.textContent = fmtClock(remaining);
    v.hud.timer.classList.toggle('urgent', remaining <= 15_000 && !v.finished);
    v.hud.pts.textContent = `${v.match.budgetLeft} pts left`;
    if (v.hud.panelPts && v.hud.panelPts.isConnected) {
      v.hud.panelPts.textContent = `${v.match.points.length} / ${LIVE_RULES.pointBudget} points placed`;
    }
  }

  private versusCoachTip(): CoachTip {
    const v = this.vs!;
    const acc = v.studio.net ? v.match.score(v.studio.forward) : 1;
    if (v.finished) {
      const w = v.winner ?? 'trainer';
      return { tone: w === 'trainer' ? 'success' : 'warn', title: w === 'trainer' ? 'Trainer wins' : 'Saboteur wins', body: `Final accuracy ${(acc * 100).toFixed(0)}% against a win line of ${(LIVE_RULES.winThreshold * 100).toFixed(0)}%.` };
    }
    if (this.humanIsTrainer()) {
      const txt = liveTrainerHint(v.match, v.studio.config, acc, v.hintLevel);
      return { tone: acc >= LIVE_RULES.winThreshold ? 'good' : 'info', title: 'Trainer hint', body: txt || 'Hints are off — you’re on your own. Good luck.' };
    }
    if (this.humanIsSaboteur()) {
      return {
        tone: acc < LIVE_RULES.winThreshold ? 'good' : 'info',
        title: acc < LIVE_RULES.winThreshold ? 'It’s working — keep painting' : 'Saboteur tip',
        body: 'Structure beats scatter: stripes, checkerboards, and islands of one class inside the other burn through the trainer’s neuron budget fastest. Drag to paint runs of points.',
      };
    }
    return { tone: 'idle', title: 'AI vs AI', body: 'The Saboteur paints, the Trainer adapts. The meter tells you who’s ahead of the win line.' };
  }
}

// ===== Small view helpers =================================================
function legendKey(color: string, label: string): HTMLElement {
  return el('span', { class: 'key' }, [el('span', { class: 'swatch', style: { background: color } }), label]);
}

/** Format milliseconds as m:ss for the match clock. */
function fmtClock(ms: number): string {
  const total = Math.max(0, Math.ceil(ms / 1000));
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${String(s).padStart(2, '0')}`;
}

function rule(label: string, value: string): HTMLElement {
  return el('div', { class: 'rule' }, [el('span', { class: 'rule-l' }, [label]), el('span', { class: 'rule-v' }, [value])]);
}

/** A small stroked lock that inherits the palette via currentColor (no OS emoji). */
function lockGlyph(): SVGElement {
  const ns = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(ns, 'svg');
  svg.setAttribute('width', '13');
  svg.setAttribute('height', '13');
  svg.setAttribute('viewBox', '0 0 24 24');
  svg.setAttribute('fill', 'none');
  svg.setAttribute('stroke', 'currentColor');
  svg.setAttribute('stroke-width', '2');
  const body = document.createElementNS(ns, 'rect');
  body.setAttribute('x', '5');
  body.setAttribute('y', '11');
  body.setAttribute('width', '14');
  body.setAttribute('height', '9');
  body.setAttribute('rx', '2');
  const shackle = document.createElementNS(ns, 'path');
  shackle.setAttribute('d', 'M8 11 V8 a4 4 0 0 1 8 0 v3');
  svg.append(body, shackle);
  return svg;
}

function loadLessons(): Set<string> {
  try {
    const raw = localStorage.getItem(LESSONS_KEY);
    if (!raw) return new Set();
    const arr: unknown = JSON.parse(raw);
    return Array.isArray(arr) ? new Set(arr.filter((x): x is string => typeof x === 'string')) : new Set();
  } catch {
    return new Set();
  }
}
function saveLessons(set: Set<string>): void {
  try {
    localStorage.setItem(LESSONS_KEY, JSON.stringify([...set]));
  } catch {
    /* ignore */
  }
}

function neuronGlyph(): SVGElement {
  const ns = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(ns, 'svg');
  svg.setAttribute('width', '24');
  svg.setAttribute('height', '24');
  svg.setAttribute('viewBox', '0 0 24 24');
  const data = [
    ['M4 6 L12 12', 0.6],
    ['M4 18 L12 12', 0.6],
    ['M12 12 L20 8', 1],
    ['M12 12 L20 16', 0.8],
  ] as const;
  for (const [d, op] of data) {
    const p = document.createElementNS(ns, 'path');
    p.setAttribute('d', d);
    p.setAttribute('stroke', '#5ce1b0');
    p.setAttribute('stroke-width', '1.6');
    p.setAttribute('opacity', String(op));
    p.setAttribute('fill', 'none');
    svg.appendChild(p);
  }
  for (const [cx, cy] of [
    [4, 6],
    [4, 18],
    [12, 12],
    [20, 8],
    [20, 16],
  ]) {
    const c = document.createElementNS(ns, 'circle');
    c.setAttribute('cx', String(cx));
    c.setAttribute('cy', String(cy));
    c.setAttribute('r', '2.2');
    c.setAttribute('fill', cx === 12 ? '#5ce1b0' : '#5b9cff');
    svg.appendChild(c);
  }
  return svg;
}
