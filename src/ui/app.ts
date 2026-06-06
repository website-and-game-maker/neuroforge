import { el, clear } from './dom';
import { renderRich } from './rich';
import { fitCanvas } from './canvas';
import { Matrix } from '../engine/matrix';
import { Rng } from '../engine/rng';
import {
  DATASET_GENERATORS,
  gaussianBlobs,
  circlesData,
  moonsData,
  xorData,
  type Dataset,
} from '../data/datasets';
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
  VersusMatch,
  VERSUS_RULES,
  VERSUS_BASE_POINTS,
  saboteurMove,
  trainerPlan,
  trainerHint,
  neuronsUsed,
  type Role,
  type Difficulty,
  type HintLevel,
  type SabotagePoint,
} from '../game/versus';
import { Tutorial, hasSeenTutorial } from './tutorial';

type Mode = 'learn' | 'challenge' | 'sandbox' | 'versus';
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
const SABOTAGE_MIN_SEP = 0.1;
const TELEMETRY_EVERY = 4; // frames between telemetry refreshes while training

interface VsState {
  match: VersusMatch;
  studio: Studio;
  humanRole: Role;
  aiVsAi: boolean;
  difficulty: Difficulty;
  hintLevel: HintLevel;
  saboClass: number;
  placedThisTurn: number;
  rng: Rng;
  queue: SabotagePoint[]; // pending AI saboteur drops
  aiStepsLeft: number; // pending AI trainer steps
  tick: number;
  finished: boolean;
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
  mount(root: HTMLElement): void {
    this.progress = loadProgress();
    this.lessonsDone = loadLessons();
    const firstOpen = CHALLENGES.findIndex((c) => !isCompleted(this.progress, c.id));
    this.challengeIndex = firstOpen === -1 ? 0 : firstOpen;
    // Land returning players who finished the tour straight in Challenges.
    this.mode = hasSeenTutorial() && this.lessonsDone.size > 0 ? 'challenge' : 'learn';

    clear(root);
    root.append(this.buildTopbar(), this.buildStage(), this.buildFooter());

    this.enterMode(this.mode);
    window.addEventListener('resize', () => this.handleResize());
    requestAnimationFrame(() => {
      this.handleResize();
      this.loop();
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

    return el('header', { class: 'topbar' }, [
      el('div', { class: 'brand' }, [
        el('div', { class: 'brand-mark' }, [neuronGlyph()]),
        el('div', { class: 'brand-titles' }, [
          el('div', { class: 'brand-title', html: 'Neuro<em>Forge</em>' }),
          el('div', { class: 'brand-tag' }, ['build · train · understand']),
        ]),
      ]),
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
        { attrs: { href: 'https://github.com/PyCoder42/neuroforge', target: '_blank', rel: 'noopener' } },
        ['source on GitHub'],
      ),
    ]);
  }

  // ===== Mode routing =====================================================
  private setMode(mode: Mode): void {
    if (this.mode === mode) return;
    this.enterMode(mode);
  }

  private enterMode(mode: Mode): void {
    this.running = false;
    this.mode = mode;
    this.probe = null;
    this.hoverNeuron = null;
    Array.from(this.refs.modeSeg.children).forEach((c, i) =>
      c.classList.toggle('active', i === ['learn', 'challenge', 'sandbox', 'versus'].indexOf(mode)),
    );

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

    const budgetBar =
      neuronBudget !== undefined
        ? el('div', { class: 'budget' }, [
            el('div', { class: 'budget-row' }, [
              el('span', {}, ['neuron budget']),
              el('span', { class: used > neuronBudget ? 'budget-n over' : 'budget-n' }, [`${used} / ${neuronBudget}`]),
            ]),
            el('div', { class: 'budget-track' }, [
              el('div', {
                class: used > neuronBudget ? 'budget-fill over' : 'budget-fill',
                style: { width: `${Math.min(100, (used / neuronBudget) * 100)}%` },
              }),
            ]),
          ])
        : null;

    return el('div', { class: 'arch' }, [rows, addBtn, budgetBar].filter(Boolean) as Node[]);
  }

  private endcap(n: string, label: string): HTMLElement {
    return el('div', { class: 'endcap' }, [
      el('span', { class: 'cap-n' }, [n]),
      el('span', { class: 'cap-l' }, [label]),
    ]);
  }

  private layerChip(width: number, idx: number): HTMLElement {
    const slider = el('input', {
      class: 'chip-range',
      attrs: { type: 'range', min: String(MIN_WIDTH), max: '32', step: '1', value: String(width), 'aria-label': `layer ${idx + 1} width` },
      on: {
        input: (e) => this.setWidth(idx, Number((e.target as HTMLInputElement).value)),
      },
    });
    return el('div', { class: 'layer-chip' }, [
      el('span', { class: 'chip-w' }, [String(width)]),
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
        el('button', { class: 'btn ghost', on: { click: () => { this.customPoints = []; this.rebuildSandboxData(); this.studio.rebuild(true); this.redrawAll(); } } }, ['Clear points']),
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
    return this.mode === 'versus' && this.vs && this.isHumanTrainerTurn() ? VERSUS_RULES.neuronBudget : Infinity;
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
  private setWidth(idx: number, width: number): void {
    const next = [...this.studioConfig().hidden];
    // Cap so the total never exceeds the budget (only finite in Versus).
    const others = next.reduce((s, w) => s + w, 0) - (next[idx] ?? 0);
    const budgetCap = this.neuronBudget() - others;
    const maxForThis = Math.min(MAX_WIDTH, budgetCap);
    next[idx] = Math.max(MIN_WIDTH, Math.min(maxForThis, width));
    this.current().setConfig({ hidden: next });
    this.applyChange(true);
    this.rerenderLeft();
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
    this.running = false;
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
    if (this.mode === 'sandbox' && this.sandboxDatasetId === DRAW_ID) this.customPoints = [];
    else this.dataNudge++;
    this.current().reseed();
    if (this.mode === 'learn') this.loadLesson(this.learnIndex, false);
    else if (this.mode === 'challenge') this.loadChallenge(this.challengeIndex, false);
    else this.loadSandbox();
  }

  // ===== Run loop ========================================================
  private loop(): void {
    // Apply at most one coalesced probe redraw per frame (set by pointermove).
    if (this.probeDirty) {
      this.probeDirty = false;
      this.redrawInspector();
      this.refreshInspectorCaption();
      if (!this.running) this.redrawArena();
    }

    if (this.mode === 'versus' && this.vs) this.vsTick();
    else if (this.running && this.current().trainer) {
      this.current().step(STEPS_PER_FRAME);
      this.frame++;
      this.redrawArena();
      if (this.frame % TELEMETRY_EVERY === 0) this.refreshTelemetry();
      if (this.frame % 6 === 0) this.redrawInspector();
    }
    requestAnimationFrame(() => this.loop());
  }

  private redrawAll(): void {
    this.redrawArena();
    this.refreshTelemetry();
    this.redrawInspector();
    this.refreshInspectorCaption();
    this.refreshIoBox();
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
        drawPoints(ctx, vp, this.vs.match.base.X, this.vs.match.base.Y, 4);
        this.drawSabotage(ctx, vp);
      } else if (studio.trainData.X.rows > 0) {
        drawPoints(ctx, vp, studio.trainData.X, studio.trainData.Y);
      }
      this.drawProbeMarker(ctx, vp);
    } else {
      this.drawRegressionGrid(ctx, w, h);
      drawRegression(ctx, vp, studio.trainData.X, studio.trainData.Y, studio.forward);
    }
  }

  private drawSabotage(ctx: CanvasRenderingContext2D, vp: ReturnType<typeof makeViewport>): void {
    if (!this.vs) return;
    const recent = new Set(this.vs.match.sabotage.slice(-VERSUS_RULES.pointsPerRound));
    for (const s of this.vs.match.sabotage) {
      const { sx, sy } = worldToScreen(vp, s.x, s.y);
      ctx.beginPath();
      ctx.arc(sx, sy, 5, 0, Math.PI * 2);
      ctx.fillStyle = rgba(classColor(s.label));
      ctx.fill();
      ctx.lineWidth = recent.has(s) ? 2.5 : 1.5;
      ctx.strokeStyle = recent.has(s) ? 'rgba(255,255,255,0.95)' : 'rgba(10,12,20,0.9)';
      ctx.stroke();
      // little spike to mark "placed", not natural
      ctx.beginPath();
      ctx.arc(sx, sy, 8.5, 0, Math.PI * 2);
      ctx.strokeStyle = recent.has(s) ? 'rgba(255,206,107,0.9)' : 'rgba(150,170,215,0.25)';
      ctx.lineWidth = 1;
      ctx.stroke();
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
      this.refreshVsHud();
      this.renderStats([
        { label: 'accuracy · all points', value: m.hasData ? `${(m.testScore * 100).toFixed(1)}%` : '—', good: false },
        { label: 'neurons used', value: `${neuronsUsed(studio.config)} / ${VERSUS_RULES.neuronBudget}` },
        { label: 'loss', value: Number.isFinite(m.loss) ? m.loss.toFixed(4) : '—' },
        { label: 'steps', value: studio.steps.toLocaleString() },
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

  private renderStats(stats: Array<{ label: string; value: string; good?: boolean }>): void {
    clear(this.refs.statGrid);
    for (const s of stats) {
      this.refs.statGrid.append(
        el('div', { class: 'stat' }, [
          el('div', { class: 'stat-label' }, [s.label]),
          el('div', { class: s.good ? 'stat-value good' : 'stat-value' }, [s.value]),
        ]),
      );
    }
  }

  private renderCoach(tip: CoachTip): void {
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
    if (this.mode === 'versus' && this.vs && this.vs.match.phase === 'saboteur' && this.isHumanSaboteurTurn()) return 'sabotage';
    return null;
  }

  private placeAt(wx: number, wy: number): void {
    const mode = this.placementMode();
    if (mode === 'draw') {
      this.customPoints.push({ x: wx, y: wy, label: this.drawClass });
      this.rebuildSandboxData();
      this.studio.rebuild(false);
      this.redrawAll();
      this.refreshRunButton();
    } else if (mode === 'sabotage' && this.vs) {
      if (this.vs.placedThisTurn >= VERSUS_RULES.pointsPerRound) return;
      // Same fairness rule the AI follows: don't let a drop sit on top of an
      // opposite-class point (that would make 100% impossible by contradiction).
      if (this.nearestOppositeDist(wx, wy, this.vs.saboClass) < SABOTAGE_MIN_SEP) {
        this.flashSaboteurReject();
        return;
      }
      this.vs.match.addSabotage([{ x: wx, y: wy, label: this.vs.saboClass, byAi: false }]);
      this.vs.placedThisTurn++;
      this.syncVersusData(false);
      this.redrawAll();
      this.refreshVsHud();
    }
  }

  /** Distance from (x,y) to the nearest existing point of the OTHER class. */
  private nearestOppositeDist(x: number, y: number, label: number): number {
    const v = this.vs;
    if (!v) return Infinity;
    let best = Infinity;
    const consider = (px: number, py: number, pl: number): void => {
      if (pl === label) return;
      const d = Math.hypot(px - x, py - y);
      if (d < best) best = d;
    };
    const base = v.match.base;
    for (let i = 0; i < base.X.rows; i++) consider(base.X.get(i, 0), base.X.get(i, 1), base.Y.data[i]!);
    for (const s of v.match.sabotage) consider(s.x, s.y, s.label);
    return best;
  }

  private flashSaboteurReject(): void {
    // Briefly surface why the drop was rejected, via the coach callout.
    this.renderCoach({
      tone: 'warn',
      title: 'Too close to an opposite point',
      body: 'You can’t drop a point right on top of an opposite-class point — that would be an impossible contradiction. Aim for open territory the model is confidently wrong about.',
    });
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

  // ===== VERSUS ==========================================================
  private vsSetup = { humanRole: 'trainer' as Role, aiVsAi: false, difficulty: 'medium' as Difficulty, base: 'moons', hintLevel: 1 as HintLevel };

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

    const baseSel = el('select', { attrs: { 'aria-label': 'Base pattern' }, on: { change: (e) => { s.base = (e.target as HTMLSelectElement).value; } } }, [
      el('option', { attrs: { value: 'moons' } }, ['Moons (curvy)']),
      el('option', { attrs: { value: 'circles' } }, ['Circles (ring)']),
      el('option', { attrs: { value: 'blobs' } }, ['Two Blobs (easy)']),
      el('option', { attrs: { value: 'xor' } }, ['XOR (corners)']),
    ]) as HTMLSelectElement;
    baseSel.value = s.base;

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
        ? 'Watch an AI Trainer defend its boundary while an AI Saboteur attacks it. Great for seeing the capacity battle play out.'
        : s.humanRole === 'trainer'
          ? 'You build and train the network. An AI Saboteur drops opposite-class points where your model is confidently wrong. Keep accuracy high to win.'
          : 'You drop points to confuse an AI’s network — place a Class A dot deep in its Class B region. Push accuracy below the threshold to win.';

    this.setLeft([
      this.panelHead('duel', 'Versus mode'),
      el('div', { class: 'group' }, [el('div', { class: 'group-label' }, ['Who are you?']), roleSeg, el('div', { class: 'mini-note' }, [roleExplain])]),
      el('div', { class: 'group' }, [el('div', { class: 'group-label' }, ['AI difficulty']), diff]),
      el('div', { class: 'group' }, [
        el('div', { class: 'group-label' }, ['Match']),
        el('div', { class: 'field' }, [el('div', { class: 'field-row' }, [el('span', { class: 'field-label' }, ['Base pattern'])]), baseSel]),
        hint,
        el('div', { class: 'rules-grid' }, [
          rule('Attacks', String(VERSUS_RULES.rounds)),
          rule('Pts / attack', String(VERSUS_RULES.pointsPerRound)),
          rule('Neuron budget', String(VERSUS_RULES.neuronBudget)),
          rule('Trainer wins ≥', `${(VERSUS_RULES.winThreshold * 100).toFixed(0)}%`),
        ].filter(Boolean) as Node[]),
      ].filter(Boolean) as Node[]),
      el('button', { class: 'btn primary', on: { click: () => this.vsStart() } }, ['Start match ▶']),
    ]);

    // Neutral arena while setting up.
    this.refs.arenaSub.textContent = 'versus · setup';
    this.refs.arenaTitle.textContent = 'Trainer vs Saboteur';
    clear(this.refs.note);
    this.refs.note.append(el('span', { class: 'note-icon' }, ['¶']), el('div', { class: 'note-body' }, [renderRich('span', '', '**The duel.** One side builds a network to classify the dots; the other drops new dots to break it. Capacity (the neuron budget) vs cunning (where the dots land).')]));
    this.refreshLegend();
    clear(this.refs.vsHud);
    clear(this.refs.coach);
    this.renderStats([{ label: 'status', value: 'ready' }, { label: 'difficulty', value: this.vsSetup.difficulty }, { label: 'base', value: this.vsSetup.base }, { label: 'mode', value: this.vsSetup.aiVsAi ? 'AI vs AI' : 'you' }]);
    this.refreshIoBox();
    this.clearCanvas(this.refs.arenaCanvas);
    this.clearCanvas(this.refs.inspectorCanvas);
    clear(this.refs.inspectorCaption);
  }

  private clearCanvas(canvas: HTMLCanvasElement): void {
    const { ctx, w, h } = fitCanvas(canvas);
    ctx.clearRect(0, 0, w, h);
  }

  private vsStart(): void {
    const s = this.vsSetup;
    const baseGen: Record<string, (n: number, seed: number) => Dataset> = {
      moons: moonsData,
      circles: circlesData,
      blobs: gaussianBlobs,
      xor: xorData,
    };
    const base = (baseGen[s.base] ?? moonsData)(VERSUS_BASE_POINTS, 7);
    const studio = new Studio({ ...defaultConfig(), hidden: [8], activation: 'tanh', lr: 0.15 });
    studio.task = 'classification';
    const match = new VersusMatch(VERSUS_RULES, base);
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
      placedThisTurn: 0,
      rng: new Rng(20260605),
      queue: [],
      aiStepsLeft: 0,
      tick: 0,
      finished: false,
    };
    this.running = false;
    this.vsEnterPhase();
    this.refreshArenaHeaderVersus();
    this.redrawAll();
  }

  private aiIsTrainer(): boolean {
    return !!this.vs && (this.vs.aiVsAi || this.vs.humanRole === 'saboteur');
  }
  private aiIsSaboteur(): boolean {
    return !!this.vs && (this.vs.aiVsAi || this.vs.humanRole === 'trainer');
  }
  private isHumanTrainerTurn(): boolean {
    return !!this.vs && this.vs.match.phase === 'trainer' && !this.aiIsTrainer();
  }
  private isHumanSaboteurTurn(): boolean {
    return !!this.vs && this.vs.match.phase === 'saboteur' && !this.aiIsSaboteur();
  }

  /** Set up whatever the new phase needs (AI plans, counters). */
  private vsEnterPhase(): void {
    const v = this.vs;
    if (!v) return;
    v.placedThisTurn = 0;
    v.queue = [];
    v.aiStepsLeft = 0;
    this.running = false;

    if (v.match.phase === 'trainer') {
      if (this.aiIsTrainer()) {
        const plan = trainerPlan(v.match, v.difficulty);
        // Warm-start: only re-initialise weights when the architecture actually changes;
        // otherwise keep the trained net and continue from where the last defense left off
        // (mirrors the human Trainer, who keeps weights across turns).
        const archChanged =
          plan.config.hidden.join(',') !== v.studio.config.hidden.join(',') ||
          plan.config.activation !== v.studio.config.activation;
        v.studio.config = { ...plan.config };
        if (archChanged) {
          v.studio.reseed();
          v.studio.rebuild(true);
        } else {
          v.studio.rebuild(false);
        }
        v.aiStepsLeft = plan.steps;
      }
    } else if (v.match.phase === 'saboteur') {
      if (this.aiIsSaboteur()) {
        v.queue = saboteurMove(v.studio.forward, v.match, v.difficulty, v.rng);
        if (v.queue.length === 0) {
          // Nothing feasible — skip straight to ending the attack.
          v.match.endSaboteurTurn();
          this.syncVersusData(false);
          this.vsEnterPhase();
          return;
        }
      }
    } else {
      v.finished = true;
    }
    this.renderVersusMatchPanel();
    this.refreshArenaHeaderVersus();
    this.refreshRunButton();
    this.refreshVsHud();
  }

  private syncVersusData(resetWeights: boolean): void {
    const v = this.vs;
    if (!v) return;
    const ds = v.match.dataset();
    v.studio.setData(ds, ds);
    v.studio.rebuild(resetWeights);
  }

  private vsTick(): void {
    const v = this.vs;
    if (!v || v.finished) return;
    v.tick++;

    if (v.match.phase === 'trainer') {
      if (this.aiIsTrainer()) {
        if (v.aiStepsLeft > 0) {
          const n = Math.min(STEPS_PER_FRAME, v.aiStepsLeft);
          v.studio.step(n);
          v.aiStepsLeft -= n;
          this.frame++;
          this.redrawArena();
          if (this.frame % TELEMETRY_EVERY === 0) this.refreshTelemetry();
          if (this.frame % 6 === 0) this.redrawInspector();
        } else {
          v.match.endTrainerTurn();
          this.vsEnterPhase();
          this.redrawAll();
        }
      } else if (this.running && v.studio.trainer) {
        v.studio.step(STEPS_PER_FRAME);
        this.frame++;
        this.redrawArena();
        if (this.frame % TELEMETRY_EVERY === 0) this.refreshTelemetry();
        if (this.frame % 6 === 0) this.redrawInspector();
      }
    } else if (v.match.phase === 'saboteur') {
      if (this.aiIsSaboteur()) {
        // Drop one queued point every few frames for watchability.
        if (v.tick % 10 === 0 && v.queue.length > 0) {
          const p = v.queue.shift()!;
          v.match.addSabotage([p]);
          v.placedThisTurn++;
          this.syncVersusData(false);
          this.redrawArena();
          this.refreshVsHud();
        } else if (v.queue.length === 0) {
          v.match.endSaboteurTurn();
          this.syncVersusData(false);
          this.vsEnterPhase();
          this.redrawAll();
        }
      }
    }
  }

  private vsEndTrainerTurn(): void {
    const v = this.vs;
    if (!v) return;
    this.running = false;
    v.match.endTrainerTurn();
    this.vsEnterPhase();
    this.redrawAll();
  }

  private vsEndSaboteurTurn(): void {
    const v = this.vs;
    if (!v) return;
    v.match.endSaboteurTurn();
    this.syncVersusData(false);
    this.vsEnterPhase();
    this.redrawAll();
  }

  private renderVersusMatchPanel(): void {
    const v = this.vs;
    if (!v) return this.renderVersusSetup();

    const phase = v.match.phase;
    const humanTrainer = this.isHumanTrainerTurn();
    const humanSaboteur = this.isHumanSaboteurTurn();

    const children: Node[] = [this.panelHead('duel', 'Versus mode')];

    if (phase === 'done') {
      const acc = v.match.score(v.studio.forward);
      const winner = v.match.winner(v.studio.forward);
      children.push(
        el('div', { class: `vs-result ${winner}` }, [
          el('div', { class: 'vs-result-tag' }, [winner === 'trainer' ? 'Trainer wins' : 'Saboteur wins']),
          el('div', { class: 'vs-result-acc' }, [`${(acc * 100).toFixed(1)}% accuracy`]),
          el('div', { class: 'mini-note' }, [
            winner === 'trainer'
              ? 'The network had the capacity and tuning to absorb every attack. Nicely defended.'
              : 'The boundary couldn’t stretch to cover the saboteur’s points — capacity or tuning fell short.',
          ]),
        ]),
        el('button', { class: 'btn primary', on: { click: () => { this.vs = null; this.renderVersusSetup(); } } }, ['New match']),
      );
      this.setLeft(children);
      this.refreshArenaHeaderVersus();
      return;
    }

    if (humanTrainer) {
      children.push(
        el('div', { class: 'turn-banner trainer' }, [`Your turn — Trainer · Defense ${v.match.attacksDone + 1}`]),
        this.buildControls({ neuronBudget: VERSUS_RULES.neuronBudget, allowNewData: false }),
        el('button', { class: 'btn primary wide', on: { click: () => this.vsEndTrainerTurn() } }, ['End turn — let the Saboteur attack ▶']),
      );
    } else if (humanSaboteur) {
      children.push(
        el('div', { class: 'turn-banner saboteur' }, [`Your turn — Saboteur · Attack ${v.match.attackNumber} / ${VERSUS_RULES.rounds}`]),
        el('div', { class: 'mini-note' }, ['Drop points where the model is confidently **wrong** — a Class A dot deep in an orange region. Click or drag the arena.']),
        el('div', { class: 'field' }, [el('div', { class: 'field-row' }, [el('span', { class: 'field-label' }, ['Dropping'])]), this.classToggle(v.saboClass, (c) => { v.saboClass = c; this.renderVersusMatchPanel(); })]),
        el('div', { class: 'vs-budget' }, [`${v.placedThisTurn} / ${VERSUS_RULES.pointsPerRound} points placed`]),
        el('button', { class: 'btn primary wide', on: { click: () => this.vsEndSaboteurTurn() } }, ['End attack ▶']),
      );
    } else {
      // AI is acting.
      children.push(
        el('div', { class: `turn-banner ${phase}` }, [
          phase === 'trainer' ? 'AI Trainer is building & training…' : 'AI Saboteur is choosing targets…',
        ]),
        el('div', { class: 'mini-note' }, ['Watch the arena. ', phase === 'trainer' ? 'The boundary is forming.' : 'New points are landing where the model was wrong.']),
      );
      if (v.aiVsAi) children.push(el('button', { class: 'btn ghost', on: { click: () => { this.vs = null; this.renderVersusSetup(); } } }, ['Stop · new setup']));
    }

    this.setLeft(children);
    this.refreshArenaHeaderVersus();
    this.refreshRunButton();
  }

  private refreshArenaHeaderVersus(): void {
    const v = this.vs;
    if (!v) return;
    this.refs.arenaSub.textContent = `versus · ${v.aiVsAi ? 'AI vs AI' : `you are the ${v.humanRole}`} · ${v.difficulty}`;
    this.refs.arenaTitle.textContent = 'Trainer vs Saboteur';
    this.refreshLegend();
    this.refs.legend.append(legendKey('rgba(255,206,107,0.95)', 'just-placed sabotage'));
    this.refreshStatus();
  }

  private refreshVsHud(): void {
    const v = this.vs;
    if (!v) {
      clear(this.refs.vsHud);
      return;
    }
    const acc = v.studio.net ? v.match.score(v.studio.forward) : 0;
    const thr = VERSUS_RULES.winThreshold;
    const pct = Math.max(0, Math.min(100, acc * 100));
    const hud = this.refs.vsHud;
    clear(hud);
    hud.append(
      el('div', { class: 'vs-meter' }, [
        el('div', { class: 'vs-meter-head' }, [
          el('span', { class: 'vs-side trainer' }, ['Trainer']),
          el('span', { class: 'vs-acc' }, [`${pct.toFixed(0)}%`]),
          el('span', { class: 'vs-side saboteur' }, ['Saboteur']),
        ]),
        el('div', { class: 'vs-track' }, [
          el('div', { class: 'vs-fill', style: { width: `${pct}%` } }),
          el('div', { class: 'vs-threshold', style: { left: `${thr * 100}%` } }),
        ]),
        el('div', { class: 'vs-rounds' }, [`${v.match.attacksDone} / ${VERSUS_RULES.rounds} attacks survived · win line ${(thr * 100).toFixed(0)}%`]),
      ]),
    );
  }

  private versusCoachTip(): CoachTip {
    const v = this.vs!;
    const acc = v.studio.net ? v.match.score(v.studio.forward) : 0;
    if (v.match.phase === 'done') {
      const w = v.match.winner(v.studio.forward);
      return { tone: w === 'trainer' ? 'success' : 'warn', title: w === 'trainer' ? 'Trainer wins' : 'Saboteur wins', body: `Final accuracy ${(acc * 100).toFixed(0)}%.` };
    }
    if (this.isHumanTrainerTurn()) {
      const txt = trainerHint(v.match, v.studio.config, acc, v.hintLevel);
      return { tone: acc >= VERSUS_RULES.winThreshold ? 'good' : 'info', title: 'Trainer hint', body: txt || 'Hints are off — you’re on your own. Good luck.' };
    }
    if (this.isHumanSaboteurTurn()) {
      return { tone: 'info', title: 'Saboteur tip', body: 'Aim for regions the shading is *strongly* one colour but should be the other. Spread your points — clustering is easy for the network to wall off.' };
    }
    return { tone: 'idle', title: v.match.phase === 'trainer' ? 'AI Trainer working' : 'AI Saboteur working', body: 'Watch the accuracy meter move as the duel plays out.' };
  }
}

// ===== Small view helpers =================================================
function legendKey(color: string, label: string): HTMLElement {
  return el('span', { class: 'key' }, [el('span', { class: 'swatch', style: { background: color } }), label]);
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
