import { el, clear } from './dom';
import { fitCanvas } from './canvas';
import { mlp, type Network } from '../engine/network';
import { activationByName, Sigmoid, Identity } from '../engine/activations';
import { BCE, MSE, type Loss } from '../engine/losses';
import { Rng } from '../engine/rng';
import { Trainer, accuracy, mseMetric } from '../engine/trainer';
import { Matrix } from '../engine/matrix';
import {
  DATASET_GENERATORS,
  type Dataset,
  type TaskKind,
} from '../data/datasets';
import { CHALLENGES, type Challenge } from '../game/challenges';
import {
  emptyProgress,
  isCompleted,
  isUnlocked,
  markComplete,
  type Progress,
} from '../game/progress';
import { loadProgress, saveProgress, clearProgress } from '../state/persistence';
import { makeViewport, screenToWorld } from '../viz/coords';
import { drawBoundary } from '../viz/boundary';
import { drawPoints } from '../viz/points';
import { drawRegression } from '../viz/regression';
import { drawLossChart } from '../viz/chart';
import { drawNetwork } from '../viz/netdiagram';
import { CLASS_A, CLASS_B, rgba } from '../viz/colors';

interface Config {
  hidden: number[];
  activation: string;
  lr: number;
  l2: number;
  batchSize: number;
  momentum: number;
}

const STEPS_PER_FRAME = 12;
const MAX_LAYERS = 4;
const MAX_WIDTH = 48;
const MIN_WIDTH = 1;
const DRAW_ID = 'draw';
const SANDBOX_N = 240;

export class App {
  private mode: 'challenge' | 'sandbox' = 'challenge';
  private challengeIndex = 0;
  private sandboxDatasetId = 'circles';
  private drawClass = 0;
  private customPoints: Array<{ x: number; y: number; label: number }> = [];

  private config: Config = { ...CHALLENGES[0]!.starter };
  private task: TaskKind = 'classification';
  private trainData!: Dataset;
  private testData!: Dataset;

  private net: Network | null = null;
  private trainer: Trainer | null = null;
  private loss: Loss = BCE;
  private running = false;
  private weightSeed = 1;
  private shuffleSeed = 1;
  private frame = 0;

  private progress: Progress = emptyProgress();

  // DOM references populated during build.
  private refs: {
    runBtn: HTMLButtonElement;
    status: HTMLElement;
    solvedFlag: HTMLElement;
    arenaTitle: HTMLElement;
    arenaSub: HTMLElement;
    note: HTMLElement;
    legend: HTMLElement;
    arenaCanvas: HTMLCanvasElement;
    lossCanvas: HTMLCanvasElement;
    diagramCanvas: HTMLCanvasElement;
    statPrimary: { label: HTMLElement; value: HTMLElement };
    statSecondary: { label: HTMLElement; value: HTMLElement };
    statLoss: HTMLElement;
    statSteps: HTMLElement;
    challengeList: HTMLElement;
    archRows: HTMLElement;
    controlsExtra: HTMLElement;
    fieldValues: Record<string, HTMLElement>;
    modeSeg: HTMLElement;
  } = {} as never;

  mount(root: HTMLElement): void {
    this.progress = loadProgress();
    // Pick the first uncompleted challenge as the starting point.
    const firstOpen = CHALLENGES.findIndex((c) => !isCompleted(this.progress, c.id));
    this.challengeIndex = firstOpen === -1 ? 0 : firstOpen;

    clear(root);
    root.append(this.buildTopbar(), this.buildStage(), this.buildFooter());

    this.loadFromMode(true);
    window.addEventListener('resize', () => this.handleResize());
    requestAnimationFrame(() => {
      this.handleResize();
      this.loop();
    });
  }

  // ----- Layout ----------------------------------------------------------
  private buildTopbar(): HTMLElement {
    const modeSeg = el('div', { class: 'seg', attrs: { role: 'tablist' } }, [
      this.segButton('Challenges', () => this.setMode('challenge'), true),
      this.segButton('Sandbox', () => this.setMode('sandbox'), false),
    ]);
    this.refs.modeSeg = modeSeg;

    const resetBtn = el(
      'button',
      {
        class: 'seg-btn',
        attrs: { title: 'Clear saved progress' },
        on: {
          click: () => {
            clearProgress();
            this.progress = emptyProgress();
            this.challengeIndex = 0;
            this.refreshChallengeList();
          },
        },
      },
      ['Reset progress'],
    );

    return el('header', { class: 'topbar' }, [
      el('div', { class: 'brand' }, [
        el('div', { class: 'brand-mark' }, [neuronGlyph()]),
        el('div', { class: 'brand-titles' }, [
          el('div', { class: 'brand-title', html: 'Neuro<em>Forge</em>' }),
          el('div', { class: 'brand-tag' }, ['a neural net, trained from scratch']),
        ]),
      ]),
      el('div', { class: 'topbar-actions' }, [
        modeSeg,
        el('div', { class: 'seg' }, [resetBtn]),
      ]),
    ]);
  }

  private segButton(label: string, onClick: () => void, active: boolean): HTMLButtonElement {
    return el(
      'button',
      { class: active ? 'seg-btn active' : 'seg-btn', on: { click: onClick } },
      [label],
    );
  }

  private buildStage(): HTMLElement {
    return el('main', { class: 'stage' }, [
      this.buildControls(),
      this.buildArena(),
      this.buildMetrics(),
    ]);
  }

  // ----- Controls panel --------------------------------------------------
  private buildControls(): HTMLElement {
    const challengeList = el('div', { class: 'challenges' });
    this.refs.challengeList = challengeList;

    const archRows = el('div', { class: 'arch-rows' });
    this.refs.archRows = archRows;
    const addLayer = el(
      'button',
      {
        class: 'add-layer',
        on: { click: () => this.addLayer() },
      },
      ['+ add hidden layer'],
    );

    const controlsExtra = el('div', { class: 'group' });
    this.refs.controlsExtra = controlsExtra;

    this.refs.fieldValues = {};
    const lrField = this.sliderField('lr', 'Learning rate', 0.005, 0.6, 0.005, (v) =>
      v.toFixed(3),
    );
    const momField = this.sliderField('momentum', 'Momentum', 0, 0.95, 0.05, (v) => v.toFixed(2));
    const l2Field = this.sliderField('l2', 'L2 (weight decay)', 0, 0.02, 0.0005, (v) =>
      v.toFixed(4),
    );

    const activationSel = el(
      'select',
      {
        attrs: { 'aria-label': 'Hidden activation' },
        on: {
          change: (e) => {
            this.config.activation = (e.target as HTMLSelectElement).value;
            this.rebuildEngine();
          },
        },
      },
      ['relu', 'tanh', 'sigmoid'].map((a) =>
        el('option', { attrs: { value: a } }, [a.toUpperCase()]),
      ),
    );
    (activationSel as HTMLSelectElement).value = this.config.activation;

    const batchSel = el(
      'select',
      {
        attrs: { 'aria-label': 'Batch size' },
        on: {
          change: (e) => {
            this.config.batchSize = Number((e.target as HTMLSelectElement).value);
            this.rebuildEngine(false); // batch size only affects the trainer
          },
        },
      },
      [4, 8, 16, 32, 64].map((b) => el('option', { attrs: { value: String(b) } }, [String(b)])),
    );
    (batchSel as HTMLSelectElement).value = String(this.config.batchSize);
    this.refs.fieldValues['activation'] = activationSel;
    this.refs.fieldValues['batch'] = batchSel;

    const runBtn = el(
      'button',
      { class: 'btn primary', on: { click: () => this.toggleRun() } },
      ['Train'],
    );
    this.refs.runBtn = runBtn as HTMLButtonElement;

    return el('section', { class: 'panel' }, [
      el('div', { class: 'panel-head' }, [
        el('div', { class: 'eyebrow' }, ['control deck']),
        el('h2', {}, ['Build & tune']),
      ]),

      // Challenge list / sandbox dataset picker live here.
      el('div', { class: 'group' }, [
        el('div', { class: 'group-label' }, ['Task']),
        challengeList,
        controlsExtra,
      ]),

      // Architecture
      el('div', { class: 'group' }, [
        el('div', { class: 'group-label' }, ['Architecture']),
        el('div', { class: 'arch' }, [archRows, addLayer]),
        el('div', { class: 'field' }, [
          el('div', { class: 'field-row' }, [el('span', { class: 'field-label' }, ['Hidden activation'])]),
          activationSel,
        ]),
      ]),

      // Hyperparameters
      el('div', { class: 'group' }, [
        el('div', { class: 'group-label' }, ['Hyperparameters']),
        lrField,
        momField,
        l2Field,
        el('div', { class: 'field' }, [
          el('div', { class: 'field-row' }, [el('span', { class: 'field-label' }, ['Batch size'])]),
          batchSel,
        ]),
      ]),

      // Actions
      el('div', { class: 'group' }, [
        runBtn,
        el('div', { class: 'btns' }, [
          el('button', { class: 'btn ghost', on: { click: () => this.stepOnce() } }, ['Step']),
          el('button', { class: 'btn ghost', on: { click: () => this.resetWeights() } }, [
            'Reset',
          ]),
          el('button', { class: 'btn ghost', on: { click: () => this.newData() } }, ['New data']),
        ]),
      ]),
    ]);
  }

  private sliderField(
    key: 'lr' | 'momentum' | 'l2',
    label: string,
    min: number,
    max: number,
    step: number,
    fmt: (v: number) => string,
  ): HTMLElement {
    const valueEl = el('span', { class: 'field-value' }, [fmt(this.config[key])]);
    this.refs.fieldValues[key] = valueEl;
    const input = el('input', {
      attrs: {
        type: 'range',
        min: String(min),
        max: String(max),
        step: String(step),
        value: String(this.config[key]),
        'aria-label': label,
      },
      on: {
        input: (e) => {
          const v = Number((e.target as HTMLInputElement).value);
          this.config[key] = v;
          valueEl.textContent = fmt(v);
          this.rebuildEngine(false); // keep weights; only the optimizer changes
        },
      },
    });
    this.refs.fieldValues[`${key}__input`] = input;
    return el('div', { class: 'field' }, [
      el('div', { class: 'field-row' }, [
        el('span', { class: 'field-label' }, [label]),
        valueEl,
      ]),
      input,
    ]);
  }

  // ----- Arena panel -----------------------------------------------------
  private buildArena(): HTMLElement {
    const arenaCanvas = el('canvas', { attrs: { 'aria-label': 'training arena' } });
    const solvedFlag = el('div', { class: 'solved-flag' }, ['Solved ✓']);
    this.refs.arenaCanvas = arenaCanvas as HTMLCanvasElement;
    this.refs.solvedFlag = solvedFlag;

    const screen = el('div', { class: 'screen' }, [arenaCanvas, solvedFlag]);
    screen.addEventListener('click', (e) => this.handleScreenClick(e));

    const status = el('span', { class: 'status' }, [el('span', { class: 'dot' }), 'idle']);
    this.refs.status = status;
    const arenaTitle = el('h2', {}, ['—']);
    const arenaSub = el('div', { class: 'eyebrow' }, ['challenge']);
    this.refs.arenaTitle = arenaTitle;
    this.refs.arenaSub = arenaSub;

    const note = el('div', { class: 'note' }, []);
    this.refs.note = note;
    const legend = el('div', { class: 'legend' }, []);
    this.refs.legend = legend;

    return el('section', { class: 'panel arena-panel' }, [
      el('div', { class: 'arena-head' }, [
        el('div', { class: 'arena-title' }, [arenaSub, arenaTitle]),
        status,
      ]),
      el('div', { class: 'screen-wrap' }, [screen]),
      legend,
      note,
    ]);
  }

  // ----- Metrics panel ---------------------------------------------------
  private buildMetrics(): HTMLElement {
    const primaryLabel = el('div', { class: 'stat-label' }, ['test accuracy']);
    const primaryValue = el('div', { class: 'stat-value' }, ['—']);
    const secondaryLabel = el('div', { class: 'stat-label' }, ['train accuracy']);
    const secondaryValue = el('div', { class: 'stat-value' }, ['—']);
    const lossValue = el('div', { class: 'stat-value' }, ['—']);
    const stepsValue = el('div', { class: 'stat-value' }, ['0']);
    this.refs.statPrimary = { label: primaryLabel, value: primaryValue };
    this.refs.statSecondary = { label: secondaryLabel, value: secondaryValue };
    this.refs.statLoss = lossValue;
    this.refs.statSteps = stepsValue;

    const lossCanvas = el('canvas', { attrs: { 'aria-label': 'loss over time' } });
    const diagramCanvas = el('canvas', { attrs: { 'aria-label': 'network diagram' } });
    this.refs.lossCanvas = lossCanvas as HTMLCanvasElement;
    this.refs.diagramCanvas = diagramCanvas as HTMLCanvasElement;

    return el('section', { class: 'panel' }, [
      el('div', { class: 'panel-head' }, [
        el('div', { class: 'eyebrow' }, ['telemetry']),
        el('h2', {}, ['How it is doing']),
      ]),
      el('div', { class: 'stat-grid' }, [
        el('div', { class: 'stat' }, [primaryLabel, primaryValue]),
        el('div', { class: 'stat' }, [secondaryLabel, secondaryValue]),
        el('div', { class: 'stat' }, [el('div', { class: 'stat-label' }, ['loss']), lossValue]),
        el('div', { class: 'stat' }, [el('div', { class: 'stat-label' }, ['steps']), stepsValue]),
      ]),
      el('div', { class: 'subpanel' }, [
        el('div', { class: 'subpanel-head' }, [el('div', { class: 'eyebrow' }, ['loss trace'])]),
        el('div', { class: 'canvas-frame trace' }, [lossCanvas]),
      ]),
      el('div', { class: 'subpanel' }, [
        el('div', { class: 'subpanel-head' }, [
          el('div', { class: 'eyebrow' }, ['network']),
          el('span', { class: 'hint' }, ['edge = weight']),
        ]),
        el('div', { class: 'canvas-frame diagram' }, [diagramCanvas]),
      ]),
      el('div', { class: 'hint' }, [
        'Every weight is updated by hand-derived backpropagation — gradient-checked, no ML libraries.',
      ]),
    ]);
  }

  private buildFooter(): HTMLElement {
    return el('footer', { class: 'footer' }, [
      el('span', {}, ['NeuroForge — built from scratch in TypeScript · ']),
      el(
        'a',
        { attrs: { href: 'https://github.com/PyCoder42/neuroforge', target: '_blank', rel: 'noopener' } },
        ['source on GitHub'],
      ),
    ]);
  }

  // ----- Mode / data / engine lifecycle ----------------------------------
  private setMode(mode: 'challenge' | 'sandbox'): void {
    if (this.mode === mode) return;
    this.mode = mode;
    Array.from(this.refs.modeSeg.children).forEach((c, i) =>
      c.classList.toggle('active', (i === 0) === (mode === 'challenge')),
    );
    this.loadFromMode(true);
  }

  /** Build data + engine for the current mode, optionally applying the starter config. */
  private loadFromMode(applyStarter: boolean): void {
    this.running = false;
    if (this.mode === 'challenge') {
      const ch = CHALLENGES[this.challengeIndex]!;
      this.task = ch.task;
      if (applyStarter) this.config = { ...ch.starter };
    } else {
      const gen = DATASET_GENERATORS.find((g) => g.id === this.sandboxDatasetId);
      this.task = this.sandboxDatasetId === DRAW_ID ? 'classification' : gen!.kind;
    }
    this.rebuildData();
    this.rebuildEngine();
    this.renderControlsForMode();
    this.refreshArenaHeader();
    this.refreshFields();
    this.refreshRunButton();
  }

  private rebuildData(): void {
    if (this.mode === 'challenge') {
      const ch = CHALLENGES[this.challengeIndex]!;
      this.trainData = ch.makeData(ch.trainSeed + this.dataNudge);
      this.testData = ch.makeData(ch.testSeed + this.dataNudge);
    } else if (this.sandboxDatasetId === DRAW_ID) {
      this.trainData = this.datasetFromPoints();
      this.testData = this.trainData;
    } else {
      const gen = DATASET_GENERATORS.find((g) => g.id === this.sandboxDatasetId)!;
      this.trainData = gen.make(SANDBOX_N, 100 + this.dataNudge);
      this.testData = gen.make(SANDBOX_N, 200 + this.dataNudge);
    }
  }

  private dataNudge = 0;

  private datasetFromPoints(): Dataset {
    const pts = this.customPoints;
    const X = pts.length
      ? Matrix.fromRows(pts.map((p) => [p.x, p.y]))
      : Matrix.zeros(0, 2);
    const Y = pts.length ? Matrix.fromRows(pts.map((p) => [p.label])) : Matrix.zeros(0, 1);
    return { kind: 'classification', name: 'Your points', inDim: 2, X, Y };
  }

  /**
   * (Re)build the network and trainer. `resetWeights` controls whether the network is
   * re-initialised: structural changes (architecture, activation, dataset, mode) reset
   * the weights, while hyperparameter tweaks (lr, momentum, L2, batch) keep the
   * learned weights and only swap in a fresh trainer/optimizer.
   */
  private rebuildEngine(resetWeights = true): void {
    const inDim = this.task === 'classification' ? 2 : 1;
    if (resetWeights || !this.net) {
      const outAct = this.task === 'classification' ? Sigmoid : Identity;
      const act = activationByName(this.config.activation);
      this.net = mlp(inDim, this.config.hidden, 1, act, outAct, new Rng(this.weightSeed));
    }
    this.loss = this.task === 'classification' ? BCE : MSE;

    if (this.trainData.X.rows > 0 && this.net) {
      this.trainer = new Trainer(
        this.net,
        this.loss,
        this.trainData.X,
        this.trainData.Y,
        {
          lr: this.config.lr,
          momentum: this.config.momentum,
          l2: this.config.l2,
          batchSize: this.config.batchSize,
        },
        new Rng(this.shuffleSeed),
      );
    } else {
      this.trainer = null;
    }
    this.hideSolved();
    this.redrawArena();
    this.refreshTelemetry();
    this.redrawDiagram();
  }

  // ----- Actions ---------------------------------------------------------
  private toggleRun(): void {
    if (!this.trainer) return;
    this.running = !this.running;
    this.refreshRunButton();
  }

  private stepOnce(): void {
    if (!this.trainer) return;
    this.trainer.step(40);
    this.redrawAll();
  }

  private resetWeights(): void {
    this.running = false;
    this.weightSeed++;
    this.shuffleSeed++;
    this.rebuildEngine();
    this.refreshRunButton();
  }

  private newData(): void {
    this.running = false;
    if (this.mode === 'sandbox' && this.sandboxDatasetId === DRAW_ID) {
      this.customPoints = [];
    } else {
      this.dataNudge++;
    }
    this.weightSeed++;
    this.shuffleSeed++;
    this.rebuildData();
    this.rebuildEngine();
    this.refreshRunButton();
  }

  // ----- Run loop --------------------------------------------------------
  private loop(): void {
    if (this.running && this.trainer) {
      this.trainer.step(STEPS_PER_FRAME);
      this.frame++;
      this.redrawArena();
      this.refreshTelemetry();
      if (this.frame % 6 === 0) this.redrawDiagram();
    }
    requestAnimationFrame(() => this.loop());
  }

  private redrawAll(): void {
    this.redrawArena();
    this.refreshTelemetry();
    this.redrawDiagram();
  }

  private predictor = (X: Matrix): Matrix => this.net!.forward(X);

  private redrawArena(): void {
    const canvas = this.refs.arenaCanvas;
    const { ctx, w, h } = fitCanvas(canvas);
    ctx.clearRect(0, 0, w, h);
    if (!this.net) return;

    if (this.task === 'classification') {
      const vp = makeViewport(w, h, 12, -1.3, 1.3, -1.3, 1.3);
      drawBoundary(ctx, vp, this.predictor, 52);
      if (this.trainData.X.rows > 0) drawPoints(ctx, vp, this.trainData.X, this.trainData.Y);
    } else {
      const vp = makeViewport(w, h, 16, -1.15, 1.15, -1.15, 1.15);
      this.drawRegressionGrid(ctx, vp.width, vp.height);
      drawRegression(ctx, vp, this.trainData.X, this.trainData.Y, this.predictor);
    }
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

  /**
   * Update all telemetry from a single evaluation pass: the step counter, loss
   * readout + trace, the primary/secondary metric tiles, and — in challenge mode —
   * completion detection (persisted once, on first solve).
   */
  private refreshTelemetry(): void {
    const steps = this.trainer?.stepCount ?? 0;
    this.refs.statSteps.textContent = steps.toLocaleString();
    const lossHist = this.trainer?.lossHistory ?? [];
    const lastLoss = lossHist.length ? lossHist[lossHist.length - 1]! : NaN;
    this.refs.statLoss.textContent = Number.isFinite(lastLoss) ? lastLoss.toFixed(4) : '—';

    const lc = fitCanvas(this.refs.lossCanvas);
    drawLossChart(lc.ctx, lossHist, lc.w, lc.h);

    if (!this.net || this.trainData.X.rows === 0) {
      this.refs.statPrimary.value.textContent = '—';
      this.refs.statSecondary.value.textContent = '—';
      this.refs.statPrimary.value.classList.remove('good');
      return;
    }

    const target = this.currentTarget();
    let testMetric: number;
    let passed = false;
    if (this.task === 'classification') {
      testMetric = accuracy(this.net.forward(this.testData.X), this.testData.Y);
      const trainAcc = accuracy(this.net.forward(this.trainData.X), this.trainData.Y);
      this.refs.statPrimary.value.textContent = `${(testMetric * 100).toFixed(1)}%`;
      this.refs.statSecondary.value.textContent = `${(trainAcc * 100).toFixed(1)}%`;
      passed = target?.kind === 'accuracy' ? testMetric >= target.min : false;
    } else {
      testMetric = mseMetric(this.net.forward(this.testData.X), this.testData.Y);
      const trainMse = mseMetric(this.net.forward(this.trainData.X), this.trainData.Y);
      this.refs.statPrimary.value.textContent = testMetric.toFixed(4);
      this.refs.statSecondary.value.textContent = trainMse.toFixed(4);
      passed = target?.kind === 'mse' ? testMetric <= target.max : false;
    }
    this.refs.statPrimary.value.classList.toggle('good', passed);

    if (this.mode === 'challenge' && target) this.handleCompletion(passed, target, testMetric);
  }

  private handleCompletion(
    passed: boolean,
    target: NonNullable<ReturnType<App['currentTarget']>>,
    testMetric: number,
  ): void {
    const ch = CHALLENGES[this.challengeIndex]!;
    if (!passed) {
      this.hideSolved();
      return;
    }
    this.showSolved();
    if (!isCompleted(this.progress, ch.id)) {
      const score = target.kind === 'accuracy' ? testMetric : Math.max(0, 1 - testMetric);
      this.progress = markComplete(this.progress, ch.id, score);
      saveProgress(this.progress);
      this.refreshChallengeList();
    }
  }

  private redrawDiagram(): void {
    if (!this.net) return;
    const dc = fitCanvas(this.refs.diagramCanvas);
    drawNetwork(dc.ctx, this.net, dc.w, dc.h);
  }

  // ----- Challenge progression -------------------------------------------
  private currentTarget(): Challenge['target'] | null {
    return this.mode === 'challenge' ? CHALLENGES[this.challengeIndex]!.target : null;
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

  // ----- Dynamic control rendering ---------------------------------------
  private renderControlsForMode(): void {
    this.refreshChallengeList();
    this.renderControlsExtra();
    this.renderArchRows();
  }

  private refreshChallengeList(): void {
    const list = this.refs.challengeList;
    clear(list);
    if (this.mode !== 'challenge') {
      list.style.display = 'none';
      return;
    }
    list.style.display = 'flex';
    CHALLENGES.forEach((ch, i) => {
      const unlocked = isUnlocked(this.progress, CHALLENGES, i);
      const done = isCompleted(this.progress, ch.id);
      const cls = ['challenge'];
      if (i === this.challengeIndex) cls.push('active');
      if (!unlocked) cls.push('locked');
      if (done) cls.push('done');
      const targetTxt =
        ch.target.kind === 'accuracy'
          ? `≥ ${(ch.target.min * 100).toFixed(0)}% accuracy`
          : `≤ ${ch.target.max} MSE`;
      const item = el(
        'button',
        {
          class: cls.join(' '),
          attrs: unlocked ? {} : { disabled: 'true' },
          on: {
            click: () => {
              if (!unlocked) return;
              this.challengeIndex = i;
              this.dataNudge = 0;
              this.loadFromMode(true);
            },
          },
        },
        [
          el('span', { class: 'challenge-idx' }, [done ? '✓' : String(i + 1)]),
          el('span', { class: 'challenge-meta' }, [
            el('span', { class: 'challenge-name' }, [ch.title]),
            el('span', { class: 'challenge-target' }, [targetTxt]),
          ]),
          el('span', { class: 'challenge-state' }, [unlocked ? (done ? '' : '○') : '🔒']),
        ],
      );
      list.append(item);
    });
  }

  private renderControlsExtra(): void {
    const box = this.refs.controlsExtra;
    clear(box);
    if (this.mode !== 'sandbox') {
      box.style.display = 'none';
      return;
    }
    box.style.display = 'flex';

    const select = el(
      'select',
      {
        attrs: { 'aria-label': 'Dataset' },
        on: {
          change: (e) => {
            this.sandboxDatasetId = (e.target as HTMLSelectElement).value;
            this.dataNudge = 0;
            this.loadFromMode(false);
          },
        },
      },
      [
        ...DATASET_GENERATORS.map((g) =>
          el('option', { attrs: { value: g.id } }, [`${g.label} (${g.kind === 'classification' ? 'classify' : 'regress'})`]),
        ),
        el('option', { attrs: { value: DRAW_ID } }, ['Draw your own (classify)']),
      ],
    );
    (select as HTMLSelectElement).value = this.sandboxDatasetId;

    box.append(
      el('div', { class: 'field' }, [
        el('div', { class: 'field-row' }, [el('span', { class: 'field-label' }, ['Dataset'])]),
        select,
      ]),
    );

    if (this.sandboxDatasetId === DRAW_ID) {
      const classSeg = el('div', { class: 'seg' }, [
        el(
          'button',
          {
            class: this.drawClass === 0 ? 'seg-btn active' : 'seg-btn',
            on: { click: () => this.setDrawClass(0) },
          },
          ['Class A'],
        ),
        el(
          'button',
          {
            class: this.drawClass === 1 ? 'seg-btn active' : 'seg-btn',
            on: { click: () => this.setDrawClass(1) },
          },
          ['Class B'],
        ),
      ]);
      box.append(
        el('div', { class: 'field' }, [
          el('div', { class: 'field-row' }, [
            el('span', { class: 'field-label' }, ['Click the screen to add points']),
          ]),
          classSeg,
        ]),
      );
    }
  }

  private setDrawClass(c: number): void {
    this.drawClass = c;
    this.renderControlsExtra();
  }

  private renderArchRows(): void {
    const rows = this.refs.archRows;
    clear(rows);
    const inDim = this.task === 'classification' ? 2 : 1;
    rows.append(this.endcap(String(inDim), 'in'));

    this.config.hidden.forEach((width, idx) => {
      rows.append(el('span', { class: 'arrow' }, ['→']));
      rows.append(this.layerChip(width, idx));
    });

    rows.append(el('span', { class: 'arrow' }, ['→']));
    rows.append(this.endcap('1', 'out'));

    const addBtn = rows.parentElement?.querySelector('.add-layer') as HTMLButtonElement | null;
    if (addBtn) addBtn.disabled = this.config.hidden.length >= MAX_LAYERS;
  }

  private endcap(n: string, label: string): HTMLElement {
    return el('div', { class: 'endcap' }, [
      el('span', { class: 'cap-n' }, [n]),
      el('span', { class: 'cap-l' }, [label]),
    ]);
  }

  private layerChip(width: number, idx: number): HTMLElement {
    return el('div', { class: 'layer-chip' }, [
      el('span', { class: 'chip-w' }, [String(width)]),
      el('div', { class: 'chip-controls' }, [
        el('button', { class: 'chip-btn', on: { click: () => this.changeWidth(idx, -2) } }, ['−']),
        el('button', { class: 'chip-btn', on: { click: () => this.changeWidth(idx, +2) } }, ['+']),
      ]),
      el('button', { class: 'chip-remove', on: { click: () => this.removeLayer(idx) } }, ['remove']),
    ]);
  }

  private addLayer(): void {
    if (this.config.hidden.length >= MAX_LAYERS) return;
    this.config.hidden = [...this.config.hidden, 8];
    this.renderArchRows();
    this.rebuildEngine();
  }

  private removeLayer(idx: number): void {
    this.config.hidden = this.config.hidden.filter((_, i) => i !== idx);
    this.renderArchRows();
    this.rebuildEngine();
  }

  private changeWidth(idx: number, delta: number): void {
    const next = [...this.config.hidden];
    next[idx] = Math.max(MIN_WIDTH, Math.min(MAX_WIDTH, next[idx]! + delta));
    this.config.hidden = next;
    this.renderArchRows();
    this.rebuildEngine();
  }

  // ----- Header / legend / fields ----------------------------------------
  private refreshArenaHeader(): void {
    if (this.mode === 'challenge') {
      const ch = CHALLENGES[this.challengeIndex]!;
      this.refs.arenaSub.textContent = `challenge ${this.challengeIndex + 1} / ${CHALLENGES.length} · ${ch.task}`;
      this.refs.arenaTitle.textContent = ch.title;
      const targetTxt =
        ch.target.kind === 'accuracy'
          ? `Goal: reach ≥ ${(ch.target.min * 100).toFixed(0)}% accuracy on the held-out test set.`
          : `Goal: get test MSE ≤ ${ch.target.max}.`;
      clear(this.refs.note);
      this.refs.note.append(
        el('span', { class: 'note-icon' }, ['¶']),
        el('div', { class: 'note-body', html: `<strong>${targetTxt}</strong> ${escapeHtml(ch.why)}` }),
      );
    } else {
      this.refs.arenaSub.textContent = `sandbox · ${this.task}`;
      const name =
        this.sandboxDatasetId === DRAW_ID
          ? 'Draw your own'
          : DATASET_GENERATORS.find((g) => g.id === this.sandboxDatasetId)?.label ?? '—';
      this.refs.arenaTitle.textContent = name;
      clear(this.refs.note);
      this.refs.note.append(
        el('span', { class: 'note-icon' }, ['¶']),
        el('div', { class: 'note-body' }, [
          'Free play — change the architecture and hyperparameters and watch how the model copes.',
        ]),
      );
    }
    this.refreshLegend();
    this.refreshStatus();
  }

  private refreshLegend(): void {
    const lg = this.refs.legend;
    clear(lg);
    if (this.task === 'classification') {
      lg.append(
        legendKey(rgba(CLASS_A), 'Class A'),
        legendKey(rgba(CLASS_B), 'Class B'),
        el('span', { class: 'key' }, ['shaded = model confidence']),
      );
    } else {
      lg.append(
        legendKey('rgba(150,170,210,0.9)', 'samples'),
        legendKey(rgba(CLASS_B), 'model prediction'),
      );
    }
  }

  private refreshFields(): void {
    const setRange = (key: string, value: number, fmt: (v: number) => string): void => {
      const input = this.refs.fieldValues[`${key}__input`] as HTMLInputElement | undefined;
      const valEl = this.refs.fieldValues[key];
      if (input) input.value = String(value);
      if (valEl) valEl.textContent = fmt(value);
    };
    setRange('lr', this.config.lr, (v) => v.toFixed(3));
    setRange('momentum', this.config.momentum, (v) => v.toFixed(2));
    setRange('l2', this.config.l2, (v) => v.toFixed(4));
    const actSel = this.refs.fieldValues['activation'] as HTMLSelectElement | undefined;
    if (actSel) actSel.value = this.config.activation;
    const batchSel = this.refs.fieldValues['batch'] as HTMLSelectElement | undefined;
    if (batchSel) batchSel.value = String(this.config.batchSize);
  }

  private refreshRunButton(): void {
    const btn = this.refs.runBtn;
    btn.disabled = !this.trainer;
    btn.classList.toggle('running', this.running);
    btn.textContent = this.running ? 'Pause' : 'Train';
    this.refreshStatus();
  }

  // ----- Interaction -----------------------------------------------------
  private handleScreenClick(e: MouseEvent): void {
    if (this.mode !== 'sandbox' || this.sandboxDatasetId !== DRAW_ID) return;
    const canvas = this.refs.arenaCanvas;
    const rect = canvas.getBoundingClientRect();
    const sx = e.clientX - rect.left;
    const sy = e.clientY - rect.top;
    const vp = makeViewport(rect.width, rect.height, 12, -1.3, 1.3, -1.3, 1.3);
    const { wx, wy } = screenToWorld(vp, sx, sy);
    this.customPoints.push({ x: wx, y: wy, label: this.drawClass });
    this.rebuildData();
    this.rebuildEngine(false); // keep the boundary; let it adapt as points are added
    this.refreshRunButton();
  }

  private handleResize(): void {
    this.redrawAll();
  }
}

// ----- Small view helpers -------------------------------------------------
function legendKey(color: string, label: string): HTMLElement {
  return el('span', { class: 'key' }, [
    el('span', { class: 'swatch', style: { background: color } }),
    label,
  ]);
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

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
