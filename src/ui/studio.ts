import { mlp, type Network } from '../engine/network';
import { activationByName, Sigmoid, Identity } from '../engine/activations';
import { BCE, MSE, type Loss } from '../engine/losses';
import { Rng } from '../engine/rng';
import { Trainer, accuracy, mseMetric } from '../engine/trainer';
import { OPTIMIZER_NAMES, type OptimizerName } from '../engine/optimizer';
import { Matrix } from '../engine/matrix';
import type { Dataset, TaskKind } from '../data/datasets';

/** The full set of knobs a player can turn. Shared by every mode. */
export interface StudioConfig {
  hidden: number[];
  activation: string;
  lr: number;
  l2: number;
  batchSize: number;
  momentum: number;
  optimizer: OptimizerName;
}

/**
 * Presentation facts about each update rule: what to call it, whether the Momentum
 * slider applies, and the learning-rate range that actually behaves for it. Adam
 * normalises its step by the gradient's own scale, so its useful lr band sits about an
 * order of magnitude below SGD's — showing both on one slider would make one of them
 * feel broken.
 */
export interface OptimizerInfo {
  name: OptimizerName;
  label: string;
  usesMomentum: boolean;
  defaultLr: number;
  lrMin: number;
  lrMax: number;
  lrStep: number;
  /** One-line "why you'd pick this", shown under the selector. */
  blurb: string;
}

export const OPTIMIZER_INFO: Record<OptimizerName, OptimizerInfo> = {
  sgd: {
    name: 'sgd',
    label: 'SGD + momentum',
    usesMomentum: true,
    defaultLr: 0.2,
    lrMin: 0.005,
    lrMax: 0.6,
    lrStep: 0.005,
    blurb: 'One step size for every weight. Simple, and the learning rate matters a lot.',
  },
  adam: {
    name: 'adam',
    label: 'Adam',
    usesMomentum: false,
    defaultLr: 0.03,
    lrMin: 0.001,
    lrMax: 0.2,
    lrStep: 0.001,
    blurb: 'Adapts the step size per weight. Usually trains faster and forgives a badly tuned lr.',
  },
};

export function optimizerInfo(name: string): OptimizerInfo {
  return OPTIMIZER_INFO[name as OptimizerName] ?? OPTIMIZER_INFO.sgd;
}

export { OPTIMIZER_NAMES, type OptimizerName };

export interface StudioMetrics {
  task: TaskKind;
  steps: number;
  loss: number;
  /** Accuracy in [0,1] for classification, MSE for regression. */
  trainScore: number;
  testScore: number;
  hasData: boolean;
}

export function defaultConfig(): StudioConfig {
  return { hidden: [8], activation: 'tanh', lr: 0.2, l2: 0, batchSize: 16, momentum: 0.9, optimizer: 'sgd' };
}

/**
 * A self-contained neural-network training session: owns the network, trainer, data and
 * seeds, and knows how to (re)build itself and report metrics. Deliberately DOM-free so
 * it can back the challenge/sandbox playground *and* both sides of Versus mode.
 */
export class Studio {
  task: TaskKind = 'classification';
  config: StudioConfig;
  trainData: Dataset;
  testData: Dataset;

  net: Network | null = null;
  trainer: Trainer | null = null;
  loss: Loss = BCE;
  weightSeed = 1;
  shuffleSeed = 1;

  constructor(config: StudioConfig = defaultConfig()) {
    this.config = { ...config };
    this.trainData = emptyDataset('classification');
    this.testData = emptyDataset('classification');
  }

  get inDim(): number {
    return this.task === 'classification' ? 2 : 1;
  }

  setConfig(patch: Partial<StudioConfig>): void {
    this.config = { ...this.config, ...patch };
  }

  setData(train: Dataset, test: Dataset): void {
    this.trainData = train;
    this.testData = test;
  }

  /** Advance the weight + shuffle seeds so the next rebuild starts somewhere new. */
  reseed(): void {
    this.weightSeed++;
    this.shuffleSeed++;
  }

  /**
   * (Re)build the network and trainer. Structural changes (architecture, activation,
   * task, data) reset the weights; hyperparameter tweaks pass `resetWeights = false` to
   * keep the learned weights and only swap in a fresh optimizer/trainer.
   */
  rebuild(resetWeights = true): void {
    if (resetWeights || !this.net) {
      const outAct = this.task === 'classification' ? Sigmoid : Identity;
      const act = activationByName(this.config.activation);
      this.net = mlp(this.inDim, this.config.hidden, 1, act, outAct, new Rng(this.weightSeed));
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
          optimizer: this.config.optimizer,
          l2: this.config.l2,
          batchSize: this.config.batchSize,
        },
        new Rng(this.shuffleSeed),
      );
    } else {
      this.trainer = null;
    }
  }

  step(n: number): void {
    this.trainer?.step(n);
  }

  get steps(): number {
    return this.trainer?.stepCount ?? 0;
  }

  forward = (X: Matrix): Matrix => this.net!.forward(X);

  metrics(): StudioMetrics {
    const lossHist = this.trainer?.lossHistory ?? [];
    const loss = lossHist.length ? lossHist[lossHist.length - 1]! : NaN;
    const hasData = !!this.net && this.trainData.X.rows > 0;
    if (!hasData) {
      return { task: this.task, steps: this.steps, loss, trainScore: NaN, testScore: NaN, hasData };
    }
    if (this.task === 'classification') {
      return {
        task: this.task,
        steps: this.steps,
        loss,
        trainScore: accuracy(this.net!.forward(this.trainData.X), this.trainData.Y),
        testScore: accuracy(this.net!.forward(this.testData.X), this.testData.Y),
        hasData,
      };
    }
    return {
      task: this.task,
      steps: this.steps,
      loss,
      trainScore: mseMetric(this.net!.forward(this.trainData.X), this.trainData.Y),
      testScore: mseMetric(this.net!.forward(this.testData.X), this.testData.Y),
      hasData,
    };
  }
}

export function emptyDataset(kind: TaskKind): Dataset {
  const inDim = kind === 'classification' ? 2 : 1;
  return { kind, name: '—', inDim, X: Matrix.zeros(0, inDim), Y: Matrix.zeros(0, 1) };
}
