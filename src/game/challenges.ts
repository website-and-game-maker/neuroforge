import {
  gaussianBlobs,
  xorData,
  circlesData,
  moonsData,
  spiralsData,
  regLinear,
  regSine,
  type Dataset,
  type TaskKind,
} from '../data/datasets';
import type { OptimizerName } from '../engine/optimizer';

/** Pass condition: a minimum accuracy (classification) or a maximum MSE (regression). */
export type Target = { kind: 'accuracy'; min: number } | { kind: 'mse'; max: number };

/** Suggested starting configuration the player can tweak. */
export interface Starter {
  hidden: number[];
  activation: string;
  lr: number;
  l2: number;
  batchSize: number;
  /** Only read by SGD. */
  momentum: number;
  optimizer: OptimizerName;
}

export interface Challenge {
  id: string;
  title: string;
  task: TaskKind;
  makeData: (seed: number) => Dataset;
  trainSeed: number;
  testSeed: number;
  target: Target;
  /** The reasoning hint — *why* this dataset needs what it needs. */
  why: string;
  starter: Starter;
}

export const CHALLENGES: Challenge[] = [
  {
    id: 'blobs',
    title: 'Two Blobs',
    task: 'classification',
    makeData: (s) => gaussianBlobs(200, s),
    trainSeed: 11,
    testSeed: 12,
    target: { kind: 'accuracy', min: 0.97 },
    why: 'These two clouds are linearly separable — a single straight line splits them. A network with NO hidden layer (plain logistic regression) is enough. Start simple.',
    starter: { hidden: [], activation: 'tanh', lr: 0.5, l2: 0, batchSize: 16, momentum: 0.9, optimizer: 'sgd' },
  },
  {
    id: 'xor',
    title: 'XOR',
    task: 'classification',
    makeData: (s) => xorData(240, s),
    trainSeed: 21,
    testSeed: 22,
    target: { kind: 'accuracy', min: 0.95 },
    why: "The classic. No single straight line can separate XOR — opposite corners share a class. You need at least one HIDDEN layer so the network can combine two lines into a bent boundary.",
    starter: { hidden: [4], activation: 'tanh', lr: 0.3, l2: 0, batchSize: 16, momentum: 0.9, optimizer: 'sgd' },
  },
  {
    id: 'circles',
    title: 'Circles',
    task: 'classification',
    makeData: (s) => circlesData(240, s),
    trainSeed: 31,
    testSeed: 32,
    target: { kind: 'accuracy', min: 0.95 },
    why: 'A ring inside a ring. The boundary is a closed curve, not a line. A hidden layer with a nonlinearity can wrap a circular boundary around the inner class.',
    starter: { hidden: [8], activation: 'tanh', lr: 0.3, l2: 0, batchSize: 16, momentum: 0.9, optimizer: 'sgd' },
  },
  {
    id: 'moons',
    title: 'Moons',
    task: 'classification',
    makeData: (s) => moonsData(240, s),
    trainSeed: 41,
    testSeed: 42,
    target: { kind: 'accuracy', min: 0.93 },
    why: 'Two interleaving crescents. The boundary has to curve between them. A small hidden layer handles it; try widening it if the moons stay tangled.',
    starter: { hidden: [8, 8], activation: 'tanh', lr: 0.2, l2: 0, batchSize: 16, momentum: 0.9, optimizer: 'sgd' },
  },
  {
    id: 'spirals',
    title: 'Spirals',
    task: 'classification',
    makeData: (s) => spiralsData(300, s),
    trainSeed: 51,
    testSeed: 52,
    target: { kind: 'accuracy', min: 0.9 },
    why: 'Two arms winding around each other — the boundary spirals too, so it needs real capacity. Tip: ReLU tiles the plane with sharp creases that wrap a winding boundary far better than smooth tanh here. Go wide (e.g. 32×32) and train a while. This is the hard one.',
    starter: { hidden: [32, 32], activation: 'relu', lr: 0.05, l2: 0, batchSize: 16, momentum: 0.9, optimizer: 'sgd' },
  },
  {
    id: 'reg-linear',
    title: 'Fit a Line',
    task: 'regression',
    makeData: (s) => regLinear(160, s),
    trainSeed: 61,
    testSeed: 62,
    target: { kind: 'mse', max: 0.02 },
    why: 'Regression now: predict a continuous value, not a class. The data is a straight line, so an Identity output with no hidden layer (linear regression) nails it.',
    starter: { hidden: [], activation: 'tanh', lr: 0.1, l2: 0, batchSize: 16, momentum: 0.9, optimizer: 'sgd' },
  },
  {
    id: 'reg-sine',
    title: 'Fit a Wave',
    task: 'regression',
    makeData: (s) => regSine(200, s),
    trainSeed: 71,
    testSeed: 72,
    target: { kind: 'mse', max: 0.02 },
    why: 'A sine wave bends — a straight line can never fit it. Hidden tanh units each contribute a bend; stack enough of them and the network traces the curve.',
    starter: { hidden: [16, 16], activation: 'tanh', lr: 0.05, l2: 0, batchSize: 16, momentum: 0.9, optimizer: 'sgd' },
  },
];
