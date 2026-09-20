import type { StudioConfig } from '../ui/studio';
import type { TaskKind } from '../data/datasets';
import type { Target } from './challenges';

export type CoachTone = 'idle' | 'info' | 'warn' | 'good' | 'success';

export interface CoachTip {
  tone: CoachTone;
  title: string;
  body: string;
}

export interface CoachInput {
  task: TaskKind;
  steps: number;
  trainScore: number;
  testScore: number;
  lossHistory: number[];
  config: StudioConfig;
  /** The optional pass condition for the current challenge. */
  target?: Target;
  /** How many training points exist (datasets with very few points behave oddly). */
  pointCount: number;
}

const sum = (a: number[]): number => a.reduce((s, v) => s + v, 0);
const mean = (a: number[]): number => (a.length ? sum(a) / a.length : NaN);

/**
 * Classify the recent loss curve into a coarse trend the coach can reason about.
 * Compares the mean of the most recent window against the window before it.
 */
function lossTrend(history: number[]): 'rising' | 'flat' | 'falling' | 'unknown' {
  const w = 24;
  if (history.length < 2 * w) return 'unknown';
  const recent = mean(history.slice(-w));
  const prior = mean(history.slice(-2 * w, -w));
  if (!Number.isFinite(recent) || !Number.isFinite(prior)) return 'unknown';
  const rel = (recent - prior) / (Math.abs(prior) + 1e-9);
  // A higher 'rising' bar avoids flagging noisy minibatch blips as divergence.
  if (rel > 0.06) return 'rising';
  if (rel < -0.04) return 'falling';
  return 'flat';
}

function totalNeurons(hidden: number[]): number {
  return sum(hidden);
}

/**
 * Produce a single, situation-aware coaching tip from the live training state. The whole
 * point is to replace "I pressed Train and it worked, but I don't know why" with a
 * concrete, causal nudge: *what* is happening and *which knob* changes it.
 */
export function coach(input: CoachInput): CoachTip {
  const { task, steps, trainScore, testScore, lossHistory, config, target } = input;
  const trend = lossTrend(lossHistory);
  const hidden = config.hidden;
  const depth = hidden.length;
  const neurons = totalNeurons(hidden);

  if (steps === 0) {
    return {
      tone: 'idle',
      title: 'Ready to train',
      body:
        task === 'classification'
          ? 'Press Train. The shaded background is the model guessing a class for every point on the plane — watch it bend to fit the dots.'
          : 'Press Train. The bright line is the model’s predicted output for every input x — watch it bend toward the samples.',
    };
  }

  // --- Diverging: the single most common beginner failure. ---
  if (trend === 'rising' || !Number.isFinite(input.lossHistory[input.lossHistory.length - 1]!)) {
    // Momentum is an SGD-only knob, so don't send Adam players looking for a slider
    // that isn't on screen.
    const alsoTry =
      config.optimizer === 'adam'
        ? 'or raise the batch size for a steadier gradient'
        : 'or reduce momentum';
    return {
      tone: 'warn',
      title: 'Learning rate too high',
      body: `The loss is climbing instead of falling — the optimizer is overshooting. Lower the learning rate (try ${(config.lr / 2).toFixed(3)}) ${alsoTry}, then Reset and train again.`,
    };
  }

  if (task === 'classification') {
    const acc = trainScore;
    const goal = target?.kind === 'accuracy' ? target.min : 0.95;

    if (acc >= goal) {
      const gap = trainScore - testScore;
      if (gap > 0.08) {
        return {
          tone: 'good',
          title: 'Fits — but watch overfitting',
          body: `Training accuracy is ${(trainScore * 100).toFixed(0)}% but held-out test is ${(testScore * 100).toFixed(0)}%. The boundary is memorizing noise. Trim a few neurons or add a little L2 weight decay to generalize better.`,
        };
      }
      return {
        tone: 'success',
        title: 'Solved cleanly',
        body: `${(testScore * 100).toFixed(0)}% on points it never trained on. The boundary captured the real pattern, not just the dots. Try the next challenge — or break this one in Sandbox to see what fails.`,
      };
    }

    // Underfitting — diagnose by capacity.
    if (depth === 0) {
      return {
        tone: 'info',
        title: 'No hidden layer = one straight line',
        body: `With zero hidden layers the network can only split the plane with a single straight line. Stuck near ${(acc * 100).toFixed(0)}%? This data needs a *bent* boundary — add a hidden layer.`,
      };
    }
    if (trend === 'flat' && acc < goal - 0.03) {
      if (neurons < 16) {
        return {
          tone: 'info',
          title: 'Not enough capacity',
          body: `Accuracy stalled at ${(acc * 100).toFixed(0)}%. The boundary can’t bend sharply enough. Widen a layer (more neurons) or add another — each neuron adds one more crease the boundary can use.`,
        };
      }
      // With capacity to spare, a plateau is a descent problem, not a model problem —
      // and swapping the update rule is the biggest lever left.
      if (config.optimizer !== 'adam') {
        return {
          tone: 'info',
          title: 'Plenty of neurons — change how it steps',
          body: `You have capacity (${neurons} neurons) but it’s plateaued at ${(acc * 100).toFixed(0)}%. SGD moves every weight by the same step size, and some of them need a different one. Switch **Optimizer** to **Adam** (it picks a step size per weight), or raise the learning rate a little and try ReLU for sharper creases.`,
        };
      }
      return {
        tone: 'info',
        title: 'Plenty of neurons — help it learn',
        body: `You have capacity (${neurons} neurons) but it’s plateaued at ${(acc * 100).toFixed(0)}%. Nudge the learning rate, or switch the activation (ReLU carves sharp creases; tanh makes smooth bends).`,
      };
    }
    return {
      tone: 'info',
      title: 'Learning…',
      body: `Accuracy ${(acc * 100).toFixed(0)}% and the loss is ${trend === 'falling' ? 'still dropping' : 'settling'}. Give it more steps — if it stops improving, you’ll need more capacity.`,
    };
  }

  // --- Regression ---
  const mse = trainScore;
  const goal = target?.kind === 'mse' ? target.max : 0.02;
  if (mse <= goal) {
    return {
      tone: 'success',
      title: 'Curve fit',
      body: `Test MSE ${testScore.toFixed(4)} — the predicted line traces the samples. Each hidden neuron contributed one bend; together they shaped the curve.`,
    };
  }
  if (depth === 0) {
    return {
      tone: 'info',
      title: 'A line can’t bend',
      body: `With no hidden layer the model can only draw a straight line (linear regression). If the data curves, MSE will floor out around ${mse.toFixed(3)}. Add hidden tanh neurons so it can bend.`,
    };
  }
  if (trend === 'flat' && mse > goal * 1.5) {
    return {
      tone: 'info',
      title: 'Underfitting the curve',
      body: `MSE stuck at ${mse.toFixed(4)}. Add hidden neurons (more bends) or train longer. tanh activations are ideal for smooth curves.`,
    };
  }
  return {
    tone: 'info',
    title: 'Fitting…',
    body: `MSE ${mse.toFixed(4)} and ${trend === 'falling' ? 'still falling' : 'settling'}. Keep training; raise the learning rate if it crawls.`,
  };
}
