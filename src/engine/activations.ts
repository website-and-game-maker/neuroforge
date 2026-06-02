/**
 * Scalar activation functions and their derivatives w.r.t. the pre-activation input.
 *
 * `df` is expressed as a function of the *input* x (not the output), which is what an
 * activation layer needs for backprop: dL/dx = df(x) ⊙ dL/dy.
 */
export interface Activation {
  readonly name: string;
  f(x: number): number;
  df(x: number): number;
}

export const ReLU: Activation = {
  name: 'relu',
  f: (x) => (x > 0 ? x : 0),
  df: (x) => (x > 0 ? 1 : 0),
};

export const Tanh: Activation = {
  name: 'tanh',
  f: (x) => Math.tanh(x),
  df: (x) => {
    const t = Math.tanh(x);
    return 1 - t * t;
  },
};

function sigmoid(x: number): number {
  // Numerically stable for large |x|.
  if (x >= 0) {
    const z = Math.exp(-x);
    return 1 / (1 + z);
  }
  const z = Math.exp(x);
  return z / (1 + z);
}

export const Sigmoid: Activation = {
  name: 'sigmoid',
  f: sigmoid,
  df: (x) => {
    const s = sigmoid(x);
    return s * (1 - s);
  },
};

export const Identity: Activation = {
  name: 'identity',
  f: (x) => x,
  df: () => 1,
};

const REGISTRY: Record<string, Activation> = {
  relu: ReLU,
  tanh: Tanh,
  sigmoid: Sigmoid,
  identity: Identity,
};

export function activationByName(name: string): Activation {
  const act = REGISTRY[name.toLowerCase()];
  if (!act) {
    throw new Error(`Unknown activation: ${name}`);
  }
  return act;
}
