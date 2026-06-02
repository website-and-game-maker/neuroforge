import type { Matrix } from './matrix';
import { Dense, ActivationLayer, type Layer, type Param } from './layers';
import type { Activation } from './activations';
import type { Rng } from './rng';

/** A sequential stack of layers: forward composes them, backward reverses them. */
export class Network {
  constructor(readonly layers: Layer[]) {}

  forward(x: Matrix): Matrix {
    let a = x;
    for (const layer of this.layers) a = layer.forward(a);
    return a;
  }

  backward(dOut: Matrix): Matrix {
    let g = dOut;
    for (let i = this.layers.length - 1; i >= 0; i--) {
      g = this.layers[i]!.backward(g);
    }
    return g;
  }

  params(): Param[] {
    const out: Param[] = [];
    for (const layer of this.layers) {
      for (const p of layer.params()) out.push(p);
    }
    return out;
  }
}

/**
 * Build a multilayer perceptron: `hidden` widths each get a Dense + activation, then a
 * final Dense + output activation (Sigmoid for binary classification, Identity for
 * regression).
 */
export function mlp(
  inDim: number,
  hidden: number[],
  outDim: number,
  act: Activation,
  outAct: Activation,
  rng: Rng,
): Network {
  const layers: Layer[] = [];
  let prev = inDim;
  for (const width of hidden) {
    layers.push(new Dense(prev, width, rng));
    layers.push(new ActivationLayer(act));
    prev = width;
  }
  layers.push(new Dense(prev, outDim, rng));
  layers.push(new ActivationLayer(outAct));
  return new Network(layers);
}
