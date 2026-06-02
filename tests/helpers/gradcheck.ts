/**
 * Numerical gradient-checking utilities.
 *
 * The engine's correctness rests on these: we compare each hand-derived analytic
 * gradient against a centered finite-difference estimate of the same quantity.
 */

/**
 * Centered finite-difference gradient of `loss` with respect to each entry of the
 * (mutable) `param` buffer. `loss` must perform a *fresh* forward pass each call so
 * that perturbations to `param` are reflected.
 */
export function numericalGradient(
  loss: () => number,
  param: Float64Array,
  eps = 1e-5,
): Float64Array {
  const grad = new Float64Array(param.length);
  for (let i = 0; i < param.length; i++) {
    const orig = param[i]!;
    param[i] = orig + eps;
    const lp = loss();
    param[i] = orig - eps;
    const lm = loss();
    param[i] = orig;
    grad[i] = (lp - lm) / (2 * eps);
  }
  return grad;
}

/** Maximum relative error between two equal-length vectors. */
export function maxRelError(a: Float64Array, b: Float64Array): number {
  if (a.length !== b.length) {
    throw new Error(`maxRelError: length mismatch ${a.length} vs ${b.length}`);
  }
  let worst = 0;
  for (let i = 0; i < a.length; i++) {
    const denom = Math.max(1e-8, Math.abs(a[i]!) + Math.abs(b[i]!));
    const rel = Math.abs(a[i]! - b[i]!) / denom;
    if (rel > worst) worst = rel;
  }
  return worst;
}

/** Sum of elementwise products — a simple scalar reduction of a forward output. */
export function dot(aData: Float64Array, bData: Float64Array): number {
  let s = 0;
  for (let i = 0; i < aData.length; i++) s += aData[i]! * bData[i]!;
  return s;
}
