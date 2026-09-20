import type { StudioConfig } from '../ui/studio';
import type { TaskKind } from '../data/datasets';
import type { Target } from '../game/challenges';

/* ===========================================================================
   Instructional content. Pure data — no DOM, no engine. Rendered by the views.
   Light markup supported by the renderer: **bold**, *italic*, and `code`.
   ========================================================================= */

/** A single guided lesson in Learn mode: it configures the playground, then teaches. */
export interface Lesson {
  id: string;
  title: string;
  /** One-sentence takeaway shown up top. */
  bigIdea: string;
  /** Teaching paragraphs. */
  body: string[];
  task: TaskKind;
  /** Dataset generator id (see data/datasets DATASET_GENERATORS) the lesson loads. */
  dataset: string;
  config: StudioConfig;
  /** The concrete action to take in this lesson. */
  doThis: string;
  /** Optional pass condition that marks the lesson "got it". */
  check?: Target;
  /** What to notice / why it matters. */
  reflect: string;
}

const C = (
  hidden: number[],
  activation: string,
  lr: number,
  extra: Partial<StudioConfig> = {},
): StudioConfig => ({
  hidden,
  activation,
  lr,
  l2: 0,
  batchSize: 16,
  momentum: 0.9,
  optimizer: 'sgd',
  ...extra,
});

export const LESSONS: Lesson[] = [
  {
    id: 'l-what',
    title: 'What you are actually building',
    bigIdea: 'A neural network here is just a function: feed it a point, it returns a guess.',
    body: [
      'The dark square is a **map of every possible point**. Left–right is one number, up–down is another. Every dot has a position, and a true colour: blue (Class A) or orange (Class B).',
      'The network’s job: look at a point’s position and **guess its colour**. The shaded background shows its guess *everywhere at once* — blue regions mean “I think A here”, orange means “I think B”. The fuzzy seam between them is the **decision boundary**.',
      'You don’t program the boundary. You press Train, and the network nudges its internal numbers (its **weights**) over and over until its guesses match the dots.',
    ],
    task: 'classification',
    dataset: 'blobs',
    config: C([], 'tanh', 0.5),
    doThis: 'Press **Train**. Watch the background split into a blue side and an orange side.',
    check: { kind: 'accuracy', min: 0.97 },
    reflect:
      'These two clouds can be split by a single straight line, so the network needs no hidden layers at all — this is the simplest possible case.',
  },
  {
    id: 'l-io',
    title: 'The 2 inputs and 1 output, concretely',
    bigIdea: 'Input neurons = the point’s coordinates. The output neuron = the probability of orange.',
    body: [
      'Open the **Inspector** on the right. The leftmost column has **2 input neurons**. They are not mysterious — input 1 is literally the point’s **x** (horizontal position) and input 2 is its **y** (vertical position). That’s the only thing the network ever sees.',
      'The rightmost column is **1 output neuron**. It emits a single number between 0 and 1: the network’s score for **orange**. Near 0 it’s calling the point blue, near 1 orange, and exactly 0.5 means the point sits right on the boundary — which is precisely where the seam is drawn.',
      'So the whole network is a function `f(x, y) → probability`. The shaded arena is just that function evaluated at every pixel.',
    ],
    task: 'classification',
    dataset: 'blobs',
    config: C([4], 'tanh', 0.4),
    doThis:
      'Train briefly, then **hover your mouse over the arena**. The Inspector lights up each neuron with its value for the point under your cursor, and shows the final probability. Move toward each cloud and watch the output swing toward 0 or 1.',
    reflect:
      'Hovering feeds one point through the network by hand. That flow — coordinates in, numbers through the middle, probability out — is the entire idea of a “forward pass”.',
  },
  {
    id: 'l-xor',
    title: 'Why one line is not enough',
    bigIdea: 'Some data can’t be split by any straight line. That’s what hidden layers are for.',
    body: [
      'This is **XOR**: orange sits in two opposite corners, blue in the other two. Try to separate them with one straight line — you can’t. No angle works.',
      'The network starts with **no hidden layer**, so it can *only* draw one straight line. It will get stuck around 50–75% no matter how long it trains. That’s not a bug; it’s a capacity limit.',
      'A **hidden layer** lets the network draw several lines and then **combine** them. Two lines can carve an X-shaped boundary that XOR actually needs.',
    ],
    task: 'classification',
    dataset: 'xor',
    config: C([], 'tanh', 0.3),
    doThis:
      'First press **Train** and watch it fail (stuck, one flat line). Then **add a hidden layer** (the “+ add hidden layer” button), Reset, and Train again. Now it solves it.',
    check: { kind: 'accuracy', min: 0.92 },
    reflect:
      'This is the single most important idea in the whole app: depth buys you the ability to bend. Feeling *why* one line fails here is worth more than any definition.',
  },
  {
    id: 'l-neurons',
    title: 'Each neuron is one crease',
    bigIdea: 'More neurons = more folds the boundary can use. Watch them add up.',
    body: [
      'This is **Circles**: an inner ring of one class, an outer ring of the other. The boundary it needs is a closed loop — definitely not a line.',
      'Each hidden neuron contributes **one fold** in the plane (with tanh, a soft bend; with ReLU, a sharp crease). One neuron isn’t enough to close a loop. Several, working together, can.',
      'In the Inspector, **hover a hidden neuron** to see the region *it* responds to across the arena (orange where it fires). A first-layer neuron carves a soft straight ridge; the final boundary blends all of them together.',
    ],
    task: 'classification',
    dataset: 'circles',
    config: C([1], 'tanh', 0.3),
    doThis:
      'Train with just **1 neuron** — it can’t wrap the ring. Bump the hidden layer up to **8** neurons, Reset, Train. Hover individual neurons in the Inspector to see each one’s contribution.',
    check: { kind: 'accuracy', min: 0.93 },
    reflect:
      'Capacity is not magic — it’s a budget of folds. Too few and the boundary can’t take the shape it needs.',
  },
  {
    id: 'l-lr',
    title: 'Learning rate: the size of each step',
    bigIdea: 'Too big and it explodes; too small and it crawls. There’s a sweet spot.',
    body: [
      'Training takes small steps downhill on the loss. The **learning rate** is the step size.',
      'Too **high** and each step overshoots — the loss bounces or rockets upward (the Coach will warn you). Too **low** and it inches along, taking forever. Most of “tuning” is finding the rate that drops the loss fast without blowing up.',
      'Watch the **loss trace** in the telemetry panel: a healthy run slides smoothly down; a too-high rate looks jagged or climbs.',
    ],
    task: 'classification',
    dataset: 'moons',
    config: C([8, 8], 'tanh', 0.05),
    doThis:
      'Train at the low rate (0.05) and notice how slowly the loss falls. Now drag **Learning rate** up high (past 0.4), Reset, Train, and watch it destabilize. Settle on a rate in between.',
    reflect:
      'You changed nothing about the network — same neurons, same data. Only the step size. That alone is the difference between “learns fast” and “never learns”.',
  },
  {
    id: 'l-optimizer',
    title: 'The optimizer: how each step is chosen',
    bigIdea: 'The learning rate says how far to step. The optimizer decides how that distance is spent.',
    body: [
      'So far every weight in the network has moved by the same rule: **step size × its gradient**. That’s **SGD**. It works, but it forces one step size to suit thousands of weights at once — and different weights want wildly different step sizes. A weight deep in the network with tiny gradients crawls, while one with big gradients is already overshooting.',
      '**Adam** fixes that by tracking two running averages per weight: the average *gradient* (the same “keep going that way” idea as **momentum**) and the average *squared* gradient — how big that weight’s gradients have been lately. It then divides the step by the square root of the second one. Loud weights get reined in, quiet weights get amplified, and every weight ends up moving at a useful pace.',
      'Because the step is normalized by the gradient’s own size, Adam’s step length is roughly the learning rate itself — which is why its useful range sits about ten times lower than SGD’s. Switching optimizers moves the slider for you.',
      'The catch: Adam getting there fast isn’t the same as getting somewhere better. On easy data both land in the same place, and Adam only earns its keep when the problem is hard or badly scaled — like these spirals.',
    ],
    task: 'classification',
    dataset: 'spirals',
    config: C([32, 32], 'relu', 0.05),
    doThis:
      'Train on **SGD** and count how long the spiral arms take to separate. Now switch **Optimizer** to **Adam**, press Reset, and train again — same network, same data, same number of steps.',
    check: { kind: 'accuracy', min: 0.95 },
    reflect:
      'Same architecture, same gradients, same loss — only the rule that turns gradients into steps changed. Optimizers don’t make a network smarter; they make the descent less dependent on you guessing one number correctly.',
  },
  {
    id: 'l-overfit',
    title: 'Underfitting vs overfitting',
    bigIdea: 'Too little capacity misses the pattern; too much memorizes the noise.',
    body: [
      'Two crescents (**Moons**) with some noise. There are two ways to do badly.',
      '**Underfit:** too few neurons — the boundary is too stiff to follow the curves, and *both* train and test accuracy stay low.',
      '**Overfit:** way too many neurons and no regularization — the boundary contorts to snag every noisy point. Train accuracy looks great, but **test accuracy (held-out points) drops**. The gap between train and test is the tell. A little **L2 weight decay** pulls the boundary back toward something smooth.',
    ],
    task: 'classification',
    dataset: 'moons',
    config: C([8, 8], 'tanh', 0.15),
    doThis:
      'Train and compare **train vs test** accuracy in the telemetry tiles. Then crank neurons way up with L2 at 0 and watch the gap widen. Add a touch of L2 and watch test recover.',
    check: { kind: 'accuracy', min: 0.9 },
    reflect:
      'The goal was never to ace the dots on screen — it’s to capture the *rule* that generalizes to points you’ve never seen. That’s why we always hold out a test set.',
  },
  {
    id: 'l-reg',
    title: 'Regression: predicting a number, not a class',
    bigIdea: 'Same machine, different output: now the single output neuron emits a value, and the curve is its prediction.',
    body: [
      'Switch tasks. Here there’s **1 input** (x) and the output is no longer a probability — it’s a **predicted number** y. The bright curve is the network’s output for every x.',
      'A sine wave is curved, so a straight line can’t fit it. Hidden **tanh** neurons each add a bend; stack enough and the curve traces the sine.',
      'There are no “classes” here, so instead of accuracy we measure **MSE** (mean squared error) — how far the curve sits from the dots on average. Lower is better.',
    ],
    task: 'regression',
    dataset: 'sine',
    config: C([16, 16], 'tanh', 0.05),
    doThis:
      'Press Train and watch the flat line bend into a wave. Try removing neurons to see it lose the ability to curve.',
    check: { kind: 'mse', max: 0.02 },
    reflect:
      'Classification and regression are the same network with a different last step: squash to a probability, or read the raw number. Everything in the middle is identical.',
  },
];

/* ----- Concept glossary --------------------------------------------------- */

export interface Concept {
  term: string;
  short: string;
  long: string;
}

export const CONCEPTS: Concept[] = [
  {
    term: 'Input neuron',
    short: 'One number the network reads in.',
    long: 'For the 2-D arena there are two inputs: the point’s x (horizontal) and y (vertical) position. For regression there’s one input, x. Inputs are the only thing the network ever sees.',
  },
  {
    term: 'Output neuron',
    short: 'The network’s answer.',
    long: 'For classification it’s a probability in [0,1] (chance the point is orange); ≥0.5 is called orange. For regression it’s the predicted value itself.',
  },
  {
    term: 'Hidden neuron',
    short: 'A learned feature in the middle.',
    long: 'Each hidden neuron computes a weighted sum of its inputs, then squashes it through an activation. A first-layer neuron splits the plane with one soft ridge; deeper neurons combine those ridges into curved features. More neurons = more pieces the boundary can use.',
  },
  {
    term: 'Weight',
    short: 'How strongly one neuron listens to another.',
    long: 'Every connection has a weight. Training is the process of adjusting all the weights so the outputs match the data. In the Inspector, edge thickness shows magnitude and colour shows sign.',
  },
  {
    term: 'Bias',
    short: 'A neuron’s baseline offset.',
    long: 'Added to the weighted sum before the activation. It lets a neuron shift its fold left/right instead of always passing through the origin.',
  },
  {
    term: 'Activation',
    short: 'The squashing function that adds nonlinearity.',
    long: 'ReLU keeps positives and zeroes negatives (sharp creases). tanh squashes to (−1,1) (smooth bends). Sigmoid squashes to (0,1) and is used on the output for probabilities. Without an activation, stacking layers would still only make a straight line.',
  },
  {
    term: 'Learning rate',
    short: 'The size of each training step.',
    long: 'Too high overshoots and the loss diverges; too low and training crawls. The most impactful single knob.',
  },
  {
    term: 'Momentum',
    short: 'Carries velocity between steps.',
    long: 'Lets updates build up speed in a consistent direction and smooths out noisy gradients — usually speeds up training.',
  },
  {
    term: 'L2 (weight decay)',
    short: 'Gently pulls weights toward zero.',
    long: 'A regularizer that discourages over-complex boundaries, trading a little training accuracy for better generalization to unseen points.',
  },
  {
    term: 'Optimizer',
    short: 'The rule that turns gradients into weight updates.',
    long: 'Backprop says which way each weight should move; the optimizer decides how far. SGD uses one step size for every weight; Adam gives each weight its own, scaled by that weight’s recent gradient size.',
  },
  {
    term: 'Adam',
    short: 'Per-weight adaptive step sizes.',
    long: 'Tracks a running average of each weight’s gradient (like momentum) and of its squared gradient, then divides the step by the square root of the second. Weights with big gradients take smaller steps, quiet weights take bigger ones. Its steps are already normalized, so it wants a much smaller learning rate than SGD.',
  },
  {
    term: 'Batch size',
    short: 'How many points per update.',
    long: 'Each step estimates the gradient from this many randomly chosen points. Smaller is noisier but can escape bad spots; larger is smoother.',
  },
  {
    term: 'Loss',
    short: 'How wrong the network is right now.',
    long: 'Training minimizes this. Cross-entropy for classification, mean-squared-error for regression. The loss trace should slide downward.',
  },
  {
    term: 'Decision boundary',
    short: 'Where the model flips its guess.',
    long: 'The set of points where the output is exactly 0.5 — the seam between the blue and orange regions.',
  },
  {
    term: 'Overfitting',
    short: 'Memorizing instead of learning.',
    long: 'The boundary contorts to fit every noisy point; training accuracy is high but held-out test accuracy drops. Fix with fewer neurons or some L2.',
  },
];

/* ----- Input / output explainer (request: "what do the neurons represent") - */

export interface IOExplainer {
  title: string;
  lines: string[];
}

export function ioExplainer(task: TaskKind): IOExplainer {
  if (task === 'classification') {
    return {
      title: 'Reading this network',
      lines: [
        '**2 inputs** = the point’s position: input 1 is x (left↔right), input 2 is y (down↔up).',
        '**Hidden neurons** = learned folds. Each bends the plane a little; together they shape the boundary.',
        '**1 output** = probability the point is **orange** (0 → blue, 1 → orange, 0.5 → the boundary).',
        'The shaded arena is this function drawn for every point at once.',
      ],
    };
  }
  return {
    title: 'Reading this network',
    lines: [
      '**1 input** = x, the position along the horizontal axis.',
      '**Hidden neurons** = learned bends that let the curve flex.',
      '**1 output** = the predicted value y (not a probability — the number itself).',
      'The bright curve is this function drawn across every x.',
    ],
  };
}

/* ----- First-run tutorial (coachmarks over real UI) ----------------------- */

export interface TutorialStep {
  /** CSS selector of the element to spotlight (first match). Empty = centered. */
  target: string;
  title: string;
  body: string;
}

export const TUTORIAL: TutorialStep[] = [
  {
    target: '',
    title: 'Welcome to NeuroForge',
    body: 'You’re about to build and train a real neural network — coded from scratch, no libraries. This 60-second tour shows where everything is. Use ← → or the buttons.',
  },
  {
    target: '[data-tour="modes"]',
    title: 'Four ways to play',
    body: '**Learn** is a guided course. **Challenges** are puzzles. **Sandbox** is free play. **Versus** pits a network-builder against a saboteur. New here? Start in Learn.',
  },
  {
    target: '[data-tour="arena"]',
    title: 'The arena',
    body: 'This is the data. Each dot is a point with a true colour. The shaded background is the network’s current guess for every position. Hover it to probe the network live.',
  },
  {
    target: '[data-tour="arch"]',
    title: 'Build the network',
    body: 'Add or remove hidden layers and change their width. The endcaps show the **2 inputs** (x, y) and **1 output** (the probability). This is the network’s shape.',
  },
  {
    target: '[data-tour="run"]',
    title: 'Train it',
    body: 'Press Train to start learning (Pause to stop). Step takes one batch at a time. Reset re-randomizes the weights. New data reshuffles the points.',
  },
  {
    target: '[data-tour="coach"]',
    title: 'The Coach is watching',
    body: 'As you train, the Coach reads the situation and tells you *what’s happening and which knob to turn* — overshooting, underfitting, overfitting, and how to fix it.',
  },
  {
    target: '[data-tour="inspector"]',
    title: 'Look inside',
    body: 'The Inspector shows every neuron and weight. Hover a neuron to see the region it responds to; hover the arena to watch a point flow through. This is how you really understand what changed.',
  },
  {
    target: '',
    title: 'You’re set',
    body: 'Head to **Learn** for the guided course, or dive into Challenges. You can replay this tour anytime from the “?” in the top bar.',
  },
];
