import { el } from './dom';
import { renderRich } from './rich';
import type { TutorialStep } from '../content/curriculum';

const SEEN_KEY = 'neuroforge.tutorialSeen';

export function hasSeenTutorial(): boolean {
  try {
    return localStorage.getItem(SEEN_KEY) === '1';
  } catch {
    return false;
  }
}

function markSeen(): void {
  try {
    localStorage.setItem(SEEN_KEY, '1');
  } catch {
    /* ignore */
  }
}

/**
 * A lightweight guided overlay: dims the page, spotlights the element named by each
 * step's CSS selector, and floats a card beside it. Keyboard: ← → to move, Esc to exit.
 */
export class Tutorial {
  private root: HTMLElement | null = null;
  private spotlight!: HTMLElement;
  private card!: HTMLElement;
  private i = 0;
  private keyHandler = (e: KeyboardEvent): void => {
    if (e.key === 'ArrowRight' || e.key === 'Enter') this.go(1);
    else if (e.key === 'ArrowLeft') this.go(-1);
    else if (e.key === 'Escape') this.close();
  };
  private resizeHandler = (): void => this.position();

  constructor(private readonly steps: TutorialStep[]) {}

  start(): void {
    if (this.root) return;
    this.i = 0;
    this.spotlight = el('div', { class: 'tut-spotlight' });
    this.card = el('div', { class: 'tut-card' });
    const overlay = el('div', { class: 'tut-overlay' }, [this.spotlight, this.card]);
    overlay.addEventListener('click', (e) => {
      if (e.target === overlay) this.go(1);
    });
    this.root = overlay;
    document.body.appendChild(overlay);
    window.addEventListener('keydown', this.keyHandler);
    window.addEventListener('resize', this.resizeHandler);
    this.render();
  }

  private go(delta: number): void {
    const next = this.i + delta;
    if (next < 0) return;
    if (next >= this.steps.length) {
      this.close();
      return;
    }
    this.i = next;
    this.render();
  }

  private render(): void {
    const step = this.steps[this.i]!;
    this.card.replaceChildren(
      el('div', { class: 'tut-eyebrow' }, [`Tour · ${this.i + 1}/${this.steps.length}`]),
      el('h3', { class: 'tut-title' }, [step.title]),
      renderRich('p', 'tut-body', step.body),
      el('div', { class: 'tut-actions' }, [
        el(
          'button',
          { class: 'tut-skip', on: { click: () => this.close() } },
          ['Skip tour'],
        ),
        el('div', { class: 'tut-nav' }, [
          this.i > 0 &&
            el('button', { class: 'btn ghost tut-btn', on: { click: () => this.go(-1) } }, ['Back']),
          el('button', { class: 'btn primary tut-btn', on: { click: () => this.go(1) } }, [
            this.i === this.steps.length - 1 ? 'Done' : 'Next',
          ]),
        ]),
      ]),
    );
    this.position();
  }

  private position(): void {
    const step = this.steps[this.i]!;
    const target = step.target ? document.querySelector(step.target) : null;
    const margin = 10;
    if (target) {
      const r = target.getBoundingClientRect();
      this.spotlight.style.display = 'block';
      this.spotlight.style.left = `${r.left - margin}px`;
      this.spotlight.style.top = `${r.top - margin}px`;
      this.spotlight.style.width = `${r.width + 2 * margin}px`;
      this.spotlight.style.height = `${r.height + 2 * margin}px`;
      target.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
      this.placeCardNear(r);
    } else {
      this.spotlight.style.display = 'none';
      this.card.style.left = '50%';
      this.card.style.top = '50%';
      this.card.style.transform = 'translate(-50%, -50%)';
    }
  }

  private placeCardNear(r: DOMRect): void {
    const cw = 320;
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    this.card.style.transform = 'none';
    // Prefer right of the target; fall back to left, then below.
    let left = r.right + 18;
    if (left + cw > vw - 12) left = r.left - cw - 18;
    if (left < 12) left = Math.min(Math.max(12, r.left), vw - cw - 12);
    let top = r.top;
    const ch = this.card.offsetHeight || 220;
    if (top + ch > vh - 12) top = Math.max(12, vh - ch - 12);
    this.card.style.left = `${left}px`;
    this.card.style.top = `${top}px`;
  }

  private close(): void {
    markSeen();
    window.removeEventListener('keydown', this.keyHandler);
    window.removeEventListener('resize', this.resizeHandler);
    this.root?.remove();
    this.root = null;
  }
}
