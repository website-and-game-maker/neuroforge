// @vitest-environment jsdom
import { describe, it, expect, vi } from 'vitest';
import { el, clear } from '../../src/ui/dom';

describe('el', () => {
  it('builds a nested tree with classes and text', () => {
    const tree = el('div', { class: 'card' }, [el('span', { text: 'hi' })]);
    expect(tree.tagName).toBe('DIV');
    expect(tree.className).toBe('card');
    expect(tree.children.length).toBe(1);
    expect(tree.firstElementChild!.tagName).toBe('SPAN');
    expect(tree.firstElementChild!.textContent).toBe('hi');
  });

  it('applies attrs, dataset, and style', () => {
    const node = el('button', {
      attrs: { type: 'button', 'aria-label': 'go' },
      dataset: { id: 'x1' },
      style: { color: 'red' },
    });
    expect(node.getAttribute('type')).toBe('button');
    expect(node.getAttribute('aria-label')).toBe('go');
    expect(node.dataset.id).toBe('x1');
    expect(node.style.color).toBe('red');
  });

  it('wires event handlers', () => {
    const onClick = vi.fn();
    const node = el('button', { on: { click: onClick } });
    node.dispatchEvent(new Event('click'));
    expect(onClick).toHaveBeenCalledOnce();
  });

  it('appends string children as text nodes and skips falsy', () => {
    const node = el('p', {}, ['a', false, null, undefined, 'b']);
    expect(node.textContent).toBe('ab');
  });
});

describe('clear', () => {
  it('removes all children', () => {
    const node = el('div', {}, [el('span'), el('span')]);
    expect(node.children.length).toBe(2);
    clear(node);
    expect(node.children.length).toBe(0);
  });
});
