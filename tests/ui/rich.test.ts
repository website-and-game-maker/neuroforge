// @vitest-environment jsdom
import { describe, it, expect } from 'vitest';
import { richInline, renderRich } from '../../src/ui/rich';

describe('richInline', () => {
  it('parses **bold** and `code` into elements, rest as text', () => {
    const nodes = richInline('a **b** and `c` end');
    expect(nodes.length).toBe(5);
    expect(nodes[0]!.textContent).toBe('a ');
    expect((nodes[1] as HTMLElement).tagName).toBe('STRONG');
    expect(nodes[1]!.textContent).toBe('b');
    expect((nodes[3] as HTMLElement).tagName).toBe('CODE');
    expect(nodes[3]!.textContent).toBe('c');
  });

  it('never interprets content as HTML', () => {
    const nodes = richInline('<img src=x onerror=alert(1)> **safe**');
    const text = nodes.map((n) => n.textContent).join('');
    expect(text).toContain('<img');
    // The literal angle brackets survive as text, not as an element.
    expect(nodes.some((n) => n.nodeType === 1 && (n as HTMLElement).tagName === 'IMG')).toBe(false);
  });

  it('renderRich wraps in the requested tag with a class', () => {
    const p = renderRich('p', 'lesson-p', 'hello **world**');
    expect(p.tagName).toBe('P');
    expect(p.className).toBe('lesson-p');
    expect(p.querySelector('strong')!.textContent).toBe('world');
  });
});
