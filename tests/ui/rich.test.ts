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

  it('parses *italic* into <em>, and lets **bold** win on the same asterisks', () => {
    const nodes = richInline('plain *lean* and **loud**');
    expect((nodes[1] as HTMLElement).tagName).toBe('EM');
    expect(nodes[1]!.textContent).toBe('lean');
    expect((nodes[3] as HTMLElement).tagName).toBe('STRONG');
    expect(nodes[3]!.textContent).toBe('loud');
  });

  it('leaves stray asterisks alone rather than italicising across them', () => {
    for (const text of ['3 * 4 * 5', 'a lone * asterisk', '2*3']) {
      const nodes = richInline(text);
      expect(nodes.length).toBe(1);
      expect(nodes[0]!.nodeType).toBe(3); // a single text node — nothing was marked up
      expect(nodes[0]!.textContent).toBe(text);
    }
  });

  it('renders multi-word italics spanning punctuation', () => {
    const nodes = richInline('the *boundary, not the dots* here');
    expect((nodes[1] as HTMLElement).tagName).toBe('EM');
    expect(nodes[1]!.textContent).toBe('boundary, not the dots');
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
