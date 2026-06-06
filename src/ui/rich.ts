/**
 * Render the light inline markup used in curriculum/coach content into real DOM nodes
 * (no innerHTML, so content is never interpreted as HTML). Supports **bold** and `code`.
 */
export function richInline(text: string): Node[] {
  const out: Node[] = [];
  // Split on **bold** and `code`, keeping the delimiters' captured groups.
  const re = /\*\*([^*]+)\*\*|`([^`]+)`/g;
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(document.createTextNode(text.slice(last, m.index)));
    if (m[1] !== undefined) {
      const strong = document.createElement('strong');
      strong.textContent = m[1];
      out.push(strong);
    } else if (m[2] !== undefined) {
      const code = document.createElement('code');
      code.className = 'inl-code';
      code.textContent = m[2];
      out.push(code);
    }
    last = re.lastIndex;
  }
  if (last < text.length) out.push(document.createTextNode(text.slice(last)));
  return out;
}

/** Build an element of `tag` with `text` rendered as rich inline content. */
export function renderRich(tag: keyof HTMLElementTagNameMap, className: string, text: string): HTMLElement {
  const node = document.createElement(tag);
  if (className) node.className = className;
  for (const child of richInline(text)) node.appendChild(child);
  return node;
}
