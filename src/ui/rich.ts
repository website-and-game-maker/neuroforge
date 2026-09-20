/**
 * Render the light inline markup used in curriculum/coach content into real DOM nodes
 * (no innerHTML, so content is never interpreted as HTML). Supports **bold**, *italic*
 * and `code`.
 */
export function richInline(text: string): Node[] {
  const out: Node[] = [];
  // Split on **bold**, *italic* and `code`, keeping the delimiters' captured groups.
  // **bold** is listed first so it wins over *italic* on the same asterisks; the italic
  // body may not begin or end with whitespace, so prose like "3 * 4 * 5" stays literal.
  const re = /\*\*([^*]+)\*\*|\*(\S|\S[^*]*\S)\*|`([^`]+)`/g;
  const wrap = (tag: 'strong' | 'em' | 'code', content: string, className?: string): void => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    node.textContent = content;
    out.push(node);
  };
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(document.createTextNode(text.slice(last, m.index)));
    if (m[1] !== undefined) wrap('strong', m[1]);
    else if (m[2] !== undefined) wrap('em', m[2]);
    else if (m[3] !== undefined) wrap('code', m[3], 'inl-code');
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
