/** Tiny declarative element builder — keeps UI construction readable without a framework. */
export type Child = Node | string | number | null | undefined | false;

export interface ElProps {
  class?: string;
  text?: string;
  html?: string;
  attrs?: Record<string, string>;
  style?: Record<string, string>;
  dataset?: Record<string, string>;
  on?: Partial<{ [K in keyof HTMLElementEventMap]: (e: HTMLElementEventMap[K]) => void }>;
}

export function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  props: ElProps = {},
  children: Child[] = [],
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (props.class) node.className = props.class;
  if (props.text !== undefined) node.textContent = props.text;
  if (props.html !== undefined) node.innerHTML = props.html;
  if (props.attrs) {
    for (const [k, v] of Object.entries(props.attrs)) node.setAttribute(k, v);
  }
  if (props.style) {
    for (const [k, v] of Object.entries(props.style)) node.style.setProperty(k, v);
  }
  if (props.dataset) {
    for (const [k, v] of Object.entries(props.dataset)) node.dataset[k] = v;
  }
  if (props.on) {
    for (const [k, handler] of Object.entries(props.on)) {
      node.addEventListener(k, handler as EventListener);
    }
  }
  for (const child of children) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : String(child));
  }
  return node;
}

/** Remove all children of a node. */
export function clear(node: Element): void {
  while (node.firstChild) node.removeChild(node.firstChild);
}
