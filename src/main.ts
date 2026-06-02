import './styles/theme.css';
import './styles/app.css';
import { App } from './ui/app';

const root = document.querySelector<HTMLDivElement>('#app');
if (root) {
  root.className = 'app';
  new App().mount(root);
}
