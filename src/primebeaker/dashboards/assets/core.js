// Shared presentation primitives for dashboard views. No run names or API keys.
export const el = id => document.getElementById(id);
export const escape = value => String(value ?? '—').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
export const number = value => value == null ? '—' : Number(value).toLocaleString(undefined, {maximumFractionDigits: 1});
export const duration = value => value == null ? '—' : value >= 60 ? (value / 60).toFixed(1) + ' min' : value.toFixed(2) + ' s';
export const palette = ['#56B4E9', '#E69F00', '#009E73', '#CC79A7', '#F0E442', '#D55E00'];

export async function getJSON(url) {
  const endpoint = new URL(url.replace(/^\//, ''), new URL('../', import.meta.url));
  endpoint.searchParams.set('t', Date.now());
  const response = await fetch(endpoint, {cache: 'no-store'});
  if (!response.ok) {
    const message = await response.json().catch(() => ({}));
    throw Error(message.error || ('HTTP ' + response.status));
  }
  return response.json();
}

export function renderPlot(target, series, {label = 'Metric', scale = 1, xMode = 'step', zeroFloor = false, smoothingSteps = 1, xLabel: suppliedXLabel} = {}) {
  series = series.map((s, i) => ({...s, color: s.color || palette[i % palette.length], points: (s.points || [])
    .filter(p => Number.isFinite(p.step)).map(p => ({x: p.step, y: Number.isFinite(p.value) ? p.value / scale : null, raw: p.value}))}));
  const points = series.flatMap(s => s.points).filter(p => p.y !== null);
  if (!points.length) {
    target.innerHTML = '<p class="hint">No recorded points yet.</p>';
    return;
  }
  let xmin = Math.min(...points.map(p => p.x)), xmax = Math.max(...points.map(p => p.x));
  let ymin = Math.min(...points.map(p => p.y)), ymax = Math.max(...points.map(p => p.y));
  if (xmin === xmax) { xmin = Math.max(0, xmin - 1); xmax += 1; }
  const pad = (ymax - ymin) * .12 || Math.max(Math.abs(ymax) * .08, .1);
  ymin -= pad; ymax += pad;
  if (zeroFloor) ymin = 0;
  const xLabel = suppliedXLabel || (xMode === 'time' ? 'Time (local)' : 'Training step (W&B)');
  const xText = value => xMode === 'time' ? new Date(value).toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'}) : value;
  const w = 560, h = 310, left = 65, right = 16, top = 20, bottom = 48;
  const px = x => left + (x - xmin) / (xmax - xmin) * (w - left - right);
  const py = y => h - bottom - (y - ymin) / (ymax - ymin) * (h - top - bottom);
  let svg = '<svg viewBox="0 0 ' + w + ' ' + h + '" role="img" aria-label="' + escape(label + ' against ' + xLabel) + '">';
  for (let i = 0; i <= 4; i++) {
    const value = ymin + (ymax - ymin) * i / 4, y = py(value);
    svg += '<line x1="' + left + '" y1="' + y + '" x2="' + (w - right) + '" y2="' + y + '" stroke="#2d3b50"/>';
    svg += '<text x="' + (left - 9) + '" y="' + (y + 4) + '" text-anchor="end" fill="#a6b4c9" font-size="13">' + number(value) + '</text>';
  }
  const ticks = [...new Set(Array.from({length: 5}, (_, i) => Math.round(xmin + (xmax - xmin) * i / 4)))];
  for (const value of ticks) svg += '<text x="' + px(value) + '" y="' + (h - bottom + 24) + '" text-anchor="middle" fill="#a6b4c9" font-size="13">' + escape(xText(value)) + '</text>';
  svg += '<text x="' + ((w + left - right) / 2) + '" y="' + (h - 5) + '" text-anchor="middle" fill="#a6b4c9" font-size="13">' + escape(xLabel) + '</text>';
  for (const source of series) {
    const draw = (points, opacity, width, markers) => {
      let segment = [];
      const flush = () => {
        if (segment.length > 1) svg += '<polyline fill="none"' + (source.lineStyle === 'dashed' ? ' stroke-dasharray="8 5"' : '') + ' stroke="' + source.color + '" stroke-opacity="' + opacity + '" stroke-width="' + width + '" points="' + segment.map(p => px(p.x) + ',' + py(p.y)).join(' ') + '"/>';
        segment = [];
      };
      for (const p of points) {
        if (p.y === null) { flush(); continue; }
        segment.push(p);
        if (markers) svg += '<circle cx="' + px(p.x) + '" cy="' + py(p.y) + '" r="' + (xMode === 'time' ? 2 : smoothingSteps > 1 ? 2 : 4) + '" fill="' + source.color + '" fill-opacity="' + opacity + '"><title>' + escape(source.name + ' · ' + (xMode === 'time' ? new Date(p.x).toLocaleString() : 'step ' + p.x) + ' · ' + number(p.raw)) + '</title></circle>';
      }
      flush();
    };
    if (smoothingSteps > 1 && xMode === 'step') {
      let window = [];
      const smooth = source.points.map(p => {
        if (p.y === null) { window = []; return p; }
        window = window.filter(v => v.x > p.x - smoothingSteps);
        window.push(p);
        return {...p, y: window.reduce((sum, v) => sum + v.y, 0) / window.length};
      });
      draw(source.points, .18, 1, true);
      draw(smooth, 1, 2.5, false);
    } else draw(source.points, 1, 2.5, true);
  }
  target.innerHTML = svg + '</svg>';
}
